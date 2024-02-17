#include "rtdetr_detect.hpp"
namespace tensorrt_infer
{
    namespace rtdetr_cuda
    {
        void RTDETRDetect::initParameters(const std::string &engine_file, float score_thr)
        {
            if (!file_exist(engine_file))
            {
                INFO("Error: engine_file is not exist!!!");
                exit(0);
            }

            this->model_info = std::make_shared<ModelInfo>();
            // Configuración de parámetros entrantes
            model_info->m_modelPath = engine_file;
            model_info->m_postProcCfg.confidence_threshold_ = score_thr;

            this->model_ = trt::infer::load(engine_file); // Carga el modelo serializado
            this->model_->print();                        // Imprime información básica sobre el modelo

            // Obtener la información del tamaño de entrada
            auto input_dim = this->model_->get_network_dims(0); // Información de dimensión de entrada
            model_info->m_preProcCfg.infer_batch_size = input_dim[0]; // Batch
            model_info->m_preProcCfg.network_input_channels_ = input_dim[1]; // Canales
            model_info->m_preProcCfg.network_input_height_ = input_dim[2]; // Alto
            model_info->m_preProcCfg.network_input_width_ = input_dim[3]; // Ancho
            model_info->m_preProcCfg.network_input_numel = input_dim[1] * input_dim[2] * input_dim[3]; // Dimension de entrada (canales * Alto * Ancho)
            model_info->m_preProcCfg.isdynamic_model_ = this->model_->has_dynamic_dim(); // Determina si el modelo es dinamico o estatico
            model_info->m_preProcCfg.normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::RGB); // Parametros para la normalización y preprocesamiento de la imágen

            // Obtener información sobre el tamaño de salida
            auto output_dim = this->model_->get_network_dims(1);  // Información de dimensión de salida
            model_info->m_postProcCfg.bbox_head_dims_ = output_dim; // ([bbox], [classes])
            model_info->m_postProcCfg.bbox_head_dims_output_numel_ = output_dim[1] * output_dim[2]; // Object_queries(300) * output([bbox(4) + classes(80)]) = 300*84
            if (model_info->m_postProcCfg.num_classes_ == 0) // Si el numero de clases no se definio anteriormente
                model_info->m_postProcCfg.num_classes_ = model_info->m_postProcCfg.bbox_head_dims_[2] - 4; // Numero de clases (output(84) - 4(bbox)) = 80
            model_info->m_postProcCfg.MAX_IMAGE_BOXES = output_dim[1]; // Object_queries
            model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT = model_info->m_postProcCfg.MAX_IMAGE_BOXES * model_info->m_postProcCfg.NUM_BOX_ELEMENT; // output final = 300*(7)
            INFO("Classes: %d \n", model_info->m_postProcCfg.num_classes_);

            CHECK(cudaStreamCreate(&cu_stream)); // Crear el cuda stream
        }

        RTDETRDetect::~RTDETRDetect()
        {
            CHECK(cudaStreamDestroy(cu_stream)); // Destruir el cuda stream
        }

        void RTDETRDetect::adjust_memory(int batch_size)
        {
            // Solicitar la memoria utilizada por la entrada y la salida del modelo
            input_buffer_.gpu(batch_size * model_info->m_preProcCfg.network_input_numel);           // Solicitar memoria GPU para entrada de modelo por lotes
            bbox_predict_.gpu(batch_size * model_info->m_postProcCfg.bbox_head_dims_output_numel_); // Solicitar memoria GPU para la salida del modelo por lotes

            /* Solicite la memoria que debe almacenarse cuando el modelo se analiza en cuadros. +32 se debe a que el primer número debe establecerse 
            en el número de cuadros para evitar el desbordamiento de la memoria. */
            output_boxarray_.gpu(batch_size * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT));
            output_boxarray_.cpu(batch_size * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT));

            if ((int)preprocess_buffers_.size() < batch_size)
            {
                for (int i = preprocess_buffers_.size(); i < batch_size; ++i)
                    preprocess_buffers_.push_back(make_shared<Memory<unsigned char>>()); // Agregar buffers de memoria para cada lote(batch_size)
            }
        }

        // Preprocesado con GPU
        void RTDETRDetect::preprocess_gpu(int ibatch, const Image &image,
                                          shared_ptr<Memory<unsigned char>> preprocess_buffer, cudaStream_t stream_)
        {
            if (image.channels != model_info->m_preProcCfg.network_input_channels_)
            {
                INFO("Warning : %d", model_info->m_preProcCfg.infer_batch_size);
                INFO("Warning : %d", model_info->m_preProcCfg.network_input_channels_);
                INFO("Warning : %d", model_info->m_preProcCfg.network_input_height_);
                INFO("Warning : %d", model_info->m_preProcCfg.network_input_width_);
                INFO("Warning : Number of channels wanted differs from number of channels in the actual image \n");
                exit(-1);
            }

            size_t size_image = image.width * image.height * image.channels;
            float *input_device = input_buffer_.gpu() + ibatch * model_info->m_preProcCfg.network_input_numel; // Puntero de memoria de la GPU para el batch actual

            uint8_t *image_device = preprocess_buffer->gpu(size_image); // Se crea para solicitar memoria para la imagen en GPU
            uint8_t *image_host = preprocess_buffer->cpu(size_image); // Se crea para solicitar memoria para la imagen en CPU

            /* El paso de asignación no es redundante, esta es la transferencia de datos desde la memoria paginada a la memoria de página fija, lo que puede acelerar
            la transferencia de datos a la memoria de la GPU. */
            memcpy(image_host, image.bgrptr, size_image); // Asignar valor a la memoria de imagen
            // Desde cpu-->gpu, image_host también se puede reemplazar con image.bgrptr y luego eliminar las líneas anteriores, pero será aproximadamente 0,02 ms más lento.
            checkRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_)); // Sube la imagen desde la memoria de la CPU a la memoria de la GPU

            // Resize y normalizacion para preparar la imagen
            resize_bilinear_and_normalize(image_device, image.width * image.channels, image.width, image.height, input_device,
                                          model_info->m_preProcCfg.network_input_width_, model_info->m_preProcCfg.network_input_height_,
                                          model_info->m_preProcCfg.normalize_, stream_);

            float* tmp = (float*)malloc(model_info->m_preProcCfg.network_input_numel);
            cudaMemcpy(tmp, input_device, 1 * model_info->m_preProcCfg.network_input_numel, cudaMemcpyDeviceToHost);
        }

        // Posprocesado con GPU
        void RTDETRDetect::postprocess_gpu(int ibatch, cudaStream_t stream_)
        {
            // boxarray_device：El puntero de la gpu que se almacenará después de analizar los resultados de la inferencia
            float *boxarray_device = output_boxarray_.gpu() + ibatch * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT);
            // image_based_bbox_output: Puntero de GPU de todos los cuadros de predicción generados por los resultados de la inferencia
            float *image_based_bbox_output = bbox_predict_.gpu() + ibatch * model_info->m_postProcCfg.bbox_head_dims_output_numel_;

            checkRuntime(cudaMemsetAsync(boxarray_device, 0, sizeof(int), stream_));
            decode_detect_rtdetr_kernel_invoker(image_based_bbox_output, model_info->m_postProcCfg.bbox_head_dims_[1], model_info->m_postProcCfg.num_classes_,
                                                model_info->m_postProcCfg.bbox_head_dims_[2], model_info->m_postProcCfg.confidence_threshold_,
                                                model_info->m_preProcCfg.network_input_width_, boxarray_device, model_info->m_postProcCfg.MAX_IMAGE_BOXES,
                                                model_info->m_postProcCfg.NUM_BOX_ELEMENT, stream_);
        }

        BatchBoxArray RTDETRDetect::parser_box(const std::vector<Image> &images)
        {
            int num_image = images.size();
            BatchBoxArray arrout(num_image);
            for (int ib = 0; ib < num_image; ++ib)
            {
                float ratio_h = model_info->m_preProcCfg.network_input_height_ * 1.0f / images[ib].height;
                float ratio_w = model_info->m_preProcCfg.network_input_width_ * 1.0f / images[ib].width;
                float *parray = output_boxarray_.cpu() + ib * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT);


                int count = min(model_info->m_postProcCfg.MAX_IMAGE_BOXES, (int)*parray);

                BoxArray &output = arrout[ib];
                output.reserve(count); // Asigna memoria para el vector de salida, generalmente es (7)
                for (int i = 0; i < count; ++i) // Itera "i" para cada elemento del vector de salida
                {
                    float *pbox = parray + 1 + i * model_info->m_postProcCfg.NUM_BOX_ELEMENT;
                    int label = pbox[5];
                    int keepflag = pbox[6];
                    if (keepflag == 1) //keepflag == 1
                    {
                        //cout<<pbox[0]<<" "<<pbox[1]<<" "<<" "<<pbox[2]<<" "<<pbox[3]<<" :::::: "<<pbox[4]<<endl;
                        int left = std::min(std::max(1, (int)(pbox[0] / ratio_w)), images[ib].width - 1);
                        int top = std::min(std::max(1, (int)(pbox[1] / ratio_h)), images[ib].height - 1);
                        int right = std::min(std::max(1, (int)(pbox[2] / ratio_w)), images[ib].width - 1);
                        int bottom = std::min(std::max(1, (int)(pbox[3] / ratio_h)), images[ib].height - 1);
                        // left,top,right,bottom,confidence,class_label
                        output.emplace_back(left, top, right, bottom, pbox[4], label);
                    }
                }
            }

            return arrout;
        }

        // Forward para una sola imagen
        BoxArray RTDETRDetect::forward(const Image &image)
        {
            auto output = forwards({image});
            if (output.empty())
                return {};
            return output[0];
        }

        // Forward para un lote de imagenes
        BatchBoxArray RTDETRDetect::forwards(const std::vector<Image> &images)
        {
            
            int num_image = images.size();
            if (num_image == 0)
                return {};
            
            // Establecer dinámicamente el tamaño del lote
            auto input_dims = model_->get_network_dims(0);
            if (model_info->m_preProcCfg.infer_batch_size != num_image)
            {
                if (model_info->m_preProcCfg.isdynamic_model_)
                {
                    model_info->m_preProcCfg.infer_batch_size = num_image;
                    input_dims[0] = num_image;
                    if (!model_->set_network_dims(0, input_dims)) // Vuelve a vincular el lote de entrada, el tipo de valor de retorno es bool
                        return {};
                }
                else
                {
                    if (model_info->m_preProcCfg.infer_batch_size < num_image)
                    {
                        INFO(
                            "When using static shape model, number of images[%d] must be "
                            "less than or equal to the maximum batch[%d].",
                            num_image, model_info->m_preProcCfg.infer_batch_size);
                        return {};
                    }
                }
            }


            // Dado que el tamaño del lote es dinámico, es necesario solicitar dinámicamente la memoria gpu/cpu.
            adjust_memory(model_info->m_preProcCfg.infer_batch_size);


            // Preprocesar la imagen y medir el tiempo
            tmpTimer.start(); 
            for (int i = 0; i < num_image; ++i)
                preprocess_gpu(i, images[i], preprocess_buffers_[i], cu_stream); // input_buffer_会获取到图片预处理好的值
            preprocess_time += tmpTimer.stop("Timer", 1, false);

            
            // Para la inferencia (forward)
            tmpTimer.start();
            float *bbox_output_device = bbox_predict_.gpu();                  // Obtiene el puntero de la gpu para almacenar los resultados después de la inferencia
            vector<void *> bindings{input_buffer_.gpu(), bbox_output_device}; // Los bindings se utilizan como entrada para reenviar
             
            if (!model_->forward(bindings, cu_stream))
            {
                INFO("Failed to tensorRT forward.");
                return {};
            }
            inference_time += tmpTimer.stop("Timer", 1, false); 


            // Analizar los resultados de la inferencia (Posprocesamiento)
            tmpTimer.start();
            for (int ib = 0; ib < num_image; ++ib)
                postprocess_gpu(ib, cu_stream);
            //Transfiere el resultado después del posprocesamiento desde la memoria GPU a la memoria CPU
            checkRuntime(cudaMemcpyAsync(output_boxarray_.cpu(), output_boxarray_.gpu(),
                                         output_boxarray_.gpu_bytes(), cudaMemcpyDeviceToHost, cu_stream));
            checkRuntime(cudaStreamSynchronize(cu_stream)); // Bloquea el stream asincrono y espera hasta que se completen todas las operaciones en el stream antes de continuar.
            postprocess_time += tmpTimer.stop("Timer", 1, false);


            return parser_box(images);
        }
    }
}