#ifndef _RTDETR_DETECT_CUDA_HPP_
#define _RTDETR_DETECT_CUDA_HPP_
#include <memory>
#include <algorithm>
#include "backend/tensorrt/trt_infer.hpp"
#include "common/model_info.hpp"
#include "common/utils.hpp"
#include "common/cv_cpp_utils.hpp"
#include "common/memory.hpp"
#include "pre_process/pre_process.cuh"
#include "post_process/post_process.cuh"

namespace tensorrt_infer
{
    namespace rtdetr_cuda
    {
        using namespace ai::modelInfo;
        using namespace ai::utils;
        using namespace ai::cvUtil;
        using namespace ai::memory;
        using namespace ai::preprocess;
        using namespace ai::postprocess;

        class RTDETRDetect
        {
        public:
            RTDETRDetect() = default;
            ~RTDETRDetect();
            
            // Parámetros de inicialización
            void initParameters(const std::string &engine_file, float score_thr = 0.5f);
            void adjust_memory(int batch_size);  // Si el tamaño del lote es dinámico, es necesario solicitar dinámicamente la memoria gpu/cpu.

            // Forward para una sola entrada
            BoxArray forward(const Image &image);
            // Forward para un lote(batch) de entradas
            BatchBoxArray forwards(const std::vector<Image> &images);

            // Para el preprocesamiento y postprocesamiento
            void preprocess_gpu(int ibatch, const Image &image,
                                shared_ptr<Memory<unsigned char>> preprocess_buffer, cudaStream_t stream_);
            void postprocess_gpu(int ibatch, cudaStream_t stream_);
            BatchBoxArray parser_box(const std::vector<Image> &images);

            // Para medir el tiempo en cada fase
            float preprocess_time = 0; //tiempo de preprocesamiento
            float inference_time = 0; //tiempo del forward
            float postprocess_time = 0; //tiempo de posprocesamiento

        private:
            std::shared_ptr<ai::backend::Infer> model_;
            std::shared_ptr<ModelInfo> model_info = nullptr;

            // Utilice una clase Memory personalizada para solicitar memoria gpu/cpu
            std::vector<std::shared_ptr<Memory<unsigned char>>> preprocess_buffers_;
            Memory<float> input_buffer_, bbox_predict_, output_boxarray_;

            // Para usar CUDA streams
            cudaStream_t cu_stream;

            // time
            Timer timer;

            // Part timer
            Timer tmpTimer;
        };
    }
}

#endif // _RTDETR_DETECT_CUDA_HPP_