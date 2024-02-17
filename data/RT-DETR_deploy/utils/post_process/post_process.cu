#include "post_process.cuh"
namespace ai
{
    namespace postprocess
    {
        static __global__ void decode_kernel_rtdetr(float *predict, int num_bboxes, int num_classes,
                                                    int output_cdim, float confidence_threshold,
                                                    int scale_expand, float *parray,
                                                    int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT)
        {
            int position = blockDim.x * blockIdx.x + threadIdx.x;
            if (position >= num_bboxes)
                return;

            float *pitem = predict + output_cdim * position;

            // A partir de puntuaciones de varias categorías, busca la puntuación de clase+etiqueta de la categoría más grande
            float *class_confidence = pitem + 4; //4
            float confidence = *class_confidence++;
            //float confianza = *class_confidence++; // Lleva la clase1 a la confianza y class_confidence aumenta en 1
            
            int label = 0;
            // ++class_confidence y class_confidence++ tienen el mismo resultado cuando se ejecutan en un bucle. Ambos agregan uno después de ejecutar el cuerpo del bucle.
            for (int i = 1; i < num_classes; ++i, ++class_confidence) // i = 1
            {
                if (*class_confidence > confidence)
                {
                    confidence = *class_confidence;
                    //printf("CONFIDENCE %f \n",confidence);
                    label = i;
                }
            }

            if (confidence < confidence_threshold)
                return;

            // Operaciones atómicas de CUDA: int atomicAdd(int *M,int V); toman una ubicación de memoria M y un valor V como entrada.
            // Las operaciones asociadas con la función atómica se realizan en V, el valor V ya está almacenado en la dirección de memoria *M, y el resultado de la suma se escribe en la misma ubicación de memoria.
            int index = atomicAdd(parray, 1); //Entonces este código significa usar parray[0] para calcular el número total de cajas
            if (index >= MAX_IMAGE_BOXES)
                return;

            float cx = *pitem++;
            float cy = *pitem++;
            float width = *pitem++;
            float height = *pitem++;
            float left = (cx - width * 0.5f) * scale_expand;
            float top = (cy - height * 0.5f) * scale_expand;
            float right = (cx + width * 0.5f) * scale_expand;
            float bottom = (cy + height * 0.5f) * scale_expand;

            // Todos los valores después de parray+1 se utilizan para almacenar elementos de cuadros. Cada cuadro tiene NUM_BOX_ELEMENT elementos.
            float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
            *pout_item++ = left;
            *pout_item++ = top;
            *pout_item++ = right;
            *pout_item++ = bottom;
            *pout_item++ = confidence;
            *pout_item++ = label;
            *pout_item++ = 1; // 1 = keep, 0 = ignore
        }

        void decode_detect_rtdetr_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                                 float confidence_threshold, int scale_expand, float *parray, int MAX_IMAGE_BOXES,
                                                 int NUM_BOX_ELEMENT, cudaStream_t stream)
        {
            auto grid = CUDATools::grid_dims(num_bboxes);
            auto block = CUDATools::block_dims(num_bboxes);
            checkCudaKernel(decode_kernel_rtdetr<<<grid, block, 0, stream>>>(
                predict, num_bboxes, num_classes, output_cdim, confidence_threshold, scale_expand,
                parray, MAX_IMAGE_BOXES, NUM_BOX_ELEMENT));
        }
    }
}