#ifndef _POST_PROCESS_HPP_CUDA_
#define _POST_PROCESS_HPP_CUDA_

#include <iostream>
#include <cuda_runtime.h>
#include "common/cuda_utils.hpp"

#define BLOCK_SIZE 32

namespace ai
{
    namespace postprocess
    {
        // Generalmente se usa para analizar yolov3/v5/v7/yolox. Si tiene otro posprocesamiento de modelo de tarea que requiere aceleración CUDA, también puede escribirlo aquí.
        // El número máximo predeterminado de cuadros de detección para una imagen es 1024, que se puede cambiar pasando parámetros o modificando directamente los parámetros predeterminados.
        void decode_detect_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                          float confidence_threshold, float *invert_affine_matrix,
                                          float *parray, int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT, cudaStream_t stream);
        // implementación cuda de nms
        void nms_kernel_invoker(float *parray, float nms_threshold, int max_objects, int NUM_BOX_ELEMENT, cudaStream_t stream);

        // yolov8 detecta análisis de posprocesamiento
        void decode_detect_yolov8_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                                 float confidence_threshold, float *invert_affine_matrix,
                                                 float *parray, int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT, cudaStream_t stream);

        // postprocesamiento de la rama del segmento yolov8
        void decode_single_mask(float left, float top, float *mask_weights, float *mask_predict,
                                int mask_width, int mask_height, unsigned char *mask_out,
                                int mask_dim, int out_width, int out_height, cudaStream_t stream);

        // análisis de posprocesamiento de pose yolov8
        void decode_pose_yolov8_kernel_invoker(float *predict, int num_bboxes, int pose_num, int output_cdim,
                                               float confidence_threshold, float *invert_affine_matrix,
                                               float *parray, int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT, cudaStream_t stream);

        // análisis de posprocesamiento rtdetr
        void decode_detect_rtdetr_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                                 float confidence_threshold, int scale_expand, float *parray, int MAX_IMAGE_BOXES,
                                                 int NUM_BOX_ELEMENT, cudaStream_t stream);
    }
}
#endif // _POST_PROCESS_HPP_CUDA_