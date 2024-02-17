#ifndef _PRE_PROCESS_HPP_CUDA_
#define _PRE_PROCESS_HPP_CUDA_

#include <iostream>
#include <cuda_runtime.h>
#include "common/cuda_utils.hpp"
#include "common/cv_cpp_utils.hpp"


namespace ai
{
    namespace preprocess
    {
        using namespace ai::cvUtil;
        //Utiliza cuda para implementar el método de interpolación bilineal y cambio de tamaño de opencv
        void resize_bilinear_and_normalize(
            uint8_t *src, int src_line_size, int src_width, int src_height, float *dst, int dst_width, int dst_height,
            const Norm &norm,
            cudaStream_t stream);
    }
}
#endif // _PRE_PROCESS_HPP_CUDA_