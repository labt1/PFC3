#include "pre_process.cuh"

namespace ai
{
    namespace preprocess
    {
        // same to opencv
        // reference: https://github.com/opencv/opencv/blob/24fcb7f8131f707717a9f1871b17d95e7cf519ee/modules/imgproc/src/resize.cpp
        // reference: https://github.com/openppl-public/ppl.cv/blob/04ef4ca48262601b99f1bb918dcd005311f331da/src/ppl/cv/cuda/resize.cu
        __global__ void resize_bilinear_and_normalize_kernel(
            uint8_t *src, int src_line_size, int src_width, int src_height, float *dst, int dst_width, int dst_height,
            float sx, float sy, Norm norm, int edge)
        {
            int position = blockDim.x * blockIdx.x + threadIdx.x;
            if (position >= edge)
                return;

            int dx = position % dst_width;
            int dy = position / dst_width;
            float src_x = (dx + 0.5f) * sx - 0.5f;
            float src_y = (dy + 0.5f) * sy - 0.5f;
            float c0, c1, c2;

            int y_low = floorf(src_y);
            int x_low = floorf(src_x);
            int y_high = limit(y_low + 1, 0, src_height - 1);
            int x_high = limit(x_low + 1, 0, src_width - 1);
            y_low = limit(y_low, 0, src_height - 1);
            x_low = limit(x_low, 0, src_width - 1);

            int ly = rint((src_y - y_low) * INTER_RESIZE_COEF_SCALE);
            int lx = rint((src_x - x_low) * INTER_RESIZE_COEF_SCALE);
            int hy = INTER_RESIZE_COEF_SCALE - ly;
            int hx = INTER_RESIZE_COEF_SCALE - lx;
            int w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
            float *pdst = dst + dy * dst_width + dx * 3;
            uint8_t *v1 = src + y_low * src_line_size + x_low * 3;
            uint8_t *v2 = src + y_low * src_line_size + x_high * 3;
            uint8_t *v3 = src + y_high * src_line_size + x_low * 3;
            uint8_t *v4 = src + y_high * src_line_size + x_high * 3;

            c0 = resize_cast(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0]);
            c1 = resize_cast(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1]);
            c2 = resize_cast(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2]);

            if (norm.channel_type == ChannelType::RGB)
            {
                float t = c2;
                c2 = c0;
                c0 = t;
            }

            if (norm.type == NormType::MeanStd)
            {
                c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
                c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
                c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
            }
            else if (norm.type == NormType::AlphaBeta)
            {
                c0 = c0 * norm.alpha + norm.beta;
                c1 = c1 * norm.alpha + norm.beta;
                c2 = c2 * norm.alpha + norm.beta;
            }

            int area = dst_width * dst_height;
            float *pdst_c0 = dst + dy * dst_width + dx;
            float *pdst_c1 = pdst_c0 + area;
            float *pdst_c2 = pdst_c1 + area;
            *pdst_c0 = c0;
            *pdst_c1 = c1;
            *pdst_c2 = c2;
        }

        void resize_bilinear_and_normalize(
            uint8_t *src, int src_line_size, int src_width, int src_height, float *dst, int dst_width, int dst_height,
            const Norm &norm,
            cudaStream_t stream)
        {

            int jobs = dst_width * dst_height;
            auto grid = CUDATools::grid_dims(jobs);
            auto block = CUDATools::block_dims(jobs);

            checkCudaKernel(resize_bilinear_and_normalize_kernel<<<grid, block, 0, stream>>>(
                src, src_line_size,
                src_width, src_height, dst,
                dst_width, dst_height, src_width / (float)dst_width, src_height / (float)dst_height, norm, jobs));
        }
    }
}