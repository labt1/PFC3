#ifndef _CV_CPP_UTILS_HPP_
#define _CV_CPP_UTILS_HPP_

#include <tuple>
#include <string.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "utils.hpp"

namespace ai
{
    namespace cvUtil
    {

        using namespace std;
        using namespace ai::utils;

        // Unifica el formato de entrada del modelo para facilitar la configuración de entrada posterior
        struct Image
        {
            const void *bgrptr = nullptr;
            int width = 0, height = 0, channels = 0;

            Image() = default;
            Image(const void *bgrptr, int width, int height, int channels) : bgrptr(bgrptr), width(width), height(height), channels(channels) {}
        };

        Image cvimg_trans_func(const cv::Mat &image);

        // Configuración de flag para normalizar la entrada
        enum class NormType : int
        {
            None = 0,
            MeanStd = 1,  // out = (x * alpha - mean) / std
            AlphaBeta = 2 // out = x * alpha + beta
        };

        // Establece si el canal de entrada es RGB o BGR
        enum class ChannelType : int
        {
            BGR = 0,
            RGB = 1
        };

        // Para inicializar la configuración de entrada
        struct Norm
        {
            float mean[3];
            float std[3];
            float alpha, beta;
            NormType type = NormType::None;
            ChannelType channel_type = ChannelType::BGR;

            // out = (x * alpha - mean) / std
            static Norm mean_std(const float mean[3], const float std[3], float alpha = 1 / 255.0f, ChannelType channel_type = ChannelType::BGR);
            // out = x * alpha + beta
            static Norm alpha_beta(float alpha, float beta = 0.0f, ChannelType channel_type = ChannelType::BGR);
            // None
            static Norm None();
        };

        /* Dado que la transformación afín se implementa usando CUDA, esta estructura se usa para calcular la matriz y la matriz inversa 
           de la transformación afín */
        struct AffineMatrix
        {
            float i2d[6]; // image to dst(network), 2x3 matrix
            float d2i[6]; // dst to image, 2x3 matrix
            void compute(const std::tuple<int, int> &from, const std::tuple<int, int> &to);
        };

        static void affine_project(float *matrix, float x, float y, float *ox, float *oy)
        {
            *ox = matrix[0] * x + matrix[1] * y + matrix[2];
            *oy = matrix[3] * x + matrix[4] * y + matrix[5];
        }

        // Box es la salida de la deteccion, incluye left, top, right, bottom y confidence
        struct Box
        {
            float left, top, right, bottom, confidence;
            int class_label;

            Box() = default;
            Box(float left, float top, float right, float bottom, float confidence, int class_label)
                : left(left),
                  top(top),
                  right(right),
                  bottom(bottom),
                  confidence(confidence),
                  class_label(class_label) {}
        };
        typedef std::vector<Box> BoxArray;
        typedef std::vector<BoxArray> BatchBoxArray;

        // Dibuja los cuadros delimitadores del resultado de la inferencia en una imagen
        void draw_one_image_rectangle(cv::Mat &image, BoxArray &result, const std::vector<std::string> &classlabels);
        void draw_one_image_rectangle(cv::Mat &image, BoxArray &result, const std::string &save_dir, const std::vector<std::string> &classlabels);
        
        // Dibuja los cuadros delimitadores del resultado de la inferencia en un lote de imagenes
        void draw_batch_rectangle(std::vector<cv::Mat> &images, BatchBoxArray &batched_result, const std::string &save_dir, const std::vector<std::string> &classlabels);
    }
}

#endif // _CV_CPP_UTILS_HPP_