#ifndef _MODEL_INFO_HPP_
#define _MODEL_INFO_HPP_

#include <string>
#include <vector>
#include "cv_cpp_utils.hpp"

namespace ai
{
    namespace modelInfo
    {
        struct PreprocessImageConfig
        {
            int32_t infer_batch_size{0};        // El lote de entrada del modelo se obtiene automáticamente y no se requiere configuración.
            int32_t network_input_width_{0};    // El ancho de la entrada del modelo se obtiene automáticamente y no se requiere configuración.
            int32_t network_input_height_{0};   // La alto de la entrada del modelo se obtiene automáticamente y no se requiere configuración.
            int32_t network_input_channels_{0}; // El número de canales de entrada al modelo se obtiene automáticamente y no se requiere configuración.
            bool isdynamic_model_ = false;      // Si es un modelo dinámico, se obtendrá automáticamente y no requiere configuración.

            size_t network_input_numel{0}; // El tamaño de la entrada del modelo (h x w x c), no requiere configuración

            ai::cvUtil::Norm normalize_ = ai::cvUtil::Norm::None(); // Configurar el preprocesamiento de la imagen de entrada
        };

        struct PostprocessImageConfig
        {
            float confidence_threshold_{0.5f};
            float nms_threshold_{0.45f};

            // Detectar rama
            std::vector<int> bbox_head_dims_; // Vector de dimensiones de salida del modelo
            size_t bbox_head_dims_output_numel_{0}; // Tamaño de la salida del modelo, no se requiere configuración

            // Algunas configuraciones de parámetros al analizar los resultados de salida del modelo se configuran mejor en tipo constante para evitar cambios.
            int MAX_IMAGE_BOXES = 1024; // Numero maximo de bboxes por imagen
            int NUM_BOX_ELEMENT = 7;               // izquierda, arriba, derecha, abajo, confidence, clase, keepflag
            size_t IMAGE_MAX_BOXES_ADD_ELEMENT{0}; // MAX_IMAGE_BOXES * NUM_BOX_ELEMENT

            int num_classes_ = 0; // Las categorías se pueden derivar automáticamente de las dimensiones de salida del modelo o se pueden establecer
        };

        struct ModelInfo
        {
            std::string m_modelPath; // ruta del modelo serializado (model.engine, model.trt, etc), no se requiere configuración

            PreprocessImageConfig m_preProcCfg;   // Configuración de preprocesamiento
            PostprocessImageConfig m_postProcCfg; // Configuración de posprocesamiento
        };
    }
}
#endif // _MODEL_INFO_HPP_