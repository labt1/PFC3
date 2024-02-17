#ifndef _TRT_INFER_HPP_
#define _TRT_INFER_HPP_

// tensorrt 导入的库
#include <logger.h>
#include <parserOnnxConfig.h>
#include <NvInfer.h>

#include <memory>
#include "backend/backend_infer.hpp"
#include "common/utils.hpp"
namespace trt
{
    namespace infer
    {
        using namespace nvinfer1;
        using namespace ai::backend;
        
        template <typename _T>
        static void destroy_nvidia_pointer(_T *ptr)
        {
            if (ptr)
                ptr->destroy();
        }

        /* Esta clase se utiliza para inicializar el tiempo, el motor y el contexto de inferencia tensorrt */
        class __native_engine_context
        {
        public:
            __native_engine_context() = default;
            virtual ~__native_engine_context();
            bool construct(std::vector<unsigned char> &trtFile);

        private:
            void destroy();

        public:
            shared_ptr<IExecutionContext> context_;
            shared_ptr<ICudaEngine> engine_;
            shared_ptr<IRuntime> runtime_ = nullptr;
        };

        /* Esta clase muestra cierta configuración de información e implementación de inferencia de tensorrt, 
            principalmente la implementación específica de Infer*/
        class InferTRT : public Infer
        {
        public:
            InferTRT() = default;
            virtual ~InferTRT() = default;

            void setup();                                                // Inicialice el enlace_name_to_index_, utilizado para vincular el nombre de entrada y el índice
            bool construct_context(std::vector<unsigned char> &trtFile); // Deserializar el motor y asignar su contexto context_
            bool load(const string &engine_file);                        // Cargue el archivo_motor, que llama al método construct_context
            std::string format_shape(const Dims &shape);                 // Convierta la forma de salida a str, que simplemente se usa para imprimir la forma y no tiene significado.

            virtual bool forward(const std::vector<void *> &bindings, void *stream,
                                 void *input_consum_event) override; //Realizar operaciones de inferencia

            virtual int index(const std::string &name) override; // Buscar índice basado en el nombre
            virtual std::vector<int> get_network_dims(const std::string &name) override;
            virtual std::vector<int> get_network_dims(int ibinding) override; // Obtenga la información de las dimensiones de entrada y salida del motor del modelo, que se usa más comúnmente.
            virtual bool set_network_dims(const std::string &name, const std::vector<int> &dims) override;
            virtual bool set_network_dims(int ibinding, const std::vector<int> &dims) override; // Establecer el modelo de forma dinámica

            virtual bool has_dynamic_dim() override; // Determinar si la entrada del modelo es una forma dinámica
            virtual void print() override;           // Imprima información sobre las dimensiones de entrada y salida del modelo actual, etc.

        public:
            shared_ptr<__native_engine_context> context_ = nullptr; //Puntero de contexto
            unordered_map<string, int> binding_name_to_index_;      //Objeto de mapa de {name:index}
        };

        // Polimorfismo, carga el modelo y devuelve el puntero del objeto instanciado
        Infer *loadraw(const std::string &file);
        std::shared_ptr<Infer> load(const std::string &file);

    }
}

#endif // _TRT_INFER_HPP_