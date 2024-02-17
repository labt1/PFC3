#ifndef _BACKEND_HPP_
#define _BACKEND_HPP_
#include <vector>
#include <string>
namespace ai
{
    namespace backend
    {

        /* La clase base de inferencia del modelo solo se puede heredar mediante herencia. Puede configurar cierta información a través de esta clase 
            para facilitar el uso de herencia multiclase en el futuro. */
        class Infer
        {
        public:
            virtual int index(const std::string &name) = 0; // Nombre de entrada y salida para indexar
            virtual bool forward(const std::vector<void *> &bindings, void *stream = nullptr,
                                 void *input_consum_event = nullptr) = 0;           // Para realizar la inferencia
            virtual std::vector<int> get_network_dims(const std::string &name) = 0; // Obtenga dimensiones de entrada y salida del modelo basadas en nombres de entrada y salida
            virtual std::vector<int> get_network_dims(int ibinding) = 0;            // Obtenga las dimensiones de entrada y salida del modelo según el índice
            virtual bool set_network_dims(const std::string &name, const std::vector<int> &dims) = 0;
            virtual bool set_network_dims(int ibinding, const std::vector<int> &dims) = 0; // Establecer la forma dinámica de la entrada.
            virtual bool has_dynamic_dim() = 0;                                            // Determinar si se trata de una entrada de forma dinámica o una entrada de forma estática
            virtual void print() = 0;                                                      // Imprimir información de dimensiones de entrada y salida
        };
    }
}
#endif // _BACKEND_HPP_