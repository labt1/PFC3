#ifndef _MEMORY_HPP_
#define _MEMORY_HPP_
#include "utils.hpp"

namespace ai
{
    namespace memory
    {
        /* Para facilitar la aplicación, reutilización y liberación de la memoria de la gpu y la cpu, 
        su función se implementa por separado con una clase BaseMemory */
        class BaseMemory
        {
        public:
            BaseMemory() = default; // Constructor por defecto
            BaseMemory(void *cpu, size_t cpu_bytes,
                       void *gpu, size_t gpu_bytes);
            virtual ~BaseMemory();

            // Solicitar memoria GPU
            virtual void *gpu_realloc(size_t bytes);
            // Solicitar memoria CPU
            virtual void *cpu_realloc(size_t bytes);
            // Libera la memoria de la gpu. Si se va a reutilizar, la memoria no se liberará.
            void release_gpu();
            // Libere la memoria de la CPU. Si se va a reutilizar, la memoria no se liberará.                    
            void release_cpu();               
            void release();

            inline bool owner_gpu() const { return owner_gpu_; }
            inline bool owner_cpu() const { return owner_cpu_; }
            inline size_t cpu_bytes() const { return cpu_bytes_; }
            inline size_t gpu_bytes() const { return gpu_bytes_; }
            virtual inline void *get_gpu() const { return gpu_; }
            virtual inline void *get_cpu() const { return cpu_; }
            void reference(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes);

        protected:
            // Algunas propiedades necesarias para las operaciones de memoria de la CPU
            void *cpu_ = nullptr;     // Puntero de dirección de memoria de la CPU
            size_t cpu_bytes_ = 0;    // El tamaño de memoria que la CPU debe solicitar, generalmente el número de bytes.
            size_t cpu_capacity_ = 0; // El número máximo de bytes que debe solicitar la memoria de la CPU, similar a la capacidad del vector
            bool owner_cpu_ = true;   // Identificador de si la memoria de la CPU está libre.Este parámetro generalmente se utiliza para controlar la reutilización de la memoria de la CPU.

            // Algunos atributos necesarios para las operaciones de memoria de la gpu, las funciones de cada atributo son las mismas que las de la cpu
            void *gpu_ = nullptr;
            size_t gpu_bytes_ = 0, gpu_capacity_ = 0;   
            bool owner_gpu_ = true;
        };

        /* Clase de plantilla para aplicación de memoria, utilizada para solicitar varios tipos de memoria de CPU/GPU*/
        template <typename _DT>
        class Memory : public BaseMemory
        {
        public:
            Memory() = default;
            Memory(const Memory &other) = delete;
            Memory &operator=(const Memory &other) = delete;

            // Solicitar memoria GPU para un tipo <_DT>
            virtual _DT *gpu(size_t size) { return (_DT *)BaseMemory::gpu_realloc(size * sizeof(_DT)); }
            // Solicitar memoria CPU para un tipo <_DT>
            virtual _DT *cpu(size_t size) { return (_DT *)BaseMemory::cpu_realloc(size * sizeof(_DT)); } 

            inline size_t cpu_size() const { return cpu_bytes_ / sizeof(_DT); }
            inline size_t gpu_size() const { return gpu_bytes_ / sizeof(_DT); }

            // Se utiliza para obtener la dirección del puntero de la memoria en GPU.
            virtual inline _DT *gpu() const { return (_DT *)gpu_; }
            // Se utiliza para obtener la dirección del puntero de la memoria en CPU.
            virtual inline _DT *cpu() const { return (_DT *)cpu_; }
        };
    }
}
#endif // _MEMORY_HPP_