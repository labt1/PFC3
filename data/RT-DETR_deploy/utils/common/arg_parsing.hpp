#ifndef UTILS_ARG_PARSING_H_
#define UTILS_ARG_PARSING_H_
#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <getopt.h>

#define RETURN_SUCCESS (0)
#define RETURN_FAIL (-1)
namespace ai
{
    namespace arg_parsing
    {
        struct Settings
        {
            // Parámetros requeridos
            std::string model_path = ""; // Rutal del modelo
            
            // Parámetros opcionales
            std::string image_path = "";   // Ruta de la imágen
            int batch_size = 1;            // Por defecto es 1, si el modelo es dinámico ingresa el tamaño del lote 
            float score_thr = 0.6f;        // Umbral para filtrar los resultados
            int device_id = 0;             // ID del GPU por si se tienen varias GPUs
            int loop_count = 10;           // La cantidad de veces que la tarea de inferencia se ejecuta en un bucle (No se usa)
            int number_of_warmup_runs = 2; // El nímero de inferencias para calentar el nucleo CUDA (No se usa)
            std::string output_dir = "";   // Dirección de salida donde se guardan los resultados
            std::string camera_ip = "";    // Direccion de la camara IP
            std::string video_path = "";    // Direccion de la camara IP

            // Etiquetas
            const std::vector<std::string> classlabels{"Pistola"};
        };
        int parseArgs(int argc, char **argv, Settings *s); // Analiza los parámetros ingresados ​​en la línea de comando y los asigna a Configuración
        void printArgs(Settings *s);                       // Imprimir todos los parámetros
    }
}

#endif // UTILS_ARG_PARSING_H_