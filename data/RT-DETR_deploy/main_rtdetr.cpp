#include "common/arg_parsing.hpp"
#include "rtdetr_cuda/infer_image.cpp"
#include "rtdetr_cuda/infer_camera.cpp"
#include "rtdetr_cuda/infer_video.cpp"

/*
--model_path, -f: Para ingresar la ruta del modelo, (obligatorio)
--image_path, -i: imagen de entrada, (obligatorio)
--batch_size, -b: tamaño_de_lote a usar[>=1], opcional, predeterminado=1
--score_thr, -s: se refiere al umbral para filtrar en el posprocesamiento, opcional, predeterminado = 0.5f
--device_id, -g: ID de la tarjeta gráfica, opcional, predeterminado = 0
--loop_count, -c: el número de veces a inferir, generalmente usado para cronometrar, opcional, predeterminado = 10
--warmup_runs, -w: Número de tiempos de calentamiento para la inferencia del modelo (activando cuda core), opcional, predeterminado = 2
--output_dir, -o: Directorio para almacenar resultados, opcional, predeterminado =''
--help, -h: use -h para ver qué comandos están disponibles
*/

int main(int argc, char *argv[])
{
    ai::arg_parsing::Settings s;
    if (parseArgs(argc, argv, &s) == RETURN_FAIL)
    {
        INFO("Failed to parse the args\n");
        return RETURN_FAIL;
    }
    ai::arg_parsing::printArgs(&s);

    CHECK(cudaSetDevice(s.device_id)); // Determina que GPU usar

    //infer_video(&s);
    
    if (s.camera_ip != ""){
        trt_cuda_video_stream_inference(&s);
    } 
    if (s.image_path != ""){
        trt_cuda_image_inference(&s);
    } 
    if (s.video_path != ""){
        trt_cuda_video_inference(&s);
    } 
    
    return RETURN_SUCCESS;
}