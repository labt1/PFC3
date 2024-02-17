#include "arg_parsing.hpp"
namespace ai
{
    namespace arg_parsing
    {
        void displayUsage()
        {
            std::cout
                << "--model_path, -f: model artifacts folder path\n"
                << "--image_path, -i: input image with full path\n"
                << "--batch_size, -b: batch size can be [1=default]\n"
                << "--score_thr, -s: filter threshold during post-processing of model results can be [0.5f=default]\n"
                << "--device_id, -g: serial number of the gpu graphics card can be [0=defualt]\n"
                << "--loop_count, -c: infer loop iteration times can be [10=default]\n"
                << "--warmup_runs, -w: number of warmup runs can be [2=default]\n"
                << "--output_dir, -o: storage path for model inference results can be [''=default]\n"
                << "--camera_ip, -p: storage path for camera \n"
                << "--video_path, -v: input video \n"
                << "--help, -h: output help command info\n"
                << "\n";
        }

        int parseArgs(int argc, char **argv, Settings *s)
        {
            int c;
            while (1)
            {
                static struct option long_options[] = {
                    {"model_path", required_argument, nullptr, 'f'},
                    {"image_path", optional_argument, nullptr, 'i'},
                    {"batch_size", required_argument, nullptr, 'b'},
                    {"score_thr", required_argument, nullptr, 's'},
                    {"device_id", required_argument, nullptr, 'g'},
                    {"loop_count", optional_argument, nullptr, 'c'},
                    {"warmup_runs", optional_argument, nullptr, 'w'},
                    {"output_dir", optional_argument, nullptr, 'o'},
                    {"camera_ip", optional_argument, nullptr, 'p'},
                    {"video_path", optional_argument, nullptr, 'v'},
                    {nullptr, 0, nullptr, 0}};

                /* getopt_long almacena el índice de opciones */
                int option_index = 0;

                c = getopt_long(argc, argv,
                                "f:i:b:s:g:c:w:o:p:v:", long_options,
                                &option_index);

                /* Detectar el final de las opciones */
                if (c == -1)
                    break;

                switch (c)
                {
                case 'f':
                    s->model_path = optarg;
                    break;
                case 'i':
                    s->image_path = optarg;
                    break;
                case 'b':
                    s->batch_size = strtol(optarg, nullptr, 10); // Convierte caracteres char a int/long
                    break;
                case 's':
                    s->score_thr = strtod(optarg, nullptr); // Convertir caracteres char a flotantes/dobles
                    break;
                case 'g':
                    s->device_id = strtol(optarg, nullptr, 10);
                    break;
                case 'c':
                    s->loop_count = strtol(optarg, nullptr, 10);
                    break;
                case 'w':
                    s->number_of_warmup_runs = strtol(optarg, nullptr, 10);
                    break;
                case 'o':
                    s->output_dir = optarg;
                    break;
                case 'p':
                    s->camera_ip = optarg;
                    break;
                case 'v':
                    s->video_path = optarg;
                    break;
                case 'h':
                case '?':
                    displayUsage();
                    exit(0);
                default:
                    return RETURN_FAIL;
                }
            }
            return RETURN_SUCCESS;
        }

        void printArgs(Settings *s)
        {
            std::cout << "\n***** Display run Config: start *****\n";

            std::cout << "model path set to: " << s->model_path << "\n";
            if (s->image_path != "")
                std::cout << "image path set to: " << s->image_path << "\n";
            std::cout << "batch size set to: " << s->batch_size << "\n";
            std::cout << "score threshold set to: " << s->score_thr << "\n";
            std::cout << "device id set to: " << s->device_id << "\n";
            if (s->image_path != ""){
                std::cout << "loop count set to: " << s->loop_count << "\n";
                std::cout << "num of warmup runs set to: " << s->number_of_warmup_runs << "\n";
                std::cout << "output directory set to: " << s->output_dir << "\n";
            }
            if (s->camera_ip != "")
                std::cout << "Camera IP: " << s->camera_ip << "\n";
            if (s->video_path != "")
                std::cout << "video path set to: " << s->video_path << "\n";

            std::cout << "***** Display run Config: end *****\n\n";
        }
    }
}
