#ifndef _UTILS_HPP_
#define _UTILS_HPP_
#include <string>
#include <fstream>
#include <numeric>
#include <sstream>
#include <memory>
#include <vector>
#include <stack>
#include <unordered_map>
#include <initializer_list>

#include <unistd.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <stdarg.h>
#include "cuda_utils.hpp"
#define strtok_s strtok_r

using namespace std;

namespace ai
{
    namespace utils
    {
        // Definición de algunas funciones de uso común

        std::string file_name(const std::string &path, bool include_suffix);
        void __log_func(const char *file, int line, const char *fmt, ...);
        std::vector<unsigned char> load_file(const std::string &file);
        std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id);
        std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v);

        // Algunas funciones de procesamiento de archivos

        bool file_exist(const string &path); // Determinar si el archivo existe
        bool dir_mkdir(const string &path);
        bool mkdirs(const string &path); // Si la carpeta no existe, se volverá a crear
        bool begin_with(const string &str, const string &with);
        bool end_with(const string &str, const string &with);
        string path_join(const char *fmt, ...);
        bool rmtree(const string &directory, bool ignore_fail = false);
        bool alphabet_equal(char a, char b, bool ignore_case);
        bool pattern_match(const char *str, const char *matcher, bool igrnoe_case = true);
        bool pattern_match_body(const char *str, const char *matcher, bool igrnoe_case);
        vector<string> find_files(
            const string &directory,
            const string &filter = "*", bool findDirectory = false, bool includeSubDirectory = false);

        // Clase para medir los tiempos
        class Timer
        {
        public:
            Timer();
            virtual ~Timer();
            void start(void *stream = nullptr);
            float stop(const char *prefix = "Timer", int loop_iters = 1, bool print = true);

        private:
            void *start_, *stop_;
            void *stream_;
        };

    }
}
#define INFO(...) ai::utils::__log_func(__FILE__, __LINE__, __VA_ARGS__)
#endif // _UTILS_HPP_
