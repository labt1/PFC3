cmake_minimum_required(VERSION 3.10.0)
project(infer)
add_definitions(-std=c++11 -w)

# 1. Configuracion del directorio de trabajo
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/output)
set(CMAKE_INSTALL_PREFIX ${EXECUTABLE_OUTPUT_PATH}/install/) #Ruta de almacenamiento al realizar make install

# set(CMAKE_BUILD_TYPE "Release") # Activar al ejecutar el proyecto
set(CMAKE_BUILD_TYPE "Debug") # Habilitar la depuración

# 2. Configurar la arquitectura de la GPU usada
# https://developer.nvidia.com/zh-cn/cuda-gpus#compute
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_89,code=compute_89") #RTX 4090 Ada Lovelace Architecture
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_72,code=compute_400") #Nvidia Jetson AGX Xavier Architecture

# 3. Encuentra las bibliotecas cuda y opencv
find_package(CUDA REQUIRED) # Instalado de forma predeterminada
find_package(OpenCV REQUIRED) # Puede instalarse con, sudo apt-get install libopencv-dev

find_package(PkgConfig REQUIRED) # Para llamar a las bibliotecas instaladas

# Omitir los siguientes paquetes si se usa un Jetson, o si ya estan instalados para evitar conflictos
pkg_check_modules(gtk3 REQUIRED IMPORTED_TARGET gtk+-3.0) # Solo si se usa una imagen docker que no tiene una interfaz grafica
# Para conectar el stream de la camara IP
pkg_search_module(gstreamer REQUIRED IMPORTED_TARGET gstreamer-1.0>=1.4)
pkg_search_module(gstreamer-sdp REQUIRED IMPORTED_TARGET gstreamer-sdp-1.0>=1.4)
pkg_search_module(gstreamer-app REQUIRED IMPORTED_TARGET gstreamer-app-1.0>=1.4)
pkg_search_module(gstreamer-video REQUIRED IMPORTED_TARGET gstreamer-video-1.0>=1.4)

# 4. Configure el directorio principal de tensorrt
set(TensorRT_ROOT "/opt/tensorrt") #Establezca el directorio raíz tensorrt8.xx
#set(TensorRT_ROOT "/usr/src/tensorrt") #Para el Nvidia Jetson que fue instalado por medio de JetPack

# 5. Incluye las rutas de encabezados hpp que se utilizarán
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}

    # tensorrt
    ${TensorRT_ROOT}/include
    ${TensorRT_ROOT}/samples/common # Para adaptarse a la importación del logger de múltiples versiones de trt [v7.xx, v8.xx].

    # Para la implementación
    ${PROJECT_SOURCE_DIR}/utils
    ${PROJECT_SOURCE_DIR}/rtdetr_cuda
)

# 6. Enlazar la ruta de las biblioteca que se utilizarán
# Consulte https://cmake.org/cmake/help/latest/module/FindCUDA.html para conocer el nombre especifico de las librerias CUDA
link_directories(
    #Incluir librerias de forma manual
     ${CUDA_LIBRARIES}
    # ${CUDA_cublas_LIBRARY}
    # ${CUDA_cudart_static_LIBRARY}

    # Si tiene instalado CUDA Toolkit
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64

    # Para TensorRT
    ${TensorRT_ROOT}/lib
)


# 7. Compile los archivos cu y cpp escritos en utils, application y en los ejemplos de TensorRT para poder llamarlos fácilmente más adelante.
file(GLOB_RECURSE cpp_cuda_srcs
    ${PROJECT_SOURCE_DIR}/utils/*.cpp
    ${PROJECT_SOURCE_DIR}/rtdetr_cuda/*.cpp
    ${PROJECT_SOURCE_DIR}/utils/*.cu
    ${TensorRT_ROOT}/samples/common/logger.cpp # Haga referencia a la versión correspondiente de logger.cpp, por defecto para TensorRT_8.xx 
    ${TensorRT_ROOT}/samples/common/sampleOptions.cpp 
    ${TensorRT_ROOT}/samples/common/sampleUtils.cpp
)
cuda_add_library(utils_cu_cpp SHARED ${cpp_cuda_srcs})

# El main para el ejecutable
add_executable(infer main_rtdetr.cpp)

# 8. Vincula todo para que se utilicen las bibliotecas
target_link_libraries(infer
    utils_cu_cpp # Llama a la biblioteca so compilada en el paso anterior
    cuda
    cudart
    cudnn
    pthread
    ${OpenCV_LIBS}
    nvinfer
    nvinfer_plugin
    # nvonnxparser

    PkgConfig::gtk3
    PkgConfig::gstreamer
    PkgConfig::gstreamer-sdp
    PkgConfig::gstreamer-app
    PkgConfig::gstreamer-video
)

# Necesario al realizar la instalación
install(TARGETS infer  utils_cu_cpp
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)

install(DIRECTORY
        ${PROJECT_SOURCE_DIR}/utils/
        ${PROJECT_SOURCE_DIR}/rtdetr_cuda/
        DESTINATION include/
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h" PATTERN "*.cuh")

# Compile y ejecute el programa con make auto -j
# add_custom_target(
#     auto
#     DEPENDS infer
#     WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/output
#     COMMAND ./infer
# )
