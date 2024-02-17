FROM nvcr.io/nvidia/tensorrt:23.11-py3

USER root

###Para agregar un grupo y un nuevo usuario
#RUN addgroup --gid 1001 nonroot && \
#    adduser --uid 1000 --gid 1001 --disabled-password --gecos "" nonroot && \
#    echo 'nonroot ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers
# Para agregar al usuario al grupo conda (opcional)
#RUN usermod -a -G conda rapids

#Para modificar un usuario existente
### El UID y GID del host deben ser los mismos que el user del contenedor para tener los mismos permisos

RUN apt-get update
RUN apt-get install build-essential
RUN apt-get install -y gdb
RUN apt-get install -y libopencv-dev

RUN apt install libeigen3-dev

#Para la camara IP
RUN \
	apt-get install -y \
	libgstreamer1.0-0 \
	gstreamer1.0-plugins-base \
	gstreamer1.0-plugins-good \
	gstreamer1.0-plugins-bad \
	gstreamer1.0-plugins-ugly \
	gstreamer1.0-libav \
	gstreamer1.0-tools \
	libgstreamer1.0-dev \
	libgstreamer-plugins-base1.0-dev

RUN apt-get install -y libgtk-3-dev

RUN groupadd --gid 1001 tensortt-group && \
    useradd -rm -d /home/TensorRT -s /bin/bash -g tensortt-group -u 1000 TensorRT

RUN usermod -u 1000 -g 1001 TensorRT
#RUN usermod -a -G conda TensorRT

USER TensorRT

WORKDIR /home/TensorRT/data
