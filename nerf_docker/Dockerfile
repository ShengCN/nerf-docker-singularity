FROM nvidia/cuda:11.8.0-devel-ubuntu22.04  as base

ENV NV_CUDA_LIB_VERSION "11.8.0"

FROM base as base-amd6

LABEL maintainer "Yichen Sheng <sheng30@purdue.edu>"

ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ARG CUDA_ARCHITECTURES=90;89;86;80;75;70;61;52;37
ARG CUDA_ARCHITECTURES=90;89;86;80

# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    htop \ 
    wget \
    colmap \ 
    libatlas-base-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libhdf5-dev \
    libcgal-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libgflags-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libmetis-dev \
    libprotobuf-dev \
    libqt5opengl5-dev \
    libsqlite3-dev \
    libsuitesparse-dev \
    nano \
    protobuf-compiler \
    python-is-python3 \
    python3.10-dev \
    python3-pip \
    qtbase5-dev \
    sudo \
    vim-tiny \
    wget && \
    rm -rf /var/lib/apt/lists/*

# colmap seems to have a bug in the docker image, a workaround is this: 
# https://stackoverflow.com/questions/63627955/cant-load-shared-library-libqt5core-so-5 
RUN strip --remove-section=.note.ABI-tag /usr/lib/x86_64-linux-gnu/libQt5Core.so.5
RUN ldconfig

RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository universe \
    && add-apt-repository multiverse
RUN apt update
RUN rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
   && mkdir /root/.conda \
   && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 \
   && rm -f Miniconda3-latest-Linux-x86_64.sh


# setup nerfstudio
ENV PATH /opt/miniconda3/bin:$PATH

# setup nerfstudio 
RUN conda create --name nerfstudio -y python=3.8 

RUN /bin/bash -c "source activate nerfstudio && pip install --upgrade pip setuptools pathtools promise pybind11"

RUN /bin/bash -c "source activate nerfstudio && python -m pip install --upgrade pip && pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118"
ENV PATH /usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH
# RUN /bin/bash -c "source activate nerfstudio && MAKEFLAGS=\"-j12\" pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

# install tiny cuda nn
ENV TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
RUN /bin/bash -c "source activate nerfstudio && python -m pip install git+https://github.com/NVlabs/tiny-cuda-nn.git@v1.6#subdirectory=bindings/torch"

# install hloc
RUN git clone --recursive https://github.com/cvg/Hierarchical-Localization.git /home/root/hloc 
RUN /bin/bash -c "source activate nerfstudio && cd /home/root/hloc && python -m pip install -e ."

# install omegaconf and nerfstudio
RUN /bin/bash -c "source activate nerfstudio && pip install omegaconf"

# install nerfstudio from src
COPY nerfstudio /home/root/nerfstudio
RUN /bin/bash -c "source activate nerfstudio && cd /home/root/nerfstudio && python -m pip install -e ."

WORKDIR /home/root/
ENTRYPOINT [  "/bin/bash" ]
