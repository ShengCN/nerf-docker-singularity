FROM nvidia/cuda:11.8.0-devel-ubuntu22.04  as base

ENV NV_CUDA_LIB_VERSION "11.8.0"

FROM base as base-amd64

LABEL maintainer "Yichen Sheng <sheng30@purdue.edu>"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
	vim \
	git \
	cmake \
	build-essential \
	curl \
	wget \
	libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \ 
	nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc \ 
    colmap \ 
	ffmpeg \
    && rm -rf /var/lib/apt/lists/*

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

# COPY zsh-in-docker.sh /tmp
# RUN chmod +x /tmp/zsh-in-docker.sh
# RUN /tmp/zsh-in-docker.sh \
#     -t https://github.com/denysdovhan/spaceship-prompt \
#     -a 'SPACESHIP_PROMPT_ADD_NEWLINE="false"' \
#     -a 'SPACESHIP_PROMPT_SEPARATE_LINE="false"' \
#     -p git \
#     -p https://github.com/zsh-users/zsh-autosuggestions \
#     -p https://github.com/zsh-users/zsh-completions \
#     -p https://github.com/zsh-users/zsh-history-substring-search \
#     -p https://github.com/zsh-users/zsh-syntax-highlighting \
#     -p 'history-substring-search' \
#     -a 'bindkey "\$terminfo[kcuu1]" history-substring-search-up' \
#     -a 'bindkey "\$terminfo[kcud1]" history-substring-search-down'


# setup nerfstudio
ENV PATH /opt/miniconda3/bin:$PATH

# setup nerfstudio 
RUN conda create --name nerfstudio -y python=3.8 
RUN /bin/bash -c "source activate nerfstudio && python -m pip install --upgrade pip && pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118"
#RUN /bin/bash -c "source activate nerfstudio && conda install -c \"nvidia/label/cuda-11.7.1\" cuda-toolkit"
# RUN /bin/bash -c "nvcc --version"
ENV PATH /usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH
# RUN /bin/bash -c "source activate nerfstudio && MAKEFLAGS=\"-j12\" pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
RUN /bin/bash -c "source activate nerfstudio && pip install nerfstudio"

RUN git clone https://github.com/cvg/Hierarchical-Localization.git /tmp/hloc 
RUN /bin/bash -c "source activate nerfstudio && cd /tmp/hloc && python -m pip install ."
RUN rm -rf /tmp/*

WORKDIR /home/root/
# ENTRYPOINT [  "/bin/zsh" ]
ENTRYPOINT [  "/bin/bash" ]
