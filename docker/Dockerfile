FROM nvcr.io/nvidia/pytorch:24.08-py3

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libglvnd-dev \
        libgl1-mesa-dev \
        libegl1-mesa-dev \
        libgles2-mesa-dev && \
    rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

RUN apt-get update && apt-get install -y \
    tzdata \
    software-properties-common \
    curl wget git build-essential cmake ffmpeg \
    libeigen3-dev libssl-dev \
    python3-pip python3-tk \
    sudo bash \
    && rm -rf /var/lib/apt/lists/* \
    && ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata

ENV TORCH_CUDA_ARCH_LIST "7.0 8.6+PTX"
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

ARG CACHE_DATE=2024-07-19

RUN mkdir /pkgs && cd /pkgs && git clone https://github.com/NVlabs/curobo.git
RUN cd /pkgs/curobo && pip3 install -e . --no-build-isolation

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install \
    pybullet \
    transformers \
    scipy \
    matplotlib \
    tqdm \
    trimesh \
    accelerate \
    plotly \
    Pillow \
    openai

WORKDIR /pkgs/IMPACT
CMD ["bash"]
