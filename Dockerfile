FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    cmake \
    git \
    ffmpeg \
    libcairo2-dev \
    pkg-config \
    libgeos-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set python3.8 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    python -m pip install --upgrade pip==22.3.1

WORKDIR /app

# Install PyTorch first (needed for diffvg build)
RUN pip install --no-cache-dir \
    torch==1.12.1+cu116 \
    torchvision==0.13.1+cu116 \
    torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy==1.22.3 \
    ipywidgets \
    diffusers \
    easydict \
    torchsummary \
    cssutils \
    shapely \
    lightning \
    imageio==2.19.3 \
    imageio-ffmpeg==0.4.7 \
    scikit-image \
    wandb \
    moviepy \
    matplotlib \
    cairosvg \
    einops \
    transformers \
    accelerate \
    svgpathtools \
    xformers

# Copy diffvg and build it
COPY diffvg/ /app/diffvg/

# Fix pybind11 CMake version compatibility with newer CMake
RUN sed -i 's/cmake_minimum_required(VERSION 3.4)/cmake_minimum_required(VERSION 3.5)/' \
    /app/diffvg/pybind11/CMakeLists.txt

# Build and install diffvg with CUDA support
RUN cd /app/diffvg && \
    DIFFVG_CUDA=1 python setup.py install

# Copy the rest of the project
COPY . /app/

# Default entrypoint
ENTRYPOINT ["python", "animate_svg.py"]
