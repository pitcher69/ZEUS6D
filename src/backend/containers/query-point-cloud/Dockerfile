FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for point cloud processing
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    libtbb-dev \
    libeigen3-dev \
    libflann-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Simplified environment variables (no aggressive threading for development)
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Create symbolic links for python3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Install essential libraries
RUN pip install --no-cache-dir \
    wheel \
    setuptools \
    cython \
    numba

RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python dependencies
COPY query-point-cloud/requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Create directories and set ownership
RUN mkdir -p /app/output/point_clouds /app/data /app/logs && \
    chown -R user:user /app

COPY query-point-cloud/ /app/
COPY shared-libs /app/shared-libs
RUN chown -R user:user /app

USER user

EXPOSE 5008

CMD ["python", "main.py"]
