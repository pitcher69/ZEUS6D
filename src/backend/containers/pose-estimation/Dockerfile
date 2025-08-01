FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libosmesa6-dev \
    freeglut3-dev \
    libglfw3-dev \
    libglew-dev \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PYOPENGL_PLATFORM=osmesa

RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python dependencies
COPY containers/pose-estimation/requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Create directories
RUN mkdir -p /app/output/pose_results /app/temp /app/logs && \
    chown -R user:user /app

# Copy application files
COPY containers/pose-estimation/ /app/
COPY shared-libs /app/shared-libs
RUN chown -R user:user /app

USER user

EXPOSE 5012

CMD ["python", "main.py"]
