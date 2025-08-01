FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
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
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Simplified environment variables (no aggressive threading)
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Create symbolic links for python3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python dependencies
COPY query-dinov2/requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Create directories and set ownership
RUN mkdir -p /app/output/dino_features /app/data /app/logs && \
    chown -R user:user /app

COPY query-dinov2/ /app/
COPY shared-libs /app/shared-libs
RUN chown -R user:user /app

USER user

EXPOSE 5009

CMD ["python", "main.py"]
