FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for rendering
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
    libglew-dev \
    libglfw3-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    mesa-common-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Simplified environment variables (no aggressive threading for development)
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PYOPENGL_PLATFORM=egl
ENV DISPLAY=:99

# Create symbolic links for python3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN useradd -m -u 1000 user
WORKDIR /app

# Clone CNOS repository
RUN git clone https://github.com/pitcher69/cnos.git /app/cnos

# Install Python dependencies
COPY query-cnos/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Create output directories
RUN mkdir -p /app/output/renders /app/output/gifs /app/logs && \
    chown -R user:user /app

COPY query-cnos/ /app/
COPY shared-libs /app/shared-libs
RUN chown -R user:user /app

USER user

EXPOSE 5007

CMD ["python", "main.py"]
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for rendering
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
    libglew-dev \
    libglfw3-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    mesa-common-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Simplified environment variables (no aggressive threading for development)
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PYOPENGL_PLATFORM=egl
ENV DISPLAY=:99

# Create symbolic links for python3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN useradd -m -u 1000 user
WORKDIR /app

# Clone CNOS repository
RUN git clone https://github.com/pitcher69/cnos.git /app/cnos

# Install Python dependencies
COPY query-cnos/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Create output directories
RUN mkdir -p /app/output/renders /app/output/gifs /app/logs && \
    chown -R user:user /app

COPY query-cnos/ /app/
COPY shared-libs /app/shared-libs
RUN chown -R user:user /app

USER user

EXPOSE 5007

CMD ["python", "main.py"]
