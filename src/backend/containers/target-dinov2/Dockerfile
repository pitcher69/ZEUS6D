FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libgomp1 \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for Python 3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Simplified environment variables (no aggressive threading)
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python dependencies
COPY target-dinov2/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Create directories
RUN mkdir -p /app/output/dinov2_features /app/data /app/logs && \
    chown -R user:user /app

COPY target-dinov2/ /app/
COPY shared-libs /app/shared-libs
RUN chown -R user:user /app

USER user

EXPOSE 5004

CMD ["python", "main.py"]
