FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.8 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
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
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for Python 3.8
RUN ln -sf /usr/bin/python3.8 /usr/bin/python3
RUN ln -sf /usr/bin/python3.8 /usr/bin/python

# Install pip for Python 3.8
RUN curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8

# Set environment variables (simplified)
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

RUN useradd -m -u 1000 user
WORKDIR /app

# Install PyTorch 1.8.1 with CUDA 11.1
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Clone and install PointNet2
RUN git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git /app/pointnet2
WORKDIR /app/pointnet2/pointnet2_ops_lib
RUN pip install .

# Clone GeDi repository
WORKDIR /app
RUN git clone https://github.com/fabiopoiesi/gedi.git /app/gedi-repo

# Install Python dependencies
COPY containers/query-gedi/requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Create directories and set ownership
RUN mkdir -p /app/output/gedi_features /app/data /app/logs && \
    chown -R user:user /app

COPY containers/query-gedi/ /app/
COPY shared-libs /app/shared-libs
RUN chown -R user:user /app

USER user

# Set Python path to include GeDi and PointNet2
ENV PYTHONPATH="/app/gedi-repo:/app/pointnet2"

EXPOSE 5010

CMD ["python", "main.py"]

