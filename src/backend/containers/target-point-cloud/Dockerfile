FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libgomp1 \
    libopencv-dev \
    cmake \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Remove multithreading environment variables for simplified processing
ENV PYTHONUNBUFFERED=1

RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python dependencies
COPY target-point-cloud/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Create directories
RUN mkdir -p /app/output/point_clouds /app/data /app/logs /app/temp && \
    chown -R user:user /app

COPY target-point-cloud/ /app/
COPY shared-libs /app/shared-libs
RUN chown -R user:user /app

USER user

EXPOSE 5003

CMD ["python", "main.py"]
