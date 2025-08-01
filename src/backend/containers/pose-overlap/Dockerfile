FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY containers/pose-overlap/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY containers/pose-overlap/ /app/
COPY shared-libs /app/shared-libs

EXPOSE 5013

CMD ["python", "main.py"]
