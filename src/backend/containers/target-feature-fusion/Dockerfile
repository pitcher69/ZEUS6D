FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Simplified environment variables (no aggressive threading)
ENV PYTHONUNBUFFERED=1

RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python dependencies
COPY target-feature-fusion/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Create directories
RUN mkdir -p /app/output/fused_features /app/data /app/logs && \
    chown -R user:user /app

COPY target-feature-fusion/ /app/
COPY shared-libs /app/shared-libs
RUN chown -R user:user /app

USER user

EXPOSE 5006

CMD ["python", "main.py"]
