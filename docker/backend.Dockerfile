FROM python:3.10-slim

# System deps: curl (healthchecks), build tools (hnswlib compilation), Java (PySpark)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl build-essential default-jre procps && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . /app

# Create output directories
RUN mkdir -p outputs/eda outputs/uploads outputs/metadata/basic \
             outputs/metadata/enriched outputs/metadata/uploads \
             outputs/metadata/stories models

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
