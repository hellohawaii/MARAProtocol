FROM python:3.11-slim

# Customize this image to control the shell tool runtime.
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*
