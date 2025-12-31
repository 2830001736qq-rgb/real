FROM nvcr.io/nvidia/pytorch:24.10-py3

# Install everything in one layer to speed up builds
RUN pip install --no-cache-dir \
    git+https://github.com/huggingface/diffusers \
    gradio \
    pillow \
    transformers>=4.44.0 \
    accelerate>=0.33.0 \
    huggingface-hub \
    qwen-vl-utils \
    timm \
    einops \
    torchvision \
    torch \
    safetensors



WORKDIR /app
COPY app.py .
CMD ["python3", "app.py"]
