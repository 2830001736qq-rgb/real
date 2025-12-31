FROM nvcr.io/nvidia/pytorch:24.10-py3

RUN pip install git+https://github.com/huggingface/diffusers gradio pillow

WORKDIR /app
COPY app.py .
CMD ["python3", "app.py"]
