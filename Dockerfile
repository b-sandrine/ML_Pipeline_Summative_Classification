# API image: FastAPI + TensorFlow feature extractor + trained model
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TF_CPP_MIN_LOG_LEVEL=2
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the ImageNet weights at build time so containers start offline-fast
RUN python -c "from tensorflow.keras.applications import MobileNetV2; MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(160,160,3))"

COPY src ./src
COPY models ./models
COPY data/train/.gitkeep data/train/.gitkeep
COPY data/test/.gitkeep data/test/.gitkeep

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
