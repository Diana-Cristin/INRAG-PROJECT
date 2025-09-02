# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONUNBUFFERED=1

# Si tu app es Gradio/FastAPI, asegúrate de escuchar en $PORT
CMD ["python", "index.py"]