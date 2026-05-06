FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1   

ENV PYTHONUNBUFFERED=1          

ENV PYTHONPATH=/app             

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY predict.py .
COPY best_efficientnet_raw.pth .

RUN mkdir -p tmp_uploads

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]