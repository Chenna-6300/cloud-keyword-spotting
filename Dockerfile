FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

# Explicitly guarantee this dependency exists
RUN pip install --no-cache-dir six

COPY . .

RUN chmod +x modelfile.eim

ENV PORT=10000

CMD ["sh", "-c", "gunicorn -w 1 -b 0.0.0.0:${PORT} cloud_server:app"]
