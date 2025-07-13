# /opt/swabscan/Dockerfile

FROM python:3.12-slim

# Install build tools + OpenCV deps in one RUN
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      libgl1 \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

