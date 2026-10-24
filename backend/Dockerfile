# =====================================================================
# SETU Backend — Dockerfile
# Serves the 26-Class RealSign ISL Recognition Model via FastAPI
# =====================================================================

FROM python:3.10-slim

WORKDIR /app

# Prevent Python from writing bytecode and enable unbuffered terminal output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies required by OpenCV, MediaPipe, and Healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    curl \
    build-essential \
    && (apt-get install -y --no-install-recommends libgl1-mesa-glx || apt-get install -y --no-install-recommends libgl1) \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code, model checkpoints, and top-level launcher
COPY backend/ ./backend/
COPY models/ ./models/
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
