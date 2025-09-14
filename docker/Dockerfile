# =========================
# Stage 1: Base Image
# =========================
FROM python:3.10-slim

# Avoid interactive tzdata prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (OpenCV, Mediapipe, TensorFlow needs these)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# =========================
# Stage 2: Dependencies
# =========================
# Copy requirements first for caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# =========================
# Stage 3: Application Code
# =========================
# Copy the full app
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
