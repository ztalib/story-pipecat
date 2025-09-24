FROM python:3.11-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and audio
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    libasound2-dev \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with specific order
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-dotenv websockets aiohttp
RUN pip install --no-cache-dir pipecat-ai[webrtc,daily,silero,deepgram,openai,cartesia,runner]>=0.0.83

# Copy application code
COPY . .

# Create conversations directory
RUN mkdir -p conversations

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
