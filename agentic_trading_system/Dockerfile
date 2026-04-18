# Use the official, lightweight Python 3.11 image
FROM python:3.11-slim

# STRICT REQUIREMENT: Force Python to print logs immediately to Docker
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the core logic and all services into the container
COPY core/ ./core/
COPY services/ ./services/
COPY backtests/ ./backtests/

# Python path configuration
ENV PYTHONPATH=/app
