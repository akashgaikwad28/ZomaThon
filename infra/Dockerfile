# Use official Python slim image for production-ready size
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (needed for lightgbm)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY infra/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/
COPY data/ data/

# Expose port
EXPOSE 8000

# Start application (using uvicorn directly for demo)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
