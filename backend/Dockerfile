FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for build and geospatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements list
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend source files, datasets, and ML configurations
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Set environment variable defaults
ENV PORT=8000

# Start application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
