FROM python:3.9-slim

WORKDIR /app

# 1. Install system dependencies more efficiently
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

# 2. Copy only requirements first for better caching
COPY requirements.txt .

# 3. Install with timeout settings and retries
RUN pip install --no-cache-dir \
    --default-timeout=100 \
    --retries 5 \
    -r requirements.txt

# 4. Copy the rest
COPY . .

# 5. Explicitly copy static files
COPY static/ /app/static/

EXPOSE 8000
CMD ["uvicorn", "students:app", "--host", "0.0.0.0", "--port", "8000"]