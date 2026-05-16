FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose ports: 8000 for API, 8501 for Streamlit
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000')" || exit 1

# Default: run both API and dashboard
CMD ["sh", "-c", "uvicorn deployment.api.main:app --host 0.0.0.0 --port 8000 & streamlit run deployment/frontend/app.py --server.port=8501 --server.address=0.0.0.0"]
