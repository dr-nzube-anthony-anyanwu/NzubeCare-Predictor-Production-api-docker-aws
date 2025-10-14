FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY streamlit_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY streamlit_app .
COPY ml_models ./ml_models/

# Create .streamlit directory and copy secrets
RUN mkdir -p /app/.streamlit
COPY .streamlit/secrets.toml /app/.streamlit/secrets.toml

# Create non-root user
RUN groupadd -r streamlituser && useradd -r -g streamlituser streamlituser
RUN chown -R streamlituser:streamlituser /app
USER streamlituser

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app_api_version.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]
