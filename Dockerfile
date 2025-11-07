# Dockerfile for Portfolio Manager Agent testing
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install Databricks CLI
RUN curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY tests/ tests/
COPY infra/ infra/
COPY scripts/ scripts/
COPY notebooks/ notebooks/

# Set Python path
ENV PYTHONPATH=/app/src

# Default command
CMD ["python", "-m", "pytest", "tests/", "-v"]