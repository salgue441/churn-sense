FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md setup.py ./
COPY src/ ./src/
COPY dashboard.py .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Expose port for dashboard
EXPOSE 8050

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CHURNSENSE_ENV=production

# Run dashboard
CMD ["python", "dashboard.py"]