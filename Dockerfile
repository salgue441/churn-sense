# churnsense/docker/Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.5.1 \
  CHURNSENSE_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"
RUN poetry config virtualenvs.create false

# Copy project files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy application code
COPY churnsense/ /app/churnsense/

# Copy data directory structure
COPY data/ /app/data/

# Copy models directory
COPY models/ /app/models/

# Create reports directory if doesn't exist
RUN mkdir -p /app/reports/figures /app/reports/results

# Expose port for the dashboard
EXPOSE 8050

# Create a non-root user and switch to it
RUN useradd --create-home appuser
USER appuser

# Set the entrypoint
ENTRYPOINT ["python", "-m", "churnsense.cli.main"]

# Set the default command
CMD ["dashboard", "--port", "8050"]