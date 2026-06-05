# ==============================================================================
# STAGE 1: BUILD STAGE
# ==============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install compilation tools necessary for libraries like tiktoken and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements to leverage Docker cache
COPY requirements.txt .

# Install dependencies into a virtual environment to keep runtime footprint small
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install wheel and key dependencies (using CPU-only index for PyTorch to reduce image size)
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# STAGE 2: RUNTIME STAGE
# ==============================================================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime system packages (e.g. libmagic or other dependencies if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy python virtual environment from build stage
COPY --from=builder /opt/venv /opt/venv

# Set environment paths to use virtual environment packages
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Copy application source code
COPY app/ ./app/

# Create a non-root security user to execute the container
RUN useradd -u 10001 researcher && \
    chown -R researcher:researcher /app

USER researcher

EXPOSE 8000

# Health check utilizing built-in python urllib instead of requiring curl package bloat
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
