FROM ghcr.io/astral-sh/uv:bookworm-slim

# Install build tools required for httptools / uvicorn
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the project into the image
ADD . /app

# Sync the project into a new environment, asserting the lockfile is up to date
WORKDIR /app
RUN uv sync --locked

# Start proxy
CMD ["uv", "run", "start_proxy.py"]
