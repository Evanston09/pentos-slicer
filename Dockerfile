FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends prusa-slicer \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev

COPY . .

EXPOSE 8080

CMD ["uv", "run", "main.py"]
