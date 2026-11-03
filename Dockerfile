FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

RUN mkdir -p data/processed

CMD ["uv", "run", "--no-sync", "uvicorn", "sbp_regulations_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]