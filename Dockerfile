FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    WEB_SERVER_HOST=0.0.0.0

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .

EXPOSE 8080
VOLUME ["/app/logs"]

CMD ["uv", "run", "--frozen", "python", "main.py", "--strategy", "merton"]
