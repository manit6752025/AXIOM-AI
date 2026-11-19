FROM python:3.11-slim

LABEL org.opencontainers.image.title="AXIOM AI" \
      org.opencontainers.image.description="Local-first AI CLI, Control Center, and workspace API." \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/manit6752025/AXIOM-AI"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ARG AXIOM_EXTRAS=""

WORKDIR /app
COPY pyproject.toml README.md ./
COPY axiom ./axiom

RUN set -eu; \
    if [ -n "$AXIOM_EXTRAS" ]; then \
      python -m pip install --no-cache-dir ".[$AXIOM_EXTRAS]"; \
    else \
      python -m pip install --no-cache-dir .; \
    fi; \
    useradd --system --uid 10001 --create-home --home-dir /home/axiom axiom; \
    install -d -o 10001 -g 10001 /workspace

WORKDIR /workspace
USER 10001:10001

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=3).read()"]

# The service binds inside the container. Publish the host port to 127.0.0.1
# unless you have deliberately reviewed and secured LAN access.
CMD ["axiom", "daemon", "--host", "0.0.0.0", "--port", "8765", "--allow-network"]
