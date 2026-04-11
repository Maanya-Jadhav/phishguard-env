# BUG FIX: pin full patch version for reproducible image builds.
# `python:3.10-slim` floats to whatever the latest 3.10.x is at build time.
FROM python:3.10.14-slim

WORKDIR /app

# Install dependencies before copying source so Docker layer cache is reused
# on code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# BUG FIX: run as a non-root user — running as root in a container is a
# security risk and violates the principle of least privilege.
RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 7860

# BUG FIX: declare a HEALTHCHECK so HF Spaces / orchestrators can detect
# if the server has crashed and restart the container automatically.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" \
  || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
