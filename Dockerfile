# Build:  docker build -t wdbc-api .
# Run:    docker run -p 8000:8000 wdbc-api      then open http://localhost:8000/docs
#
# The model is trained during the build, so the image is self-contained and
# every build from the same code + data gives the same model (fixed seeds).

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Editable install (-e) keeps the code in /app/src, so the package finds
# /app/configs and /app/data relative to itself.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[api]"

COPY configs ./configs
COPY data ./data
RUN python -m wdbc.train

# Run as a non-root user (a basic container-security habit).
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "wdbc.api:app", "--host", "0.0.0.0", "--port", "8000"]
