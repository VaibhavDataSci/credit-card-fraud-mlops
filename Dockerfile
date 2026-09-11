FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_ALLOW_FILE_STORE=true

WORKDIR /app

RUN groupadd --system appuser && useradd --system --gid appuser appuser

COPY requirements-api.txt ./
RUN pip install --no-cache-dir --timeout 180 --retries 5 -r requirements-api.txt

COPY --chown=appuser:appuser params.yaml ./
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser src/__init__.py ./src/__init__.py
COPY --chown=appuser:appuser src/features ./src/features
COPY --chown=appuser:appuser src/monitoring ./src/monitoring
COPY --chown=appuser:appuser src/models ./src/models
COPY --chown=appuser:appuser scripts/detect_drift.py ./scripts/detect_drift.py
COPY --chown=appuser:appuser scripts/retrain_model.py ./scripts/retrain_model.py
COPY --chown=appuser:appuser scripts/approve_model.py ./scripts/approve_model.py

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
