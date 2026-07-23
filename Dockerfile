# Slim base: the API needs sklearn + mlflow at runtime, nothing that compiles.
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/exported_model

# Requirements first so the dependency layer caches across code edits.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then the model's own pins, which MLflow regenerates on every export.
# requirements.txt already pins the model-critical libraries to these same
# versions, so this layer is normally a no-op. It stays because it is the file
# MLflow rewrites automatically on every retrain: if a future export moves to a
# newer scikit-learn and requirements.txt is not updated to match, this line
# still pins serving to the environment that actually wrote the pickle, instead
# of failing at load with "Can't get attribute '_RemainderColsList'".
COPY exported_model/ ./exported_model/
RUN pip install --no-cache-dir -r exported_model/requirements.txt

COPY app/ ./app/

# Non-root: the container has no reason to run privileged.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.model_server:app", "--host", "0.0.0.0", "--port", "8000"]
