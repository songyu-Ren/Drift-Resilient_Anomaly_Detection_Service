FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY config /app/config
COPY src /app/src
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
RUN mkdir -p /app/artifacts

ENV DDS_SETTINGS=/app/config/settings.yaml

# Train during build so model artifacts are present at runtime
RUN python -c 'from src.drift_detect_service.train import train_and_save; train_and_save()'
EXPOSE 8000

CMD ["uvicorn", "src.drift_detect_service.api:app", "--host", "0.0.0.0", "--port", "8000"]