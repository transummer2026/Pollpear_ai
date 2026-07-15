# syntax=docker/dockerfile:1

FROM python:3.14-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1


COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

FROM builder AS dev
COPY . .
EXPOSE 3002
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "3002", "--reload"]

FROM builder AS prod
COPY . .
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 3002
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "3002", "--workers", "2"]
