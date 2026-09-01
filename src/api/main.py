"""FastAPI entrypoint for the Pollpear AI feed service.

Exposes ``/feed/{user_id}`` plus health/root probes, trains the model lazily
on first use, and retrains it in a background loop so the served model stays
fresh without blocking requests. Prometheus metrics are exposed at /metrics.
"""

import asyncio
import threading
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Query
from prometheus_fastapi_instrumentator import Instrumentator
from uvicorn import run

from src.api.logger import get_logger, setup_logging
from src.scoring.recommender import recommend
from src.training.trainer import train

setup_logging()
logger = get_logger(__name__)

# Interval between background retrains, in seconds.
RETRAIN_INTERVAL_SECONDS = 300

_train_lock = threading.Lock()


def _locked_train():
	"""Train under a lock so at most one training run happens at a time."""
	with _train_lock:
		return train()


async def _retrain_forever() -> None:
	"""Retrain on a fixed interval, swapping in the new model on success.

	Training runs in a worker thread so the event loop keeps serving requests.
	If a retrain fails the previous model is kept and the loop continues, so a
	transient database outage degrades freshness rather than availability.
	"""
	while True:
		await asyncio.sleep(RETRAIN_INTERVAL_SECONDS)
		try:
			model = await asyncio.to_thread(_locked_train)
		except Exception:
			logger.exception("Periodic retraining failed, keeping previous model")
			continue
		app.state.model = model
		logger.info(f"Model retrained: {len(model.poll_ids)} candidate polls, {len(model.interactions)} interactions")


@asynccontextmanager
async def lifespan(app: FastAPI):
	"""Start the background retrain task for the app's lifetime and cancel it on shutdown."""
	task = asyncio.create_task(_retrain_forever())
	yield
	task.cancel()


app = FastAPI(title="Pollpear AI", version="0.1.0", lifespan=lifespan)

Instrumentator().instrument(app).expose(app)  # Expose Prometheus metrics at /metrics


def _model():
	"""Return the current model, training it synchronously on first access.

	The first request before the background loop has produced a model trains
	one under the lock (double-checked) instead of failing.
	"""
	if getattr(app.state, "model", None) is None:
		with _train_lock:
			if getattr(app.state, "model", None) is None:
				logger.info("No model in memory, training")
				app.state.model = train()
	return app.state.model


@app.get("/health")
def health() -> dict[str, str]:
	"""Liveness probe: always returns ok if the process is up."""
	logger.info("Health check requested")
	return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
	"""Return a small service identity banner."""
	logger.info("Root endpoint requested")
	return {"service": "pollpear-ai", "status": "ok"}


@app.get("/feed/{user_id}")
def get_feed(user_id: int, limit: Annotated[int, Query(ge=1, le=100)] = 9):
	"""Return the recommended feed for ``user_id`` (``limit`` polls, 1..100)."""
	logger.info(f"Feed requested for user_id: {user_id} (limit={limit})")
	feed = recommend(_model(), user_id, limit=limit)
	logger.info(f"Feed returned for user_id: {user_id} ({len(feed)} polls)")
	return {"user_id": user_id, "feed": feed}


def start() -> None:
	"""Launch the ASGI server (development entrypoint, autoreload enabled)."""
	run("src.api.main:app", host="0.0.0.0", port=3002, log_level="info", reload=True)


if __name__ == "__main__":
	logger.info("Pollpear AI service started")
	start()
