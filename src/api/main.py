import asyncio
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from uvicorn import run

from src.api.logger import get_logger, setup_logging
from src.scoring.recommender import recommend
from src.training.trainer import train

setup_logging()
logger = get_logger(__name__)

RETRAIN_INTERVAL_SECONDS = 300

_train_lock = threading.Lock()


def _locked_train():
	with _train_lock:
		return train()


async def _retrain_forever() -> None:
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
	task = asyncio.create_task(_retrain_forever())
	yield
	task.cancel()


app = FastAPI(title="Pollpear AI", version="0.1.0", lifespan=lifespan)

Instrumentator().instrument(app).expose(app) #route /metrics for prometheus metrics


def _model():
	if getattr(app.state, "model", None) is None:
		with _train_lock:
			if getattr(app.state, "model", None) is None:
				logger.info("No model in memory, training")
				app.state.model = train()
	return app.state.model


@app.get("/health")
def health() -> dict[str, str]:
	logger.info("Health check requested")
	return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
	logger.info("Root endpoint requested")
	return {"service": "pollpear-ai", "status": "ok"}


@app.get("/feed/{user_id}")
def get_feed(user_id: int):
	logger.info(f"Feed requested for user_id: {user_id}")
	feed = recommend(_model(), user_id)
	logger.info(f"Feed returned for user_id: {user_id} ({len(feed)} polls)")
	return {"user_id": user_id, "feed": feed}


def start() -> None:
	run("src.api.main:app", host="0.0.0.0", port=3002, log_level="info", reload=True)


if __name__ == "__main__":
	logger.info("Pollpear AI service started")
	start()
