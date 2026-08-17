from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from uvicorn import run

from src.api.logger import get_logger, setup_logging
from src.scoring.recommender import recommend
from src.training.trainer import train

setup_logging()
logger = get_logger(__name__)

app = FastAPI(title="Pollpear AI", version="0.1.0")

Instrumentator().instrument(app).expose(app) #route /metrics for prometheus metrics


def _model():
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


@app.post("/train")
def post_train():
	model = train()
	app.state.model = model
	logger.info(f"Model trained: {len(model.poll_ids)} candidate polls, {len(model.interactions)} interactions")
	return {
		"status": "trained",
		"trained_at": model.trained_at.isoformat(),
		"candidate_polls": len(model.poll_ids),
		"interactions": len(model.interactions),
	}


def start() -> None:
	run("src.api.main:app", host="0.0.0.0", port=3002, log_level="info", reload=True)


if __name__ == "__main__":
	logger.info("Pollpear AI service started")
	start()
