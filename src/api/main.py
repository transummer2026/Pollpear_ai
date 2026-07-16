from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from uvicorn import run

from src.api.logger import get_logger, setup_logging
from src.data.loader import load_feed

setup_logging()
logger = get_logger(__name__)

app = FastAPI(title="Pollpear AI", version="0.1.0")

Instrumentator().instrument(app).expose(app) #route /metrics for prometheus metrics


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
	rows = load_feed(9)
	pairs = zip(rows.poll_id.tolist(), rows.title.tolist(), strict=True)
	feed = [{"poll_id": int(pid), "title": str(title)} for pid, title in pairs]
	logger.info(f"Feed returned for user_id: {user_id} ({len(feed)} polls)")
	return {"user_id": user_id, "feed": feed}



def start() -> None:
	run("src.api.main:app", host="0.0.0.0", port=3002, log_level="info", reload=True)


if __name__ == "__main__":
	logger.info("Pollpear AI service started")
	start()
