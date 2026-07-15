from uvicorn import run
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from logging import getLogger, StreamHandler, Formatter
from prometheus_fastapi_instrumentator import Instrumentator


logger = getLogger(__name__)
handler = StreamHandler()
formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel("INFO")

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
async def get_feed(user_id: str):
	logger.info(f"Feed requested for user_id: {user_id}")
	if user_id not in ["1", "2", "3"]:
		logger.warning(f"User not found for user_id: {user_id}")
		return JSONResponse(status_code=404, content={"error": "User not found"})
	logger.info(f"Feed returned for user_id: {user_id}")
	return JSONResponse(content={"user_id": user_id, "feed": f"Feed for user {user_id}"})



def start() -> None:
	run("src.api.main:app", host="0.0.0.0", port=3002, log_level="info", reload=True)


if __name__ == "__main__":
	start()
	logger.info("Pollpear AI service started on http://localhost:3002")
