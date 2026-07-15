from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"service": "pollpear-ai", "status": "ok"}


def test_feed_known_user():
    r = client.get("/feed/1")
    assert r.status_code == 200
    assert r.json() == {"user_id": "1", "feed": "Feed for user 1"}


def test_feed_unknown_user():
    r = client.get("/feed/99")
    assert r.status_code == 404
    assert r.json() == {"error": "User not found"}
