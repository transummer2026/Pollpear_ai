from unittest.mock import patch

import pandas as pd
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


def test_feed_returns_polls():
    fake = pd.DataFrame([{"poll_id": 7, "title": "a"}, {"poll_id": 8, "title": "b"}])
    with patch("src.api.main.load_feed", return_value=fake):
        r = client.get("/feed/1")
    assert r.status_code == 200
    body = r.json()
    assert body["user_id"] == 1
    assert body["feed"] == [{"poll_id": 7, "title": "a"}, {"poll_id": 8, "title": "b"}]
