from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app
from src.training.trainer import build_model

client = TestClient(app)


def _toy_model():
    interactions = pd.DataFrame({"user_id": [1, 2], "poll_id": [10, 11]})
    poll_hashtags = pd.DataFrame({"poll_id": [10, 11], "hashtag": ["tech", "food"]})
    candidates = pd.DataFrame(
        {
            "poll_id": [10, 11],
            "title": ["a", "b"],
            "created_by": [3, 4],
            "published_at": [pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-02")],
            "created_at": [pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-02")],
        }
    )
    return build_model(interactions, poll_hashtags, candidates)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"service": "pollpear-ai", "status": "ok"}


def test_feed_returns_polls():
    app.state.model = None
    with patch("src.api.main.train", return_value=_toy_model()):
        r = client.get("/feed/1")
    assert r.status_code == 200
    body = r.json()
    assert body["user_id"] == 1
    assert [p["poll_id"] for p in body["feed"]] == [11]
