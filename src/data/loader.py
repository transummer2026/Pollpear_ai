import threading
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
from sqlalchemy import Engine, create_engine, text

from src.vault.client import from_env

INTERACTIONS_SQL = """
SELECT DISTINCT v.created_by AS user_id, o.poll_id AS poll_id
FROM votes v
JOIN vote_options vo ON vo.vote_id = v.id
JOIN options o ON o.id = vo.option_id
"""

POLL_HASHTAGS_SQL = """
SELECT ph.poll_id AS poll_id, h.name AS hashtag
FROM poll_hashtags ph
JOIN hashtags h ON h.id = ph.hashtag_id
"""

CANDIDATE_POLLS_SQL = """
SELECT id AS poll_id, title, type, status, visibility
FROM polls
WHERE status = 'active' AND visibility = 'public'
"""

FEED_SQL = """
SELECT id AS poll_id, title
FROM polls
WHERE status = 'active' AND visibility = 'public'
ORDER BY id
LIMIT :limit
"""

_engine: Engine | None = None
_engine_lock = threading.Lock()


def _dsn() -> str: #data source name - strip schema from url
    raw = from_env().read_kv("pollpear/db")["database_url"]
    parts = urlsplit(raw)
    params = [(k, v) for k, v in parse_qsl(parts.query) if k != "schema"]
    return urlunsplit(("postgresql+psycopg", parts.netloc, parts.path, urlencode(params), ""))


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = create_engine(_dsn(), pool_pre_ping=True)
    return _engine


def _query(sql: str, params: dict[str, int] | None = None) -> pd.DataFrame:
    return pd.read_sql_query(text(sql), get_engine(), params=params)


def load_interactions() -> pd.DataFrame:
    return _query(INTERACTIONS_SQL)


def load_poll_hashtags() -> pd.DataFrame:
    return _query(POLL_HASHTAGS_SQL)


def load_candidate_polls() -> pd.DataFrame:
    return _query(CANDIDATE_POLLS_SQL)


def load_feed(limit: int = 9) -> pd.DataFrame:
    return _query(FEED_SQL, params={"limit": limit})
