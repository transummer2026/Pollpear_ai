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
SELECT id AS poll_id, title, created_by, published_at, created_at
FROM polls
WHERE status = 'active' AND visibility = 'public' AND deleted_at IS NULL
"""

FRIENDSHIPS_SQL = """
SELECT user_a_id, user_b_id
FROM friendships
WHERE status = 'accepted' AND deleted_at IS NULL
"""

BLOCKS_SQL = """
SELECT blocker_id, blocked_id
FROM blocks
"""

TIMED_INTERACTIONS_SQL = """
SELECT v.created_by AS user_id, o.poll_id AS poll_id, MIN(v.created_at) AS voted_at
FROM votes v
JOIN vote_options vo ON vo.vote_id = v.id
JOIN options o ON o.id = vo.option_id
GROUP BY v.created_by, o.poll_id
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


def load_friendships() -> pd.DataFrame:
    return _query(FRIENDSHIPS_SQL)


def load_blocks() -> pd.DataFrame:
    return _query(BLOCKS_SQL)


def load_timed_interactions() -> pd.DataFrame:
    return _query(TIMED_INTERACTIONS_SQL)
