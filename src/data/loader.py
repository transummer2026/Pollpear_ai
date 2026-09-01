"""Read-only loaders that pull training data from the Postgres database.

The connection string is fetched from Vault at first use; each loader runs
one query and returns a pandas DataFrame. Every query filters out
soft-deleted rows (``deleted_at IS NULL``) so the model never trains on
deleted content.
"""

import threading
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
from sqlalchemy import Engine, create_engine, text

from src.vault.client import from_env

# One (user_id, poll_id) row per user/poll pair the user has voted on.
# Note: ``pollId`` is a quoted camelCase column, hence the double quotes.
INTERACTIONS_SQL = """
SELECT DISTINCT v.created_by AS user_id, v."pollId" AS poll_id
FROM votes v
WHERE v.deleted_at IS NULL
"""

POLL_HASHTAGS_SQL = """
SELECT ph.poll_id AS poll_id, h.name AS hashtag
FROM poll_hashtags ph
JOIN hashtags h ON h.id = ph.hashtag_id
"""

# Polls eligible to appear in a feed: active, public, and either not tied
# to a room or tied to a public room (private-room polls are excluded).
CANDIDATE_POLLS_SQL = """
SELECT p.id AS poll_id, p.title, p.created_by, p.published_at, p.created_at, p.room_id
FROM polls p
WHERE p.status IN ('active', 'pending') AND p.visibility = 'public' AND p.deleted_at IS NULL
  AND (
    p.room_id IS NULL
    OR EXISTS (
      SELECT 1 FROM rooms r
      WHERE r.id = p.room_id AND r.visibility = 'public'
    )
  )
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

ROOM_MEMBERSHIPS_SQL = """
SELECT room_id, user_id
FROM room_members
WHERE deleted_at IS NULL
"""

TIMED_INTERACTIONS_SQL = """
SELECT v.created_by AS user_id, v."pollId" AS poll_id, MIN(v.created_at) AS voted_at
FROM votes v
WHERE v.deleted_at IS NULL
GROUP BY v.created_by, v."pollId"
"""

_engine: Engine | None = None
_engine_lock = threading.Lock()


def _dsn() -> str:
    """Build the SQLAlchemy DSN from the ``pollpear/db`` secret in Vault.

    The stored URL is normalized to the ``postgresql+psycopg`` driver and its
    ``schema`` query parameter is dropped, since libpq/psycopg does not accept
    it as a connection option.
    """
    raw = from_env().read_kv("pollpear/db")["database_url"]
    parts = urlsplit(raw)
    params = [(k, v) for k, v in parse_qsl(parts.query) if k != "schema"]
    return urlunsplit(("postgresql+psycopg", parts.netloc, parts.path, urlencode(params), ""))


def get_engine() -> Engine:
    """Return the shared SQLAlchemy engine, creating it once on first call.

    Uses double-checked locking so concurrent callers build a single engine.
    ``pool_pre_ping`` validates pooled connections before reuse, tolerating
    connections dropped by the database between retraining cycles.
    """
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = create_engine(_dsn(), pool_pre_ping=True)
    return _engine


def _query(sql: str, params: dict[str, int] | None = None) -> pd.DataFrame:
    """Run ``sql`` against the shared engine and return the result as a DataFrame."""
    return pd.read_sql_query(text(sql), get_engine(), params=params)


def load_interactions() -> pd.DataFrame:
    """Return every distinct (user_id, poll_id) vote used as positive signal."""
    return _query(INTERACTIONS_SQL)


def load_poll_hashtags() -> pd.DataFrame:
    """Return (poll_id, hashtag) rows used to build the content features."""
    return _query(POLL_HASHTAGS_SQL)


def load_candidate_polls() -> pd.DataFrame:
    """Return the polls eligible to be recommended, with their metadata."""
    return _query(CANDIDATE_POLLS_SQL)


def load_friendships() -> pd.DataFrame:
    """Return accepted friendships as undirected (user_a_id, user_b_id) pairs."""
    return _query(FRIENDSHIPS_SQL)


def load_blocks() -> pd.DataFrame:
    """Return (blocker_id, blocked_id) rows used to filter blocked users out."""
    return _query(BLOCKS_SQL)


def load_room_memberships() -> pd.DataFrame:
    return _query(ROOM_MEMBERSHIPS_SQL)


def load_timed_interactions() -> pd.DataFrame:
    """Return each user/poll vote with the earliest vote time (for the eval split)."""
    return _query(TIMED_INTERACTIONS_SQL)
