import os
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import pandas as pd
import psycopg

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


def _dsn() -> str:
    raw = os.environ.get("POSTGRES_URL")
    if not raw:
        raise RuntimeError("POSTGRES_URL is not set")
    parts = urlsplit(raw)
    params = [(k, v) for k, v in parse_qsl(parts.query) if k != "schema"]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), parts.fragment))


def _query(sql: str) -> pd.DataFrame:
    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            columns = [c.name for c in cur.description]
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


def load_interactions() -> pd.DataFrame:
    return _query(INTERACTIONS_SQL)


def load_poll_hashtags() -> pd.DataFrame:
    return _query(POLL_HASHTAGS_SQL)


def load_candidate_polls() -> pd.DataFrame:
    return _query(CANDIDATE_POLLS_SQL)
