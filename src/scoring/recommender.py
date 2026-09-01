"""Hybrid feed recommender.

Each candidate poll is scored on five independent signals -- content
similarity, co-vote similarity, recency, popularity and social proximity --
which are normalized and blended with :data:`WEIGHTS`. Every signal degrades
to zeros when its inputs are missing, so a partially trained model still
returns a usable ranking.
"""

import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.models.feed_model import FeedModel

# Default blend of the five signals; the weights sum to 1.0.
WEIGHTS = {"content": 0.1, "covotes": 0.2, "recency": 0.3, "popularity": 0.3, "social": 0.1}
# Days after which the recency score halves (exponential decay).
RECENCY_HALF_LIFE_DAYS = 3.0
# Pseudo-observation window (days) that smooths the popularity rate toward the global mean.
POPULARITY_PRIOR_DAYS = 2.0


def _zeros(candidate_ids: list[int]) -> pd.Series:
    """Return an all-zero score Series indexed by candidate poll id."""
    return pd.Series(0.0, index=candidate_ids)


def _normalized(series: pd.Series) -> pd.Series:
    """Scale scores to [0, 1] by dividing by the max (no-op when all zero)."""
    top = series.max()
    return series / top if top > 0 else series


def _content_scores(model: FeedModel, voted: set[int], candidate_ids: list[int]) -> pd.Series:
    """Score candidates by TF-IDF similarity to the user's hashtag profile.

    The profile is the mean TF-IDF vector of the polls the user voted on;
    each candidate scores by cosine similarity to it. Returns zeros when the
    model has no hashtag features or the user has no usable voting history.
    """
    if model.vectorizer is None or model.hashtag_matrix.shape[1] == 0:
        return _zeros(candidate_ids)
    rows = {poll_id: i for i, poll_id in enumerate(model.hashtag_poll_ids)}
    voted_rows = [rows[poll_id] for poll_id in voted if poll_id in rows]
    if not voted_rows:
        return _zeros(candidate_ids)
    # Average the voted polls' TF-IDF vectors into a single taste profile.
    profile = model.hashtag_matrix[voted_rows].mean(axis=0, keepdims=True)
    if not profile.any():
        return _zeros(candidate_ids)
    sims = cosine_similarity(profile, model.hashtag_matrix)[0]
    return pd.Series({poll_id: float(sims[rows[poll_id]]) for poll_id in candidate_ids})


def _covote_scores(model: FeedModel, voted: set[int], candidate_ids: list[int]) -> pd.Series:
    """Score each candidate by its strongest co-vote link to a voted poll.

    Uses the precomputed item-item similarity: a candidate scores as the
    maximum similarity between it and any poll the user has already voted on.
    """
    idx = {poll_id: i for i, poll_id in enumerate(model.interaction_poll_ids)}
    voted_rows = [idx[poll_id] for poll_id in voted if poll_id in idx]
    if not voted_rows:
        return _zeros(candidate_ids)
    return pd.Series(
        {
            poll_id: float(model.item_similarity[voted_rows, idx[poll_id]].max()) if poll_id in idx else 0.0
            for poll_id in candidate_ids
        }
    )


def _age_days(candidates: pd.DataFrame, now: datetime) -> np.ndarray:
    """Return each poll's age in days, using published_at (else created_at).

    Ages are clipped at zero so a future-dated timestamp cannot yield a
    negative age.
    """
    published = pd.to_datetime(candidates["published_at"].fillna(candidates["created_at"]), utc=True)
    return ((now - published).dt.total_seconds() / 86400.0).clip(lower=0.0).to_numpy()


def _recency_scores(candidates: pd.DataFrame, now: datetime) -> pd.Series:
    """Score newer polls higher via exponential decay with a fixed half-life."""
    # ln(2)/half_life is the decay constant that makes the score halve every half-life.
    decay = np.exp(-(math.log(2) / RECENCY_HALF_LIFE_DAYS) * _age_days(candidates, now))
    return pd.Series(decay, index=[int(poll_id) for poll_id in candidates["poll_id"]])


def _popularity_scores(
    model: FeedModel, candidates: pd.DataFrame, candidate_ids: list[int], now: datetime
) -> pd.Series:
    """Score candidates by an age-adjusted, smoothed vote rate.

    A raw count favors old polls that had more time to accumulate votes, so
    we use votes-per-day instead. The rate is Bayesian-smoothed toward the
    global rate with a ``POPULARITY_PRIOR_DAYS`` prior, which keeps a brand
    new poll with one vote from dominating on an enormous instantaneous rate.
    """
    counts = model.popularity.reindex(candidate_ids).fillna(0.0).to_numpy()
    # Floor age at one hour so a just-published poll has a finite vote rate.
    ages = np.maximum(_age_days(candidates, now), 1.0 / 24.0)
    global_rate = counts.sum() / ages.sum() if ages.sum() > 0 else 0.0
    rates = (counts + POPULARITY_PRIOR_DAYS * global_rate) / (ages + POPULARITY_PRIOR_DAYS)
    return _normalized(pd.Series(rates, index=candidate_ids))


def _friends_of(model: FeedModel, user_id: int) -> set[int]:
    """Return the user's friend ids.

    Friendships are stored once as an unordered pair, so the user may appear
    on either side; collect partners from both columns.
    """
    friendships = model.friendships
    if friendships.empty:
        return set()
    partners_b = friendships.loc[friendships["user_a_id"] == user_id, "user_b_id"]
    partners_a = friendships.loc[friendships["user_b_id"] == user_id, "user_a_id"]
    return {int(user) for user in partners_a} | {int(user) for user in partners_b}


def _social_scores(model: FeedModel, user_id: int, candidate_ids: list[int]) -> pd.Series:
    """Give a candidate a full score if a friend created or voted on it.

    Social proximity is binary (1.0 or 0.0): a poll counts as socially
    relevant when its creator is a friend or any friend voted on it.
    """
    friends = _friends_of(model, user_id)
    if not friends:
        return _zeros(candidate_ids)
    interactions = model.interactions
    friend_votes = (
        set()
        if interactions.empty
        else {int(poll_id) for poll_id in interactions.loc[interactions["user_id"].isin(friends), "poll_id"]}
    )
    creators = {
        int(poll_id): int(creator)
        for poll_id, creator in zip(model.candidates["poll_id"], model.candidates["created_by"], strict=True)
    }
    return pd.Series(
        {
            poll_id: 1.0 if creators.get(poll_id) in friends or poll_id in friend_votes else 0.0
            for poll_id in candidate_ids
        }
    )


def _excluded_users(model: FeedModel, user_id: int) -> set[int]:
    """Return every user whose polls must be hidden from ``user_id``.

    Blocking is treated symmetrically for the feed: exclude users this user
    blocked *and* users who blocked this user.
    """
    blocks = model.blocks
    if blocks.empty:
        return set()
    blocked = blocks.loc[blocks["blocker_id"] == user_id, "blocked_id"]
    blockers = blocks.loc[blocks["blocked_id"] == user_id, "blocker_id"]
    return {int(user) for user in blocked} | {int(user) for user in blockers}


def _rooms_of(model: FeedModel, user_id: int) -> set[int]:
    memberships = model.room_memberships
    if memberships.empty:
        return set()
    return {int(room) for room in memberships.loc[memberships["user_id"] == user_id, "room_id"]}


def recommend(
    model: FeedModel,
    user_id: int,
    limit: int = 9,
    now: datetime | None = None,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """Return the top ``limit`` recommended polls for ``user_id``.

    Polls the user already voted on, and polls by excluded (blocked) users,
    are removed before scoring. The five signals are blended with ``weights``
    (default :data:`WEIGHTS`) and the highest-scoring polls are returned as
    ``{poll_id, title, score}`` dicts. ``now`` is injectable so evaluation
    and tests can score against a fixed point in time.
    """
    if model.candidates.empty:
        return []
    blend = weights if weights is not None else WEIGHTS
    at = now if now is not None else datetime.now(UTC)
    interactions = model.interactions
    voted = (
        set()
        if interactions.empty
        else {int(poll_id) for poll_id in interactions.loc[interactions["user_id"] == user_id, "poll_id"]}
    )
    excluded = _excluded_users(model, user_id)
    # Keep only polls the user should see: not already voted on, not authored by a
    # blocked user, and — for room polls — only rooms the user belongs to. Public-feed
    # polls carry no room_id and always pass the membership test.
    rooms = _rooms_of(model, user_id)
    room_ids = model.candidates["room_id"]
    keep = (
        ~model.candidates["poll_id"].astype(int).isin(voted)
        & ~model.candidates["created_by"].astype(int).isin(excluded)
        & (room_ids.isna() | room_ids.fillna(-1).astype(int).isin(rooms))
    )
    candidates = model.candidates[keep]
    candidate_ids = [int(poll_id) for poll_id in candidates["poll_id"]]
    if not candidate_ids:
        return []
    scores = pd.DataFrame(
        {
            "content": _content_scores(model, voted, candidate_ids),
            "covotes": _covote_scores(model, voted, candidate_ids),
            "recency": _recency_scores(candidates, at),
            "popularity": _popularity_scores(model, candidates, candidate_ids, at),
            "social": _social_scores(model, user_id, candidate_ids),
        }
    )
    # Weighted blend of the per-signal score columns into a single ranking score.
    total = sum(weight * scores[name] for name, weight in blend.items())
    top = total.sort_values(ascending=False).head(limit)
    titles = dict(zip(candidate_ids, candidates["title"], strict=True))
    return [
        {"poll_id": int(poll_id), "title": str(titles[int(poll_id)]), "score": round(float(score), 4)}
        for poll_id, score in top.items()
    ]
