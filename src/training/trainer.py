"""Offline training: turn raw database tables into a :class:`FeedModel`.

This builds the two precomputed structures the scorer needs -- a TF-IDF
matrix over poll hashtags (content signal) and an item-item co-vote
similarity matrix (collaborative signal) -- plus the raw frames the scorer
reads at request time.
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.api.logger import setup_logging
from src.data.loader import (
    load_blocks,
    load_candidate_polls,
    load_friendships,
    load_interactions,
    load_poll_hashtags,
    load_room_memberships,
)
from src.models.feed_model import FeedModel, identity_analyzer


def _hashtag_documents(candidates: pd.DataFrame, poll_hashtags: pd.DataFrame) -> list[list[str]]:
    """Return one hashtag-token list per poll, in ``candidates`` row order.

    Polls with no hashtags yield an empty list, keeping the output aligned
    one-to-one with ``candidates`` so it can index the TF-IDF matrix.
    """
    if poll_hashtags.empty:
        return [[] for _ in range(len(candidates))]
    grouped = poll_hashtags.groupby("poll_id")["hashtag"].apply(list)
    return [list(grouped.get(poll_id, [])) for poll_id in candidates["poll_id"]]


def _build_hashtag_matrix(documents: list[list[str]]) -> tuple[TfidfVectorizer | None, np.ndarray]:
    """Fit a TF-IDF vectorizer over hashtag documents and return a dense matrix.

    When every document is empty the vocabulary would be empty and
    ``TfidfVectorizer`` would raise, so return ``(None, zero-width matrix)``
    as a degraded feature set that the scorer treats as "no content signal".
    """
    if not any(documents):
        return None, np.zeros((len(documents), 0))
    vectorizer = TfidfVectorizer(analyzer=identity_analyzer)
    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix.toarray()


def _build_item_similarity(interactions: pd.DataFrame) -> tuple[list[int], np.ndarray]:
    """Compute item-item cosine similarity from co-voting behavior.

    Builds a binary user x poll matrix and takes the cosine similarity of its
    transpose, so two polls score high when they share many voters. Returns
    the poll_ids aligned to the similarity matrix rows/columns.
    """
    if interactions.empty:
        return [], np.zeros((0, 0))
    votes = pd.crosstab(interactions["user_id"], interactions["poll_id"])
    similarity = cosine_similarity(votes.to_numpy().T)
    return [int(poll_id) for poll_id in votes.columns], similarity


def build_model(
    interactions: pd.DataFrame,
    poll_hashtags: pd.DataFrame,
    candidates: pd.DataFrame,
    friendships: pd.DataFrame | None = None,
    blocks: pd.DataFrame | None = None,
    room_memberships: pd.DataFrame | None = None,
) -> FeedModel:
    """Assemble a complete :class:`FeedModel` from the loaded frames.

    The hashtag vocabulary spans candidate polls *plus* every other poll that
    carries hashtags, so a poll a user voted on still has a TF-IDF vector for
    building their content profile even once it is no longer an eligible
    candidate. ``friendships``/``blocks`` default to empty frames when a caller
    omits them (e.g. during evaluation).
    """
    candidate_ids = [int(poll_id) for poll_id in candidates["poll_id"]]
    tagged_ids: set[int] = set() if poll_hashtags.empty else {int(poll_id) for poll_id in poll_hashtags["poll_id"]}
    # Candidates first (their rows line up with the feed), then any extra
    # tagged polls so voted-but-ineligible polls still get a TF-IDF vector.
    hashtag_poll_ids = candidate_ids + sorted(tagged_ids - set(candidate_ids))
    hashtag_polls = pd.DataFrame({"poll_id": hashtag_poll_ids})
    vectorizer, hashtag_matrix = _build_hashtag_matrix(_hashtag_documents(hashtag_polls, poll_hashtags))
    interaction_poll_ids, item_similarity = _build_item_similarity(interactions)
    # Raw vote count per poll; the scorer turns this into an age-adjusted rate.
    popularity = interactions.groupby("poll_id").size().astype("int64")
    return FeedModel(
        trained_at=datetime.now(UTC),
        poll_ids=candidate_ids,
        vectorizer=vectorizer,
        hashtag_matrix=hashtag_matrix,
        hashtag_poll_ids=hashtag_poll_ids,
        interaction_poll_ids=interaction_poll_ids,
        item_similarity=item_similarity,
        popularity=popularity,
        interactions=interactions,
        candidates=candidates,
        friendships=friendships if friendships is not None else pd.DataFrame(columns=["user_a_id", "user_b_id"]),
        blocks=blocks if blocks is not None else pd.DataFrame(columns=["blocker_id", "blocked_id"]),
        room_memberships=(
            room_memberships if room_memberships is not None else pd.DataFrame(columns=["room_id", "user_id"])
        ),
    )


def train() -> FeedModel:
    """Load every input table from the database and build the model."""
    return build_model(
        load_interactions(),
        load_poll_hashtags(),
        load_candidate_polls(),
        load_friendships(),
        load_blocks(),
        load_room_memberships(),
    )


if __name__ == "__main__":
    setup_logging()
    train()
