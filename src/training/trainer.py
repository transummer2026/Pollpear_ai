from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.api.logger import setup_logging
from src.data.loader import load_candidate_polls, load_interactions, load_poll_hashtags
from src.models.feed_model import FeedModel, identity_analyzer


def _hashtag_documents(candidates: pd.DataFrame, poll_hashtags: pd.DataFrame) -> list[list[str]]:
    if poll_hashtags.empty:
        return [[] for _ in range(len(candidates))]
    grouped = poll_hashtags.groupby("poll_id")["hashtag"].apply(list)
    return [list(grouped.get(poll_id, [])) for poll_id in candidates["poll_id"]]


def _build_hashtag_matrix(documents: list[list[str]]) -> tuple[TfidfVectorizer | None, np.ndarray]:
    if not any(documents):
        return None, np.zeros((len(documents), 0))
    vectorizer = TfidfVectorizer(analyzer=identity_analyzer)
    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix.toarray()


def _build_item_similarity(interactions: pd.DataFrame) -> tuple[list[int], np.ndarray]:
    if interactions.empty:
        return [], np.zeros((0, 0))
    votes = pd.crosstab(interactions["user_id"], interactions["poll_id"])
    similarity = cosine_similarity(votes.to_numpy().T)
    return [int(poll_id) for poll_id in votes.columns], similarity


def build_model(
    interactions: pd.DataFrame,
    poll_hashtags: pd.DataFrame,
    candidates: pd.DataFrame,
) -> FeedModel:
    vectorizer, hashtag_matrix = _build_hashtag_matrix(_hashtag_documents(candidates, poll_hashtags))
    interaction_poll_ids, item_similarity = _build_item_similarity(interactions)
    popularity = interactions.groupby("poll_id").size().astype("int64")
    return FeedModel(
        trained_at=datetime.now(UTC),
        poll_ids=[int(poll_id) for poll_id in candidates["poll_id"]],
        vectorizer=vectorizer,
        hashtag_matrix=hashtag_matrix,
        interaction_poll_ids=interaction_poll_ids,
        item_similarity=item_similarity,
        popularity=popularity,
        interactions=interactions,
    )


def train() -> FeedModel:
    interactions = load_interactions()
    poll_hashtags = load_poll_hashtags()
    candidates = load_candidate_polls()
    model = build_model(interactions, poll_hashtags, candidates)
    return model


if __name__ == "__main__":
    setup_logging()
    train()
