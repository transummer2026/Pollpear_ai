"""Definition of the trained feed model artifact and its analyzer helper."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def identity_analyzer(tokens: list[str]) -> list[str]:
    """Pass an already-tokenized hashtag list straight through to TfidfVectorizer.

    Hashtags are pre-split, so no text analysis is needed. This must be a
    module-level named function (not a lambda) so the fitted vectorizer, and
    therefore the whole model, can be pickled by joblib.
    """
    return tokens


@dataclass(frozen=True)
class FeedModel:
    """Immutable snapshot of everything the scorer needs at request time.

    Frozen so a trained model can be shared across threads without copying.
    Several fields are parallel arrays whose row order is significant:

    - ``hashtag_matrix`` rows align to ``hashtag_poll_ids`` (TF-IDF vectors).
    - ``item_similarity`` rows/columns align to ``interaction_poll_ids``
      (item-item co-vote cosine similarity).
    - ``popularity`` is a Series of vote counts indexed by poll_id.
    """

    trained_at: datetime
    poll_ids: list[int]
    vectorizer: TfidfVectorizer | None
    hashtag_matrix: np.ndarray
    hashtag_poll_ids: list[int]
    interaction_poll_ids: list[int]
    item_similarity: np.ndarray
    popularity: pd.Series
    interactions: pd.DataFrame
    candidates: pd.DataFrame
    friendships: pd.DataFrame
    blocks: pd.DataFrame
    room_memberships: pd.DataFrame
