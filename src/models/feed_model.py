from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def identity_analyzer(tokens: list[str]) -> list[str]:
    return tokens


@dataclass(frozen=True)
class FeedModel:
    trained_at: datetime
    poll_ids: list[int]
    vectorizer: TfidfVectorizer | None
    hashtag_matrix: np.ndarray
    interaction_poll_ids: list[int]
    item_similarity: np.ndarray
    popularity: pd.Series
    interactions: pd.DataFrame
