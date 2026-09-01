"""Offline evaluation harness for the feed recommender.

Uses a temporal train/test split (train on the earliest votes, test on the
latest) and reports precision@k for several weight presets against a plain
popularity baseline, so weight changes can be compared before shipping.
"""

import pandas as pd

from src.api.logger import get_logger, setup_logging
from src.data.loader import (
    load_blocks,
    load_candidate_polls,
    load_friendships,
    load_poll_hashtags,
    load_room_memberships,
    load_timed_interactions,
)
from src.scoring.recommender import WEIGHTS, recommend
from src.training.trainer import build_model

logger = get_logger(__name__)

TRAIN_RATIO = 0.8
FEED_SIZE = 9

# Weight blends to compare; "current" mirrors the production WEIGHTS.
PRESETS = {
    "current": WEIGHTS,
    "content-heavy": {"content": 0.6, "covotes": 0.2, "recency": 0.1, "popularity": 0.05, "social": 0.05},
    "balanced": {"content": 0.35, "covotes": 0.25, "recency": 0.2, "popularity": 0.1, "social": 0.1},
    "covotes-heavy": {"content": 0.2, "covotes": 0.5, "recency": 0.1, "popularity": 0.1, "social": 0.1},
}


def split_interactions(timed: pd.DataFrame, ratio: float = TRAIN_RATIO):
    """Split votes in time at the ``ratio`` quantile of ``voted_at``.

    Everything up to the cutoff becomes training data and everything after
    becomes the test set, mimicking prediction of future votes and avoiding
    look-ahead leakage. The cutoff timestamp is returned so scoring can use
    it as "now".
    """
    cutoff = timed["voted_at"].quantile(ratio)
    train = timed.loc[timed["voted_at"] <= cutoff, ["user_id", "poll_id"]].reset_index(drop=True)
    test = timed.loc[timed["voted_at"] > cutoff, ["user_id", "poll_id"]].reset_index(drop=True)
    return train, test, pd.Timestamp(cutoff)


def popularity_baseline(train: pd.DataFrame, voted: set[int], candidate_ids: list[int], limit: int) -> list[int]:
    """Rank candidates by training-set vote count as a non-personalized baseline.

    Already-voted polls are excluded; if fewer than ``limit`` polls have any
    votes, the list is padded with remaining candidates.
    """
    counts = train.groupby("poll_id").size().sort_values(ascending=False)
    allowed = set(candidate_ids) - voted
    ranked = [int(poll_id) for poll_id in counts.index if int(poll_id) in allowed]
    fill = [poll_id for poll_id in candidate_ids if poll_id in allowed and poll_id not in ranked]
    return (ranked + fill)[:limit]


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fraction of the top-``k`` recommendations that are relevant.

    Divides by ``k`` (not by the number returned), so short feeds are
    penalized rather than flattered.
    """
    return len(set(recommended[:k]) & relevant) / k


def evaluate() -> pd.DataFrame:
    """Score every preset and the baseline per user and return a report frame.

    The model is trained only on the training split and scored at the split
    cutoff time, so recency/popularity reflect what was known then. Users are
    scored only when they have at least one test vote on a reachable
    candidate poll.
    """
    timed = load_timed_interactions()
    if timed.empty:
        logger.warning("No interactions in database, nothing to evaluate")
        return pd.DataFrame()
    candidates = load_candidate_polls()
    poll_hashtags = load_poll_hashtags()
    friendships = load_friendships()
    blocks = load_blocks()
    room_memberships = load_room_memberships()

    train, test, cutoff = split_interactions(timed)
    # Score at the split boundary; naive cutoffs are assumed UTC.
    at = cutoff.tz_localize("UTC").to_pydatetime() if cutoff.tzinfo is None else cutoff.to_pydatetime()
    model = build_model(train, poll_hashtags, candidates, friendships, blocks, room_memberships)
    candidate_ids = model.poll_ids
    candidate_set = set(candidate_ids)

    rows = []
    evaluated_users = 0
    for user_id, group in test.groupby("user_id"):
        # A test vote counts only if that poll is still a reachable candidate.
        relevant = {int(poll_id) for poll_id in group["poll_id"]} & candidate_set
        if not relevant:
            continue
        evaluated_users += 1
        voted = {int(poll_id) for poll_id in train.loc[train["user_id"] == user_id, "poll_id"]}
        baseline = popularity_baseline(train, voted, candidate_ids, FEED_SIZE)
        row = {
            "user_id": int(user_id),
            "relevant": len(relevant),
            "popularity-baseline": precision_at_k(baseline, relevant, FEED_SIZE),
        }
        for name, weights in PRESETS.items():
            feed = recommend(model, int(user_id), FEED_SIZE, now=at, weights=weights)
            row[f"hybrid-{name}"] = precision_at_k([p["poll_id"] for p in feed], relevant, FEED_SIZE)
        rows.append(row)

    report = pd.DataFrame(rows)
    return report


if __name__ == "__main__":
    setup_logging()
    report = evaluate()
    if not report.empty:
        systems = [column for column in report.columns if column not in ("user_id", "relevant")]
        logger.info(f"Users evaluated: {len(report)}, mean reachable test votes: {report['relevant'].mean():.2f}")
        summary = report[systems].mean().sort_values(ascending=False)
        for name, value in summary.items():
            logger.info(f"Mean precision@{FEED_SIZE} {name}: {value:.4f}")
