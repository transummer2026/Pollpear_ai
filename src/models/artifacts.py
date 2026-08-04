import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import joblib

from src.models.feed_model import FeedModel

_MODELS_DIR = Path(__file__).resolve().parent
_PREFIX = "feed-model-"
_SUFFIX = ".joblib"


def save(model: FeedModel, directory: Path | None = None) -> Path:
    target_dir = directory if directory is not None else _MODELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    target = target_dir / f"{_PREFIX}{stamp}{_SUFFIX}"
    fd, tmp_name = tempfile.mkstemp(dir=target_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            joblib.dump(model, tmp_file)
        os.replace(tmp_name, target)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target


def load_latest(directory: Path | None = None) -> FeedModel | None:
    target_dir = directory if directory is not None else _MODELS_DIR
    paths = sorted(target_dir.glob(f"{_PREFIX}*{_SUFFIX}"))
    if not paths:
        return None
    return joblib.load(paths[-1])
