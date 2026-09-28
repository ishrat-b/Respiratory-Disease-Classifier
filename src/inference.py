"""Small inference interface for the caller-supplied two-stage classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from src.models.stacked_pipeline import Prediction, TwoStageClassifier


PathLike = Union[str, Path]


def predict_one(
    pipeline: TwoStageClassifier,
    features: Any,
    cough_path: PathLike,
    vowel_path: PathLike,
) -> Prediction:
    """Predict one patient using an already-constructed two-stage pipeline.

    Checkpoint loading, feature preparation, and audio/Mel processing remain in
    their existing modules. The caller must provide compatible loaded models.
    """
    return pipeline.predict(
        features=features,
        cough_path=str(cough_path),
        vowel_path=str(vowel_path),
    )
