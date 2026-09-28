"""Reusable loading and alignment helpers for the respiratory pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import pandas as pd


PathLike = Union[str, Path]

# Repository-level label contract. These constants document the intended
# labels; this module does not rewrite labels inside input CSV files.
CANONICAL_LABELS = {
    0: "healthy",
    1: "asthma",
    2: "copd",
    3: "covid",
}


@dataclass
class FeatureData:
    """Feature matrix and the identifiers/labels kept beside it."""

    features: pd.DataFrame
    labels: pd.Series
    ids: pd.Series


@dataclass
class AlignedAudioData:
    """Feature rows and audio paths in the same patient-ID order."""

    features: pd.DataFrame
    audio: pd.DataFrame


def load_feature_data(
    csv_path: PathLike,
    id_col: str = "id",
    label_col: str = "disease",
) -> FeatureData:
    """Load a feature CSV and separate IDs, labels, and model features.

    This follows the original stacked pipeline: the ID column is removed from
    the feature matrix, the label column is kept separately, and labels are
    returned as stored rather than re-encoded here.
    """
    data = pd.read_csv(csv_path)

    if label_col not in data.columns:
        raise ValueError(f"Label column '{label_col}' not found in {csv_path}.")

    if id_col not in data.columns:
        raise ValueError(f"ID column '{id_col}' not found in {csv_path}.")

    ids = data[id_col].copy()
    data = data.drop(columns=[id_col])

    labels = data[label_col].copy()
    features = data.drop(columns=[label_col]).copy()
    return FeatureData(features=features, labels=labels, ids=ids)


def load_audio_metadata(
    csv_path: PathLike,
    id_col: str = "id",
    cough_col: str = "cough_path",
    vowel_col: str = "vowel_path",
) -> pd.DataFrame:
    """Load audio metadata containing patient IDs and WAV paths."""
    audio = pd.read_csv(csv_path)
    required = [id_col, cough_col, vowel_col]
    missing = [column for column in required if column not in audio.columns]
    if missing:
        raise ValueError(f"Missing required audio columns: {missing}")
    return audio


def align_features_and_audio(
    feature_data: FeatureData,
    audio_data: pd.DataFrame,
    id_col: str = "id",
    cough_col: str = "cough_path",
    vowel_col: str = "vowel_path",
) -> AlignedAudioData:
    """Join feature rows and audio paths by patient ID.

    The original pipeline uses an inner join because feature CSVs can contain
    healthy patients while audio-path CSVs may contain only patients with both
    disease-specific recordings. No missing records are silently fabricated.
    """
    validate_audio_metadata(audio_data, id_col, cough_col, vowel_col)

    if len(feature_data.features) != len(feature_data.ids):
        raise ValueError("Feature rows and feature IDs must have the same length.")

    feature_rows = feature_data.features.copy()
    path_columns = [cough_col, vowel_col]
    overlapping_paths = [
        column for column in path_columns if column in feature_rows.columns
    ]
    if overlapping_paths:
        raise ValueError(
            "Feature data already contains audio-path columns: "
            f"{overlapping_paths}. Keep paths in the audio table for alignment."
        )
    feature_rows[id_col] = feature_data.ids.to_numpy()

    joined = audio_data[[id_col, cough_col, vowel_col]].merge(
        feature_rows,
        on=id_col,
        how="inner",
    )

    if joined.empty:
        raise ValueError("No patient IDs matched between feature and audio data.")

    aligned_audio = joined[[id_col, cough_col, vowel_col]].reset_index(drop=True)
    aligned_features = joined.drop(
        columns=[id_col, cough_col, vowel_col], errors="ignore"
    ).reset_index(drop=True)
    validate_alignment(aligned_features, aligned_audio, id_col, cough_col, vowel_col)
    return AlignedAudioData(features=aligned_features, audio=aligned_audio)


def validate_audio_metadata(
    audio_data: pd.DataFrame,
    id_col: str = "id",
    cough_col: str = "cough_path",
    vowel_col: str = "vowel_path",
) -> None:
    """Validate required audio columns, IDs, and path values."""
    required = [id_col, cough_col, vowel_col]
    missing = [column for column in required if column not in audio_data.columns]
    if missing:
        raise ValueError(f"Missing required audio columns: {missing}")
    if audio_data[id_col].isna().any():
        raise ValueError(f"Audio column '{id_col}' contains missing IDs.")
    for column in (cough_col, vowel_col):
        if audio_data[column].isna().any():
            raise ValueError(f"Audio column '{column}' contains missing paths.")


def validate_alignment(
    features: pd.DataFrame,
    audio: pd.DataFrame,
    id_col: str = "id",
    cough_col: str = "cough_path",
    vowel_col: str = "vowel_path",
) -> None:
    """Ensure aligned feature and audio tables have matching row order."""
    if id_col not in audio.columns:
        raise ValueError(f"Aligned audio data must contain '{id_col}'.")
    if len(features) != len(audio):
        raise ValueError("Aligned feature and audio tables must have equal lengths.")
    validate_audio_metadata(audio, id_col, cough_col, vowel_col)
