"""Acoustic feature extraction used by the respiratory ML experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import librosa
import numpy as np
from scipy.stats import kurtosis, skew


PathLike = Union[str, Path]


def cough_feature_names() -> list[str]:
    """Return the original 66-feature cough ordering."""
    names = []
    for prefix in ("mfcc", "mfcc_delta"):
        for statistic in ("mean", "std"):
            names.extend(f"{prefix}_{index}_{statistic}" for index in range(1, 14))

    names.extend(
        [
            "spectral_centroid_mean",
            "spectral_centroid_std",
            "spectral_bandwidth_mean",
            "spectral_bandwidth_std",
            "spectral_rolloff_mean",
            "spectral_rolloff_std",
            "spectral_flatness_mean",
            "spectral_flatness_std",
            "zcr_mean",
            "zcr_std",
            "rms_mean",
            "rms_std",
            "skewness",
            "kurtosis",
        ]
    )
    return names


def extract_cough_features(
    file_path: PathLike,
    sr: int = 22050,
) -> np.ndarray:
    """Extract the original 66 aggregate features from a cough recording."""
    y, sr = librosa.load(file_path, sr=sr)
    y = librosa.util.normalize(y)

    stft = np.abs(librosa.stft(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)

    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr)
    flatness = librosa.feature.spectral_flatness(S=stft)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    features = []
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    features.extend(np.mean(mfcc_delta, axis=1))
    features.extend(np.std(mfcc_delta, axis=1))

    for values in (centroid, bandwidth, rolloff, flatness, zcr, rms):
        features.extend((np.mean(values), np.std(values)))

    features.extend((skew(y), kurtosis(y)))
    return np.asarray(features)


def safe_extract_cough_features(
    file_path: PathLike,
    sr: int = 22050,
) -> np.ndarray:
    """Return 66 NaNs when cough extraction fails, matching the notebook helper."""
    try:
        return extract_cough_features(file_path, sr=sr)
    except Exception as exc:
        print(f"Error processing {file_path}: {exc}")
        return np.full(66, np.nan)


def vowel_feature_names() -> list[str]:
    """Return the original 47-feature vowel ordering."""
    names = ["duration"]
    for index in range(1, 14):
        names.extend((f"mfcc_{index}_mean", f"mfcc_{index}_std"))
    names.extend(
        [
            "rms_mean",
            "rms_std",
            "zcr_mean",
            "zcr_std",
            "spectral_centroid_mean",
            "spectral_flatness_mean",
            "f0_mean",
            "f0_std",
            "f0_min",
            "f0_max",
            "f0_range",
            "jitter",
            "shimmer",
            "hnr",
            "F1_mean",
            "F1_std",
            "F2_mean",
            "F2_std",
            "F3_mean",
            "F3_std",
        ]
    )
    return names


def extract_vowel_features(
    file_path: PathLike,
    sr: int = 16000,
    max_duration: int = 5,
) -> dict[str, float] | None:
    """Extract the original 47 acoustic and Praat vowel features."""
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=max_duration)
        y, _ = librosa.effects.trim(y, top_db=30)
        if len(y) == 0:
            return None

        features: dict[str, float] = {
            "duration": librosa.get_duration(y=y, sr=sr),
        }

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for index in range(13):
            features[f"mfcc_{index + 1}_mean"] = np.mean(mfcc[index])
            features[f"mfcc_{index + 1}_std"] = np.std(mfcc[index])

        rms = librosa.feature.rms(y=y)[0]
        features["rms_mean"] = np.mean(rms)
        features["rms_std"] = np.std(rms)

        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features["zcr_mean"] = np.mean(zcr)
        features["zcr_std"] = np.std(zcr)

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        features["spectral_centroid_mean"] = np.mean(centroid)
        features["spectral_flatness_mean"] = np.mean(flatness)

        import parselmouth
        from parselmouth.praat import call

        sound = parselmouth.Sound(file_path)
        praat_duration = min(sound.get_total_duration(), max_duration)
        sound = sound.extract_part(
            from_time=0,
            to_time=praat_duration,
            preserve_times=False,
        )

        pitch = call(sound, "To Pitch", 0.0, 75, 500)
        f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
        f0_std = call(pitch, "Get standard deviation", 0, 0, "Hertz")
        f0_min = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
        f0_max = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
        features["f0_mean"] = f0_mean
        features["f0_std"] = f0_std
        features["f0_min"] = f0_min
        features["f0_max"] = f0_max
        features["f0_range"] = f0_max - f0_min

        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
        features["jitter"] = call(
            point_process,
            "Get jitter (local)",
            0,
            0,
            0.0001,
            0.02,
            1.3,
        )
        features["shimmer"] = call(
            [sound, point_process],
            "Get shimmer (local)",
            0,
            0,
            0.0001,
            0.02,
            1.3,
            1.6,
        )

        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        features["hnr"] = call(harmonicity, "Get mean", 0, 0)

        formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        time_points = np.linspace(0.05, praat_duration - 0.05, 20)
        for formant_number in (1, 2, 3):
            values = []
            for time_point in time_points:
                value = call(
                    formant,
                    "Get value at time",
                    formant_number,
                    time_point,
                    "Hertz",
                    "Linear",
                )
                if not np.isnan(value):
                    values.append(value)
            features[f"F{formant_number}_mean"] = (
                np.mean(values) if values else np.nan
            )
            features[f"F{formant_number}_std"] = (
                np.std(values) if values else np.nan
            )

        return features
    except Exception as exc:
        print(f"Error processing {file_path}: {exc}")
        return None


def extract_features_from_audio(
    file_path: PathLike,
    audio_type: str,
) -> np.ndarray | dict[str, float] | None:
    """Dispatch to the cough or vowel extractor used by the project."""
    if audio_type == "cough":
        return extract_cough_features(file_path)
    if audio_type == "vowel":
        return extract_vowel_features(file_path)
    raise ValueError("audio_type must be 'cough' or 'vowel'.")
