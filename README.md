# Respiratory Disease Classification from Cough and Voice

A student research project investigating respiratory disease classification from cough and vowel recordings, patient metadata, engineered acoustic features, and learned audio representations. The project is exploratory and is not a clinical diagnostic system.

## Problem

The project investigates whether respiratory-related information in cough and voice recordings can support classification of healthy recordings and disease categories. It combines signal processing, classical machine learning, and CNN-based spectrogram modelling.

The repository preserves both the reusable implementation candidate and the historical experiments that led to it. The experiments are not all synchronized and should not be treated as one reproducible training run.

## Dataset

Raw audio and source datasets are intentionally not included. The original workspace contains tables derived from cough and vowel recordings, metadata, audio paths, segmented records, and train/validation/test splits.

Verified table fields include combinations of:

- patient or recording `id`
- `disease` or historical variants such as `disease_y`
- cough and vowel audio paths
- age, gender, smoking, cough, fever, and cold metadata where available
- engineered acoustic features

Patient IDs are used to align feature records with cough and vowel audio-path records. No authoritative dataset URL is claimed because the original evidence does not establish one redistribution source.

## Approach

The project contains two main modelling pathways.

### 1. Acoustic feature extraction

The classical path summarizes audio into tabular features:

- cough recordings produce 66 aggregate features
- vowel recordings produce 47 features
- metadata can be joined to these tables
- SVM, XGBoost, HistGradientBoosting, and other classical models were explored

### 2. Mel-spectrogram CNN

The learned-audio path uses cough and vowel recordings directly:

- audio is loaded and converted to mono
- it is resampled to 16 kHz
- each recording is padded or truncated to five seconds
- a 128-bin Mel spectrogram is created
- cough and vowel spectrograms are combined along the time dimension
- a CNN predicts the disease subtype for diseased cases

### Two-stage architecture

The canonical architecture candidate separates screening from subtype classification:

```text
acoustic features and metadata
            |
            v
Stage 1: binary XGBoost
            |
      healthy vs diseased
            |
     diseased cases only
            v
Stage 2: CNN on cough/vowel Mel spectrograms
            |
      asthma / copd / covid
```

The reusable implementation is in `src/models/stacked_pipeline.py`. The small interface in `src/inference.py` accepts caller-supplied compatible models and does not assume that any stored checkpoint is the verified final model.

## Label Mapping

The repository-level label contract is:

```text
0 = healthy
1 = asthma
2 = copd
3 = covid
```

Historical notebooks and saved artifacts contain other encoding conventions. This mapping is the canonical repository contract, not proof that every historical checkpoint uses the same class order.

## Pipeline

```text
raw audio and metadata
        |
        +--> segmentation and path preparation
        |
        +--> cough features: 66 values
        |        or vowel features: 47 values
        |        |
        |        +--> classical model experiments
        |
        +--> cough/vowel waveforms
                 |
                 +--> 16 kHz, five-second preparation
                 +--> 128-bin Mel spectrograms
                 +--> Stage 2 CNN pathway

features --> Stage 1 binary decision
                 |
                 +--> healthy: final healthy prediction
                 +--> diseased: Stage 2 subtype prediction
```

The separate `audio_cleaning.ipynb` workflow applies additional denoising and
fixed-length cleaning, but the available evidence does not establish that its
`clean_path`/`dataset_cleaned.csv` outputs feed the canonical feature or
stacked-model pipeline. That workflow is preserved as experimental history.

## Features

### Cough features

The verified cough extractor uses Librosa at 22,050 Hz and peak-normalizes the waveform before calculating:

- 13 MFCC means and 13 MFCC standard deviations
- 13 MFCC-delta means and 13 MFCC-delta standard deviations
- mean and standard deviation for spectral centroid, bandwidth, rolloff, flatness, zero-crossing rate, and RMS
- waveform skewness and kurtosis

Together these produce 66 ordered values.

### Vowel features

The verified vowel extractor uses Librosa at 16 kHz, loads up to five seconds, and trims silence with `top_db=30`. It then uses Parselmouth/Praat for voice measurements. The 47 ordered features include:

- duration
- 13 MFCC mean/std pairs
- RMS and zero-crossing-rate summaries
- spectral centroid and flatness
- F0 mean, standard deviation, minimum, maximum, and range
- jitter, shimmer, and HNR
- mean/std values for formants F1, F2, and F3

## Models

Stage 1 is represented as a binary XGBoost component whose class-1 probability is compared with a `0.5` threshold. Healthy predictions stop at Stage 1. Diseased predictions are passed to Stage 2.

Stage 2 is the `RespiratoryNet` CNN defined in `src/models/stacked_pipeline.py`. Its output mapping is:

```text
0 = asthma
1 = copd
2 = covid
```

The mapping describes the canonical Stage 2 interface. Existing checkpoint files are not claimed to use this ordering unless their training provenance is established.

## Results

Results are experiment-specific. The notebooks contain validation/test reports, confusion matrices, ROC plots, and training figures from different branches and configurations. There is no single repository-wide final accuracy in this README.

See [results/README.md](results/README.md) and the notebooks for the context of any reported result.

## Repository Structure

```text
README.md
requirements.txt
.gitignore
src/                         reusable canonical Python code
docs/                        methodology, approaches, and known issues
data/                        data policy and expected table formats
models/                      model-artifact policy
results/                     traceable result placeholders
scripts/                     reserved for supported command-line entry points
notebooks/exploration/       early data and feature exploration
notebooks/experiments/       model comparisons and alternate branches
notebooks/final_pipeline/    curated preprocessing and split notebooks
archive/original_project/    preservation notes for original working material
archive/experimental/        experimental extracted utilities
```

Large original working folders remain on the local drive and are excluded from the GitHub-facing source set. Historical notebooks are preserved without rewriting their contents.

## Reproducibility

Included:

- reusable preprocessing, feature extraction, model, and inference interfaces
- selected historical notebooks
- documented feature definitions and label contract
- methodology and known limitations

Not included:

- raw audio
- source datasets
- generated CSV tables
- model binaries and checkpoints
- local cache directories
- a verified final training run with a released checkpoint

Reproduction therefore requires obtaining the appropriate data under its original usage terms and selecting the specific historical notebook/configuration to reproduce.

## Limitations

- Some historical artifacts have incomplete or uncertain training provenance.
- Historical label mappings differ between notebooks and saved models.
- The final architecture is documented as an implementation candidate, not as a verified reproducibly trained release.
- Absolute local paths and duplicate experimental branches remain in historical notebooks.
- This is a student/research project and not a clinical diagnostic system.

See [docs/known_issues.md](docs/known_issues.md) for the detailed limitations.
