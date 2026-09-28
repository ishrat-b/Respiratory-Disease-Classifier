# Methodology

## Task definition

The canonical repository label mapping is:

```text
0 = healthy
1 = asthma
2 = copd
3 = covid
```

The final model is staged rather than a single four-class predictor:

- **Stage 1:** healthy versus diseased.
- **Stage 2:** asthma versus COPD versus COVID, applied to cases that pass Stage 1 as diseased.

The historical artifacts contain other encoding conventions. This document describes the canonical repository contract that the cleaned implementation is expected to follow; the original notebooks remain available as experimental evidence.

## Classical-feature pathway

Raw cough and vowel recordings are located, segmented, and summarized into tabular features. The resulting feature rows may also include age, gender, smoking, cough, fever, and cold indicators where available. The training notebooks apply missing-value handling, categorical encoding, scaling, variance filtering, and feature selection before fitting classical classifiers.

A separate notebook contains denoising and fixed-length cleaning that writes `clean_path` and `dataset_cleaned.csv`. The available source does not establish that those outputs feed the canonical feature or stacked-model pipeline, so that workflow is preserved as experimental history rather than required methodology.

## Spectrogram pathway

The stacked implementation loads cough and vowel WAV files, converts them to mono, resamples to 16 kHz, pads or truncates them to five seconds, computes 128-bin Mel spectrograms, converts power to decibels, normalizes each representation, and concatenates cough and vowel spectrograms along the time dimension.

## Data splitting

The project includes patient-grouped stratified split work. The grouped split is the preferred methodology because multiple segments or recordings from one patient should not cross train, validation, and test boundaries.

## Reporting

Evaluation reports use accuracy, macro precision, macro recall, macro F1, classification reports, and confusion matrices. Results from older notebooks belong to their individual experiments and should not be presented as results from the stacked pipeline unless the source run is explicitly identified.
