# Approaches explored

This project records several research paths rather than one uninterrupted model-development line.

## Classical acoustic features

Cough and vowel recordings were converted into tabular summaries alongside available patient metadata. The cough branch contains 66 aggregate features. The vowel branch contains 47 features, including Librosa summaries and Parselmouth/Praat voice measurements. These tables were used with classical scikit-learn and boosting models.

## SVM

SVM models were tested on preprocessed tabular acoustic and metadata features. The notebooks include model-comparison and held-out evaluation cells. This is an experimental classical branch, not a component loaded by the stacked inference code.

## XGBoost

XGBoost was explored as a tabular classifier and appears as the intended binary Stage 1 model in the stacked architecture. The available root XGBoost artifact is binary, but its exact training provenance and semantic class mapping are not fully established.

## HistGradientBoosting

HistGradientBoosting was compared with other tabular classifiers. Historical saved pipelines expose four numeric classes, but they are not verified components of the final stacked system.

## Vowel-only modelling

Separate notebooks extract vowel features and test classifiers using only vowel recordings. This branch asks whether sustained voice recordings provide useful information without the combined cough pathway. It remains experimental history.

## CNN and Mel-spectrogram modelling

The stacked implementation converts cough and vowel recordings into normalized 128-bin Mel spectrograms and sends the combined representation to a CNN. Separate checkpoints exist in the workspace, but their training provenance and class ordering are not fully established.

## Two-stage stacked architecture

The final architecture candidate uses tabular acoustic/metadata features for a binary healthy-versus-diseased Stage 1 model. Diseased cases are then passed to a CNN that predicts asthma, COPD, or COVID from cough and vowel spectrograms. `src/models/stacked_pipeline.py` contains the reusable implementation; `src/inference.py` provides a caller-supplied-model wrapper.

No approach is ranked here, and no single metric is presented as the project-wide result.
