# Respiratory Disease Classification from Cough and Voice

This is a project exploring whether cough and sustained vowel recordings can be used as input to machine-learning models that classify respiratory disease (asthma, COPD, and covid) categories. It is collaborative work done with 2 other team members. 

The broader idea was to investigate a lower-barrier method for people to have access to an at-home screening test for respiratory diseases by provide their cough/vowel sound audio samples. This repository is a research prototype: it is not a public-facing app, it does not establish that this approach works as a medical test, and it should not be used to make health decisions.

We worked with two kinds of recordings because they capture different sounds. A cough is short, erratic and irregular. A sustained vowel 'o'-sound gives a longer sample of the voice. The project compares hand-engineered measurements from those sounds with models that learn patterns from spectrograms. Experiments also use patient metadata when it is available.

## What goes into the models?

The feature-extraction code turns each recording into acoustic features. This gives traditional machine-learning models a compact description of the audio instead of the raw waveform (wav files).

| Recording | Features in the reusable extractor | What they describe |
| --- | --- | --- |
| Cough | 66 values | Mean and standard deviation of 13 MFCCs and their deltas; summaries of spectral centroid, bandwidth, rolloff, flatness, zero-crossing rate, and RMS energy; waveform skewness and kurtosis. |
| Vowel | 47 values | Duration; mean and standard deviation of 13 MFCCs; RMS and zero-crossing summaries; spectral centroid and flatness; pitch (F0), jitter, shimmer, harmonic-to-noise ratio, and summaries of the first three formants. |

MFCCs (Mel-frequency cepstral coefficients) are a common way to summarize the changing frequency shape of sound. In simple terms, they provide a compact description of its tone and texture. The cough extractor loads audio at 22,050 Hz and peak-normalizes it. The vowel extractor loads up to five seconds at 16 kHz, trims silence, and uses Librosa and Parselmouth/Praat for acoustic and voice measurements. These values describe recordings; they are not by themselves evidence of a diagnosis.

The feature tables can also include metadata such as age, gender, smoking, and reported cough, fever, or cold indicators, depending on the source table. The reusable preprocessing code joins feature rows to cough and vowel paths using patient ID. Its expected columns include `id`, `disease`, `cough_path`, and `vowel_path`.

## Models we explored

The notebooks contain multiple experiments rather than one controlled comparison where every model used identical data and settings. The main model families were:

- **Support Vector Machine (SVM):** Used with tabular audio features and metadata. An SVM tries to separate classes by finding boundaries between their feature patterns. It is one of the classical baselines in the notebooks; some SVM runs require missing-value handling because the estimator does not accept NaN values directly.
  
- **XGBoost:** A boosted-tree model for tabular features. It builds a sequence of decision trees, where later trees focus on examples earlier trees handled poorly. XGBoost appears in model comparisons and is the intended binary model for Stage 1 of the final two-stage design.
  
- **HistGradientBoosting:** Another tree-boosting approach explored on tabular data. It provides a point of comparison with XGBoost and SVM.
  
- **CNN on Mel spectrograms:** A convolutional neural network works with a picture-like representation of sound. The code converts audio to a Mel spectrogram, where one axis represents frequency bands and the other represents time. Convolution layers learn local patterns in this representation instead of relying on a manually supplied list of acoustic statistics.
  
- **Vowel-only and other notebook branches:** Separate experiments look at vowel features on their own, cough features on their own, and at alternative feature/model combinations. They are experimental research branches.


## Two-stage model

The reusable pipeline is an implementation candidate:

1. Stage 1: An XGBoost model uses a prepared feature row to classify a sample as healthy or diseased. The threshold defaults to 0.5.
2. Stage 2: For samples classified as diseased, a CNN uses cough and vowel Mel spectrograms to classify asthma, COPD, or COVID.

 ```text
Audio measurements and available metadata
                    |
                    v
       Stage 1: binary XGBoost model
             healthy or diseased
                    |
       healthy -----+----- diseased
          |                     |
          v                     v
   return healthy       Stage 2: CNN on audio
                        asthma / COPD / COVID
``` 
   
The CNN prepares both recordings as mono 16 kHz audio, pads or trims them to five seconds, and combines their 128-bin Mel spectrograms. It then passes the combined representation to `RespiratoryNet`, which has four convolution blocks with 32, 64, 128, and 256 channels. Each block uses convolution, batch normalization, ReLU activation, pooling, and dropout. Global average pooling reduces the learned feature maps to a vector, and fully connected layers produce three class scores. The code maps those scores to asthma, COPD, or COVID in that order.

src/inference.py wraps a prediction, but compatible trained models, prepared features, and audio paths must be supplied.

For four-class tables, the labelling convention is 0 = healthy, 1 = asthma, 2 = COPD, 3 = COVID. 
Stage 2 uses a separate order: 0 = asthma, 1 = COPD, 2 = COVID. 
Older notebooks may use different label encodings.


## Repository layout

- `src/` — reusable preprocessing, feature extraction, model, and inference code.
- `docs/` — methodology, approaches explored, and known issues.
- `data/` — notes on expected data and labels. Raw recordings and dataset tables are not included.
- `models/` — model-artifact policy and provenance notes.
- `results/` — notes on interpreting experiment-specific metrics and figures.
- `notebooks/exploration/` — early exploration, segmentation, and feature work.
- `notebooks/experiments/` — model comparisons, alternate branches, and vowel-only experiments.
- `notebooks/final_pipeline/` — curated preprocessing and patient-split notebooks.
- `scripts/` — currently reserved; there is no supported command-line runner.
- `archive/` — selected original and experimental project material.

## Setup and running the project

The Python dependencies are listed in `requirements.txt`. From the repository root, create a virtual environment and install them:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The raw audio, source datasets, processed CSV files, and verified final checkpoints are not included. The reusable functions live under `src/`, but they do not replace the data preparation and training steps in the notebooks.

## Results and limitations

The final two-stage experiment reported:
- Accuracy: 69.13%
- Recall: 69.13%
- F1 score: 68.87%
These results are specific to that experiment and should be read with its data split, preprocessing, labels, and model settings.

Raw audio and source datasets are not redistributed here.
Historical notebooks may use different label encodings or absolute local paths. 
The two-stage pipeline is not a clinical diagnostic system or released public service.
