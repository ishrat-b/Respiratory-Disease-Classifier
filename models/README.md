# Models

Model binaries and checkpoints are intentionally excluded from the GitHub-facing repository. The workspace contains artifacts such as XGBoost/joblib files, PyTorch checkpoints, and pickle files, but their provenance is incomplete for some files.

## Canonical architecture

```text
Acoustic features and audio paths
              |
              v
Stage 1 - binary XGBoost
              |
       healthy vs diseased
              |
       diseased cases only
              v
Stage 2 - CNN on cough/vowel Mel spectrograms
              |
       asthma / copd / covid
```

The reusable implementation candidate is defined in `src/models/stacked_pipeline.py` and wrapped by `src/inference.py`. It accepts caller-supplied compatible models.

## Artifact status

- `layer1_xgboost.joblib`: a loadable binary XGBoost artifact was found, but its semantic class-name mapping and exact producer are not established.
- `efficientnet_respiratory.pth`: a three-output PyTorch state dictionary was found without class-name metadata or a verified training producer.
- `inference_pipeline.pkl`: a custom serialized artifact was found, but it could not be fully inspected or tied conclusively to the canonical pipeline.
- `first_iterations/best_model.pkl`, `best_model.pth`, and `label_encoder.pkl`: historical experimental artifacts with stronger local notebook associations, not verified final stacked components.

This repository does not claim to contain a reproducibly trained final checkpoint. Do not add these binaries to Git unless their provenance, licensing, and release policy are decided separately.
