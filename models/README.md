## Final Canonical architecture

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
