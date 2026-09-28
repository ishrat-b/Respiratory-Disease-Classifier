# Data

Raw audio and dataset files are intentionally excluded from this repository. The project workspace contains source tables built from cough and vowel recordings, metadata, audio paths, segmented records, and train/validation/test split tables, but those files are not redistributed here.

The canonical source-level label contract is:

```text
0 = healthy
1 = asthma
2 = copd
3 = covid
```

Patient IDs are used to align feature rows with cough and vowel audio-path tables. Feature tables may contain metadata such as age, gender, smoking, cough, fever, and cold indicators where available.

The canonical code expects feature records and audio metadata with known columns such as `id`, `disease`, `cough_path`, and `vowel_path`, although historical tables also use names such as `disease_y` and contain segment-level fields.

Historical files use more than one label-encoding convention. Do not assume that a historical artifact or checkpoint follows the canonical mapping without checking its training provenance.

No dataset URL is provided because the original project evidence does not establish one authoritative redistribution source.
