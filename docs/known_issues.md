# Known issues

- Raw audio and processed datasets are excluded from the public repository.
- The provenance of some trained artifacts is incomplete.
- The repository-level contract of disease labelling is `0 = healthy`, `1 = asthma`, `2 = copd`, `3 = covid`, but this does not prove the class order of every historical checkpoint.
- Historical notebooks contain absolute paths, duplicate branches, and generated outputs from different experiments.
- Reported results are experiment-specific and must be read with the notebook and split that produced them.
- This is a student/research project, not a clinical diagnostic system.
