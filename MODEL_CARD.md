# Model card: WDBC breast-cancer classifier

## Model

- **Type:** L2-regularised logistic regression that I implemented in NumPy and trained by full-batch gradient descent (`src/wdbc/logistic_scratch.py`), applied after z-score scaling.
- **Version:** 1.0.0. Trained by `python -m wdbc.train`; settings are in `configs/config.yaml`.
- **Output:** the probability that a tumour is malignant, and a label using a decision threshold of 0.27.

## Intended use

I built this as a portfolio and learning project to demonstrate an end-to-end ML pipeline.
**It is not intended for clinical use** and must not inform any diagnosis or treatment.

## Data

- UCI *Breast Cancer Wisconsin (Diagnostic)*: 569 samples from fine-needle aspirates, from the University of Wisconsin, Madison (donated to UCI in 1995). 30 features are computed from digitised images of cell nuclei (10 measurements, each as a mean, a standard error and a worst value).
- 357 benign (63%), 212 malignant (37%).
- I split it into 455 training and 114 test samples, stratified by diagnosis, with seed 42.

## Performance

| Evaluation | ROC-AUC | Recall | Precision | Missed malignant | False alarms |
| --- | --- | --- | --- | --- | --- |
| Cross-validation (5-fold × 3), threshold 0.5 | 0.9955 ± 0.0050 | 0.947 | 0.982 | | |
| Test set, threshold 0.27 | 0.9977 | 0.976 | 0.932 | 1 of 42 | 3 of 72 |

I chose the threshold from out-of-fold training predictions so that recall is at least 0.98.

## Limitations and risks

- **Small, old, single-site dataset.** 569 samples from one institution. I don't know how it would perform on other populations, equipment or labs; probably worse.
- **Pre-computed features.** The model needs the same 30 image-derived measurements, produced the same way. It doesn't read images.
- **Small test set.** There are only 42 malignant test cases, so one more or one fewer miss changes recall by 2.4 points.
- **Correlated features.** The coefficients shouldn't be read as independent feature importances.
- **Distribution shift.** The API flags inputs outside the training range, but it can't detect shifts within that range.
- **The threshold is a value judgement.** Recall ≥ 0.98 favours catching cancers over avoiding follow-up tests. A real deployment would set it together with clinicians.

## Ethical considerations

A missed malignancy delays treatment; a false alarm causes anxiety and extra procedures. The dataset has no demographic attributes, so I can't assess fairness across patient groups.
