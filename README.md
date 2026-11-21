# Liver-Disease Screening from Routine Blood Tests (ILPD)

**ML Application Assignment, Problem 7** · B.Sc. (H) Computer Science, Sem V · Ramanujan College, University of Delhi
**Author:** Kshitij Khanal (Roll No. 24020570048) · **Article type:** Comparative Study

A comparison of five classical classifiers and three class-imbalance strategies for screening
liver disease from ten routine blood-test measurements, evaluated with nested cross-validation,
an untouched hold-out set, simple baselines and a threshold analysis aimed at not missing patients.

## 1. Problem
Liver disease is often found late, although the liver function tests that reveal it are cheap.
The goal is to flag likely patients so that confirmatory tests go to the people who need them.
A missed patient costs more than a false alarm, so **recall** matters most, but it must be read
together with **specificity**, because 71.4% of the records are patients and a rule that labels
everyone "patient" already has recall 1.0.

## 2. Dataset
- **Indian Liver Patient Dataset (ILPD)**, UCI Machine Learning Repository, 583 records, CC BY 4.0
  https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset
- 10 predictors (age, gender, total/direct bilirubin, ALP, ALT, AST, total proteins, albumin, A/G ratio);
  target: 1 = liver patient (416), 2 = non-patient (167).
- The raw file has **no header row**; the scripts add the column names. A copy is in `data/ILPD.csv`.

## 3. Method (short)
| Step | What is done |
|---|---|
| Split | Stratified 80/20 (seed 42): 466 train / 117 hold-out |
| Preprocessing | Median imputation, log1p of bilirubin/ALP/ALT/AST, standardisation (all inside a `Pipeline`, fitted on training folds only) |
| Models | Logistic regression, Gaussian Naive Bayes, k-NN, SVM (RBF), decision tree |
| Imbalance | none / class weights / random oversampling (own implementation, applied to training folds only) |
| Tuning + validation | Grid search (3-fold, balanced accuracy) inside repeated stratified 5-fold CV (2 repeats) |
| Baselines | Majority class, stratified random guess, one-feature model (log total bilirubin) |
| Final test | Hold-out set used once; screening threshold chosen from training out-of-fold predictions; bootstrap 95% CIs |
| Metrics | Recall, specificity, precision, F1, balanced accuracy, ROC-AUC, confusion matrix |

## 4. Main results
*(Positive class = patient. Cross-validation: mean over 10 outer folds. Hold-out: 117 records.)*

| Model | CV balanced acc. | CV ROC-AUC | Hold-out ROC-AUC |
|---|---|---|---|
| Logistic regression + oversampling (best by CV) | 0.697 | 0.726 | 0.837 |
| One-feature bilirubin baseline | 0.652 | 0.683 | 0.742 |
| Majority-class baseline | 0.500 | 0.500 | 0.500 |

- Without imbalance handling, models reach recall 0.81–0.91 only by labelling most people as patients (specificity 0.21–0.36).
- At a threshold of **0.35** (chosen on training data only), logistic regression + oversampling gives hold-out
  **recall 0.93, specificity 0.29** (77/83 patients found, 24/34 non-patients flagged).
- Classical models are suitable only as a **first-pass triage aid**, not for diagnosis.
- The hold-out AUC is higher than the CV AUC, probably because the 117-record split is easy; trust the CV figures more.

Full tables: `results/*.csv`; figures: `results/figures/`; full console output: `results/run_log.txt`.

## 5. Repository structure
```
.
├── data/ILPD.csv              # dataset copy (CC BY 4.0, see source above)
├── src/
│   ├── run_experiments.py     # preprocessing, nested CV, hold-out test, threshold, baselines, bootstrap
│   └── extra_figures.py       # EDA statistics, correlation heatmap, threshold curve, pipeline diagram
├── results/
│   ├── cv_results.csv  holdout_results.csv  baseline_results.csv
│   ├── threshold_analysis.csv  holdout_screening_threshold.csv  eda_summary.csv
│   ├── run_log.txt
│   └── figures/*.png
├── requirements.txt
└── README.md
```

## 6. How to reproduce
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/run_experiments.py      # about a few minutes; writes results/ and figures
python src/extra_figures.py        # run after the first script (it reads threshold_analysis.csv)
```
Tested with Python 3.12, scikit-learn 1.9.1, pandas 3.0.6, NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2.
All randomness uses seed 42, so the numbers in `results/` should reproduce exactly.

## 7. Limitations
Small dataset (583 records; 117 in the hold-out set, so wide confidence intervals); one region of India;
male-dominated (76%); "non-patients" are people referred for liver tests, not healthy volunteers;
13 duplicate rows were kept; no external validation.

## 8. Use of AI tools
The code and the first draft of the paper were produced with an AI assistant (Claude, Anthropic).
The scripts were run on the dataset and every number and figure in the paper comes from them.
The author is responsible for the content. *(Edit this section so it accurately describes what you did.)*

## 9. Citation / licence
Dataset: B. V. Ramana and N. B. Venkateswarlu, ILPD (Indian Liver Patient Dataset), UCI ML Repository, 2012 (CC BY 4.0).
Code: add a licence of your choice (for example MIT) as `LICENSE`.
