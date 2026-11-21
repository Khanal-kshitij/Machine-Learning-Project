"""Problem 7 - Liver-disease screening from routine blood tests (ILPD).
Classifiers: LogReg, Gaussian NB, k-NN, SVM, Decision Tree
Imbalance handling: none / class weights / random oversampling (inside each CV fold)
Metrics: recall, precision, F1, ROC-AUC, confusion matrix
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
    GridSearchCV, train_test_split, cross_validate, cross_val_predict)
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
    recall_score, precision_score, f1_score, accuracy_score, make_scorer)

import os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results") + os.sep
os.makedirs(OUT + "figures", exist_ok=True)
SEED = 42
cols = ["Age","Gender","Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase",
        "ALT","AST","Total_Proteins","Albumin","Albumin_Globulin_Ratio","Dataset"]
# the UCI file has no header row
df = pd.read_csv(os.path.join(ROOT, "data", "ILPD.csv"), header=None, names=cols)
df["Gender"] = (df["Gender"] == "Male").astype(int)
df["y"] = (df["Dataset"] == 1).astype(int)          # 1 = liver patient (positive class)
X, y = df.drop(columns=["Dataset","y"]), df["y"]
print("Shape:", df.shape, "| patients:", y.sum(), "non-patients:", (1-y).sum(),
      "| missing A/G ratio:", X.Albumin_Globulin_Ratio.isna().sum(), "| duplicate rows:", df.duplicated().sum())

# ---------- EDA figure (column-width, for the two-column paper) ----------
plt.rcParams.update({"font.size": 6, "axes.titlesize": 6.5, "axes.labelsize": 6, "xtick.labelsize": 5, "ytick.labelsize": 5,
                     "legend.fontsize": 5.5, "axes.linewidth": .5, "savefig.bbox": "tight", "savefig.pad_inches": .03})
fig, ax = plt.subplots(3, 3, figsize=(3.4, 3.3))
for a, c in zip(ax.ravel(), [c for c in X.columns if c != "Gender"]):
    for k, lab in [(1,"Patient"),(0,"Non-patient")]:
        a.hist(X.loc[y==k, c].dropna(), bins=22, alpha=.6, label=lab)
    a.set_title(c.replace("_"," "), pad=2)
ax[0,0].legend(); plt.tight_layout(pad=.4); plt.savefig(OUT+"figures/fig1_eda_distributions.png", dpi=300); plt.close()

# ---------- Preprocessing + oversampling meta-estimator ----------
def log_skewed(Z):                      # enzymes/bilirubin are heavily right-skewed
    Z = np.array(Z, dtype=float).copy()
    idx = [2,3,4,5,6]                    # TB, DB, ALP, ALT, AST
    Z[:, idx] = np.log1p(Z[:, idx]); return Z

class ROS(BaseEstimator, ClassifierMixin):
    """Random oversampling of the minority class; applied only to the training fold."""
    def __init__(self, est=None, seed=SEED): self.est, self.seed = est, seed
    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        rng = np.random.RandomState(self.seed)
        pos, neg = np.where(y==1)[0], np.where(y==0)[0]
        mi, ma = (neg, pos) if len(neg) < len(pos) else (pos, neg)
        extra = rng.choice(mi, len(ma)-len(mi), replace=True)
        idx = np.concatenate([np.arange(len(y)), extra])
        self.est_ = clone(self.est).fit(X[idx], y[idx]); self.classes_ = self.est_.classes_; return self
    def predict(self, X): return self.est_.predict(X)
    def predict_proba(self, X): return self.est_.predict_proba(X)
    def decision_function(self, X): return self.est_.decision_function(X) if hasattr(self.est_,"decision_function") else self.est_.predict_proba(X)[:,1]

def build(model, strategy):
    cw = "balanced" if strategy == "class_weight" else None
    m = {"LogReg": LogisticRegression(max_iter=2000, class_weight=cw),
         "GaussianNB": GaussianNB(),
         "kNN": KNeighborsClassifier(),
         "SVM": SVC(probability=True, class_weight=cw, random_state=SEED),
         "DecisionTree": DecisionTreeClassifier(class_weight=cw, random_state=SEED)}[model]
    if strategy == "oversample": m = ROS(m)
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("log", FunctionTransformer(log_skewed)),
                     ("sc", StandardScaler()), ("clf", m)])

grids = {"LogReg": {"C":[.01,.1,1,10]},
         "GaussianNB": {"var_smoothing":[1e-9,1e-6,1e-3,1e-1]},
         "kNN": {"n_neighbors":[3,5,9,15,25]},
         "SVM": {"C":[.1,1,10], "gamma":["scale",.01,.1]},
         "DecisionTree": {"max_depth":[2,3,4,5,8], "min_samples_leaf":[5,10,20]}}
def pgrid(model, strategy):
    pre = "clf__est__" if strategy == "oversample" else "clf__"
    return {pre+k: v for k, v in grids[model].items()}
strategies = {"LogReg":["none","class_weight","oversample"], "GaussianNB":["none","oversample"],
              "kNN":["none","oversample"], "SVM":["none","class_weight","oversample"],
              "DecisionTree":["none","class_weight","oversample"]}

def sens_spec(yt, yp):
    tn, fp, fn, tp = confusion_matrix(yt, yp).ravel(); return tp/(tp+fn), tn/(tn+fp)

# ---------- 1) Hold-out split (stratified 80/20) kept aside for final test ----------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)

# ---------- 2) Nested CV benchmark on the training set (5x2 outer, 3 inner, tuned for balanced accuracy) ----------
outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
spec = make_scorer(lambda a,b: sens_spec(a,b)[1])
scoring = {"specificity": spec, "bal_acc": "balanced_accuracy", "recall": "recall", "precision": "precision", "f1": "f1", "roc_auc": "roc_auc", "accuracy": "accuracy"}
rows = []
for model, strats in strategies.items():
    for s in strats:
        gs = GridSearchCV(build(model, s), pgrid(model, s), scoring="balanced_accuracy",
                          cv=StratifiedKFold(3, shuffle=True, random_state=SEED), n_jobs=-1)
        r = cross_validate(gs, Xtr, ytr, cv=outer, scoring=scoring, n_jobs=1)
        row = {"Model": model, "Imbalance": s}
        for k in scoring: row[k] = r["test_"+k].mean(); row[k+"_sd"] = r["test_"+k].std()
        rows.append(row); print(f"{model:12s} {s:12s} recall={row['recall']:.3f} spec={row['specificity']:.3f} prec={row['precision']:.3f} f1={row['f1']:.3f} auc={row['roc_auc']:.3f}")
res = pd.DataFrame(rows).sort_values("bal_acc", ascending=False)
res.round(4).to_csv(OUT+"cv_results.csv", index=False)

# ---------- 3) Final fit of every configuration -> hold-out test ----------
test_rows, rocs, cms, fitted = [], {}, {}, {}
for _, rw in res.iterrows():
    m, s = rw.Model, rw.Imbalance
    gs = GridSearchCV(build(m, s), pgrid(m, s), scoring="balanced_accuracy",
                      cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1).fit(Xtr, ytr)
    p, pr = gs.predict(Xte), gs.predict_proba(Xte)[:,1]
    sens, spec = sens_spec(yte, p)
    test_rows.append({"Model": m, "Imbalance": s, "Best params": {k.split("__")[-1]:v for k,v in gs.best_params_.items()},
        "BalAcc": (sens+spec)/2, "Recall": recall_score(yte,p), "Specificity": spec, "Precision": precision_score(yte,p),
        "F1": f1_score(yte,p), "ROC_AUC": roc_auc_score(yte,pr), "Accuracy": accuracy_score(yte,p)})
    rocs[(m,s)] = roc_curve(yte, pr); cms[(m,s)] = confusion_matrix(yte, p); fitted[(m,s)] = gs
test = pd.DataFrame(test_rows); test.round(4).to_csv(OUT+"holdout_results.csv", index=False)

# ---------- 4) Figures (column-width) ----------
# 4a. CV comparison (horizontal bars)
fig, ax = plt.subplots(figsize=(3.4, 4.3))
lab = (res.Model + " (" + res.Imbalance.map({"none":"none","class_weight":"cw","oversample":"ros"}) + ")").values[::-1]
yy = np.arange(len(res)); h = .2
for i,(k,c) in enumerate([("recall","Recall"),("specificity","Specificity"),("bal_acc","Balanced acc."),("roc_auc","ROC-AUC")]):
    ax.barh(yy+(i-1.5)*h, res[k].values[::-1], h, label=c)
ax.set_yticks(yy); ax.set_yticklabels(lab); ax.set_xlim(0,1); ax.set_xlabel("Score (mean over 10 outer folds)")
ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=4, frameon=False, columnspacing=.8, handlelength=1.2); plt.tight_layout(pad=.4); plt.savefig(OUT+"figures/fig2_cv_comparison.png", dpi=300); plt.close()

# 4b. ROC curves (best strategy per model on hold-out balanced accuracy; display only)
best_per = test.sort_values("BalAcc", ascending=False).groupby("Model").head(1)
fig, ax = plt.subplots(figsize=(3.4,3.1))
for _, r in best_per.iterrows():
    f, t, _ = rocs[(r.Model, r.Imbalance)]
    ax.plot(f, t, lw=1, label=f"{r.Model} ({r.Imbalance}) {r.ROC_AUC:.2f}")
ax.plot([0,1],[0,1],"k--",lw=.6); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)")
ax.legend(loc="lower right"); plt.tight_layout(pad=.4); plt.savefig(OUT+"figures/fig3_roc_holdout.png", dpi=300); plt.close()

# 4c. Confusion matrices (2 x 3 grid)
fig, ax = plt.subplots(2, 3, figsize=(3.4, 3.0)); ax = ax.ravel()
for a in ax: a.axis("off")
for a, (_, r) in zip(ax, best_per.iterrows()):
    a.axis("on"); cm = cms[(r.Model, r.Imbalance)]; a.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2): a.text(j, i, cm[i,j], ha="center", va="center", fontsize=8, color="white" if cm[i,j] > cm.max()/2 else "black")
    a.set_xticks([0,1]); a.set_yticks([0,1]); a.set_xticklabels(["N","P"]); a.set_yticklabels(["N","P"])
    a.set_title(f"{r.Model}\n({r.Imbalance})", fontsize=5.5, pad=2); a.set_xlabel("Predicted", labelpad=1); a.set_ylabel("Actual", labelpad=1)
plt.tight_layout(pad=.4, h_pad=1.6, w_pad=.8); plt.savefig(OUT+"figures/fig4_confusion_matrices.png", dpi=300); plt.close()

# ---------- 5) Interpretation: LogReg coefficients & tree importances ----------
lr = fitted[("LogReg","class_weight")].best_estimator_
coef = pd.Series(lr.named_steps["clf"].coef_[0], index=X.columns).sort_values()
imp = pd.Series(fitted[("DecisionTree","class_weight")].best_estimator_.named_steps["clf"].feature_importances_,
                index=X.columns).sort_values()
fig, ax = plt.subplots(2,1, figsize=(3.4,3.8))
coef.plot.barh(ax=ax[0]); ax[0].set_title("Logistic regression coefficients (balanced, standardised)")
imp.plot.barh(ax=ax[1]); ax[1].set_title("Decision tree feature importance (balanced, depth 2)")
plt.tight_layout(pad=.4); plt.savefig(OUT+"figures/fig5_feature_effects.png", dpi=300); plt.close()

# ---------- 6) Threshold analysis for screening (out-of-fold on training set, best config) ----------
top = res.iloc[0]; best_m, best_s = top.Model, top.Imbalance
oof = cross_val_predict(fitted[(best_m,best_s)].best_estimator_, Xtr, ytr,
                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:,1]
th = []
for t in np.arange(.2,.85,.05):
    p = (oof>=t).astype(int); se, sp = sens_spec(ytr, p)
    th.append({"threshold":round(t,2),"recall":se,"specificity":sp,"precision":precision_score(ytr,p),"f1":f1_score(ytr,p)})
th = pd.DataFrame(th).round(3); th.to_csv(OUT+"threshold_analysis.csv", index=False)

# operating point: lowest-risk-of-missing-patients threshold with recall >= 0.85 on out-of-fold training data
ok = th[th.recall >= 0.85]; t_star = float(ok.sort_values("specificity", ascending=False).iloc[0].threshold) if len(ok) else .5
pte = fitted[(best_m,best_s)].predict_proba(Xte)[:,1]; pp = (pte>=t_star).astype(int); se, sp = sens_spec(yte, pp)
op = pd.DataFrame([{"Model":best_m,"Imbalance":best_s,"threshold":t_star,"Recall":se,"Specificity":sp,
     "Precision":precision_score(yte,pp),"F1":f1_score(yte,pp),"ROC_AUC":roc_auc_score(yte,pte)}]).round(3)
op.to_csv(OUT+"holdout_screening_threshold.csv", index=False)
cmop = confusion_matrix(yte, pp); print("\nScreening operating point on hold-out:\n", op.to_string(index=False), "\nConfusion matrix [[TN FP][FN TP]]:\n", cmop)
print("\nBEST by CV balanced accuracy:", best_m, best_s); print("\n=== CV ===\n", res[["Model","Imbalance","recall","specificity","precision","f1","bal_acc","roc_auc"]].round(3).to_string(index=False))
print("\n=== HOLD-OUT ===\n", test.drop(columns="Best params").round(3).to_string(index=False))
print("\nCoefficients:\n", coef.round(3).to_string()); print("\nTree importance:\n", imp.round(3).to_string())
print("\nThreshold analysis (", best_m, best_s, "):\n", th.to_string(index=False))

# ---------- 7) Simple baselines (same outer CV and same hold-out) ----------
from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
base_models = {
    "Majority class (always 'patient')": DummyClassifier(strategy="most_frequent"),
    "Stratified random guess": DummyClassifier(strategy="stratified", random_state=SEED),
    "Single feature: log Total Bilirubin (LogReg)": Pipeline([
        ("sel", ColumnTransformer([("tb", Pipeline([("log", FunctionTransformer(np.log1p)), ("sc", StandardScaler())]),
                                    ["Total_Bilirubin"])])),
        ("clf", LogisticRegression(class_weight="balanced"))]),
}
brow = []
for name, bm in base_models.items():
    r = cross_validate(bm, Xtr, ytr, cv=outer, scoring=scoring, n_jobs=1)
    bm.fit(Xtr, ytr); p = bm.predict(Xte)
    pr = bm.predict_proba(Xte)[:,1]; se, sp = sens_spec(yte, p)
    brow.append({"Baseline": name, **{"cv_"+k: r["test_"+k].mean() for k in ["recall","specificity","precision","f1","bal_acc","roc_auc"]},
                 "test_recall": se, "test_specificity": sp, "test_precision": precision_score(yte,p, zero_division=0),
                 "test_f1": f1_score(yte,p), "test_roc_auc": roc_auc_score(yte,pr)})
base = pd.DataFrame(brow).round(3); base.to_csv(OUT+"baseline_results.csv", index=False)
print("\n=== BASELINES ===\n", base.T.to_string())

# ---------- 8) Bootstrap 95% CI on hold-out for the best config ----------
rng = np.random.RandomState(SEED); pb = fitted[(best_m,best_s)].predict_proba(Xte)[:,1]
yt = yte.values; aucs, recs, specs = [], [], []
for _ in range(2000):
    i = rng.randint(0, len(yt), len(yt))
    if len(np.unique(yt[i])) < 2: continue
    aucs.append(roc_auc_score(yt[i], pb[i])); q = (pb[i] >= t_star).astype(int)
    a, b = sens_spec(yt[i], q); recs.append(a); specs.append(b)
ci = lambda v: tuple(np.percentile(v, [2.5, 97.5]).round(3))
print("\nBootstrap 95% CI (hold-out):", best_m, best_s, "AUC", ci(aucs), "| recall@t*", ci(recs), "| spec@t*", ci(specs))
print("Hold-out class counts:", yte.value_counts().to_dict(), "| train:", ytr.value_counts().to_dict())
print("Best params:\n", test[["Model","Imbalance","Best params"]].to_string(index=False))
