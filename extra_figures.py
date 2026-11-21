"""Extra descriptive statistics and figures for the paper (EDA table, correlation heatmap, threshold curve, pipeline diagram)."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import mannwhitneyu
import os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results") + os.sep
os.makedirs(OUT + "figures", exist_ok=True)
cols = ["Age","Gender","Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase","ALT","AST",
        "Total_Proteins","Albumin","Albumin_Globulin_Ratio","Dataset"]
df = pd.read_csv(os.path.join(ROOT, "data", "ILPD.csv"), header=None, names=cols)
df["y"] = (df.Dataset == 1).astype(int); df["Male"] = (df.Gender == "Male").astype(int)
num = [c for c in cols[:-1] if c != "Gender"]
rows = []
for c in num:
    a, b = df.loc[df.y==1, c].dropna(), df.loc[df.y==0, c].dropna()
    rows.append({"Variable": c, "Missing": int(df[c].isna().sum()), "Median_all": df[c].median(),
        "Median_patient": a.median(), "Median_non": b.median(), "Skew": df[c].skew(),
        "MW_p": mannwhitneyu(a, b).pvalue})
st = pd.DataFrame(rows); st.to_csv(OUT+"eda_summary.csv", index=False); print(st.round(4).to_string(index=False))
print("Female patient %:", round(df[df.Gender=="Female"].y.mean()*100,1), "Male patient %:", round(df[df.Gender=="Male"].y.mean()*100,1),
      "| n female", (df.Gender=="Female").sum(), "n male", (df.Gender=="Male").sum())
print("Age range", df.Age.min(), df.Age.max(), "rows with Alk>... max:", df.Alkaline_Phosphotase.max(), df.AST.max(), df.ALT.max())
cm = df[num].corr(method="spearman"); print(cm.round(2).to_string())
plt.rcParams.update({"font.size": 6, "axes.titlesize": 6.5, "axes.labelsize": 6, "xtick.labelsize": 5, "ytick.labelsize": 5,
                     "legend.fontsize": 5.5, "axes.linewidth": .5, "savefig.bbox": "tight", "savefig.pad_inches": .03})
short = {"Total_Bilirubin":"TB","Direct_Bilirubin":"DB","Alkaline_Phosphotase":"ALP","Total_Proteins":"TP","Albumin":"Alb","Albumin_Globulin_Ratio":"A/G"}
lab = [short.get(c, c) for c in num]
fig, ax = plt.subplots(figsize=(3.4, 3.0)); im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(num))); ax.set_yticks(range(len(num))); ax.set_xticklabels(lab); ax.set_yticklabels(lab)
for i in range(len(num)):
    for j in range(len(num)): ax.text(j, i, f"{cm.iloc[i,j]:.2f}".replace("0.", ".").replace("1.00","1"), ha="center", va="center", fontsize=4.5, color="white" if abs(cm.iloc[i,j]) > .6 else "black")
plt.colorbar(im, fraction=.046, pad=.02); plt.tight_layout(pad=.4)
plt.savefig(OUT+"figures/fig_corr.png", dpi=300); plt.close()

th = pd.read_csv(OUT+"threshold_analysis.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.plot(th.threshold, th.recall, "o-", ms=2, lw=.8, label="Recall"); ax.plot(th.threshold, th.specificity, "s-", ms=2, lw=.8, label="Specificity")
ax.plot(th.threshold, th.precision, "^-", ms=2, lw=.8, label="Precision"); ax.axvline(.35, ls="--", c="gray", lw=.7); ax.axhline(.85, ls=":", c="gray", lw=.7)
ax.text(.36, .03, "threshold 0.35", fontsize=5.5); ax.set_xlabel("Decision threshold on P(patient)"); ax.set_ylabel("Score")
ax.legend(); plt.tight_layout(pad=.4); plt.savefig(OUT+"figures/fig_threshold.png", dpi=300); plt.close()

steps = [("1  Raw data", "ILPD.csv, 583 rows, 10 features"), ("2  Cleaning", "label 1 = patient; Gender to 0/1; duplicates kept"),
         ("3  Stratified split", "80% train (466) / 20% hold-out (117)"),
         ("4  Pipeline (fit on train only)", "median impute, log1p (TB, DB, ALP, ALT, AST), scale, [oversample]"),
         ("5  Tuning + nested CV", "grid search (3-fold) in 5x2 outer CV; metric: balanced accuracy"),
         ("6  Hold-out test + threshold", "recall, specificity, precision, F1, ROC-AUC, confusion matrix")]
fig, ax = plt.subplots(figsize=(3.4, 3.0)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, len(steps))
for i, (t, d) in enumerate(steps):
    y = len(steps) - i - 1
    ax.add_patch(FancyBboxPatch((.02, y+.14), .96, .66, boxstyle="round,pad=0.005", fc="#eef3fa", ec="#33557a", lw=.6))
    ax.text(.05, y+.56, t, fontsize=6.3, weight="bold"); ax.text(.05, y+.27, d, fontsize=5.4)
    if i < len(steps)-1: ax.annotate("", xy=(.5, y-.2), xytext=(.5, y+.14), arrowprops=dict(arrowstyle="-|>", color="#33557a", lw=.6))
plt.tight_layout(pad=.2); plt.savefig(OUT+"figures/fig_pipeline.png", dpi=300); plt.close()
