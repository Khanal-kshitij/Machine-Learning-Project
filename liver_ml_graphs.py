import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.utils import resample

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Set overall seaborn visual styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11})

# -------------------------------------------------------------------------
# 1. Load Dataset & Preprocessing
# -------------------------------------------------------------------------
columns = [
    'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 
    'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 
    'Aspartate_Aminotransferase', 'Total_Protiens', 
    'Albumin', 'Albumin_and_Globulin_Ratio', 'Target'
]

import pandas as pd
df = pd.read_csv('Indian Liver Patient Dataset (ILPD).csv')

# Target mapping: 1 -> Patient (Positive class), 2 -> Healthy (Negative class)
df['Target_Label'] = df['Target'].map({1: 'Liver Disease (1)', 2: 'Healthy (2)'})
df['Target'] = df['Target'].map({1: 1, 2: 0})

df_encoded = df.copy()
df_encoded['Gender'] = LabelEncoder().fit_transform(df_encoded['Gender'])

X = df_encoded.drop(columns=['Target', 'Target_Label'])
y = df_encoded['Target']

# Median Imputation for missing values
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Stratified Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

# Random Oversampling on training split to handle class imbalance
X_train_df = pd.DataFrame(X_train, columns=X.columns)
X_train_df['Target'] = y_train.values

df_majority = X_train_df[X_train_df.Target == 1]
df_minority = X_train_df[X_train_df.Target == 0]

df_minority_upsampled = resample(
    df_minority, replace=True, n_samples=len(df_majority), random_state=42
)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

X_train_res = df_upsampled.drop('Target', axis=1)
y_train_res = df_upsampled['Target']

# Feature Scaling
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Classifiers
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Gaussian Naive Bayes": GaussianNB(),
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# -------------------------------------------------------------------------
# GRAPH 1: Target Class Imbalance Bar Plot
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x='Target_Label', palette=['#d9534f', '#5cb85c'], ax=ax)
ax.set_title('Target Class Distribution (Class Imbalance)', fontsize=13, fontweight='bold')
ax.set_xlabel('Diagnosis')
ax.set_ylabel('Count')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', fontsize=12, color='white', fontweight='bold')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300)
plt.show()

# -------------------------------------------------------------------------
# GRAPH 2: Feature Correlation Heatmap
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(X_imputed.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax, cbar=True, square=True)
ax.set_title('Feature Correlation Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.show()

# -------------------------------------------------------------------------
# GRAPH 3: ROC Curves Comparison Across Classifiers
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
cms = {}

for name, model in models.items():
    model.fit(X_train_res_scaled, y_train_res)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    
    ax.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})', linewidth=2)
    cms[name] = confusion_matrix(y_test, y_pred)

ax.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.50)')
ax.set_xlabel('False Positive Rate (1 - Specificity)')
ax.set_ylabel('True Positive Rate (Recall / Sensitivity)')
ax.set_title('ROC Curves Comparison Across Classifiers', fontsize=13, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300)
plt.show()

# -------------------------------------------------------------------------
# GRAPH 4: Confusion Matrices Grid
# -------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for idx, (name, cm) in enumerate(cms.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                xticklabels=['Healthy (0)', 'Disease (1)'],
                yticklabels=['Healthy (0)', 'Disease (1)'])
    axes[idx].set_title(name, fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

axes[5].axis('off')  # Hide empty subplot space
plt.suptitle('Confusion Matrices for Classifiers', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300)
plt.show()

# -------------------------------------------------------------------------
# GRAPH 5: Performance Metrics Multi-Bar Chart Comparison
# -------------------------------------------------------------------------
metrics_list = []
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    metrics_list.append({
        'Model': name,
        'Recall': recall_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    })

metrics_df = pd.DataFrame(metrics_list)
metrics_melted = pd.melt(metrics_df, id_vars=['Model'], var_name='Metric', value_name='Score')

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=metrics_melted, x='Model', y='Score', hue='Metric', palette='Set2', ax=ax)
ax.set_title('Classifier Performance Metrics Comparison', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.show()
