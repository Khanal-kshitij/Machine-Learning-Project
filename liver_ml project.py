import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import RandomOverSampler

# 1. Load Dataset
# Downloaded from UCI ML Repository: https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset
columns = [
    'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 
    'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 
    'Aspartate_Aminotransferase', 'Total_Protiens', 
    'Albumin', 'Albumin_and_Globulin_Ratio', 'Dataset'
]
df = pd.read_csv(
    'Indian Liver Patient Dataset (ILPD).csv',
    names=columns
)

# Map target label: UCI dataset uses 1 (Patient) and 2 (Non-Patient). Map to 1 and 0.
df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})

# 2. Preprocessing
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

X = df.drop(columns=['Dataset'])
y = df['Dataset']

# Handle missing values
X = df.drop(columns=['Dataset'])
y = df['Dataset']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

imputer = SimpleImputer(strategy='median')

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Class Balancing via Random Oversampling
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train_scaled, y_train)

# 3. Model Initialization
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Gaussian Naive Bayes": GaussianNB(),
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF Kernel)": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# 4. Benchmarking Pipeline
results = []

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
    
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
    cm = confusion_matrix(y_test, y_pred)
    
    results.append({
        "Model": name,
        "Recall (Sensitivity)": round(rec, 4),
        "Precision": round(prec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(auc, 4),
        "Confusion Matrix": cm.tolist()
    })

# Output Results
results_df = pd.DataFrame(results)
print(results_df.to_string())
