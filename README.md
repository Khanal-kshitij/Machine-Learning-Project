# 🩺 Early Liver Disease Prediction Using Machine Learning

## 📖 Overview

This project presents a comparative machine learning study for the early detection of liver disease using the Indian Liver Patient Dataset (ILPD). The objective is to evaluate multiple classification algorithms and identify the most effective model for supporting clinical screening through routine blood test results. The project implements a complete machine learning pipeline, including data preprocessing, feature scaling, class imbalance handling, model training, and performance evaluation. 

---

## 🎯 Objectives

- Predict the presence of liver disease using patient clinical data.
- Compare the performance of multiple machine learning algorithms.
- Address missing values and class imbalance issues.
- Optimize predictive performance using healthcare-relevant metrics.
- Evaluate model suitability for real-world medical screening applications.

---

## 📊 Dataset

The project utilizes the **Indian Liver Patient Dataset (ILPD)**, which contains patient demographic information and biochemical test results commonly used in liver disease diagnosis.

### Features

- Age
- Gender
- Total Bilirubin
- Direct Bilirubin
- Alkaline Phosphotase
- Alamine Aminotransferase (ALT)
- Aspartate Aminotransferase (AST)
- Total Proteins
- Albumin
- Albumin and Globulin Ratio

### Target Variable

- Liver Disease
- No Liver Disease

---

## ⚙️ Machine Learning Pipeline

### 1. Data Preprocessing

- Missing value treatment using median imputation
- Data cleaning and validation
- Feature preparation and transformation

### 2. Feature Scaling

- Z-score normalization of numerical features to ensure consistent model performance. 

### 3. Class Imbalance Handling

- Random oversampling applied to balance the dataset and improve minority-class prediction performance. 

### 4. Model Training

The following machine learning algorithms were evaluated:

- Logistic Regression
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree Classifier

---

## 📈 Evaluation Metrics

Models were assessed using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

Special emphasis was placed on **Recall** and **ROC-AUC** because accurately identifying patients with liver disease is critical in healthcare applications. 

---

## 🔍 Key Findings

This project provides a comparative analysis of traditional machine learning algorithms for liver disease prediction and highlights:

- The impact of preprocessing techniques on model performance.
- Benefits of handling class imbalance through oversampling.
- Trade-offs between Precision and Recall.
- Model suitability for clinical screening applications.
- Comparative performance evaluation using multiple metrics.

Detailed methodology, experimental results, and analysis are available in the project report.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 📂 Repository Structure

```text
Machine-Learning-Project/
│
├── Kshitij_Khanal_Liver_Disease_ML_Submission.pdf
├── README.md
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Khanal-kshitij/Machine-Learning-Project.git
cd Machine-Learning-Project
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

## ▶️ Usage

1. Load the dataset.
2. Perform preprocessing and feature engineering.
3. Train the selected machine learning model.
4. Evaluate model performance using the defined metrics.
5. Compare results across multiple algorithms.

---

## 🔮 Future Improvements

- Random Forest and XGBoost implementation
- Hyperparameter tuning
- Feature selection techniques
- Explainable AI (SHAP and LIME)
- Model deployment as a web application
- Integration into clinical decision-support systems

---

## 🏆 Conclusion

This project demonstrates how machine learning can assist in the early detection of liver disease using routine clinical data. By comparing multiple classification algorithms and focusing on clinically relevant metrics such as Recall and ROC-AUC, the study provides valuable insights into the application of machine learning for healthcare screening and prediction. 【1-0e577e】

---

## 👨‍💻 Author

**Kshitij Khanal**

GitHub: https://github.com/Khanal-kshitij

---

## 📄 License

This project is intended for educational and academic purposes.
