Machine Learning Project: Early Liver Disease Prediction
Overview
This project presents a comparative machine learning study for the early detection of liver disease using the Indian Liver Patient Dataset (ILPD). The objective is to evaluate multiple classification algorithms and identify the most effective model for supporting early clinical screening through routine blood test data.

The project implements a complete machine learning pipeline including data preprocessing, feature scaling, imbalance handling, model training, and performance evaluation.

Objectives
Predict the presence of liver disease using patient medical records.
Compare the performance of multiple machine learning algorithms.
Address missing values and class imbalance.
Optimize models for healthcare-focused metrics such as Recall and ROC-AUC.
Provide insights into the suitability of different classifiers for clinical screening applications.
Dataset
Dataset: Indian Liver Patient Dataset (ILPD)

The dataset contains patient information including demographic attributes and blood test measurements commonly used in liver disease diagnosis.

Features
Age
Gender
Total Bilirubin
Direct Bilirubin
Alkaline Phosphotase
Alamine Aminotransferase (ALT)
Aspartate Aminotransferase (AST)
Total Proteins
Albumin
Albumin and Globulin Ratio
Target Variable
Liver Disease Status
Positive (Liver Disease)
Negative (No Liver Disease)
Machine Learning Pipeline
1. Data Preprocessing
Missing value treatment using median imputation
Categorical encoding
Data cleaning and validation
2. Feature Scaling
Z-score standardization for numerical variables
3. Class Imbalance Handling
Random oversampling to balance class distribution
4. Model Training
The following machine learning algorithms were evaluated:

Logistic Regression
Naive Bayes
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Decision Tree
Evaluation Metrics
Models were assessed using:

Accuracy
Precision
Recall
F1-Score
ROC-AUC Score
Confusion Matrix
Special emphasis was placed on Recall and ROC-AUC due to the healthcare context, where minimizing missed diagnoses is critical.

Project Structure
Machine-Learning-Project/
│
├── Kshitij_Khanal_Liver_Disease_ML_Submission.pdf
├── README.md
└── assets/ (optional)
Results
The project compares multiple classification models to determine which provides the best balance between sensitivity and overall predictive performance for liver disease detection.

Key findings include:

Performance differences among traditional machine learning classifiers.
Impact of oversampling on imbalanced healthcare data.
Trade-offs between precision and recall.
Identification of models better suited for screening applications.
Detailed results and analysis can be found in:

Kshitij_Khanal_Liver_Disease_ML_Submission.pdf

Technologies Used
Python
Pandas
NumPy
Scikit-Learn
Matplotlib
Seaborn
Jupyter Notebook
Installation
Clone the repository:

git clone https://github.com/Khanal-kshitij/Machine-Learning-Project.git
cd Machine-Learning-Project
Install required dependencies:

pip install pandas numpy scikit-learn matplotlib seaborn jupyter
Usage
Load the dataset.
Run data preprocessing and feature engineering.
Train the selected machine learning model.
Evaluate performance using the provided metrics.
Compare results across classifiers.
Future Improvements
Ensemble learning methods (Random Forest, XGBoost)
Hyperparameter optimization
Feature selection techniques
Explainable AI (SHAP, LIME)
Deployment as a web application
Author
Kshitij Khanal

GitHub: https://github.com/Khanal-kshitij

License
This project is intended for educational and academic purposes.
