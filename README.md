Machine Learning Project: Early Liver Disease Prediction
Overview

This project presents a comparative machine learning study focused on the early detection of liver disease using the Indian Liver Patient Dataset (ILPD). The primary goal is to evaluate and compare the performance of various machine learning classification algorithms to identify the most effective model for supporting clinical screening and diagnosis. The project implements a complete machine learning workflow, including data preprocessing, feature engineering, handling class imbalance, model training, and performance evaluation.

Problem Statement

Liver disease is a major global health concern, and early diagnosis can significantly improve treatment outcomes. Healthcare professionals often rely on routine blood test results to identify potential liver conditions. This project investigates how machine learning can assist in predicting liver disease by analyzing patient demographic information and biochemical attributes.

Dataset

The project uses the Indian Liver Patient Dataset (ILPD), which contains patient information and laboratory test results relevant to liver disease diagnosis.

The dataset includes features such as:

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

The target variable indicates whether a patient is diagnosed with liver disease or not.

Methodology
Data Preprocessing

The dataset undergoes several preprocessing steps to improve model performance:

Median imputation for handling missing values
Data cleaning and validation
Feature transformation and preparation
Encoding of categorical variables where necessary
Feature Scaling

Numerical features are standardized using Z-score normalization to ensure consistent feature distribution across models.

Class Imbalance Handling

To address class imbalance within the dataset, random oversampling techniques are applied to create a more balanced training set.

Machine Learning Models Evaluated

The following classification algorithms are implemented and compared:

Logistic Regression
Naive Bayes
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Decision Tree Classifier

These models are chosen because they represent a diverse set of machine learning approaches and provide a strong benchmark for medical prediction tasks.

Evaluation Metrics

Model performance is assessed using multiple evaluation metrics:

Accuracy
Precision
Recall
F1-Score
ROC-AUC Score
Confusion Matrix

Particular emphasis is placed on Recall and ROC-AUC because, in healthcare applications, minimizing false negatives is critical for ensuring that potential liver disease cases are not overlooked.

Results and Findings

The project provides a comparative analysis of various machine learning models and highlights the strengths and limitations of each approach for liver disease prediction. The findings demonstrate how preprocessing strategies, feature scaling, and oversampling influence model performance and clinical applicability. Detailed analyses and experimental results are documented in the project report.

Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Jupyter Notebook
Repository Contents
Kshitij_Khanal_Liver_Disease_ML_Submission.pdf – Detailed project report and findings.
README.md – Project documentation.
Future Improvements

Potential enhancements for future work include:

Ensemble learning techniques such as Random Forest and XGBoost
Hyperparameter optimization
Advanced feature selection methods
Explainable AI techniques (SHAP, LIME)
Model deployment through a web-based application
Integration with healthcare decision-support systems
Conclusion

This project demonstrates the application of machine learning techniques for early liver disease screening using routine clinical test data. By comparing multiple classification algorithms and emphasizing clinically relevant evaluation metrics, the study provides valuable insights into the potential of machine learning to support healthcare professionals in disease detection and risk assessment.

Author

Kshitij Khanal

GitHub: https://github.com/Khanal-kshitij

License

This project is intended for educational and academic purposes.
