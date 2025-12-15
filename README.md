# Employee Attrition Prediction Using Machine Learning
# Project Overview

Employee attrition is a critical challenge for organizations, impacting productivity, operational continuity, and recruitment costs. This project applies multiple machine learning algorithms to predict whether an employee will leave or stay at a company based on historical HR data.

The goal is to compare classical and advanced models and evaluate their effectiveness in predicting employee attrition using structured tabular data.

# Dataset Description

The dataset used in this project is the HR Capstone Dataset, where each row represents an employee and the target variable indicates whether the employee left the company.

Target Variable

left

1 → Employee left the company

0 → Employee stayed

Key Features

satisfaction_level

last_evaluation

number_project

average_monthly_hours

time_spent_company (employee tenure)

work_accident

promotion_last_5_years

department (categorical)

salary (categorical)

# Data Preprocessing

The following preprocessing steps were performed before modeling:

Data Loading:

Dataset loaded from HR_capstone_dataset.csv.

Initial Data Inspection

Dataset structure examined using .info() and .describe().

Column names reviewed for consistency.

Column Name Standardization

Corrected misspellings such as:

average_montly_hours → average_monthly_hours

time_spend_company → time_spent_company

promotion_last_5years → promotion_last_5_years

Duplicate Handling:

Duplicate rows were identified and removed to ensure data integrity.

Outlier Analysis:

A boxplot was used to visualize employee tenure.

The IQR method was applied to detect outliers in time_spent_company.

Categorical Encoding

Label Encoding was applied to:

department

salary

Feature Scaling:

Standardization was applied where required (Logistic Regression and Neural Network).

# Modeling Approach

The dataset was split into:

Training set: 80%

Testing set: 20%
(Stratified sampling was used to preserve class distribution.)

The following models were implemented:

_Logistic Regression_

Baseline, interpretable model

_Decision Tree Classifier_

Captures non-linear decision boundaries

_Random Forest Classifier_

Ensemble model for improved generalization

_Neural Network (MLP)_

Multi-layer perceptron with hidden layers

# Results and Performance
Model	Accuracy
Logistic Regression	83.66%
Decision Tree	96.99%
Random Forest	98.67%
Neural Network (MLP)	97.96%
Detailed Observations

Logistic Regression struggled to correctly identify employees who left, showing low recall for the minority class.

Decision Tree significantly improved performance by learning non-linear relationships.

Random Forest achieved the best overall results, offering high precision and recall for both classes.

Neural Network performed competitively, slightly below Random Forest, and demonstrated strong predictive capability when features were scaled.

# Conclusion

The results clearly indicate that tree-based ensemble methods, particularly Random Forest, are highly effective for employee attrition prediction on structured HR data. While Logistic Regression provides interpretability, it is limited in handling complex patterns. Neural Networks offer strong performance but require careful preprocessing and tuning.

This project highlights how machine learning can support data-driven HR decision-making by identifying employees at risk of leaving.

# Future Improvements

Hyperparameter tuning using GridSearchCV

ROC–AUC and precision–recall curve analysis

Feature importance and SHAP-based explainability

Handling class imbalance using resampling techniques

Cross-validation for more robust evaluation

# Technologies Used

Python

Pandas, NumPy

Scikit-learn

Matplotlib

# Author

Mahmood Ali Khan
Master’s in Artificial Intelligence
Machine Learning & Data Analytics Enthusiast
