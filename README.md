# Employee-Attrition-Prediction-Using-Machine-Learning-and-Neural-Networks
# Project Overview

This project focuses on predicting employee attrition—whether an employee will leave or stay—using machine learning techniques. The dataset contains employee-related attributes such as satisfaction level, performance evaluation, workload, tenure, promotion history, department, and salary.

The objective is to build and compare multiple machine learning models to identify the most effective approach for predicting attrition.

# Dataset Description

Each row in the dataset represents an employee, and the target variable is:

left → 1 if the employee left the company, 0 otherwise

Key features include:

satisfaction_level

last_evaluation

number_project

average_monthly_hours

time_spent_company

work_accident

promotion_last_5_years

department (label encoded)

salary (label encoded)

The dataset was cleaned prior to modeling, with consistent column naming, no missing values, and appropriate encoding of categorical variables.

# Methodology

Train–Test Split:

The data was split into training (80%) and testing (20%) sets.

Stratified sampling was used to preserve class distribution.

Feature Scaling:

Standardization was applied for Logistic Regression and Neural Network models.

Models Implemented:

Logistic Regression (baseline, interpretable model)

Decision Tree Classifier

Random Forest Classifier

Neural Network (Multi-Layer Perceptron)

Evaluation Metrics:

Accuracy

Precision

Recall

F1-score

# Results
#Model	Accuracy
Logistic Regression	83.66%
Decision Tree	96.99%
Random Forest	98.67%
Neural Network	97.96%
Key Observations

Logistic Regression struggled to correctly identify employees who left, indicating difficulty handling non-linear relationships.

Decision Tree significantly improved performance by capturing feature interactions.

Random Forest achieved the best overall performance, demonstrating strong generalization and robustness.

Neural Network also performed exceptionally well, slightly below Random Forest.

# Conclusion

Tree-based ensemble methods, particularly Random Forest, proved most effective for predicting employee attrition in this dataset. While Logistic Regression provided interpretability, its predictive power was limited. Neural Networks offered strong performance but required careful scaling and tuning.

This project demonstrates how machine learning can be effectively applied to HR analytics to support data-driven decision-making.
