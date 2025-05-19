# Loan Approval Prediction

This project applies machine learning techniques to predict whether a loan application will be approved based on demographic and financial attributes of applicants.

## Project Overview

Manual loan approval processes can be inconsistent and slow. This project demonstrates how supervised machine learning models can automate this decision-making process efficiently and accurately. Three models were explored:

- Logistic Regression  
- k-Nearest Neighbors (kNN)  
- Decision Tree Classifier  

The dataset was preprocessed, models were trained and fine-tuned, and their performance was evaluated using multiple metrics.

## Techniques Used

- Data cleaning and preprocessing (missing value imputation, encoding, scaling)
- Train-test split (80/20)
- Model training and evaluation
- Hyperparameter tuning using `GridSearchCV`
- Evaluation metrics: Accuracy, Precision, Recall, F1-score, Cross-validation

## Results Summary

After tuning, the Decision Tree Classifier achieved the highest performance:

| Model              | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|--------|----------|
| Logistic Regression | 92.45%   | 89.76%    | 93.20% | 91.47%   |
| kNN                | 90.12%   | 88.50%    | 91.00% | 89.73%   |
| Decision Tree      | 96.84%   | 93.96%    | 97.80% | 95.84%   |

## Dataset

- **Source**: (https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
- Contains information on loan applicants, such as credit score, income, loan amount, etc.

## Files

- `loan_approval_dataset.csv`: Raw dataset used for training
- `loan.ipynb`: Jupyter Notebook with preprocessing, model training, evaluation, and tuning steps
- `Mlnn-midterm-report.pdf`: Full midterm project report with detailed analysis and results

## Future Work

- Add ensemble models like Random Forest or Gradient Boosting
- Improve feature engineering
- Explore model interpretability tools (e.g., SHAP, LIME)
- Deploy the model with a simple UI

