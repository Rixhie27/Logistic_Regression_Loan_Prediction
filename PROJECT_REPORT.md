# Logistic Regression Project Report

## Objective

To predict whether a loan application will be approved using Logistic Regression.

## 1. Exploratory Data Analysis

The dataset was analyzed using statistical summaries, distributions,
box plots, categorical graphs, and a correlation heatmap.

## 2. Data Cleaning

Missing values were handled using median and mode imputation.
Duplicate records were also checked and removed.

## 3. Data Preprocessing

The `Loan_ID` column was removed. Categorical variables were encoded,
the target variable was converted into numerical values, and the data
was divided into training and testing sets. Feature scaling was also performed.

## 4. Model Training

A Logistic Regression model was trained using the preprocessed data.

## 5. Model Evaluation

The model achieved the following results:

- Accuracy: **86.18%**
- Precision: **84.00%**
- Recall: **98.82%**
- F1 Score: **90.81%**
- ROC-AUC: **85.14%**

## Conclusion

The Logistic Regression model successfully predicted loan approval with
good overall performance. The results show that the model was particularly
effective at identifying approved loan applications.