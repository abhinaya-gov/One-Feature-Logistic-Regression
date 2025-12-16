# One-Feature-Logistic-Regression
This project implements a very simple Logistic Regression completely from scratch using NumPy, without relying on machine learning libraries for training. The goal is to predict whether a student passes an exam based on the number of hours studied.

---

## Problem Statement

Given the number of hours a student studies, predict:
- **0** → Fail  
- **1** → Pass  

This is a binary classification problem.

---

## Key Concepts

- Logistic Regression
- Sigmoid Activation Function
- Binary Cross-Entropy Loss
- Gradient Descent
- Model Evaluation Metrics

---

## Sigmoid Activation

Logistic regression outputs a linear value \( w \cdot x + b \), which can range from negative to positive infinity.  
The **sigmoid function** converts this value into a probability between **0 and 1**, allowing the output to be interpreted as the probability of passing.

Sigmoid helps transform a linear model into a **probabilistic classifier**.

The sigmoid function is defined as:

σ(z) = 1 / (1 + e^(−z)), where z = w · x + b

---

## Binary Cross-Entropy Loss 

Binary cross-entropy measures how well the predicted probabilities match the true binary labels.
- Correct and confident predictions result in low loss
- Confident but incorrect predictions are heavily penalized

This loss function is well-suited for binary classification and enables efficient gradient-based optimization.

Binary Cross-Entropy Loss:

L(y, ŷ) = −[ y·log(ŷ) + (1−y)·log(1−ŷ) ]

Overall cost:

J = −(1/m) Σ [ yᵢ·log(ŷᵢ) + (1−yᵢ)·log(1−ŷᵢ) ]

---

## Dataset

A small synthetic dataset is used:

| Hours Studied | Pass |
|--------------|------|
| 1–4          | 0    |
| 5–10         | 1    |

---

## Implementation Details

The model is built entirely from scratch using NumPy and includes:
- Probability prediction using sigmoid
- Loss computation using binary cross-entropy
- Manual gradient calculation
- Gradient descent optimization
- Threshold-based classification

---

## Evaluation

The model is evaluated using:
- Accuracy
- Confusion Matrix
- Precision, Recall, F1-score
- MSE, RMSE, MAE (on predicted probabilities)

---

## Sklearn Comparison

For validation, the results are compared with `sklearn.linear_model.LogisticRegression`.
Weights and bias from sklearn are printed to verify that the from-scratch implementation converges to similar parameters.

---

## 🚀 Why This Project?

- Builds intuition for probabilistic classification
- Reinforces gradient descent and loss functions
- Avoids black-box ML usage
- Strong foundation before deep learning

---

## 🛠️ Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn (comparison only)

---

## 📄 License

MIT License
