# Simple Linear Regression: From Scratch vs. Scikit-Learn 

## Project Overview
This project demonstrates a deep understanding of **Simple Linear Regression** by implementing it from scratch using pure NumPy math, then verifying it against Scikit-Learn's optimized implementation. Using a Placement dataset, the notebook proves that the custom algorithm produces **identical results** to sklearn - confirming the mathematical intuition behind the algorithm.

This "from scratch" approach is the best way to understand what a library like sklearn is actually doing under the hood.

## Dataset
The dataset contains student placement data with a single feature predicting salary.

- **Feature:** CGPA (or equivalent placement score)
- **Target Variable:** Placed Salary/Package

## Libraries Used
- **Pandas & NumPy:** Data loading and array operations for the custom implementation
- **Scikit-Learn (`sklearn`):** `train_test_split`, `LinearRegression`, and evaluation metrics (`MAE`, `MSE`, `MAPE`, `R²`)

## Key Concepts & Workflow

### 1. Building the Custom Algorithm
A `Linear_Regression` class is built from scratch using only NumPy. The `.fit()` method manually computes the optimal slope and intercept using their closed-form formulas:

- **Slope (m):** $m = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}$
- **Intercept (b):** $b = \bar{y} - m\bar{x}$

The `.predict()` method then applies $y = mx + b$ to unseen test data.

### 2. Scikit-Learn Implementation
The identical task is performed using `sklearn.linear_model.LinearRegression`. A key practical note is demonstrated here: sklearn expects **2D feature arrays**, requiring an `.reshape(-1, 1)` before fitting — unlike the custom class which operates on flat 1D arrays.

### 3. Model Verification & Comparison
Both models are evaluated side-by-side using four metrics: **MAE, MSE, RMSE, and R²**. The slope, intercept, metrics, and individual predictions all match perfectly - validating that the custom math is correct.

## Key Takeaway
If your from-scratch implementation matches sklearn exactly, it means you haven't just used the algorithm you understand it. This comparison also highlights a subtle but important API difference: sklearn's requirement for 2D input vs. the raw NumPy loop operating on 1D arrays. 

## How to Run
1. Clone this repository to your local machine.
2. Ensure you have the required libraries installed:
```bash
   pip install pandas numpy scikit-learn
```
3. Open the notebook in Jupyter Notebook or JupyterLab.
4. Run cells sequentially to see the custom implementation, sklearn comparison, and metric outputs.