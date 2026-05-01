# Multiple Linear Regression: From Scratch vs. Scikit-Learn

## Project Overview
This project demonstrates a deep understanding of **Multiple Linear Regression** by implementing it from scratch using pure NumPy math, then verifying it against Scikit-Learn's optimized implementation. Using the Diabetes dataset, the notebook proves that the custom algorithm produces **identical results** to sklearn - confirming the mathematical intuition behind the algorithm.

This "from scratch" approach is the best way to understand what a library like sklearn is actually doing under the hood.

## Dataset
The dataset is the built-in **Diabetes dataset** from `sklearn.datasets`, containing 442 patient samples with 10 numeric features predicting disease progression.

- **Features:** Age, Sex, BMI, Blood Pressure, and 6 blood serum measurements (10 features total)
- **Target Variable:** Quantitative measure of disease progression one year after baseline

## Libraries Used
- **NumPy:** Array operations and matrix algebra for the custom implementation
- **Scikit-Learn (`sklearn`):** `load_diabetes`, `train_test_split`, `LinearRegression`, and evaluation metrics (`MAE`, `MSE`, `RMSE`, `R²`)

## Key Concepts & Workflow

### 1. Building the Custom Algorithm
A `Multiple_Linear_Regression` class is built from scratch using only NumPy. The `.fit()` method solves for the optimal weights analytically using the **Normal Equation**:

$$\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

A column of ones is prepended to the feature matrix so the intercept $b$ is absorbed into $\mathbf{w}$ as $w_0$, keeping the implementation clean and general. The `.predict()` method then applies $\hat{y} = \mathbf{X}\mathbf{w}$ to unseen test data.

### 2. Scikit-Learn Implementation
The identical task is performed using `sklearn.linear_model.LinearRegression`. Unlike the SLR case, **no reshape is needed** here — sklearn's MLR naturally expects 2D feature arrays, and the diabetes dataset is already in that shape.

### 3. Model Verification & Comparison
Both models are evaluated side-by-side using four metrics: **MAE, MSE, RMSE, and R²**. The intercept, all coefficients, metrics, and individual predictions all match perfectly — validating that the custom Normal Equation math is correct.

## Key Takeaway
If your from-scratch implementation matches sklearn exactly, it means you haven't just used the algorithm — you understand it. The Normal Equation is a powerful closed-form solution that scales naturally from one feature (SLR) to any number of features (MLR) with a single matrix operation, which is precisely what sklearn is computing under the hood.

## How to Run
1. Clone this repository to your local machine.
2. Ensure you have the required libraries installed:
```bash
pip install numpy scikit-learn
```
3. Open the notebook in Jupyter Notebook or JupyterLab.
4. Run cells sequentially to see the custom implementation, sklearn comparison, and metric outputs.
