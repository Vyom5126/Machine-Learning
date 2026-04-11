# Feature Engineering: Handling Missing Data 

##  Project Overview
This project demonstrates the critical role of **missing data imputation** in Machine Learning preprocessing. Using the Titanic Survival dataset, this notebook compares the mathematical intuition and visual impact of various imputation strategies on both **numerical** and **categorical** features — implemented first manually with `pandas` for intuition, then using production-ready `scikit-learn` pipelines.

Understanding how to handle missing data correctly is essential before feeding data into any ML model, since most algorithms cannot handle `NaN` values and the wrong imputation strategy can silently distort your data's variance and distribution.

##  Dataset
The dataset is a modified version of the classic Titanic Survival dataset, with values intentionally removed to simulate real-world missing data scenarios.

**Numerical features used:** `Age`, `Fare`  
**Categorical features used:** `Sex`, `Embarked`  
**Target variable:** `Survived` (0 = No, 1 = Yes)

> *Original source: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic). Values were intentionally removed from the dataset for educational demonstration purposes.*

##  Libraries Used
- **Pandas & NumPy:** Data manipulation and manual imputation calculations
- **Scikit-Learn (`sklearn`):** `train_test_split`, `SimpleImputer`, `ColumnTransformer`
- **Matplotlib:** KDE plots and bar charts for visual comparison

##  Imputation Techniques Covered

### Numerical Imputation
1. **Mean & Median Imputation**  
   Best for data that is Missing Completely at Random (MCAR, <5% missing). Mean is used for normally distributed data; Median is preferred when outliers are present. KDE and Boxplot comparisons show how the distribution shifts after imputation.

2. **Arbitrary Value Imputation**  
   Used when data is NOT missing at random. Missing values are replaced with extreme sentinel values (e.g., `99` or `-1`) that are far from the normal range, flagging missingness for the model without destroying structure.

3. **End of Distribution (EOD) Imputation**  
   A mathematically principled version of arbitrary imputation. The fill value is computed as **Mean + 3 × Standard Deviation**, placing it at the extreme tail of the distribution — signalling missingness without inventing an arbitrary number.

### Categorical Imputation
4. **Most Frequent (Mode) Imputation**  
   Used when missingness is completely random. Missing categories are replaced with the most common category (mode). A survival distribution comparison (before vs. after) visually demonstrates the impact on the `Sex` feature.

5. **"Missing" Category Imputation**  
   Used when missingness is not random and carries information. Rather than imputing a real value, a new category `"Missing"` is introduced, allowing the model to learn from the pattern of missingness itself.

##  Key Insight: Index Preservation in Scikit-Learn
A critical practical trap is demonstrated in this notebook: `ColumnTransformer` drops DataFrame indices and returns a NumPy array. Every transformed output is explicitly converted back to a `pd.DataFrame` with the **original shuffled index from `train_test_split`** preserved — preventing silent row misalignment between features and the target variable.

##  Key Visualizations
- **KDE Plots:** Show how each imputation method distorts (or preserves) the original probability density of the feature
- **Boxplots:** Compare spread and outlier behaviour before and after imputation
- **Bar Charts:** Show category frequency distributions for categorical imputation

##  How to Run
1. Clone this repository to your local machine.
2. Ensure you have the required libraries installed:
```bash
   pip install pandas numpy matplotlib scikit-learn
```
3. Open `titanic.ipynb` in Jupyter Notebook or JupyterLab.
4. Run cells sequentially to see the visual impact of each technique.