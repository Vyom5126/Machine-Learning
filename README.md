# Machine-Learning
# Feature Scaling: Standardization vs. Normalization

## 📌 Project Overview
This project demonstrates the critical role of data preprocessing in Machine Learning, specifically focusing on **Feature Scaling**. Using a Social Network Ads dataset, this notebook compares the effects of **Standardization** (Z-score scaling) and **Normalization** (Min-Max scaling) on numerical features like Age and Estimated Salary.

Understanding how to properly scale data is essential for distance-based Machine Learning algorithms (like K-Nearest Neighbors or Support Vector Machines) to perform optimally without being biased by features with larger magnitudes.

## 📊 Dataset
The dataset represents social network user data and whether they purchased a specific product. 
* **Features used:** `Age` and `EstimatedSalary`
* **Target Variable:** `Purchased` (0 = No, 1 = Yes)

## 🛠️ Libraries Used
* **Pandas & NumPy:** Data manipulation and mathematical operations
* **Scikit-Learn (`sklearn`):** Data splitting (`train_test_split`), Standardization (`StandardScaler`), and Normalization (`MinMaxScaler`)
* **Matplotlib & Seaborn:** Data visualization (Scatter plots and Kernel Density Estimate plots)

## 🔍 Key Visualizations & Takeaways
This notebook visually proves the mathematical impact of scaling through:
1. **Scatter Plots:** Showing how the overall distribution and shape of the data points remain identical before and after scaling, but the axes' scales are drastically compressed.
2. **KDE Plots (Probability Density):** * **Standardization:** Centers the data around a mean of 0 with a standard deviation of 1.
   * **Normalization:** Compresses all data points into a strict range between 0 and 1.

By comparing the pre-scaling and post-scaling graphs, we can clearly see that scaling changes the range of the data without distorting the fundamental relationship between the variables.

## 🚀 How to Run
1. Clone this repository to your local machine.
2. Ensure you have the required libraries installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
