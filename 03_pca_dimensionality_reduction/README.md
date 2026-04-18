# Dimensionality Reduction: Principal Component Analysis (PCA) 📉

## Project Overview
This project demonstrates how **Principal Component Analysis (PCA)** combats the **Curse of Dimensionality** in image classification. Using the MNIST Digit Recognizer dataset, we establish a KNN baseline on raw 784-feature image data, then apply PCA to reduce dimensionality — comparing accuracy, finding the optimal number of components, and finally projecting high-dimensional data into 2D and 3D space for visual insight.

PCA is a foundational unsupervised technique that transforms correlated features into a smaller set of linearly uncorrelated **Principal Components**, preserving maximum variance while stripping away noise.

## Dataset
The dataset is the Kaggle [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) (`train.csv`) — a variant of the classic MNIST dataset.

- **Features:** 784 pixel intensity values (one per pixel of a 28×28 grayscale image)
- **Target Variable:** `label` (digit 0–9)

## Libraries Used
- **Pandas & NumPy:** Data loading and array manipulation
- **Scikit-Learn (`sklearn`):** `StandardScaler`, `PCA`, `KNeighborsClassifier`, `accuracy_score`, `train_test_split`
- **Matplotlib:** Image rendering and Cumulative Explained Variance plot
- **Plotly Express:** Interactive 2D and 3D scatter plots for PCA visualization

## Key Concepts & Workflow

### 1. Baseline Model (Without PCA)
A standard `KNeighborsClassifier` is trained on all **784 raw pixel features**. This establishes an accuracy benchmark and illustrates the problem — KNN struggles in high-dimensional spaces because distances between all points converge, making neighbourhood-based decisions unreliable and slow.

### 2. Feature Scaling → PCA (100 Components)
PCA is highly sensitive to feature scale since it maximises variance. `StandardScaler` is applied **before** PCA to prevent high-range pixels from dominating the principal components. The 784 features are then compressed to **100 Principal Components**, and a second KNN model measures the accuracy impact of this compression.

### 3. Finding the Optimal Number of Components
A loop iterates from 1 to 49 components, training and evaluating a fresh KNN model at each step. This reveals the sweet spot:
- **Too few components** → loss of crucial digit information (underfitting)
- **Too many components** → retained noise slows and destabilises the model

PCA effectively acts as a **noise filter**, and the loop identifies the component count that maximises accuracy.

### 4. 2D & 3D Visualisation
By setting `n_components=2` and `n_components=3`, the 784-dimensional image data is projected into human-interpretable space using interactive **Plotly** scatter plots. Distinct digit clusters forming in this compressed space visually confirms that PCA is preserving class-separating structure.

### 5. Explained Variance Analysis
The notebook inspects the PCA object's `explained_variance_ratio_` and plots its **cumulative sum** across all components. This curve is the standard tool for choosing how many components to retain — typically targeting 95% cumulative explained variance.

## Key Visualizations
- **Pixel Image Render:** A sample digit reconstructed from raw pixel values using `imshow`
- **Accuracy vs. Components Plot:** Shows how KNN accuracy changes from 1 to 49 PCA components
- **Interactive 2D Scatter (Plotly):** Digit clusters projected onto 2 principal components, colour-coded by label
- **Interactive 3D Scatter (Plotly):** The same projection extended to 3 components for richer spatial intuition
- **Cumulative Explained Variance Curve:** Standard elbow-style plot to guide component selection

## How to Run
1. Clone this repository to your local machine.
2. Download `train.csv` from the [Kaggle Digit Recognizer Competition](https://www.kaggle.com/competitions/digit-recognizer) and place it in the notebook's directory.
3. Ensure you have the required libraries installed:
```bash
   pip install pandas numpy matplotlib scikit-learn plotly
```
4. Open `pca.ipynb` in Jupyter Notebook or JupyterLab.
5. Run cells sequentially to reproduce the baseline, PCA pipeline, and all visualizations.