# Facebook Seller Post Clustering with K-Means

This project uses the **K-Means clustering algorithm** to segment Facebook seller posts from Thailand based on user engagement data.

The dataset includes metrics like reactions, comments, and shares for posts from various sellers. We apply unsupervised learning to find natural groupings in the data and visualize them in 2D using scaled features.

## 📊 What It Does

- Loads the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/488/facebook+live+sellers+in+thailand)
- Cleans and scales the numeric data
- Applies K-Means clustering with 3 clusters
- Visualizes the clusters, data points, and decision boundaries using Matplotlib

## 🧠 Key Concepts

- **Unsupervised Learning**
- **K-Means Clustering**
- **Data Scaling**
- **2D Visualization of High-Dimensional Data**

## 🛠️ Requirements

- Python 3
- numpy
- matplotlib
- scikit-learn
- ucimlrepo

Install dependencies with:

```bash
pip install numpy matplotlib scikit-learn ucimlrepo
