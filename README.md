# 📦 Amazon Product Recommendation System

A Python-based machine learning project that explores and implements multiple recommendation system techniques on Amazon electronics product ratings data.

## 📌 Overview

This project demonstrates how to build product recommendation systems using real-world Amazon electronics ratings data. It covers three distinct approaches — from simple popularity-based methods to advanced collaborative filtering and matrix factorization.

## ✨ Features

- **Popularity-Based Recommendations** — Recommends the most-rated products to all users
- **Collaborative Filtering (Item-Item)** — Uses Pearson correlation via the `Surprise` library (KNNWithMeans) to find similar items and personalize recommendations
- **Model-Based Filtering (SVD)** — Applies Truncated SVD for dimensionality reduction on the user-item utility matrix to generate latent-factor recommendations

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Data visualization |
| `scikit-learn` | SVD, cosine similarity, train/test split |
| `scikit-surprise` | KNN-based collaborative filtering |

## 📂 Project Structure

```
.
├── Amazon_Recommendation_System.ipynb   # Main notebook with all models
└── README.md
```

## 🚀 Getting Started

### Prerequisites

Install the required Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scikit-surprise
```

### Dataset

This project uses the **Amazon Electronics Ratings** dataset, available on [Kaggle](https://www.kaggle.com/datasets/vibivij/amazon-electronics-rating-datasetrecommendation).

Download the dataset and place it at:
```
/kaggle/input/amazon-product-reviews/ratings_Electronics (1).csv
```

The CSV has no header row and contains the following columns:

- `userId` — Unique user identifier
- `productId` — Unique product identifier
- `Rating` — Rating value (1–5)
- `timestamp` — Unix timestamp of the rating

### Running the Notebook

1. Open `Amazon_Recommendation_System.ipynb` in Jupyter Notebook or JupyterLab (or upload to Kaggle):
   ```bash
   jupyter notebook Amazon_Recommendation_System.ipynb
   ```
2. Run all cells sequentially (Kernel → Restart & Run All).

## 📊 Notebook Walkthrough

| Section | Description |
|---|---|
| Load Dataset | Reads the CSV and assigns column headers |
| Dataset Overview | Shape, data types, rating statistics |
| Missing Values | Checks for and reports any null values |
| Rating Distribution | Bar plot of rating frequencies |
| Unique Users & Products | Summary counts |
| Ratings per User Analysis | Quantile analysis of user activity |
| Popularity-Based Model | Top products by number of ratings (≥50 ratings filter) |
| Collaborative Filtering | KNNWithMeans model trained with Surprise; RMSE evaluation |
| Model-Based SVD | Utility matrix → SVD → cosine similarity recommendations |

## 📈 Models

### 1. Popularity-Based Recommendations

Filters products with at least 50 ratings, then ranks them by average rating and total count.

### 2. Item-Item Collaborative Filtering

Uses the `Surprise` library's `KNNWithMeans` algorithm with Pearson similarity on a 70/30 train-test split. Evaluated using RMSE.

### 3. SVD-Based Model

Constructs a user-item ratings matrix from the top 10,000 records, applies `TruncatedSVD` for dimensionality reduction, and computes cosine similarity between product vectors to surface recommendations.

## 📄 License

This project is open-source and available for educational purposes.
