# K-Means Clustering: Customer Segmentation

## Project Overview

This project implements a K-Means clustering pipeline using Scikit-Learn to segment
mall customers based on Age, Annual Income, and Spending Score. The trained model
is then tested on real-world custom survey data collected from 10 individuals.

## Dataset

- **Standard Dataset:** Mall Customers Dataset (200 records) - Age, Annual Income (k$), Spending Score (1-100)
- **Custom Dataset:** Survey data from 10 real individuals with the same features

## Pipeline Steps

1. **Data Loading:** Automatically pulled both datasets from this GitHub repo via raw URLs
2. **Preprocessing:** Applied StandardScaler to normalize features (mean=0, std=1) since
   K-Means relies on Euclidean distance
3. **Optimal K Selection:** Used the Elbow Method, plotting inertia vs K (1-10).
   The elbow occurred at K=5, indicating diminishing returns beyond 5 clusters
4. **Model Training:** Trained KMeans(n_clusters=5, init='k-means++', random_state=42)
5. **Model Persistence:** Saved trained model (`kmeans_model.pkl`) and scaler
   (`scaler.pkl`) using joblib
6. **Real-World Testing:** Loaded custom survey data, applied the SAME fitted scaler
   (transform only, no refit), and predicted cluster assignments

## Results

- The model identified 5 distinct customer segments based on income and spending behavior
- All 10 custom survey individuals were successfully classified into existing clusters
- [Add 2-3 sentences here describing what clusters your custom people fell into,
  e.g., "Person 6 (low income, high spending) fell into the 'careless spenders' cluster,
  similar to young high-spenders in the standard dataset"]

## Files

- `notebook.ipynb` - Full pipeline code
- `data/Mall_Customers.csv` - Standard training dataset
- `data/custom_survey_data.csv` - Custom survey data
- `model/kmeans_model.pkl` - Trained K-Means model
- `model/scaler.pkl` - Fitted StandardScaler

## How to Run

Open `notebook.ipynb` in Google Colab. All data is fetched automatically from
this repository's raw URLs.
