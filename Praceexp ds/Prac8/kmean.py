import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# 📌 Step 1: Generate Sample Data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)  # Creating 4 clusters
X = StandardScaler().fit_transform(X)  # Normalize the data

# 📊 Step 2: Elbow Method to Find Optimal Clusters
wcss = []  # Within-Cluster Sum of Squares
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # Inertia measures cluster compactness

# 🔥 Print WCSS values for different K
print("\n📊 WCSS Values for Different K:")
for i, val in enumerate(wcss, start=1):
    print(f"K={i}: WCSS={val:.2f}")

# 🔥 Step 3: Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal K')
plt.show()

# ✅ Step 4: Apply K-Means with Optimal Clusters (e.g., K=4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# 🖨️ Print Cluster Assignments
print("\n📌 Cluster Assignments for First 10 Points:")
for i in range(10):
    print(f"Point {i+1}: Cluster {y_kmeans[i]}")

# 🖨️ Print Centroids
print("\n📍 Cluster Centroids:")
print(kmeans.cluster_centers_)

# 🎨 Step 5: Visualize Clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', edgecolors='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering Results')
plt.legend()
plt.show()
