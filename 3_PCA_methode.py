import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA  # Pour réduire à 2 dimensions pour la visualisation
from sklearn.preprocessing import StandardScaler  # Pour normaliser les données

# Charger les données depuis le fichier CSV
data = pd.read_csv("Mall_Customers.csv")#a modifier

# Garder uniquement les colonnes pertinentes
features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Normalisation des données pour que toutes les colonnes aient le même poids
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Application de Spectral Clustering
spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42)
clusters = spectral.fit_predict(features_scaled)

# Ajout des clusters aux données originales
data['Cluster'] = clusters

# Réduction des dimensions avec PCA pour visualisation
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

# Visualisation des clusters
plt.figure(figsize=(8, 6))
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap='viridis', s=50, edgecolor='k')
plt.title("Spectral Clustering des clients")
plt.xlabel("Dimension 1 (PCA)")
plt.ylabel("Dimension 2 (PCA)")
plt.colorbar(label='Cluster')
plt.show()

# Afficher les premiers résultats avec clusters
print("------ Aperçu des données avec clusters ------")
print(data.head())