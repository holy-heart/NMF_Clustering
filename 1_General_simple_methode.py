# 000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# Import des bibliothèques
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Charger les données
data = pd.read_csv("D:/Document/Univercité/MMDM/Projet/test/Mall_Customers.csv")#a modifier

features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]  # Variables pertinentes
print(features.head())

# Imputer les valeurs manquantes par la moyenne
features.fillna(features.mean(), inplace=True)

# Identification et traitement des outliers
for col in features.columns:
    q1 = features[col].quantile(0.25)
    q3 = features[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    # Imputer les outliers par la moyenne
    features[col] = features[col].apply(lambda x: features[col].mean() if x < lower_bound or x > upper_bound else x)

# Normalisation des données (0 à 1)
features = features.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

# Afficher les données après prétraitement
print("Données après prétraitement :")
print(features.head())


# 111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
# NMF : Réduction de dimensions
nmf_model = NMF(n_components=2, init='random', random_state=42)  # Réduit à 2 dimensions
W = nmf_model.fit_transform(features)  # Matrice des caractéristiques latentes
H = nmf_model.components_  # Matrice de base
print("Matrice W (caractéristiques latentes) :")
print(W[:5])

# 222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
# CLUSTER : Combien de groupes ? (1ère méthode : Méthode du coude)
inertias = []
k_range = range(1, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(W)
    inertias.append(kmeans.inertia_)

# Tracer la courbe de la méthode du coude
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertias, marker='o')
plt.title("Méthode du coude")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Inertie")
plt.grid()
plt.show()

# 333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
# CLUSTER : Combien de groupes ? (2ème méthode : Score de silhouette)
silhouette_scores = []
k_range = range(2, 10)  # Score de silhouette commence à k=2
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(W)
    score = silhouette_score(W, clusters)
    silhouette_scores.append(score)
    
# Tracer le score de silhouette
plt.figure(figsize=(8, 6))
plt.plot(k_range, silhouette_scores, marker='o')
plt.title("Score de silhouette")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Score de silhouette")
plt.grid()
plt.show()

# 444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444
# CLUSTER : Application finale
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')  # Choisir 2 clusters
clusters = kmeans.fit_predict(W)  # Clusterisation sur la matrice W

# Ajouter les clusters aux données initiales
data['Cluster'] = clusters
print("Données avec les clusters :")
print(data.head())

# Visualiser les clusters
plt.figure(figsize=(8, 6))
plt.scatter(W[:, 0], W[:, 1], c=clusters, cmap='viridis')
plt.title("Clustering via NNMF")
plt.xlabel("Caractéristique latente 1")
plt.ylabel("Caractéristique latente 2")
plt.colorbar(label='Cluster')
plt.show()
# 555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555
# Convertir les valeurs de 'Gender' en catégories
data['Genre'] = data['Genre'].map({'Male': 0, 'Female': 1})
# Créer une fonction pour calculer le pourcentage d'hommes et de femmes
def calculate_gender_percentage(df):
    male_percentage = (df['Genre'] == 0).sum() / len(df) * 100
    female_percentage = (df['Genre'] == 1).sum() / len(df) * 100
    return pd.Series({'male': male_percentage, 'female': female_percentage})

# Grouper les données par cluster et appliquer la fonction
gender_percentages = data.groupby('Cluster').apply(calculate_gender_percentage)

# Calculer les moyennes des variables pour chaque cluster
cluster_means = data.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()

# Compter le nombre d'éléments dans chaque cluster
cluster_counts = data['Cluster'].value_counts().rename("Nombre d'éléments")

# Réunir les DataFrames (moyennes, counts et pourcentages de genre)
cluster_summary = pd.concat([cluster_means, cluster_counts, gender_percentages], axis=1)

# Afficher la table combinée
print("Tableau des moyennes, des comptes et des pourcentages par cluster :")
print(cluster_summary)