import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

iris = load_iris(as_frame=True)
X = iris.data

X_scaled = StandardScaler().fit_transform(X)

inertias = []
ks = range(1, 8)
for k in ks:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(list(ks), inertias, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/elbow_method.png", dpi=160)
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

result = X.copy()
result["Cluster"] = labels
print("\nCluster counts:")
print(result["Cluster"].value_counts().sort_index())

plt.figure(figsize=(8, 5))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, s=35)
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.title("K-Means Clusters")
plt.tight_layout()
plt.savefig("outputs/kmeans_clusters.png", dpi=160)
plt.show()
