import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance retained:", round(pca.explained_variance_ratio_.sum() * 100, 2), "%")

plt.figure(figsize=(8, 5))
for cls, name in enumerate(iris.target_names):
    mask = iris.target == cls
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=name, s=35)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: Iris Dataset in Two Dimensions")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/pca_2d.png", dpi=160)
plt.show()
