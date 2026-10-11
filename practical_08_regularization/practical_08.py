import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(42)
X = np.linspace(-3, 3, 120).reshape(-1, 1)
y = np.sin(X[:, 0]) + rng.normal(0, 0.25, len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

degree = 12

models = {
    "Linear Regression": Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("scale", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Ridge": Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=10.0))
    ]),
    "Lasso": Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("scale", StandardScaler()),
        ("model", Lasso(alpha=0.01, max_iter=20000))
    ])
}

scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    scores[name] = mean_squared_error(y_test, pred)

print("\nTest MSE:")
for name, score in scores.items():
    print(f"{name:20s}: {score:.4f}")

grid = np.linspace(-3, 3, 500).reshape(-1, 1)
plt.figure(figsize=(9, 5))
plt.scatter(X_train, y_train, alpha=0.45, label="Training data")
plt.scatter(X_test, y_test, alpha=0.45, label="Test data")
for name, model in models.items():
    plt.plot(grid, model.predict(grid), linewidth=2, label=name)

plt.xlabel("X")
plt.ylabel("Target")
plt.title("Regularization and Overfitting")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/regularization_comparison.png", dpi=160)
plt.show()
