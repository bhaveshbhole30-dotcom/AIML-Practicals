import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

rng = np.random.default_rng(42)
X = np.linspace(-3, 3, 250).reshape(-1, 1)
y = np.sin(X[:, 0]) + 0.15 * X[:, 0] ** 2 + rng.normal(0, 0.08, len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = make_pipeline(
    StandardScaler(),
    SVR(kernel="rbf", C=10, epsilon=0.08, gamma="scale")
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

mse = mean_squared_error(y_test, pred)
print(f"MAE  : {mean_absolute_error(y_test, pred):.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {np.sqrt(mse):.4f}")
print(f"R²   : {r2_score(y_test, pred):.4f}")

order = np.argsort(X_test[:, 0])
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, alpha=0.5, label="Actual")
plt.plot(X_test[order], pred[order], linewidth=2, label="SVR prediction")
plt.xlabel("X")
plt.ylabel("Target")
plt.title("Support Vector Regression (RBF)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/svr_regression.png", dpi=160)
plt.show()
