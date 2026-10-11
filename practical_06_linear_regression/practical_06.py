import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

rng = np.random.default_rng(42)
n = 300

area = rng.uniform(500, 3000, n)
bedrooms = rng.integers(1, 6, n)
age = rng.uniform(0, 30, n)
price = 18 + 0.08 * area + 8 * bedrooms - 0.7 * age + rng.normal(0, 18, n)

X = np.column_stack([area, bedrooms, age])
y = price

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print(f"MAE  : {mae:.3f}")
print(f"MSE  : {mse:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")

plt.figure(figsize=(7, 5))
plt.scatter(y_test, pred, alpha=0.65)
lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
plt.plot(lims, lims, linestyle="--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression: Actual vs Predicted")
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png", dpi=160)
plt.show()
