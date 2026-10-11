import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
n = 150
hours = np.random.uniform(1, 10, n)
attendance = np.clip(np.random.normal(75, 10, n), 40, 100)
sleep = np.clip(np.random.normal(7, 1, n), 4, 10)
marks = np.clip(35 + 5.2 * hours + 0.25 * attendance + 2.5 * sleep + np.random.normal(0, 8, n), 0, 100)

df = pd.DataFrame({
    "Study_Hours": hours,
    "Attendance": attendance,
    "Sleep_Hours": sleep,
    "Final_Marks": marks
})

corr = df.corr(numeric_only=True)
print("\nCorrelation Matrix:\n")
print(corr.round(3))

target = "Final_Marks"
strength = corr[target].drop(target).abs().sort_values(ascending=False)
print(f"\nStrongest association with {target}: {strength.index[0]} "
      f"(r = {corr.loc[strength.index[0], target]:.3f})")

plt.figure(figsize=(8, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=160)
plt.show()

best = strength.index[0]
plt.figure(figsize=(7, 5))
sns.regplot(data=df, x=best, y=target, scatter_kws={"alpha": 0.55})
plt.title(f"{best} vs {target}")
plt.tight_layout()
plt.savefig("outputs/association_plot.png", dpi=160)
plt.show()
