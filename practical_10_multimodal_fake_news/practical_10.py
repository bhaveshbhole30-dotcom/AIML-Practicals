from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "data" / "multimodal_news.csv")

def image_features(path):
    img = Image.open(path).convert("L").resize((64, 64))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    hist, _ = np.histogram(arr, bins=16, range=(0, 1), density=True)
    dx = np.abs(np.diff(arr, axis=1)).mean()
    dy = np.abs(np.diff(arr, axis=0)).mean()
    return np.concatenate([hist, [dx, dy]])

vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
X_text = vectorizer.fit_transform(df["text"]).toarray()

X_image = np.vstack([
    image_features(BASE / "data" / name)
    for name in df["image_path"]
])

X_image = StandardScaler().fit_transform(X_image)
X = np.hstack([X_text, X_image])
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=500, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("\nClassification Report:\n")
print(classification_report(y_test, pred))

cm = confusion_matrix(y_test, pred, labels=["fake", "real"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=["fake", "real"], yticklabels=["fake", "real"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Multimodal Fake News Classifier")
plt.tight_layout()
plt.savefig(BASE / "outputs" / "confusion_matrix.png", dpi=160)
plt.show()

print("\nFeatures used:")
print(f"Text TF-IDF features : {X_text.shape[1]}")
print(f"Image features       : {X_image.shape[1]}")
print(f"Combined features    : {X.shape[1]}")
