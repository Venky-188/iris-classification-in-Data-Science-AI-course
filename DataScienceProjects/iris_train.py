import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# 1. Load dataset
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

# map target to names for easier analysis
target_map = dict(enumerate(iris.target_names))
y_named = y.map(target_map)

# 2. Quick exploration
print("Dataset shape:", X.shape)
print("Class distribution:\n", y_named.value_counts())

# pairplot (visualize)
sns.pairplot(pd.concat([X, y_named.rename('species')], axis=1), hue='species')
plt.suptitle("Iris Pairplot", y=1.02)
plt.tight_layout()
plt.savefig("iris_pairplot.png")
plt.close()

# 3. Preprocess & split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# We'll standardize features for models that need it
scaler = StandardScaler()

# 4. Define models (pipelines)
models = {
    "LogisticRegression": Pipeline([("scaler", scaler), ("clf", LogisticRegression(max_iter=200))]),
    "KNN": Pipeline([("scaler", scaler), ("clf", KNeighborsClassifier())]),
    "DecisionTree": Pipeline([("clf", DecisionTreeClassifier(random_state=42))]),
    "RandomForest": Pipeline([("clf", RandomForestClassifier(random_state=42))])
}

# 5. Train & evaluate
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n--- {name} ---")
    print("Accuracy:", acc)
    print("Classification report:\n", classification_report(y_test, preds, target_names=iris.target_names))
    cm = confusion_matrix(y_test, preds)
    print("Confusion matrix:\n", cm)
    results.append((name, acc, model))

# 6. Choose best model by accuracy on test set
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
best_name, best_acc, best_model = results_sorted[0]
print(f"\nBest model: {best_name} with accuracy {best_acc:.4f}")

# Save the best model
os.makedirs("models", exist_ok=True)
joblib.dump({"model": best_model, "target_map": target_map, "feature_names": list(X.columns)}, "models/iris_best.joblib")
print("Saved best model to models/iris_best.joblib")

# Also save a confusion matrix figure for the best model
import seaborn as sns
import matplotlib.pyplot as plt

best_preds = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix ({best_name})")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()