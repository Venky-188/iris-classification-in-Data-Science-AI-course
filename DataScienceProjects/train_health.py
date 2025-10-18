import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import shap
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Synthetic dataset generation (for demo)
# Features: age, sex (0/1), bmi, systolic_bp, diastolic_bp, cholesterol, glucose, smoking (0/1), physical_activity (0-10), family_history (0/1)
def generate_synthetic_data(n=5000, seed=42):
    np.random.seed(seed)
    age = np.random.randint(18, 85, size=n)
    sex = np.random.binomial(1, 0.48, size=n)
    bmi = np.round(np.random.normal(27, 5, n), 1)
    systolic = np.round(np.random.normal(130, 20, n))
    diastolic = np.round(np.random.normal(80, 10, n))
    cholesterol = np.round(np.random.normal(200, 40, n))
    glucose = np.round(np.random.normal(100, 30, n))
    smoking = np.random.binomial(1, 0.2, size=n)
    phys_act = np.clip(np.random.normal(4, 2, n), 0, 10)
    family_hist = np.random.binomial(1, 0.25, size=n)

    # Simple risk function (not real medicine)
    risk_score = (
        0.03*(age-40) + 0.1*(bmi-25) + 0.02*(systolic-120) + 0.015*(cholesterol-180)
        + 0.02*(glucose-90) + 0.5*smoking - 0.05*phys_act + 0.7*family_hist + 0.2*sex
    )
    # Convert to probability with sigmoid
    prob = 1 / (1 + np.exp(-risk_score))
    # Binary target: disease within 1 year (synthetic)
    y = np.random.binomial(1, prob)

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "systolic_bp": systolic,
        "diastolic_bp": diastolic,
        "cholesterol": cholesterol,
        "glucose": glucose,
        "smoking": smoking,
        "physical_activity": phys_act,
        "family_history": family_hist,
        "target": y
    })
    return df

df = generate_synthetic_data(4000)
print("Generated dataset shape:", df.shape)
print(df['target'].value_counts(normalize=True))

# Optional: save dataset
os.makedirs("data", exist_ok=True)
df.to_csv("data/synthetic_health.csv", index=False)

# 2. Split
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 3. Train model (RandomForest)
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 4. Evaluate
preds = rf.predict(X_test)
probs = rf.predict_proba(X_test)[:,1]
acc = accuracy_score(y_test, preds)
auc = roc_auc_score(y_test, probs)
print("Accuracy:", acc)
print("ROC AUC:", auc)
print("Classification Report:\n", classification_report(y_test, preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

# 5. Feature importance plot
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Feature importances (RandomForest)")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.close()

# 6. SHAP explainer (TreeExplainer)
explainer = shap.TreeExplainer(rf)
# I/O note: shap has heavy plotting; we'll save a global importance plot
shap_values = explainer.shap_values(X_train, check_additivity=False)  # shap_values is list for classification; shap_values[1] for class 1
# summary plot saved as image
plt.figure()
shap.summary_plot(shap_values, X_train, show=False)
plt.savefig("shap_summary.png", bbox_inches='tight')
plt.close()

# 7. Save model + explainer metadata
os.makedirs("models", exist_ok=True)
joblib.dump({
    "model": rf,
    "explainer": explainer,
    "feature_names": list(X.columns)
}, "models/health_rf.joblib")
print("Saved model and explainer to models/health_rf.joblib")