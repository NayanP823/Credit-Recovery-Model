# train_save_rf_pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, recall_score,
    precision_score, f1_score, balanced_accuracy_score
)
import joblib

# === CONFIG ===
FILE_PATH = "default of credit card clients.xls"   # same as your original
TARGET_COL = 'default payment next month'
MODEL_FILENAME = "risk_model.joblib"               # same filename so other code won't break
RANDOM_STATE = 42

# === 1) Load & basic cleaning ===
print(f"Loading data from {FILE_PATH}...")
df = pd.read_excel(FILE_PATH, header=1)
df.columns = df.columns.str.strip()

# Drop ID if present
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

# Define X, y (keep same features as before: all columns except target)
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Force numeric and handle bad imports (coerce strings -> NaN)
X = X.apply(pd.to_numeric, errors='coerce')

# Impute numeric NaNs with column median (safe for single-run)
X = X.fillna(X.median())

# Ensure target is numeric and binary 0/1
y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)
unique_y = sorted(y.unique())
print("Target unique values after cleaning:", unique_y)

# If target is not strictly 0/1, binarize (>0 -> 1)
if not set(unique_y).issubset({0, 1}):
    y = (y > 0).astype(int)
    print("Binarized target to 0/1")

# === 2) Train/test split (stratified) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# === 3) Build pipeline (scaler + RandomForest) ===
# We include scaler to keep API identical to your logistic pipeline
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features='sqrt',
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

# === 4) Train ===
print("Training Random Forest pipeline...")
rf_pipeline.fit(X_train, y_train)
print("Training completed.")

# === 5) Predict & Metrics (requested: accuracy + class metrics) ===
y_pred = rf_pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)
cm = confusion_matrix(y_test, y_pred)
recall_def = recall_score(y_test, y_pred, pos_label=1)
precision_def = precision_score(y_test, y_pred, pos_label=1)
f1_def = f1_score(y_test, y_pred, pos_label=1)
balanced_acc = balanced_accuracy_score(y_test, y_pred)

print("\n=== Evaluation (test set) ===")
print(f"Accuracy: {acc:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print("\nClassification Report:")
print(report)
print("Confusion Matrix:")
print(cm)
print(f"Default-class (1) Recall: {recall_def:.4f}")
print(f"Default-class (1) Precision: {precision_def:.4f}")
print(f"Default-class (1) F1: {f1_def:.4f}")

# Print the exact metrics you requested to compare with previous run:
print("\n=== Summary (for comparison) ===")
print(f"Accuracy (now): {acc:.4f}  (previous ~0.81 expected)")
print(f"Default recall (now): {recall_def:.4f}  (previous was ~0.24 — should be improved)")
print(f"Default precision (now): {precision_def:.4f}")

# === 6) Save pipeline (same file name to avoid breaking existing code) ===
joblib.dump(rf_pipeline, MODEL_FILENAME)
print(f"\nSaved Random Forest pipeline to: {MODEL_FILENAME}")

# === Optional: show feature importances (top 10) ===
try:
    importances = rf_pipeline.named_steps['model'].feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(10)
    print("\nTop 10 feature importances:")
    print(feat_imp)
except Exception:
    pass