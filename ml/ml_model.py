import os
import sys
import io

# Set UTF-8 encoding for Windows terminals
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import joblib
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedGroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "csi_features.csv")
ML_DIR = os.path.join(BASE_DIR, "ml")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")

os.makedirs(ML_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

LABEL_NAMES = {
    0: "0 Persons (Empty Room)",
    1: "1 Person Detected",
    2: "2 Persons Detected",
    3: "3 Persons Detected",
    4: "4 Persons Detected",
    5: "5 Persons Detected"
}

def train_and_evaluate():
    print("=" * 65)
    print("  [+] Training Multi-Person Wi-Fi CSI AI Model (0 to 5 Persons)")
    print("=" * 65)

    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"[*] Loaded dataset: {df.shape[0]} samples across {df.shape[1]-1} features.")
    print(f"[*] Class distribution:\n{df['label'].value_counts().sort_index()}")

    feature_cols = [c for c in df.columns if c not in ("label", "capture_id")]
    X = df[feature_cols].values
    y = df["label"].values.astype(int)

    # 2. Train/Test Split
    capture_ids = df.get("capture_id")
    if capture_ids is not None and df.groupby("label")["capture_id"].nunique().min() >= 2:
        # Group split by capture session
        held_out = {
            label: sorted(group["capture_id"].unique())[-1]
            for label, group in df.groupby("label")
        }
        test_mask = np.array([capture == held_out[label] for capture, label in zip(capture_ids, y)])
        X_train, X_test = X[~test_mask], X[test_mask]
        y_train, y_test = y[~test_mask], y[test_mask]
        evaluation_note = "Held-out capture session per class (generalized validation)."
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        evaluation_note = "Stratified 80/20 train/test split."

    # 3. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Model Training
    print("[*] Training RandomForest Classifier (n_estimators=150, max_depth=14)...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        min_samples_split=3,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # 5. Cross-Validation
    cv = StratifiedGroupKFold(n_splits=3) if capture_ids is not None and df.groupby("label")["capture_id"].nunique().min() >= 3 else 5
    if isinstance(cv, int):
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")
    else:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, groups=capture_ids[~test_mask], cv=cv, scoring="accuracy")
    print(f"[+] 5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

    # 6. Test Evaluation
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print("\n" + "=" * 55)
    print(f"  [RESULT] MULTI-PERSON TEST ACCURACY: {acc*100:.2f}%")
    print(f"  [RESULT] MACRO F1 SCORE:             {f1*100:.2f}%")
    print("=" * 55)

    target_names = [LABEL_NAMES[i] for i in sorted(np.unique(y))]
    print("\n[+] Detailed Multi-Person Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 7. Generate Evaluation Figures
    # A. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f"{i}P" for i in range(len(target_names))],
                yticklabels=[f"{i}P" for i in range(len(target_names))],
                cbar=False)
    plt.title(f'Multi-Person Confusion Matrix (Acc = {acc*100:.1f}%)', fontsize=12, pad=12)
    plt.xlabel('Predicted People Count', fontsize=10)
    plt.ylabel('Actual People Count', fontsize=10)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # B. Feature Importance
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:15]
    top_feats = [feature_cols[i] for i in top_indices]
    top_scores = importances[top_indices]

    plt.figure(figsize=(9, 5))
    plt.barh(top_feats[::-1], top_scores[::-1], color='#00E5FF', edgecolor='#061B33')
    plt.title('Top 15 Most Discriminative Multi-Person Features', fontsize=12, pad=12)
    plt.xlabel('Importance Score', fontsize=10)
    plt.tight_layout()
    fi_path = os.path.join(RESULTS_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    plt.close()

    # 8. Save Artifacts for ML & Dashboard
    model_path = os.path.join(ML_DIR, "csi_model.pkl")
    scaler_path = os.path.join(ML_DIR, "scaler.pkl")
    cols_path = os.path.join(ML_DIR, "feature_cols.pkl")
    meta_path = os.path.join(RESULTS_DIR, "metadata.json")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, cols_path)

    metadata = {
        "model_type": type(model).__name__,
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "test_accuracy": round(float(acc), 4),
        "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
        "cv_std_accuracy": round(float(cv_scores.std()), 4),
        "evaluation_note": evaluation_note,
        "label_mapping": LABEL_NAMES
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 9. Sync Artifacts to Dashboard Directory for 100% Zero-Error Loading
    shutil.copy(model_path, os.path.join(DASHBOARD_DIR, "csi_model.pkl"))
    shutil.copy(scaler_path, os.path.join(DASHBOARD_DIR, "scaler.pkl"))
    shutil.copy(cols_path, os.path.join(DASHBOARD_DIR, "feature_cols.pkl"))

    print("\n[SUCCESS] Multi-person model, scaler, and features saved and synced!")
    print(f"  -> Model file:  {model_path}")
    print(f"  -> Scaler file: {scaler_path}")
    print(f"  -> Sync to dashboard complete!")

if __name__ == "__main__":
    train_and_evaluate()
