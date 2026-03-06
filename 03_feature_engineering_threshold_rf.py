# ==============================
# 03 - Feature Engineering + Threshold Analysis + Random Forest Comparison
# Save figures to outputs/figures/
# ==============================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score
)

# ------------------------------
# Config
# ------------------------------
DATA_PATH = "Assignment1_mimic dataset (1).csv"
TARGET_COL = "icu_death_flag"
RANDOM_STATE = 42
TEST_SIZE = 0.2
HIGH_MISSING_THRESHOLD = 0.90

FIG_DIR = os.path.join("outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LEAKAGE_KEYWORDS = [
    "hospital_expire_flag",
    "outtime",
    "deathtime",
    "los",
    "discharge",
    "_los",
    "_expire",
]
ID_COLS = ["subject_id", "hadm_id", "stay_id", "intime"]

# ------------------------------
# Utilities
# ------------------------------
def remove_leakage_and_ids(X: pd.DataFrame) -> pd.DataFrame:
    # Remove leakage columns
    leakage_cols = []
    for col in X.columns:
        col_low = col.lower()
        for key in LEAKAGE_KEYWORDS:
            if key.lower() in col_low:
                leakage_cols.append(col)
    leakage_cols = list(set(leakage_cols))
    X = X.drop(columns=leakage_cols, errors="ignore")

    # Remove IDs
    X = X.drop(columns=[c for c in ID_COLS if c in X.columns], errors="ignore")
    return X

def handle_min_max_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace paired *_min and *_max with a single *_range = max - min.
    Drops the original *_min and *_max columns.
    """
    df = df.copy()
    min_cols = [c for c in df.columns if c.endswith("_min")]
    for min_col in min_cols:
        base = min_col[:-4]
        max_col = f"{base}_max"
        if max_col in df.columns:
            df[f"{base}_range"] = df[max_col] - df[min_col]
            df = df.drop(columns=[min_col, max_col])
    return df

def build_preprocessor(X_train: pd.DataFrame):
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor

def evaluate_at_threshold(y_true, y_score, threshold: float):
    y_pred = (y_score >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision_1": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_1": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

# ------------------------------
# Load & split
# ------------------------------
df = pd.read_csv(DATA_PATH)
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found.")

y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

X = remove_leakage_and_ids(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

# Remove high missing features based on TRAIN
missing_ratio_train = X_train.isnull().mean()
high_missing_cols = missing_ratio_train[missing_ratio_train > HIGH_MISSING_THRESHOLD].index.tolist()
X_train = X_train.drop(columns=high_missing_cols)
X_test = X_test.drop(columns=high_missing_cols)

# ------------------------------
# Baseline Logistic (for comparison inside 03)
# ------------------------------
preprocessor_base = build_preprocessor(X_train)
logistic_base = Pipeline([
    ("preprocessing", preprocessor_base),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ))
])
logistic_base.fit(X_train, y_train)
y_score_base = logistic_base.predict_proba(X_test)[:, 1]
auc_base = roc_auc_score(y_test, y_score_base)

print("\n===== 03 - Feature Engineering / Threshold / RF =====")
print("Baseline Logistic Test AUC:", float(auc_base))

# ------------------------------
# Step 1: min/max range engineering
# ------------------------------
X_train_fe = handle_min_max_range(X_train)
X_test_fe = handle_min_max_range(X_test)

preprocessor_fe = build_preprocessor(X_train_fe)

logistic_fe = Pipeline([
    ("preprocessing", preprocessor_fe),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ))
])
logistic_fe.fit(X_train_fe, y_train)
y_score_fe = logistic_fe.predict_proba(X_test_fe)[:, 1]
auc_fe = roc_auc_score(y_test, y_score_fe)
print("After min/max-range engineering Logistic Test AUC:", float(auc_fe))

# ------------------------------
# Step 2: Threshold analysis + ROC/PR curves (saved)
# ------------------------------
# ROC
fpr, tpr, roc_thresholds = roc_curve(y_test, y_score_fe)
plt.figure()
plt.plot(fpr, tpr)
plt.title("ROC Curve (Logistic + min/max-range)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.tight_layout()
roc_path = os.path.join(FIG_DIR, "03_roc_logistic_fe.png")
plt.savefig(roc_path, dpi=200)
plt.close()
print(f"Saved figure: {roc_path}")

# PR
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_score_fe)
plt.figure()
plt.plot(recall, precision)
plt.title("Precision-Recall Curve (Logistic + min/max-range)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
pr_path = os.path.join(FIG_DIR, "03_pr_logistic_fe.png")
plt.savefig(pr_path, dpi=200)
plt.close()
print(f"Saved figure: {pr_path}")

# Threshold strategies
results = []

# Strategy A: default 0.5
results.append(("default_0.5", evaluate_at_threshold(y_test, y_score_fe, 0.5)))

# Strategy B: maximize F1 (search over thresholds from PR curve)
# pr_thresholds has length len(precision)-1; safe iterate
best_f1 = -1.0
best_thr = 0.5
for thr in pr_thresholds:
    metrics = evaluate_at_threshold(y_test, y_score_fe, float(thr))
    if metrics["f1_1"] > best_f1:
        best_f1 = metrics["f1_1"]
        best_thr = float(thr)
results.append(("max_f1", evaluate_at_threshold(y_test, y_score_fe, best_thr)))

# Strategy C: Recall >= 0.80, maximize Precision
target_recall = 0.80
best_prec = -1.0
best_thr_r = 0.5
for thr in pr_thresholds:
    metrics = evaluate_at_threshold(y_test, y_score_fe, float(thr))
    if metrics["recall_1"] >= target_recall and metrics["precision_1"] > best_prec:
        best_prec = metrics["precision_1"]
        best_thr_r = float(thr)
results.append((f"recall>={target_recall}_max_precision", evaluate_at_threshold(y_test, y_score_fe, best_thr_r)))

print("\n===== Threshold Strategy Summary (Logistic + FE) =====")
for name, m in results:
    print(f"\n[{name}] threshold={m['threshold']:.4f}  precision_1={m['precision_1']:.4f}  recall_1={m['recall_1']:.4f}  f1_1={m['f1_1']:.4f}")
    print("Confusion Matrix:\n", m["confusion_matrix"])

# ------------------------------
# Step 3: Random Forest comparison (same FE data & same preprocessor)
# ------------------------------
rf = Pipeline([
    ("preprocessing", preprocessor_fe),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1
    ))
])
rf.fit(X_train_fe, y_train)
y_score_rf = rf.predict_proba(X_test_fe)[:, 1]
auc_rf = roc_auc_score(y_test, y_score_rf)

print("\n===== AUC Comparison (03) =====")
print("Baseline Logistic:", float(auc_base))
print("Logistic + min/max-range:", float(auc_fe))
print("Random Forest + min/max-range:", float(auc_rf))

print("\n===== END (03) =====")