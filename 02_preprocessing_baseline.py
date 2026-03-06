# ==============================
# 02 - Preprocessing Strategy + Baseline Logistic
# ==============================

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# ------------------------------
# Config
# ------------------------------
DATA_PATH = "Assignment1_mimic dataset (1).csv"
TARGET_COL = "icu_death_flag"
RANDOM_STATE = 42
TEST_SIZE = 0.2
HIGH_MISSING_THRESHOLD = 0.90

# Leakage removal rules (case-insensitive substring match)
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

print("\n===== PREPROCESSING + BASELINE (02) =====")

# ------------------------------
# Load
# ------------------------------
df = pd.read_csv(DATA_PATH)
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

# ------------------------------
# Remove leakage columns
# ------------------------------
leakage_cols = []
for col in X.columns:
    col_low = col.lower()
    for key in LEAKAGE_KEYWORDS:
        if key.lower() in col_low:
            leakage_cols.append(col)
leakage_cols = sorted(set(leakage_cols))

print("\nRemoved leakage-risk columns:")
print(leakage_cols if leakage_cols else "(none found)")

X = X.drop(columns=leakage_cols, errors="ignore")

# ------------------------------
# Remove ID columns
# ------------------------------
X = X.drop(columns=[c for c in ID_COLS if c in X.columns], errors="ignore")

# ------------------------------
# Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

# ------------------------------
# Remove high missing features based on TRAIN only
# ------------------------------
missing_ratio_train = X_train.isnull().mean()
high_missing_cols = missing_ratio_train[missing_ratio_train > HIGH_MISSING_THRESHOLD].index.tolist()

print(f"\nRemoved {len(high_missing_cols)} high-missing features (>{int(HIGH_MISSING_THRESHOLD*100)}%)")

X_train = X_train.drop(columns=high_missing_cols)
X_test = X_test.drop(columns=high_missing_cols)

# ------------------------------
# Feature typing
# ------------------------------
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

print(f"Numeric features after cleaning: {len(numeric_features)}")
print(f"Categorical features after cleaning: {len(categorical_features)}")

# ------------------------------
# Preprocessing pipelines
# ------------------------------
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

# ------------------------------
# Baseline Logistic Regression
# ------------------------------
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ))
])

# ------------------------------
# 5-fold CV AUC on FULL (leak-free) X,y
# NOTE: This uses the same cleaning rules already applied to X.
# ------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

print("\n===== 5-Fold CV AUC (Baseline Logistic) =====")
print("Fold AUC scores:", cv_auc)
print("Mean CV AUC:", float(np.mean(cv_auc)))
print("Std  CV AUC:", float(np.std(cv_auc)))

# ------------------------------
# Fit on train, evaluate on test
# ------------------------------
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

print("\n===== Test Set Results (Baseline Logistic) =====")
print("Test AUC:", float(roc_auc_score(y_test, y_proba)))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n===== END (02) =====")