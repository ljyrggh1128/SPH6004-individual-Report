# ==============================
# 06 - Threshold Analysis for Final Model
# Final config:
#   Feature set: corr_filter
#   Model: GradientBoostingClassifier
# Outputs:
#   outputs/tables/06_threshold_analysis.csv
#   outputs/figures/06_ROC_final_model.png
#   outputs/figures/06_PR_final_model.png
# ==============================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# ------------------------------
# Config
# ------------------------------
DATA_PATH = "Assignment1_mimic dataset (1).csv"
TARGET_COL = "icu_death_flag"
RANDOM_STATE = 42
TEST_SIZE = 0.2

HIGH_MISSING_THRESHOLD = 0.90
CORR_THRESHOLD = 0.90

RECALL_CONSTRAINT = 0.80  # for "Recall>=0.80, maximize Precision" rule

OUT_DIR = "outputs"
TABLE_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

LEAKAGE_KEYWORDS = [
    "hospital_expire_flag", "outtime", "deathtime", "los",
    "discharge", "_los", "_expire"
]
ID_COLS = ["subject_id", "hadm_id", "stay_id", "intime"]

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------
# Helpers
# ------------------------------
def remove_leakage_and_ids(X: pd.DataFrame):
    leakage_cols = []
    for col in X.columns:
        cl = col.lower()
        for key in LEAKAGE_KEYWORDS:
            if key.lower() in cl:
                leakage_cols.append(col)
    leakage_cols = sorted(set(leakage_cols))
    X = X.drop(columns=leakage_cols, errors="ignore")
    X = X.drop(columns=[c for c in ID_COLS if c in X.columns], errors="ignore")
    return X, leakage_cols


def compress_object_to_category(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = X[c].astype("category")
    return X


def correlation_filter_train_only(X_train: pd.DataFrame, threshold=0.90):
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    imputer = SimpleImputer(strategy="median")
    X_num = pd.DataFrame(imputer.fit_transform(X_train[numeric_cols]),
                         columns=numeric_cols, index=X_train.index)
    corr = X_num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [c for c in upper.columns if any(upper[c] > threshold)]
    return drop_cols


def build_preprocessor_lowmem(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Sparse-friendly preprocessing:
    - numeric: median + StandardScaler(with_mean=False)
    - categorical: most_frequent + OneHotEncoder sparse float32
    """
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first",
                              sparse_output=True, dtype=np.float32))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols)
        ],
        sparse_threshold=1.0
    )


def metrics_at_threshold(y_true, y_score, thr: float):
    y_pred = (y_score >= thr).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": float(thr),
        "precision_1": float(p),
        "recall_1": float(r),
        "f1_1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "pred_pos_rate": float(y_pred.mean()),
    }


# ------------------------------
# Load & clean
# ------------------------------
df = pd.read_csv(DATA_PATH)
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found.")

y = df[TARGET_COL]
X_raw = df.drop(columns=[TARGET_COL])

X, leakage_cols = remove_leakage_and_ids(X_raw)
X = compress_object_to_category(X)

print("\n===== 06 THRESHOLD ANALYSIS (FINAL MODEL) =====")
print("Removed leakage columns:", leakage_cols)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Remove >90% missing (train-only)
missing_ratio_train = X_train.isnull().mean()
high_missing_cols = missing_ratio_train[missing_ratio_train > HIGH_MISSING_THRESHOLD].index.tolist()
X_train = X_train.drop(columns=high_missing_cols)
X_test = X_test.drop(columns=high_missing_cols)

print(f"Removed {len(high_missing_cols)} high-missing features (>90%).")

# Corr filter (train-only)
drop_corr_cols = correlation_filter_train_only(X_train, threshold=CORR_THRESHOLD)
X_train = X_train.drop(columns=drop_corr_cols, errors="ignore")
X_test = X_test.drop(columns=drop_corr_cols, errors="ignore")
print(f"CorrFilter dropped numeric features: {len(drop_corr_cols)}")

# ------------------------------
# Build final model pipeline
# ------------------------------
pre = build_preprocessor_lowmem(X_train)

gb = GradientBoostingClassifier(random_state=RANDOM_STATE)

model = Pipeline([
    ("preprocessing", pre),
    ("classifier", gb)
])

# Fit
model.fit(X_train, y_train)

# Predict scores
y_score = model.predict_proba(X_test)[:, 1]

# Overall threshold-free metrics
test_auc = roc_auc_score(y_test, y_score)
test_pr_auc = average_precision_score(y_test, y_score)

print("\nThreshold-free performance:")
print(f"Test ROC-AUC: {test_auc:.4f}")
print(f"Test PR-AUC:  {test_pr_auc:.4f}")

# ------------------------------
# Save ROC / PR curves
# ------------------------------
fpr, tpr, _ = roc_curve(y_test, y_score)
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve (Final Model: CorrFilter + GradientBoosting)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.tight_layout()
roc_path = os.path.join(FIG_DIR, "06_ROC_final_model.png")
plt.savefig(roc_path, dpi=200)
plt.close()

prec, rec, thr_pr = precision_recall_curve(y_test, y_score)
plt.figure()
plt.plot(rec, prec)
plt.title("Precision-Recall Curve (Final Model: CorrFilter + GradientBoosting)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
pr_path = os.path.join(FIG_DIR, "06_PR_final_model.png")
plt.savefig(pr_path, dpi=200)
plt.close()

print(f"\nSaved ROC figure: {roc_path}")
print(f"Saved PR  figure: {pr_path}")

# ------------------------------
# Threshold scan table
# ------------------------------
# We scan thresholds based on PR curve thresholds (more meaningful than 0..1 uniform grid)
thresholds = np.unique(thr_pr)
# add default 0.5 explicitly
thresholds = np.unique(np.concatenate([thresholds, np.array([0.5])]))

rows = [metrics_at_threshold(y_test, y_score, float(t)) for t in thresholds]
thr_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

out_table = os.path.join(TABLE_DIR, "06_threshold_analysis.csv")
thr_df.to_csv(out_table, index=False)
print(f"\nSaved threshold table: {out_table}")

# ------------------------------
# Recommend thresholds by strategies
# ------------------------------
# Strategy 1: default 0.5
default_res = metrics_at_threshold(y_test, y_score, 0.5)

# Strategy 2: maximize F1
best_f1_idx = thr_df["f1_1"].idxmax()
best_f1_res = thr_df.loc[best_f1_idx].to_dict()

# Strategy 3: recall >= constraint, maximize precision
eligible = thr_df[thr_df["recall_1"] >= RECALL_CONSTRAINT].copy()
if eligible.shape[0] > 0:
    best_prec_idx = eligible["precision_1"].idxmax()
    best_prec_res = thr_df.loc[best_prec_idx].to_dict()
else:
    # fallback: if impossible, use best F1
    best_prec_res = best_f1_res

print("\n===== Recommended Thresholds =====")
print("[Default 0.5] ", default_res)
print("[Max F1]      ", best_f1_res)
print(f"[Recall>={RECALL_CONSTRAINT} Max Precision] ", best_prec_res)

print("\n===== END (06) =====")