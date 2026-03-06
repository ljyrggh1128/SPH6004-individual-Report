# ==============================
# 05 - Model Comparison (FAST/STABLE)
# Version = Plan A: REMOVE LASSO selector in 05 to avoid long runtime
# Compare models on two feature sets:
#   - baseline
#   - corr_filter
# Outputs:
#   outputs/tables/05_model_comparison_fast.csv
#   outputs/figures/05_FAST_ROC_<featureset>.png
#   outputs/figures/05_FAST_PR_<featureset>.png
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

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# ------------------------------
# Config
# ------------------------------
DATA_PATH = "Assignment1_mimic dataset (1).csv"
TARGET_COL = "icu_death_flag"
RANDOM_STATE = 42
TEST_SIZE = 0.2

HIGH_MISSING_THRESHOLD = 0.90
CORR_THRESHOLD = 0.90

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
    Low-memory preprocessing:
    - numeric: median + StandardScaler(with_mean=False)
    - categorical: most_frequent + OneHotEncoder sparse float32
    """
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first",
                              sparse_output=True, dtype=np.float32)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        sparse_threshold=1.0
    )

def eval_metrics(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    return {
        "AUC": float(roc_auc_score(y_true, y_score)),
        "PR_AUC": float(average_precision_score(y_true, y_score)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision_1": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall_1": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1_1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

def plot_roc_pr(y_true, model_scores_dict, title_suffix, fig_dir):
    # ROC
    plt.figure()
    for name, score in model_scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, score)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC Curves ({title_suffix})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    roc_path = os.path.join(fig_dir, f"05_FAST_ROC_{title_suffix}.png")
    plt.savefig(roc_path, dpi=200)
    plt.close()

    # PR
    plt.figure()
    for name, score in model_scores_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, score)
        plt.plot(rec, prec, label=name)
    plt.title(f"Precision-Recall Curves ({title_suffix})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    pr_path = os.path.join(fig_dir, f"05_FAST_PR_{title_suffix}.png")
    plt.savefig(pr_path, dpi=200)
    plt.close()

    return roc_path, pr_path

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

print("\n===== 05 MODEL COMPARISON (FAST/STABLE) =====")
print("Removed leakage columns:", leakage_cols)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Remove >90% missing based on train only
missing_ratio_train = X_train.isnull().mean()
high_missing_cols = missing_ratio_train[missing_ratio_train > HIGH_MISSING_THRESHOLD].index.tolist()
X_train = X_train.drop(columns=high_missing_cols)
X_test = X_test.drop(columns=high_missing_cols)

print(f"Removed {len(high_missing_cols)} high-missing features (>90%).")

# ------------------------------
# Feature sets: baseline + corr_filter only (Plan A)
# ------------------------------
feature_sets = {}
feature_sets["baseline"] = (X_train.copy(), X_test.copy())

drop_corr_cols = correlation_filter_train_only(X_train, threshold=CORR_THRESHOLD)
X_train_corr = X_train.drop(columns=drop_corr_cols, errors="ignore")
X_test_corr = X_test.drop(columns=drop_corr_cols, errors="ignore")
feature_sets["corr_filter"] = (X_train_corr, X_test_corr)

print(f"CorrFilter dropped numeric features: {len(drop_corr_cols)}")

# ------------------------------
# Models
# ------------------------------
models = {
    "Logistic": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
    "Ridge(Logistic L2)": LogisticRegression(max_iter=3000, penalty="l2", solver="lbfgs",
                                            class_weight="balanced", random_state=RANDOM_STATE),
    "SVM(RBF)": SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    "DecisionTree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                           n_jobs=1, random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostClassifier(n_estimators=300, learning_rate=0.5, random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

# ------------------------------
# Run comparisons
# ------------------------------
rows = []

for fs_name, (Xtr, Xte) in feature_sets.items():
    print(f"\n--- Feature set: {fs_name} ---")

    pre = build_preprocessor_lowmem(Xtr)
    scores_for_plots = {}

    for model_name, clf in models.items():
        pipe = Pipeline([
            ("preprocessing", pre),
            ("classifier", clf)
        ])

        pipe.fit(Xtr, y_train)
        y_score = pipe.predict_proba(Xte)[:, 1]
        scores_for_plots[model_name] = y_score

        m = eval_metrics(y_test, y_score, threshold=0.5)
        m.update({"feature_set": fs_name, "model": model_name})
        rows.append(m)

        print(f"{model_name}: AUC={m['AUC']:.4f}, PR_AUC={m['PR_AUC']:.4f}, F1_1={m['F1_1']:.4f}")

    roc_path, pr_path = plot_roc_pr(y_test, scores_for_plots, fs_name, FIG_DIR)
    print(f"Saved ROC: {roc_path}")
    print(f"Saved PR:  {pr_path}")

# Save table
result_df = pd.DataFrame(rows)
out_path = os.path.join(TABLE_DIR, "05_model_comparison_fast.csv")
result_df.to_csv(out_path, index=False)

print("\n===== DONE =====")
print(f"Saved results table: {out_path}")