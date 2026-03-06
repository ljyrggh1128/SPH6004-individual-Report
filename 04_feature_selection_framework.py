# ==============================
# 04 - Feature Selection Framework (Final, Most Stable)
# Baseline: outer 5-fold CV + test
# CorrFilter: test only (no outer CV to avoid memory failures)
# Ridge/LASSO/ENet: low-memory matrix + internal CV (LogisticRegressionCV)
# ==============================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

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

    pairs = []
    for c in upper.columns:
        high = upper[c][upper[c] > threshold]
        for r, v in high.items():
            pairs.append({"feature_1": r, "feature_2": c, "abs_corr": float(v)})
    pairs_df = pd.DataFrame(pairs).sort_values("abs_corr", ascending=False).reset_index(drop=True)
    return drop_cols, pairs_df

def build_preprocessor_standard(Xtr: pd.DataFrame):
    num_cols = Xtr.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = Xtr.select_dtypes(include=["object","category"]).columns.tolist()

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first"))])

    return ColumnTransformer([("num", num_pipe, num_cols),
                              ("cat", cat_pipe, cat_cols)])

def build_preprocessor_lowmem(Xtr: pd.DataFrame):
    num_cols = Xtr.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = Xtr.select_dtypes(include=["object","category"]).columns.tolist()

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler(with_mean=False))])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first",
                                               sparse_output=True, dtype=np.float32))])

    return ColumnTransformer([("num", num_pipe, num_cols),
                              ("cat", cat_pipe, cat_cols)],
                             sparse_threshold=1.0)

def eval_scores(y_true, y_score):
    return float(roc_auc_score(y_true, y_score)), float(average_precision_score(y_true, y_score))

def internal_cv_auc_from_clf(clf: LogisticRegressionCV) -> float:
    key = sorted(clf.scores_.keys())[-1]
    s = clf.scores_[key]
    return float(s.mean(axis=0).max())

# ------------------------------
# Load & split
# ------------------------------
df = pd.read_csv(DATA_PATH)
y = df[TARGET_COL]
X_raw = df.drop(columns=[TARGET_COL])

X, leakage_cols = remove_leakage_and_ids(X_raw)
X = compress_object_to_category(X)

print("\n===== 04 FEATURE SELECTION FRAMEWORK (FINAL) =====")
print("Removed leakage columns:", leakage_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# >90% missing filter (train-only)
missing_ratio_train = X_train.isnull().mean()
high_missing_cols = missing_ratio_train[missing_ratio_train > HIGH_MISSING_THRESHOLD].index.tolist()
X_train = X_train.drop(columns=high_missing_cols)
X_test = X_test.drop(columns=high_missing_cols)
print(f"Removed {len(high_missing_cols)} high-missing features (>90%).")

raw_cols_after_basic = X_train.shape[1]

# ------------------------------
# A) Baseline: outer 5-fold CV + test
# ------------------------------
pre_base = build_preprocessor_standard(X_train)
baseline = Pipeline([
    ("pre", pre_base),
    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
base_cv = cross_val_score(baseline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)

baseline.fit(X_train, y_train)
base_score = baseline.predict_proba(X_test)[:,1]
base_auc, base_prauc = eval_scores(y_test, base_score)

print("\n[Baseline] CV AUC:", float(base_cv.mean()), "+/-", float(base_cv.std()))
print("[Baseline] Test AUC:", base_auc, " Test PR-AUC:", base_prauc)

# ------------------------------
# B) CorrFilter: test only (no outer CV to avoid memory crash)
# ------------------------------
drop_corr_cols, corr_pairs_df = correlation_filter_train_only(X_train, threshold=CORR_THRESHOLD)
corr_pairs_df.to_csv(os.path.join(TABLE_DIR, "04_corr_high_pairs.csv"), index=False)
pd.DataFrame({"dropped_numeric_feature": drop_corr_cols}).to_csv(
    os.path.join(TABLE_DIR, "04_corr_filter_dropped_features.csv"), index=False
)

X_train_corr = X_train.drop(columns=drop_corr_cols, errors="ignore")
X_test_corr = X_test.drop(columns=drop_corr_cols, errors="ignore")

pre_corr_std = build_preprocessor_standard(X_train_corr)
corr_log = Pipeline([
    ("pre", pre_corr_std),
    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE))
])
corr_log.fit(X_train_corr, y_train)
corr_score = corr_log.predict_proba(X_test_corr)[:,1]
corr_auc, corr_prauc = eval_scores(y_test, corr_score)

print(f"\n[CorrFilter |r|>{CORR_THRESHOLD}] Dropped numeric:", len(drop_corr_cols))
print("[CorrFilter] Test AUC:", corr_auc, " Test PR-AUC:", corr_prauc)

# ------------------------------
# Low-memory transform once -> Ridge/LASSO/ENet
# ------------------------------
pre_low = build_preprocessor_lowmem(X_train_corr)
Xtr_mat = pre_low.fit_transform(X_train_corr)
Xte_mat = pre_low.transform(X_test_corr)

if sparse.issparse(Xtr_mat):
    Xtr_mat = Xtr_mat.astype(np.float32)
    Xte_mat = Xte_mat.astype(np.float32)
else:
    Xtr_mat = Xtr_mat.astype(np.float32, copy=False)
    Xte_mat = Xte_mat.astype(np.float32, copy=False)

feature_names = pre_low.get_feature_names_out()
Cs = np.logspace(-3, 2, 8)

# Ridge
ridge = LogisticRegressionCV(Cs=Cs, cv=5, penalty="l2", solver="lbfgs",
                            scoring="roc_auc", class_weight="balanced",
                            max_iter=6000, n_jobs=1, refit=True)
ridge.fit(Xtr_mat, y_train)
ridge_cv_auc = internal_cv_auc_from_clf(ridge)
ridge_score = ridge.predict_proba(Xte_mat)[:,1]
ridge_auc, ridge_prauc = eval_scores(y_test, ridge_score)

print("\n[Ridge L2] Internal CV AUC:", ridge_cv_auc)
print("[Ridge L2] Test AUC:", ridge_auc, " Test PR-AUC:", ridge_prauc)

# LASSO
lasso = LogisticRegressionCV(Cs=Cs, cv=5, penalty="l1", solver="saga",
                            scoring="roc_auc", class_weight="balanced",
                            max_iter=8000, n_jobs=1, refit=True)
lasso.fit(Xtr_mat, y_train)
lasso_cv_auc = internal_cv_auc_from_clf(lasso)
lasso_score = lasso.predict_proba(Xte_mat)[:,1]
lasso_auc, lasso_prauc = eval_scores(y_test, lasso_score)

coef = lasso.coef_[0]
coef_df = pd.DataFrame({"feature": feature_names, "coef": coef, "abs_coef": np.abs(coef)})
coef_df = coef_df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
selected_df = coef_df[coef_df["coef"] != 0].copy().reset_index(drop=True)

coef_df.to_csv(os.path.join(TABLE_DIR, "04_l1_all_coefficients.csv"), index=False)
selected_df.to_csv(os.path.join(TABLE_DIR, "04_l1_selected_features.csv"), index=False)
coef_df.head(30).to_csv(os.path.join(TABLE_DIR, "04_l1_top30_features.csv"), index=False)

print("\n[LASSO L1] Internal CV AUC:", lasso_cv_auc)
print("[LASSO L1] Test AUC:", lasso_auc, " Test PR-AUC:", lasso_prauc)
print("[LASSO L1] Non-zero selected features:", int(selected_df.shape[0]))

# Plot top30
top30 = coef_df.head(30).iloc[::-1]
plt.figure(figsize=(10,8))
plt.barh(top30["feature"], top30["coef"])
plt.title("Top 30 Coefficients (LASSO Logistic)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_lasso_top30_coeffs.png"), dpi=200)
plt.close()

# Elastic Net (optional; comment if needed)
enet = LogisticRegressionCV(Cs=Cs, cv=5, penalty="elasticnet", solver="saga",
                           l1_ratios=[0.3,0.5,0.7],
                           scoring="roc_auc", class_weight="balanced",
                           max_iter=9000, n_jobs=1, refit=True)
enet.fit(Xtr_mat, y_train)
enet_cv_auc = internal_cv_auc_from_clf(enet)
enet_score = enet.predict_proba(Xte_mat)[:,1]
enet_auc, enet_prauc = eval_scores(y_test, enet_score)

enet_coef = enet.coef_[0]
enet_df = pd.DataFrame({"feature": feature_names, "coef": enet_coef, "abs_coef": np.abs(enet_coef)})
enet_df = enet_df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
enet_sel = enet_df[enet_df["coef"] != 0].copy().reset_index(drop=True)

enet_df.to_csv(os.path.join(TABLE_DIR, "04_enet_all_coefficients.csv"), index=False)
enet_sel.to_csv(os.path.join(TABLE_DIR, "04_enet_selected_features.csv"), index=False)
enet_df.head(30).to_csv(os.path.join(TABLE_DIR, "04_enet_top30_features.csv"), index=False)

print("\n[Elastic Net] Internal CV AUC:", enet_cv_auc)
print("[Elastic Net] Test AUC:", enet_auc, " Test PR-AUC:", enet_prauc)
print("[Elastic Net] Non-zero selected features:", int(enet_sel.shape[0]))

# Summary
summary = pd.DataFrame([
    {"method":"baseline_logistic (outer CV)", "raw_columns_after_basic_cleaning":raw_cols_after_basic,
     "corr_dropped_numeric":0, "selected_nonzero_features_after_ohe":np.nan,
     "cv_auc_mean":float(base_cv.mean()), "cv_auc_std":float(base_cv.std()),
     "test_auc":base_auc, "test_pr_auc":base_prauc},
    {"method":f"corr_filter(|r|>{CORR_THRESHOLD})_logistic (test only)", "raw_columns_after_basic_cleaning":raw_cols_after_basic,
     "corr_dropped_numeric":len(drop_corr_cols), "selected_nonzero_features_after_ohe":np.nan,
     "cv_auc_mean":np.nan, "cv_auc_std":np.nan,
     "test_auc":corr_auc, "test_pr_auc":corr_prauc},
    {"method":"corr_filter + ridge_l2 (internal CV)", "raw_columns_after_basic_cleaning":raw_cols_after_basic,
     "corr_dropped_numeric":len(drop_corr_cols), "selected_nonzero_features_after_ohe":np.nan,
     "cv_auc_mean":ridge_cv_auc, "cv_auc_std":np.nan,
     "test_auc":ridge_auc, "test_pr_auc":ridge_prauc},
    {"method":"corr_filter + lasso_l1 (internal CV)", "raw_columns_after_basic_cleaning":raw_cols_after_basic,
     "corr_dropped_numeric":len(drop_corr_cols), "selected_nonzero_features_after_ohe":int(selected_df.shape[0]),
     "cv_auc_mean":lasso_cv_auc, "cv_auc_std":np.nan,
     "test_auc":lasso_auc, "test_pr_auc":lasso_prauc},
    {"method":"corr_filter + elastic_net (internal CV)", "raw_columns_after_basic_cleaning":raw_cols_after_basic,
     "corr_dropped_numeric":len(drop_corr_cols), "selected_nonzero_features_after_ohe":int(enet_sel.shape[0]),
     "cv_auc_mean":enet_cv_auc, "cv_auc_std":np.nan,
     "test_auc":enet_auc, "test_pr_auc":enet_prauc},
])
summary_path = os.path.join(TABLE_DIR, "04_feature_selection_summary.csv")
summary.to_csv(summary_path, index=False)

print("\n===== SUMMARY (saved) =====")
print(summary)
print("Saved:", summary_path)

print("\n===== Classification report (LASSO, threshold=0.5) =====")
y_pred_l1 = (lasso_score >= 0.5).astype(int)
print(classification_report(y_test, y_pred_l1))

print("\n===== END (FINAL) =====")