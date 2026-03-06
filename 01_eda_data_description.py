# ==============================
# 01 - Data Description (EDA)
# Save figures to outputs/figures/
# ==============================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Config
# ------------------------------
DATA_PATH = "Assignment1_mimic dataset (1).csv"
TARGET_COL = "icu_death_flag"
FIG_DIR = os.path.join("outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ------------------------------
# Load Data
# ------------------------------
df = pd.read_csv(DATA_PATH)

print("\n===== DATA DESCRIPTION (01) =====")

# ------------------------------
# Basic Info
# ------------------------------
n_samples = df.shape[0]
n_cols = df.shape[1]
print(f"Total samples: {n_samples}")
print(f"Total columns (including target): {n_cols}")

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

# ------------------------------
# Target distribution
# ------------------------------
target_counts = df[TARGET_COL].value_counts().sort_index()
target_ratio = df[TARGET_COL].value_counts(normalize=True).sort_index()

print("\nTarget Distribution (Count):")
print(target_counts)

print("\nTarget Distribution (Ratio):")
print(target_ratio)

# Save bar plot: target distribution
plt.figure()
target_counts.plot(kind="bar")
plt.title("ICU Death Distribution")
plt.xlabel("ICU Death Flag")
plt.ylabel("Count")
plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "01_target_distribution.png")
plt.savefig(fig_path, dpi=200)
plt.close()
print(f"\nSaved figure: {fig_path}")

# ------------------------------
# Variable type distribution
# ------------------------------
type_counts = df.dtypes.value_counts()
print("\nVariable Type Distribution:")
print(type_counts)

# ------------------------------
# Missing value structure
# ------------------------------
missing_ratio = df.isnull().mean().sort_values(ascending=False)
print("\nTop 20 Features with Highest Missing Ratio:")
print(missing_ratio.head(20))

# Save bar plot: top 30 missing ratios
plt.figure(figsize=(12, 4))
missing_ratio.head(30).plot(kind="bar")
plt.title("Top 30 Missing Ratios")
plt.ylabel("Missing Ratio")
plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "01_top30_missing_ratio.png")
plt.savefig(fig_path, dpi=200)
plt.close()
print(f"Saved figure: {fig_path}")

# ------------------------------
# Detect min/max paired variables
# ------------------------------
min_cols = [c for c in df.columns if c.endswith("_min")]
paired_vars = []
for min_col in min_cols:
    base = min_col[:-4]  # remove "_min"
    if f"{base}_max" in df.columns:
        paired_vars.append(base)

paired_vars = sorted(set(paired_vars))
print("\nDetected min/max paired variables (show up to 50):")
print(paired_vars[:50])
print(f"Total paired bases detected: {len(paired_vars)}")

print("\n===== END (01) =====")