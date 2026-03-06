# SPH6004 Individual Assignment: ICU Mortality Prediction

## Project Overview
This repository contains my individual assignment for **SPH6004 Advanced Statistical Learning**.  
The task is to predict **ICU mortality** using the provided MIMIC-derived ICU admission snapshot dataset.

The project includes:
- data description
- preprocessing and leakage removal
- feature engineering
- feature selection
- model comparison
- threshold analysis for the final model

The target variable is:

- `icu_death_flag`
  - `1` = patient died in ICU
  - `0` = patient survived ICU

---

## Main Findings
Key results from this project are:

- The dataset is **class-imbalanced**, with ICU death rate around **8.68%**
- Correlation-based filtering removed **14 redundant numeric features**
- This feature reduction caused **almost no loss in predictive performance**
- The final selected setting was:
  - **Feature set:** `corr_filter`
  - **Final model:** `GradientBoosting`
- Final model performance:
  - **ROC-AUC:** 0.9161
  - **PR-AUC:** 0.6358
- Threshold analysis showed:
  - default threshold 0.5 gives high precision but low recall
  - threshold around **0.237** gives the best F1 balance

---

## Repository Structure
```text
.
├── 01_eda_data_description.py
├── 02_preprocessing_baseline.py
├── 03_feature_engineering_threshold_rf.py
├── 04_feature_selection_framework.py
├── 05_model_comparison.py
├── 06_threshold_analysis_final_model.py
├── outputs/
│   ├── figures/
│   └── tables/
└── README.md
