# Project Tasks & Milestones: Credit Risk XAI Framework (v1.0)

## Phase 1: Data Ingestion & Leakage Prevention (Notebook 01)
- [x] Ingest OpenIntro Loans Full Schema dataset (10,000 records).
- [x] Define binary target variable (`loan_status_binary`): 0 = Good, 1 = Default.
- [x] Impute missing values (e.g., employment length, delinquency history).
- [x] **Crucial:** Identify and drop post-origination repayment artifacts (e.g., `paid_principal`, `paid_total`) to prevent target leakage.

## Phase 2: Exploratory Data Analysis (Notebook 02)
- [x] Visualize the extreme class imbalance (1.8% default rate).
- [x] Perform risk ordering sanity check: validate that empirical default rates scale monotonically across loan grades (A to G).
- [x] Generate feature distributions and correlation heatmaps to identify extreme financial outliers (e.g., DTI up to 469).

## Phase 3: Preprocessing & Resampling (Notebook 03)
- [x] Apply One-Hot Encoding to categorical variables.
- [x] Execute a static 70/30 stratified train-test split.
- [x] Implement Synthetic Minority Over-sampling Technique (SMOTE) strictly on the training partition to achieve a 50/50 class balance.
- [x] Save processed arrays (`X_train_resampled`, `X_test`, etc.) for downstream modeling.

## Phase 4: Model Training & Evaluation (Notebook 04)
- [x] Train baseline parametric model: Logistic Regression.
- [x] Train non-linear ensemble models: Random Forest, XGBoost, and LightGBM.
- [x] Evaluate global ranking metrics (ROC-AUC) and demonstrate the baseline threshold collapse (0% minority recall for Random Forest at 0.50 threshold).
- [x] Select LightGBM as the primary architecture due to leaf-wise growth efficiency and minority-class sensitivity.

## Phase 5: Threshold Optimization (Notebook 08)
- [x] Generate Precision-Recall curves.
- [x] Determine the optimal decision threshold (~3.4%) to prioritize minority-class default detection over raw accuracy.
- [x] Formulate Expected Loss (EL) binary linear programming solver to simulate portfolio allocation under budget constraints.

## Phase 6: Explainable AI & Transparency (Notebook 05)
- [x] Integrate SHAP (Shapley Additive Explanations) via `TreeExplainer`.
- [x] Generate global summary plots to identify primary risk drivers (e.g., `interest_rate`, `debt_to_income`, `grade`).
- [x] Generate local SHAP force plots for individual borrowers to simulate ECOA-compliant adverse action reasoning.

## Phase 7: Fairness & Compliance Auditing (Notebook 06)
- [x] Define economic proxy groups (e.g., High Income vs. Low Income).
- [x] Calculate False Positive Rate (FPR) parity to audit for disparate impact.
- [x] Generate probability calibration curves (reliability diagrams) to evaluate SMOTE-induced probability distortion.

## Phase 8: Reporting & Dashboard Visuals (Notebook 07)
- [x] Export final demographics tables and performance comparisons.
- [x] Generate production-grade visualizations (Confusion Matrices, Precision-Recall optimum).
- [x] Consolidate methodology into formal thesis documentation.