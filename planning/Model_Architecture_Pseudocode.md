# Model Architecture Pseudocode

## 1. Data Ingestion & Leakage Prevention

```text
Load OpenIntro_Loans_Data

Drop post-origination columns
    - paid_principal
    - paid_total
    - etc.

Purpose:
    Prevent target leakage by removing variables that contain
    information generated after loan origination.
```

## 2. Preprocessing & Data Splitting

```text
Encode categorical variables using One-Hot Encoding

Split data into:
    X_train
    X_test
    y_train
    y_test

Split ratio:
    70% training
    30% testing

Use stratification to preserve the target-class distribution.
```

## 3. Class Imbalance Mitigation

Apply SMOTE only to the training data to prevent information from the test set from influencing model training.

```text
Initialize SMOTE()

X_train_balanced, y_train_balanced =
    SMOTE.fit_resample(X_train, y_train)
```

## 4. Model Training

```text
Initialize LightGBM_Classifier(
    random_state = 42,
    objective = "binary"
)

LightGBM_Classifier.fit(
    X_train_balanced,
    y_train_balanced
)
```

## 5. Evaluation & Threshold Optimization

```text
Generate predictions on X_test

Calculate:
    - ROC-AUC
    - Precision
    - Recall

Determine the optimal decision threshold by:
    - Maximizing F1-score
    - OR optimizing for cost sensitivity
```

## 6. Explainability & Fairness

### Model Explainability

```text
Initialize SHAP_TreeExplainer(
    LightGBM_Classifier
)

Calculate SHAP values for X_test

Identify key risk drivers, including:
    - interest_rate
    - DTI
```

### Fairness Analysis

```text
Define income-based proxy groups:
    - Low-income
    - High-income

Calculate False Positive Rate (FPR)
for each income proxy group

Compare FPR across groups
to assess potential disparities in model behavior.
```

## End-to-End Pipeline

```text
OpenIntro_Loans_Data
        │
        ▼
Remove Post-Origination Columns
        │
        ▼
One-Hot Encode Categorical Variables
        │
        ▼
70/30 Stratified Train-Test Split
        │
        ├──────────────► X_test / y_test
        │
        ▼
Apply SMOTE to Training Data Only
        │
        ▼
Train LightGBM Classifier
        │
        ▼
Generate Predictions on X_test
        │
        ▼
Evaluate ROC-AUC, Precision, Recall
        │
        ▼
Optimize Classification Threshold
        │
        ├──────────────► SHAP Explainability
        │
        └──────────────► Fairness Analysis
                              │
                              ▼
                         Compare FPR
                    Across Income Groups
```