"""
Credit Risk Model Dashboard
============================
Interactive Streamlit dashboard for the LightGBM credit-risk classifier
built in notebooks 03 (feature engineering), 04 (model training), 05
(explainability) and 08 (threshold optimization).

Run from the project root with:
    streamlit run app.py

Expected project layout (paths are relative to the project root):
    data/clean_data.csv
    app.py                <- this file

Why this app retrains the model instead of loading a .pkl
-----------------------------------------------------------
Notebook 03 drops 'balance', 'paid_total', 'paid_principal',
'paid_interest' and 'paid_late_fees' before encoding, specifically to
prevent target leakage (these are post-origination repayment artifacts
— see Section 4.1.2 of the thesis). The originally-shared
best_model_lightgbm.pkl was trained BEFORE that correction and still
expects those 5 leaked columns (118 features instead of 113), which
would silently reproduce the leakage the thesis explicitly corrects
for (inflated AUC ~0.83 instead of the reported ~0.66).

To stay faithful to the corrected, leakage-free pipeline that Chapter 5
of the thesis actually reports on, this app reproduces notebooks 03 +
04 + 08 exactly (same column drops, same encoding, same 70/30 split,
same SMOTE-on-train-only step, same default-hyperparameter LightGBM)
and retrains once per session. This is deterministic given the fixed
random_state=42 used throughout, so results are stable across runs.

Features
--------
1. A threshold slider (0.01 - 0.99) that updates a precision/recall
   trade-off chart in real time, with the thesis's own optimal
   (max-F1) threshold from notebook 08 marked for reference.
2. An interactive SHAP force-plot viewer (notebook 05's methodology)
   where the user can pick a borrower ID from the real hold-out test
   set and see the local explanation for that borrower's prediction.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
import streamlit.components.v1 as components
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# Config — mirrors notebook 03 exactly
# --------------------------------------------------------------------------
DATA_PATH = "data/clean_data.csv"

# Notebook 03, cell "2. FEATURE SELECTION & CLEANING": drop the text
# target column before encoding.
TARGET_TEXT_COL = "loan_status"

# Notebook 03, cell "Drop other high-cardinality/unnecessary columns":
# sub_grade is redundant with grade; issue_month isn't used; the paid_*
# and balance columns are post-origination artifacts dropped to prevent
# target leakage (Section 4.1.2 of the thesis).
COLS_TO_DROP = [
    "sub_grade",
    "issue_month",
    "balance",
    "paid_total",
    "paid_principal",
    "paid_interest",
    "paid_late_fees",
]

# Notebook 03, cell "3. ONE-HOT ENCODING": the object-dtype columns that
# get one-hot encoded with drop_first=True.
CATEGORICAL_COLS = [
    "state",
    "homeownership",
    "verified_income",
    "verification_income_joint",
    "loan_purpose",
    "application_type",
    "grade",
    "initial_listing_status",
    "disbursement_method",
]

TARGET_COL = "loan_status_binary"

# Notebook 03: test_size=0.3, random_state=42, stratify=y
TEST_SIZE = 0.3
RANDOM_STATE = 42

st.set_page_config(
    page_title="Credit Risk Model Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# Cached pipeline — reproduces notebooks 03 + 04 + 08 exactly
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading clean_data.csv...")
def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner="Reproducing notebook 03 feature engineering...")
def build_features(raw_df: pd.DataFrame):
    """Notebook 03, steps 2-5: drop columns, one-hot encode, fill NaNs
    with column medians, then a stratified 70/30 train/test split.
    Index is preserved throughout so test-set rows can be traced back
    to their original clean_data.csv record."""
    df = raw_df.drop(columns=[TARGET_TEXT_COL])
    df = df.drop(columns=COLS_TO_DROP)

    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    y = df_encoded[TARGET_COL]
    X = df_encoded.drop(columns=[TARGET_COL])

    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


@st.cache_resource(show_spinner="Training LightGBM (notebook 04 pipeline)...")
def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Notebook 03, step 6 (SMOTE on training data only) + notebook 04
    (default-hyperparameter LightGBM, random_state=42)."""
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    model = LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
    model.fit(X_train_resampled, y_train_resampled)
    return model


@st.cache_resource(show_spinner=False)
def get_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_data(show_spinner="Computing SHAP values (notebook 05 methodology)...")
def compute_shap_values(_explainer, X_test: pd.DataFrame):
    """Notebook 05, cell 3: SHAP values on the raw margin (log-odds)
    output, class 1 (Risky)."""
    shap_values = _explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    expected_value = _explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = (
            expected_value[1] if len(np.atleast_1d(expected_value)) > 1 else expected_value[0]
        )
    return shap_values, expected_value


@st.cache_data(show_spinner=False)
def compute_optimal_threshold(y_test: pd.Series, proba_test: np.ndarray):
    """Notebook 08: max-F1 threshold over the hold-out test set's
    precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_test, proba_test)
    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    best_idx = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[best_idx]), float(precision[best_idx]), float(recall[best_idx]), float(f1_scores[best_idx])


def st_shap(plot, height: int = 180):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height, scrolling=True)


# --------------------------------------------------------------------------
# Load data, engineer features, train model
# --------------------------------------------------------------------------
try:
    raw_df = load_raw_data(DATA_PATH)
except FileNotFoundError as e:
    st.error(
        f"Could not find `{e.filename}`.\n\n"
        f"Make sure you're running `streamlit run app.py` from the project "
        f"root, with `{DATA_PATH}` present."
    )
    st.stop()

X_train, X_test, y_train, y_test = build_features(raw_df)
model = train_model(X_train, y_train)
proba_test = model.predict_proba(X_test)[:, 1]

explainer = get_explainer(model)
shap_values, expected_value = compute_shap_values(explainer, X_test)

opt_threshold, opt_precision, opt_recall, opt_f1 = compute_optimal_threshold(y_test, proba_test)

st.title("💳 Credit Risk Model Dashboard")
st.caption(
    "LightGBM credit-risk classifier (leakage-free, 113-feature pipeline) — "
    "threshold analysis & SHAP explanations"
)

# --------------------------------------------------------------------------
# Sidebar — global controls
# --------------------------------------------------------------------------
st.sidebar.header("Decision Threshold")

if "threshold" not in st.session_state:
    st.session_state.threshold = round(opt_threshold, 2) if opt_threshold >= 0.01 else 0.03

if st.sidebar.button(f"Use thesis-optimal threshold ({opt_threshold:.4f})", use_container_width=True):
    st.session_state.threshold = max(0.01, min(0.99, round(opt_threshold, 2)))

threshold = st.sidebar.slider(
    "Probability threshold for classifying a borrower as high-risk (1)",
    min_value=0.01,
    max_value=0.99,
    step=0.01,
    key="threshold",
)
st.sidebar.markdown(
    f"Borrowers with predicted default probability **≥ {threshold:.2f}** "
    f"are flagged as high-risk."
)
st.sidebar.divider()
st.sidebar.caption(
    f"Hold-out test set: {len(X_test):,} borrowers "
    f"({y_test.mean():.1%} observed default rate) — 70/30 split, "
    f"random_state=42, SMOTE applied to training data only."
)
st.sidebar.caption(
    f"Notebook 08's reported optimal operating point: threshold ≈ "
    f"{opt_threshold:.4f}, Precision {opt_precision:.1%}, "
    f"Recall {opt_recall:.1%}, F1 {opt_f1:.3f}."
)

# ==========================================================================
# SECTION 1 — Precision / Recall trade-off (reacts live to the slider)
# ==========================================================================
st.header("1. Precision–Recall Trade-off")

y_pred = (proba_test >= threshold).astype(int)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

metric_cols = st.columns(5)
metric_cols[0].metric("Threshold", f"{threshold:.2f}")
metric_cols[1].metric("Precision", f"{precision:.3f}")
metric_cols[2].metric("Recall", f"{recall:.3f}")
metric_cols[3].metric("F1 score", f"{f1:.3f}")
metric_cols[4].metric("Flagged high-risk", f"{y_pred.sum():,} / {len(y_pred):,}")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_test, proba_test)

    fig_pr = go.Figure()
    fig_pr.add_trace(
        go.Scatter(
            x=rec_curve,
            y=prec_curve,
            mode="lines",
            name="PR curve",
            line=dict(color="#636EFA", width=2),
        )
    )
    fig_pr.add_trace(
        go.Scatter(
            x=[recall],
            y=[precision],
            mode="markers",
            name=f"Current ({threshold:.2f})",
            marker=dict(color="#EF553B", size=14, symbol="star"),
        )
    )
    fig_pr.add_trace(
        go.Scatter(
            x=[opt_recall],
            y=[opt_precision],
            mode="markers",
            name=f"Thesis optimum ({opt_threshold:.3f})",
            marker=dict(color="#00CC96", size=11, symbol="diamond"),
        )
    )
    fig_pr.update_layout(
        title="Precision vs. Recall",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=50, b=10),
    )
    st.plotly_chart(fig_pr, use_container_width=True)

with chart_col2:
    fig_thresh = go.Figure()
    fig_thresh.add_trace(
        go.Scatter(
            x=pr_thresholds,
            y=prec_curve[:-1],
            mode="lines",
            name="Precision",
            line=dict(color="#00CC96", width=2),
        )
    )
    fig_thresh.add_trace(
        go.Scatter(
            x=pr_thresholds,
            y=rec_curve[:-1],
            mode="lines",
            name="Recall",
            line=dict(color="#AB63FA", width=2),
        )
    )
    fig_thresh.add_vline(
        x=threshold,
        line_width=2,
        line_dash="dash",
        line_color="#EF553B",
        annotation_text=f"current = {threshold:.2f}",
        annotation_position="top",
    )
    fig_thresh.add_vline(
        x=opt_threshold,
        line_width=2,
        line_dash="dot",
        line_color="#00CC96",
        annotation_text=f"thesis optimum = {opt_threshold:.3f}",
        annotation_position="bottom",
    )
    fig_thresh.update_layout(
        title="Precision & Recall vs. Threshold",
        xaxis_title="Decision threshold",
        yaxis_title="Score",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=50, b=10),
    )
    st.plotly_chart(fig_thresh, use_container_width=True)

with st.expander("Confusion matrix at the current threshold"):
    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual: Low risk (0)", "Actual: High risk (1)"],
        columns=["Predicted: Low risk (0)", "Predicted: High risk (1)"],
    )
    st.dataframe(cm_df, use_container_width=True)

st.divider()

# ==========================================================================
# SECTION 2 — SHAP force plot viewer per borrower
# ==========================================================================
st.header("2. Borrower-level SHAP Explanation")

left, right = st.columns([1, 3])

with left:
    borrower_id = st.selectbox(
        "Select a borrower ID (test set index into clean_data.csv)",
        options=X_test.index.tolist(),
        format_func=lambda i: f"Borrower #{i}",
    )
    row_pos = X_test.index.get_loc(borrower_id)
    borrower_proba = float(proba_test[row_pos])
    decision = "🔴 High risk" if borrower_proba >= threshold else "🟢 Low risk"

    st.metric("Predicted default probability", f"{borrower_proba:.1%}")
    st.metric("Decision at current threshold", decision)
    st.metric("Actual outcome", "Defaulted" if y_test.loc[borrower_id] == 1 else "Repaid")

    with st.expander("Raw borrower features (clean_data.csv)"):
        st.dataframe(
            raw_df.loc[[borrower_id]].T.rename(columns={borrower_id: "value"}).astype(str),
            use_container_width=True,
        )

with right:
    row = X_test.loc[[borrower_id]]
    shap_row = shap_values[row_pos : row_pos + 1]

    force_plot = shap.force_plot(
        expected_value,
        shap_row,
        row,
        link="identity",
    )

    st.caption(
        "Force plot in raw model output (log-odds / margin) space, matching "
        "notebook 05's methodology — red features push the prediction "
        "toward higher default risk, blue features push it toward lower "
        "risk. Base value: "
        f"{expected_value:.3f} (≈ {1 / (1 + np.exp(-expected_value)):.1%} "
        "baseline probability before this borrower's features are applied)."
    )
    st_shap(force_plot, height=170)

    st.subheader("Top contributing features")
    contrib = pd.DataFrame(
        {
            "feature": row.columns,
            "value": row.iloc[0].values,
            "shap_value": np.ravel(shap_row),
        }
    )
    contrib["abs_shap"] = contrib["shap_value"].abs()
    contrib = contrib.sort_values("abs_shap", ascending=False).head(10)

    fig_bar = go.Figure(
        go.Bar(
            x=contrib["shap_value"],
            y=contrib["feature"],
            orientation="h",
            marker_color=np.where(contrib["shap_value"] >= 0, "#EF553B", "#636EFA"),
        )
    )
    fig_bar.update_layout(
        title="Top 10 SHAP contributions for this borrower",
        xaxis_title="SHAP value (impact on model output, log-odds)",
        yaxis=dict(autorange="reversed"),
        margin=dict(t=50, b=10),
        height=380,
    )
    st.plotly_chart(fig_bar, use_container_width=True)
