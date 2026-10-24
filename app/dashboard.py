"""Streamlit dashboard for scoring transactions via the API.

Run from the project root with: streamlit run app/dashboard.py
"""

from pathlib import Path

import pandas as pd
import requests
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="DeepGuard AI",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ DeepGuard AI")
st.caption("Deep learning fraud risk scoring — MVP")

st.info(
    "Educational prototype using an anonymized historical dataset. "
    "Scores are not production banking decisions."
)

if not DATA_PATH.exists():
    st.error(
        "Dataset not found. Place creditcard.csv in data/raw/ "
        "and refresh this page."
    )
    st.stop()

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()
features = [c for c in df.columns if c != "Class"]

st.sidebar.header("Transaction selection")
row_idx = st.sidebar.number_input(
    "Dataset row index",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1,
)

row = df.iloc[int(row_idx)]
payload = {
    "features": {
        col: float(row[col]) for col in features
    }
}

left, middle, right = st.columns(3)
left.metric("Transaction amount", f"{row['Amount']:.2f}")
middle.metric("Elapsed time", f"{row['Time']:.0f} sec")
right.metric("Dataset size", f"{len(df):,}")

st.subheader("Selected transaction")
st.write(
    "The model receives 30 numeric features. "
    "The historical Class label is excluded from the API request."
)

with st.expander("Inspect model input features"):
    st.json(payload["features"])

if st.button("Analyze transaction", type="primary"):
    try:
        response = requests.post(
            f"{API_URL}/predict", json=payload, timeout=30
        )
        response.raise_for_status()
        result = response.json()

        st.subheader("Prediction result")
        a, b, c = st.columns(3)
        a.metric("Model risk score", f"{result['risk_score']:.2%}")
        b.metric("Review threshold", f"{result['threshold']:.2%}")
        c.metric("Decision", result["decision"])

        if result["flagged_for_review"]:
            st.warning("Flagged for review by the model.")
        else:
            st.success("Below the model's review threshold.")

        # Ground truth is shown only after scoring for retrospective demo.
        actual = int(row["Class"])
        st.write(
            "Historical dataset label:",
            "Fraud" if actual == 1 else "Legitimate",
        )
        st.caption(
            "A single transaction's result does not establish model quality. "
            "Use held-out evaluation metrics to assess performance."
        )

    except requests.RequestException as exc:
        st.error(
            f"Could not reach the prediction API: {exc}. "
            "Check that the API is running on port 8000."
        )

st.divider()
st.subheader("Dataset overview")
st.write(f"Total transactions: {len(df):,}")
st.write(f"Fraud examples: {int(df['Class'].sum()):,}")
st.write(f"Fraud rate: {df['Class'].mean():.3%}")

st.caption(
    "This dataset contains historical anonymized features; it does not "
    "provide merchant, account, or device details."
)

