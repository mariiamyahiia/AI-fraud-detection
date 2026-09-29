import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import xgboost as xgb

BASE = Path(__file__).resolve().parent.parent
ART = BASE / "artifacts"
V_COLS = [f"V{i}" for i in range(1, 29)]
RAW_COLS = ["Time"] + V_COLS + ["Amount"]

st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(ART / "preprocessor.joblib")
    bundle = joblib.load(ART / "final_fraud_model.joblib")
    return preprocessor, bundle


@st.cache_data
def load_json(name):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_samples():
    p = ART / "sample_transactions.csv"
    return pd.read_csv(p, index_col=0) if p.exists() else None


def score(df_raw: pd.DataFrame):
    """Raw transactions -> (transformed features, fraud probability)."""
    pre, bundle = load_artifacts()
    Xt = pd.DataFrame(pre.transform(df_raw[RAW_COLS]), columns=bundle["feature_names"], index=df_raw.index)
    return Xt, bundle["model"].predict_proba(Xt)[:, 1]


def explain(Xt_row: pd.DataFrame, top=8):
    """Per-transaction feature contributions (XGBoost native SHAP values)."""
    _, bundle = load_artifacts()
    contrib = bundle["model"].get_booster().predict(xgb.DMatrix(Xt_row), pred_contribs=True)[0]
    s = pd.Series(contrib[:-1], index=Xt_row.columns)
    return s.reindex(s.abs().sort_values(ascending=False).index).head(top)


try:
    _, bundle = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Missing artifact: {e.filename}. Run the notebooks first so `artifacts/` is populated.")
    st.stop()

st.title("🛡️ Credit Card Fraud Detection")
st.caption("XGBoost + SMOTE, tuned with RandomizedSearchCV, with a threshold tuned for the precision/recall trade-off.")

st.sidebar.header("Decision threshold")
threshold = st.sidebar.slider("Flag as fraud when probability ≥", 0.01, 0.99,
                              float(bundle["threshold"]), 0.01)
st.sidebar.caption(f"Default (tuned on validation folds): **{bundle['threshold']:.2f}**. "
                   "Lower = catch more fraud but more false alarms.")


def verdict(p):
    if p >= threshold:
        return "Potential Fraud ⚠️"
    return "Legitimate Transaction ✅"


tab_pred, tab_dash, tab_model = st.tabs(["🔍 Predict", "📊 Fraud Dashboard", "🧠 Model & Explainability"])

# Predict
with tab_pred:
    mode = st.radio("Input method", ["Example transaction", "Manual entry", "Batch CSV upload"], horizontal=True)

    if mode == "Batch CSV upload":
        st.write(f"Upload a CSV with columns: `{', '.join(RAW_COLS)}` (an extra `Class` column is ignored).")
        up = st.file_uploader("CSV file", type="csv")
        if up:
            df = pd.read_csv(up)
            missing = [c for c in RAW_COLS if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                _, proba = score(df)
                out = df.copy()
                out["fraud_probability"] = proba.round(4)
                out["prediction"] = ["Fraud" if p >= threshold else "Legitimate" for p in proba]
                c1, c2, c3 = st.columns(3)
                c1.metric("Transactions", len(out))
                c2.metric("Flagged as fraud", int((proba >= threshold).sum()))
                c3.metric("Flag rate", f"{(proba >= threshold).mean():.2%}")
                st.dataframe(out.sort_values("fraud_probability", ascending=False), use_container_width=True)
                st.download_button("Download results", out.to_csv(index=False), "predictions.csv", "text/csv")
    else:
        if mode == "Example transaction":
            samples = load_samples()
            if samples is None:
                st.warning("No example file found (`artifacts/sample_transactions.csv`).")
                st.stop()
            labels = [f"#{i} - actual: {'FRAUD' if r.true_label == 1 else 'legitimate'}" for i, r in samples.iterrows()]
            choice = st.selectbox("Pick a real test transaction", labels)
            row = samples.iloc[labels.index(choice)]
            values = {c: float(row[c]) for c in RAW_COLS}
        else:
            st.info("V1-V28 are anonymised PCA components in this dataset, so they are usually pasted or loaded, "
                    "not typed by hand. Defaults are 0.")
            c1, c2 = st.columns(2)
            values = {"Time": c1.number_input("Time (seconds since first transaction)", 0.0, 172800.0, 50000.0),
                      "Amount": c2.number_input("Amount", 0.0, 30000.0, 50.0)}
            with st.expander("V1 - V28 features"):
                cols = st.columns(4)
                for i, v in enumerate(V_COLS):
                    values[v] = cols[i % 4].number_input(v, value=0.0, format="%.4f", key=v)

        df = pd.DataFrame([values])
        Xt, proba = score(df)
        p = float(proba[0])

        st.subheader(verdict(p))
        st.progress(min(p, 1.0), text=f"Fraud probability / risk score: {p:.1%}")
        risk = "High" if p >= threshold else ("Medium" if p >= threshold / 2 else "Low")
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk level", risk)
        m2.metric("Amount", f"{values['Amount']:.2f}")
        m3.metric("Threshold", f"{threshold:.2f}")

        st.markdown("**Why this score?** Top feature contributions (positive pushes toward fraud):")
        contrib = explain(Xt)
        st.bar_chart(contrib)

# Dashboard
with tab_dash:
    stats = load_json("dashboard_stats.json")
    if not stats:
        st.warning("`dashboard_stats.json` not found - run the last cell of the Member 4 notebook with the raw CSV present.")
    else:
        a, b, c = st.columns(3)
        a.metric("Total transactions", f"{stats['total']:,}")
        b.metric("Fraudulent", f"{stats['frauds']:,}")
        c.metric("Fraud rate", f"{stats['fraud_rate_pct']:.3f}%")

        hourly = pd.DataFrame(stats["hourly"]).set_index("Hour")
        hourly["fraud_rate_%"] = hourly["frauds"] / hourly["transactions"] * 100
        left, right = st.columns(2)
        left.subheader("Transactions by hour of day")
        left.bar_chart(hourly["transactions"])
        right.subheader("Fraud rate (%) by hour of day")
        right.line_chart(hourly["fraud_rate_%"])

        st.subheader("Amount by class")
        amt = pd.DataFrame(stats["amount_by_class"]).T
        amt.index = ["Legitimate", "Fraud"]
        amt.columns = ["Mean", "Median", "Max"]
        st.dataframe(amt.round(2))
        st.caption("Hour of day is derived from the `Time` column (seconds since the first transaction, 2-day span).")

# Model
with tab_model:
    metrics = load_json("metrics.json")
    if metrics:
        st.subheader("Before vs after optimization (test set)")
        st.dataframe(pd.DataFrame(metrics["comparison"]).T.round(4), use_container_width=True)
        st.write(f"Best cross-validated PR-AUC: **{metrics['cv_best_pr_auc']:.4f}**")
        with st.expander("Best hyperparameters"):
            st.json(metrics["best_params"])
    for title, fname in [("Threshold selection", "threshold_curve.png"),
                         ("Confusion matrices", "confusion_matrices.png"),
                         ("SHAP summary", "shap_summary.png"),
                         ("SHAP feature ranking", "shap_bar.png"),
                         ("Explaining one detected fraud", "shap_waterfall_fraud.png")]:
        p = ART / "figures" / fname
        if p.exists():
            st.subheader(title)
            st.image(str(p))
