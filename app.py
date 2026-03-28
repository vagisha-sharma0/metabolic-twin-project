"""import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# =========================
# LOAD ARTIFACTS
# =========================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("artifacts/activity_model.pkl")

        with open("artifacts/feature_columns.json") as f:
            feature_cols = json.load(f)

        with open("artifacts/exertion_rules.json") as f:
            rules = json.load(f)

        # Optional activity map
        activity_map = None
        if os.path.exists("artifacts/activity_map.json"):
            with open("artifacts/activity_map.json") as f:
                activity_map = json.load(f)

        return model, feature_cols, rules, activity_map

    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        st.stop()


model, feature_cols, rules, activity_map = load_artifacts()


# =========================
# EXERTION FUNCTION
# =========================
def compute_exertion(row):
    hr = row.get("heart_rate_mean", 0)

    motion = (
        row.get("hand_acc_mag_mean", 0) +
        row.get("chest_acc_mag_mean", 0) +
        row.get("ankle_acc_mag_mean", 0)
    ) / 3

    score = (hr / 200) * 0.6 + (motion / 20) * 0.4

    # Clamp score between 0–1
    score = max(0, min(1, score))

    if score < rules["low"]:
        return "Low", score
    elif score < rules["moderate"]:
        return "Moderate", score
    elif score < rules["high"]:
        return "High", score
    else:
        return "Very High", score


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Metabolic Twin", layout="wide")

st.title("🧠 Wearable Metabolic Twin")
st.markdown("Predict activity + estimate exertion from wearable sensor features")

# =========================
# INPUT SECTION
# =========================
st.sidebar.header("Input")

uploaded_file = st.sidebar.file_uploader("Upload feature CSV")
use_sample = st.sidebar.button("Use Sample Data")

df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)

elif use_sample:
    df = pd.read_parquet("data_processed/window_features.parquet").sample(1)


# =========================
# PREDICTION
# =========================
if df is not None:
    st.subheader("Input Data")
    st.dataframe(df.head())

    # Validate columns
    missing = set(feature_cols) - set(df.columns)
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    # ✅ FIXED
    X = df[feature_cols]

    preds = model.predict(X)
    probs = model.predict_proba(X)

    for i in range(len(df)):
        st.subheader(f"Prediction #{i+1}")

        activity_id = preds[i]
        confidence = np.max(probs[i])

        exertion_label, exertion_score = compute_exertion(df.iloc[i])

        # Convert ID → name if available
        if activity_map:
            activity_name = activity_map.get(str(activity_id), str(activity_id))
        else:
            activity_name = str(activity_id)

        col1, col2, col3 = st.columns(3)

        col1.metric("Activity", activity_name)
        col2.metric("Confidence", f"{confidence:.2f}")
        col3.metric("Exertion", exertion_label)

        st.progress(exertion_score)

else:
    st.info("Upload a feature file or use sample data")"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


# =========================
# LOAD ARTIFACTS
# =========================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("artifacts/activity_model.pkl")

        with open("artifacts/feature_columns.json") as f:
            feature_cols = json.load(f)

        with open("artifacts/exertion_rules.json") as f:
            rules = json.load(f)

        activity_map = None
        if os.path.exists("artifacts/activity_map.json"):
            with open("artifacts/activity_map.json") as f:
                activity_map = json.load(f)

        return model, feature_cols, rules, activity_map

    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        st.stop()


model, feature_cols, rules, activity_map = load_artifacts()


# =========================
# EXERTION FUNCTION
# =========================
def compute_exertion(row):
    hr = row.get("heart_rate_mean", 0)

    motion = (
        row.get("hand_acc_mag_mean", 0) +
        row.get("chest_acc_mag_mean", 0) +
        row.get("ankle_acc_mag_mean", 0)
    ) / 3

    score = (hr / 200) * 0.6 + (motion / 20) * 0.4
    score = max(0, min(1, score))

    if score < rules["low"]:
        return "Low", score
    elif score < rules["moderate"]:
        return "Moderate", score
    elif score < rules["high"]:
        return "High", score
    else:
        return "Very High", score


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Metabolic Twin", layout="wide")

st.title("🧠 Wearable Metabolic Twin")
st.markdown("Predict human activity and estimate exertion using wearable sensor data")


# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("Input")

uploaded_file = st.sidebar.file_uploader("Upload feature CSV", type=["csv"])
use_sample = st.sidebar.button("Use Sample Data")

df = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

elif use_sample:
    try:
        df = pd.read_parquet("data_processed/window_features.parquet").sample(200)
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        st.stop()


# =========================
# MAIN LOGIC
# =========================
if df is not None:
    st.subheader("📂 Input Data")
    st.dataframe(df.head())

    # Validate features
    missing = set(feature_cols) - set(df.columns)
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    X = df[feature_cols]

    preds = model.predict(X)
    probs = model.predict_proba(X)

    # =========================
    # PERFORMANCE (UNCHANGED)
    # =========================
    if "activityID" in df.columns:
        st.subheader("📊 Model Performance")

        y_true = df["activityID"]
        y_pred = preds

        acc = accuracy_score(y_true, y_pred)
        st.metric("Model Accuracy", f"{acc:.2f}")

        unique_labels = sorted(set(y_true))
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

        if activity_map:
            labels = [activity_map.get(str(i), str(i)) for i in unique_labels]
        else:
            labels = [str(i) for i in unique_labels]

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        st.pyplot(fig)

    # =========================
    # PREDICTIONS (LIMIT = 20)
    # =========================
    st.subheader("🔍 Predictions (Showing first 20)")

    num_predictions = min(20, len(df))

    for i in range(num_predictions):
        st.markdown(f"### Prediction #{i+1}")

        activity_id = preds[i]
        confidence = np.max(probs[i])

        exertion_label, exertion_score = compute_exertion(df.iloc[i])

        if activity_map:
            activity_name = activity_map.get(str(activity_id), str(activity_id))
        else:
            activity_name = str(activity_id)

        col1, col2, col3 = st.columns(3)

        col1.metric("Activity", activity_name)
        col2.metric("Confidence", f"{confidence:.2f}")
        col3.metric("Exertion", exertion_label)

        st.progress(exertion_score)

else:
    st.info("Upload a feature CSV file or click 'Use Sample Data'")