import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier

from src.config import PROCESSED_DATA_PATH


# =========================
# LOAD FEATURES
# =========================
def load_features():
    path = os.path.join(PROCESSED_DATA_PATH, "window_features.parquet")
    df = pd.read_parquet(path)
    print("Loaded features:", df.shape)
    return df


# =========================
# PREPARE DATA
# =========================
def prepare_data(df):
    X = df.drop(columns=["activityID", "subject_id"])
    y = df["activityID"]
    groups = df["subject_id"]

    print("Features shape:", X.shape)

    return X, y, groups


# =========================
# TRAIN MODEL (GROUP SPLIT)
# =========================
def train_model(X, y, groups):
    print("Training model...")

    gkf = GroupKFold(n_splits=5)

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\nFold {fold+1}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        print(classification_report(y_test, preds))

    # Train final model on full data
    model.fit(X, y)

    return model


# =========================
# SAVE MODEL
# =========================
def save_model(model, feature_columns):
    os.makedirs("artifacts", exist_ok=True)

    joblib.dump(model, "artifacts/activity_model.pkl")

    with open("artifacts/feature_columns.json", "w") as f:
        json.dump(feature_columns, f)

    print("Model saved!")


# =========================
# EXERTION RULES
# =========================
def create_exertion_rules():
    rules = {
        "low": 0.3,
        "moderate": 0.6,
        "high": 0.8
    }

    os.makedirs("artifacts", exist_ok=True)

    with open("artifacts/exertion_rules.json", "w") as f:
        json.dump(rules, f)

    print("Exertion rules saved!")


# =========================
# MAIN
# =========================
def main():
    df = load_features()
    X, y, groups = prepare_data(df)

    model = train_model(X, y, groups)

    save_model(model, X.columns.tolist())
    create_exertion_rules()


if __name__ == "__main__":
    main()