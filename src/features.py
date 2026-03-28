import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.config import (
    PROCESSED_DATA_PATH,
    WINDOW_SIZE_SECONDS,
    WINDOW_STRIDE_SECONDS,
    TARGET_HZ
)


# =========================
# LOAD CLEAN DATA
# =========================
def load_data():
    path = os.path.join(PROCESSED_DATA_PATH, "clean_sensor_data.parquet")
    df = pd.read_parquet(path)
    print("Loaded data:", df.shape)
    return df


# =========================
# WINDOWING FUNCTION
# =========================
def create_windows(df):
    print("Creating windows...")

    window_size = int(WINDOW_SIZE_SECONDS * TARGET_HZ)
    stride = int(WINDOW_STRIDE_SECONDS * TARGET_HZ)

    feature_rows = []

    for subject_id, group in tqdm(df.groupby("subject_id")):
        group = group.reset_index(drop=True)

        for start in range(0, len(group) - window_size + 1, stride):
            end = start + window_size
            window = group.iloc[start:end]

            features = extract_features(window)

            # Label = majority activity
            label = window["activityID"].mode()[0]

            features["activityID"] = label
            features["subject_id"] = subject_id

            feature_rows.append(features)

    features_df = pd.DataFrame(feature_rows)
    print("Created feature dataset:", features_df.shape)

    return features_df


# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(window):
    features = {}

    # Selected columns
    cols = [
        "hand_acc_mag", "chest_acc_mag", "ankle_acc_mag",
        "hand_gyro_mag", "chest_gyro_mag", "ankle_gyro_mag",
        "heart_rate"
    ]

    for col in cols:
        data = window[col]

        features[f"{col}_mean"] = data.mean()
        features[f"{col}_std"] = data.std()
        features[f"{col}_min"] = data.min()
        features[f"{col}_max"] = data.max()
        features[f"{col}_median"] = data.median()
        features[f"{col}_range"] = data.max() - data.min()

    # Heart rate dynamics
    hr = window["heart_rate"]

    features["hr_delta"] = hr.iloc[-1] - hr.iloc[0]
    if hr.isnull().any() or len(hr) < 2:
        features["hr_slope"] = 0
    else:
        features["hr_slope"] = np.polyfit(range(len(hr)), hr, 1)[0]

    return features


# =========================
# SAVE FEATURES
# =========================
def save_features(df):
    path = os.path.join(PROCESSED_DATA_PATH, "window_features.parquet")
    df.to_parquet(path)
    print(f"Saved features to {path}")


# =========================
# MAIN
# =========================
def main():
    df = load_data()
    features_df = create_windows(df)
    save_features(features_df)


if __name__ == "__main__":
    main()