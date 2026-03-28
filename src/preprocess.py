import os
import pandas as pd
import numpy as np
from glob import glob

from src.config import (
    PROTOCOL_PATH,
    COLUMN_NAMES,
    DROP_COLUMNS,
    TARGET_HZ,
    PROCESSED_DATA_PATH,
    ACC_COLUMNS,
    GYRO_COLUMNS
)


# =========================
# LOAD SINGLE FILE
# =========================
def load_single_file(file_path):
    df = pd.read_csv(file_path, sep=" ", header=None)
    df.columns = COLUMN_NAMES

    # Extract subject ID from filename
    subject_id = int(os.path.basename(file_path).replace("subject", "").replace(".dat", ""))
    df["subject_id"] = subject_id

    return df


# =========================
# LOAD ALL FILES
# =========================
def load_all_data():
    files = glob(os.path.join(PROTOCOL_PATH, "*.dat"))
    print(f"Found {len(files)} files")

    all_dfs = []

    for f in files:
        print(f"Loading {f}")
        df = load_single_file(f)
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    print("Combined shape:", full_df.shape)

    return full_df


# =========================
# CLEAN DATA
# =========================
def clean_data(df):
    print("Cleaning data...")

    # Remove activityID = 0
    df = df[df["activityID"] != 0]

    # Drop orientation columns
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    # Sort by subject and time
    df = df.sort_values(by=["subject_id", "timestamp"])

    # Handle heart rate (forward fill per subject)
    df["heart_rate"] = df.groupby("subject_id")["heart_rate"].ffill()

    # Interpolate other signals (per subject)
    df = df.groupby("subject_id", group_keys=False).apply(lambda x: x.interpolate())

    # Drop remaining NaNs
    df = df.dropna()

    print("After cleaning:", df.shape)

    return df


# =========================
# RESAMPLE TO 10 Hz
# =========================
def resample_data(df):
    print("Resampling to", TARGET_HZ, "Hz...")

    resampled_list = []

    for subject_id, group in df.groupby("subject_id"):
        group = group.copy()

        # Convert timestamp to datetime
        group = group.sort_values("timestamp")
        group["timestamp"] = pd.to_datetime(group["timestamp"], unit="s")

        group = group.set_index("timestamp")

        # Resample
        activity = group["activityID"].resample(freq).ffill()
        group_resampled = group.resample(freq).mean()

        group_resampled["activityID"] = activity

        # Forward fill labels
        group_resampled["activityID"] = group_resampled["activityID"].ffill()
        group_resampled["subject_id"] = subject_id

        group_resampled = group_resampled.reset_index()

        resampled_list.append(group_resampled)

    df_resampled = pd.concat(resampled_list, ignore_index=True)

    print("After resampling:", df_resampled.shape)

    return df_resampled


# =========================
# CREATE MAGNITUDE FEATURES
# =========================
def create_magnitude_features(df):
    print("Creating magnitude features...")

    # Helper function
    def magnitude(x, y, z):
        return np.sqrt(x**2 + y**2 + z**2)

    # Hand
    df["hand_acc_mag"] = magnitude(df["hand_acc_x"], df["hand_acc_y"], df["hand_acc_z"])
    df["hand_gyro_mag"] = magnitude(df["hand_gyro_x"], df["hand_gyro_y"], df["hand_gyro_z"])

    # Chest
    df["chest_acc_mag"] = magnitude(df["chest_acc_x"], df["chest_acc_y"], df["chest_acc_z"])
    df["chest_gyro_mag"] = magnitude(df["chest_gyro_x"], df["chest_gyro_y"], df["chest_gyro_z"])

    # Ankle
    df["ankle_acc_mag"] = magnitude(df["ankle_acc_x"], df["ankle_acc_y"], df["ankle_acc_z"])
    df["ankle_gyro_mag"] = magnitude(df["ankle_gyro_x"], df["ankle_gyro_y"], df["ankle_gyro_z"])

    return df


# =========================
# MAIN PIPELINE
# =========================
def main():
    df = load_all_data()
    df = clean_data(df)
    df = resample_data(df)
    df = create_magnitude_features(df)

    output_path = os.path.join(PROCESSED_DATA_PATH, "clean_sensor_data.parquet")
    df.to_parquet(output_path)

    print(f"Saved cleaned data to {output_path}")


if __name__ == "__main__":
    main()