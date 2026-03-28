import pandas as pd
import os

# dataset path
DATA_PATH = "data/pamap2+physical+activity+monitoring/PAMAP2_Dataset/Protocol"

# column names (official PAMAP2 format)
columns = [
    "timestamp", "activity_id", "heart_rate",
    
    # IMU hand
    "hand_temp", "hand_acc_x", "hand_acc_y", "hand_acc_z",
    "hand_gyro_x", "hand_gyro_y", "hand_gyro_z",
    
    # IMU chest
    "chest_temp", "chest_acc_x", "chest_acc_y", "chest_acc_z",
    "chest_gyro_x", "chest_gyro_y", "chest_gyro_z",
    
    # IMU ankle
    "ankle_temp", "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
    "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z"
]

def load_all_subjects():
    all_data = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".dat"):
            file_path = os.path.join(DATA_PATH, file)

            df = pd.read_csv(
                file_path,
                sep=" ",
                header=None
            )

            # trim columns (dataset has more, we keep important ones)
            df = df.iloc[:, :len(columns)]
            df.columns = columns

            df["subject"] = file  # track subject

            all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


if __name__ == "__main__":
    df = load_all_subjects()

    print("SHAPE:", df.shape)
    print("\nCOLUMNS:", df.columns)
    print("\nHEAD:\n", df.head())
    print("\nMISSING VALUES:\n", df.isnull().sum())