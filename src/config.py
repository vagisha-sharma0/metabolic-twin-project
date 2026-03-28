import os

# =========================
# PATH CONFIGURATION
# =========================

BASE_DATA_PATH = os.path.join(
    "data",
    "pamap2+physical+activity+monitoring",
    "PAMAP2_Dataset"
)

PROTOCOL_PATH = os.path.join(BASE_DATA_PATH, "Protocol")

PROCESSED_DATA_PATH = "data_processed"
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)


# =========================
# SAMPLING CONFIG
# =========================

TARGET_HZ = 10   # resample everything to 10 Hz


# =========================
# WINDOW CONFIG (for later)
# =========================

WINDOW_SIZE_SECONDS = 5
WINDOW_STRIDE_SECONDS = 2.5


# =========================
# COLUMN NAMES (PAMAP2)
# =========================

COLUMN_NAMES = [
    "timestamp",
    "activityID",
    "heart_rate",

    # Hand IMU
    "hand_temp",
    "hand_acc_x", "hand_acc_y", "hand_acc_z",
    "hand_acc16_x", "hand_acc16_y", "hand_acc16_z",
    "hand_gyro_x", "hand_gyro_y", "hand_gyro_z",
    "hand_mag_x", "hand_mag_y", "hand_mag_z",
    "hand_ori_1", "hand_ori_2", "hand_ori_3", "hand_ori_4",

    # Chest IMU
    "chest_temp",
    "chest_acc_x", "chest_acc_y", "chest_acc_z",
    "chest_acc16_x", "chest_acc16_y", "chest_acc16_z",
    "chest_gyro_x", "chest_gyro_y", "chest_gyro_z",
    "chest_mag_x", "chest_mag_y", "chest_mag_z",
    "chest_ori_1", "chest_ori_2", "chest_ori_3", "chest_ori_4",

    # Ankle IMU
    "ankle_temp",
    "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
    "ankle_acc16_x", "ankle_acc16_y", "ankle_acc16_z",
    "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
    "ankle_mag_x", "ankle_mag_y", "ankle_mag_z",
    "ankle_ori_1", "ankle_ori_2", "ankle_ori_3", "ankle_ori_4"
]


# =========================
# DROP COLUMNS
# =========================

# Orientation is INVALID (dataset note)
DROP_COLUMNS = [
    col for col in COLUMN_NAMES if "ori" in col
]


# =========================
# IMPORTANT SIGNALS
# =========================

ACC_COLUMNS = [
    "hand_acc_x", "hand_acc_y", "hand_acc_z",
    "chest_acc_x", "chest_acc_y", "chest_acc_z",
    "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
]

GYRO_COLUMNS = [
    "hand_gyro_x", "hand_gyro_y", "hand_gyro_z",
    "chest_gyro_x", "chest_gyro_y", "chest_gyro_z",
    "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
]


# =========================
# ACTIVITY LABELS
# =========================

ACTIVITY_MAP = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    9: "watching_tv",
    10: "computer_work",
    11: "car_driving",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping"
}