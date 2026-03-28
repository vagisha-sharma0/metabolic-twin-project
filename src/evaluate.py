import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

from src.config import PROCESSED_DATA_PATH


# =========================
# LOAD DATA + MODEL
# =========================
def load_all():
    df = pd.read_parquet(os.path.join(PROCESSED_DATA_PATH, "window_features.parquet"))
    model = joblib.load("artifacts/activity_model.pkl")

    print("Loaded data:", df.shape)

    return df, model


# =========================
# PREPARE
# =========================
def prepare(df):
    X = df.drop(columns=["activityID", "subject_id"])
    y = df["activityID"]
    return X, y


# =========================
# CONFUSION MATRIX
# =========================
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    print("Saved confusion matrix!")


# =========================
# FEATURE IMPORTANCE
# =========================
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).head(20)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_imp, x="importance", y="feature")
    plt.title("Top 20 Feature Importance")

    plt.savefig("outputs/feature_importance.png")
    plt.close()

    print("Saved feature importance!")


# =========================
# MAIN
# =========================
def main():
    df, model = load_all()
    X, y = prepare(df)

    preds = model.predict(X)

    print("\nClassification Report:\n")
    print(classification_report(y, preds))

    plot_confusion_matrix(y, preds)
    plot_feature_importance(model, X.columns)


if __name__ == "__main__":
    main()