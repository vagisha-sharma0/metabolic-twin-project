# 🧠 Wearable Metabolic Twin

A machine learning project that predicts human physical activity and estimates exertion levels using wearable sensor data.

---

## 📌 Overview

This project builds a **Metabolic Twin system** that:

- Predicts human activities (e.g., walking, running, sitting)
- Estimates exertion levels (Low, Moderate, High, Very High)
- Visualizes model performance using a confusion matrix

The system is built using sensor data from wearable devices and demonstrates an end-to-end ML pipeline.

---

## ⚙️ Features

- 📊 Activity prediction using trained ML model  
- 🔥 Exertion estimation using rule-based system  
- 📉 Confusion matrix visualization  
- 📈 Accuracy evaluation  
- 🖥️ Interactive UI using Streamlit  

---

## 🏗️ Project Structure
metabolic_twin_project/
│── src/ # Core ML pipeline
│── app.py # Streamlit app
│── README.md
│── requirements.txt
│── data/ # (not included)
│── artifacts/ # (not included)
│── data_processed/ # (not included)

---

## 📂 Dataset

Due to GitHub size limitations, the dataset is not included.

You can download it from:
https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring

### 📁 After downloading:
Place it in:
data/
└── pamap2+physical+activity+monitoring/
└── PAMAP2_Dataset/

---

## 🚀 How to Run

### 1️⃣ Clone the repository
### 2️⃣ Install dependencies

pip install -r requirements.txt


### 3️⃣ Run the app

streamlit run app.py

---

## 🧠 Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  
- Matplotlib, Seaborn  

---

## 📊 Model Details

- Trained on wearable sensor features  
- Uses classification model for activity prediction  
- Exertion computed using heart rate + motion intensity  

---

## 🎯 Results

- Achieves good accuracy on activity classification  
- Provides real-time predictions via UI  
- Displays confusion matrix for evaluation  

---

## 📌 Future Improvements

- Real-time sensor integration  
- Deployment on cloud  
- Advanced deep learning models  
- Personalized health insights  

---

## 👤 Author

Vagisha Sharma

---

## ⭐ Note

This project demonstrates an **end-to-end machine learning pipeline** including:
data preprocessing, feature engineering, model training, evaluation, and deployment.