# 📩 SMS Spam Detector

A **machine learning–powered web application** that classifies SMS messages as **Spam** or **Ham (Not Spam)**.  
Built using **Python**, **scikit-learn**, and **Streamlit**, and trained on a real-world dataset from Kaggle.

🚀 **Live App:**  
👉 https://smsspamdetector-rddswm7fxetzc2g9b7xzce.streamlit.app/

---

## 🧠 Project Overview

Spam messages are a real problem—annoying at best, dangerous at worst.  
This project applies **Natural Language Processing (NLP)** and **Machine Learning** to automatically detect spam SMS messages.

The model is trained on labeled SMS data and deployed as an interactive web app where users can test messages in real time.

---

## 📊 Dataset

- **Source:** Kaggle  
- **Dataset Name:** SMS Spam Collection Dataset  
- **Link:** https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset  
- **Size:** 5,572 SMS messages  
- **Classes:** `spam`, `ham`

---

## ⚙️ Tech Stack

- **Language:** Python 🐍  
- **Machine Learning:** scikit-learn  
- **NLP:** TF-IDF Vectorization  
- **Web Framework:** Streamlit  
- **Model Serialization:** Pickle  
- **Deployment:** Streamlit Cloud  

---

## 🧪 Model Pipeline

1. Text Cleaning & Preprocessing  
2. TF-IDF Vectorization  
3. Supervised Classification  
4. Model Evaluation  
5. Model & Vectorizer Serialization  

Saved files:
- `model.pkl` → trained ML model  
- `vectorizer.pkl` → TF-IDF vectorizer  

---

## 🗂️ Repository Structure

```text
SMS_Spam_Detector/
│
├── Data/                      # Dataset files
├── app.py                     # Streamlit web app
├── sms_spam_detection.ipynb   # Model training & analysis
├── model.pkl                  # Trained ML model
├── vectorizer.pkl             # TF-IDF vectorizer
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
└── LICENSE                    # License file
