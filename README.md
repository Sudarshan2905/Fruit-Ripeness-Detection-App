# 🍌 Fruit Ripeness Detection – ML + Streamlit

A simple and accurate **Fruit Ripeness Detection** web application built using **Machine Learning**, **OpenCV**, **LBP texture features**, and **Streamlit**.  
This app predicts whether a fruit (banana) is:

- 🟢 Unripe  
- 🟡 Ripe  
- 🟠 Overripe  
- 🔴 Rotten  

It uses a trained **Random Forest Classifier** on a banana ripeness dataset.

---

## 🚀 Features

- Upload any fruit image (banana)
- Automatic feature extraction (HSV + LBP)
- ML model prediction with confidence %
- Clean and fast Streamlit user interface
- Works entirely locally or on Streamlit Cloud

---

## 🧠 Tech Stack

- **Python**
- **Streamlit**
- **OpenCV (HSV color histogram)**
- **LBP (Local Binary Pattern)**
- **Random Forest Classifier**
- **Joblib**

---

## 📂 Project Structure

```
Fruit-Ripeness-Detection/
│── app.py
│── rf_ripeness_model.joblib
│── label_encoder.joblib
│── requirements.txt
│── README.md
```

> ⚠️ Dataset is **not required** for deployment.  
> The model is already trained and saved as `.joblib`.

---

## 🔧 Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 🪄 How It Works

1. Image is resized to **128×128**
2. Extract HSV color histogram features  
3. Extract LBP texture features  
4. Combine features into a single vector  
5. Apply trained Random Forest model  
6. Show predicted ripeness + confidence  

---

## 📸 Demo

Upload a fruit image and get instant ripeness result!

---

## 🌐 Deployment (Streamlit Cloud)

1. Push code to GitHub  
2. Create new app on Streamlit Cloud  
3. Select your repo  
4. Choose `app.py` as the main file  
5. Deploy 🎉  

---

## 📜 License

This project is for educational and research purposes.

---

## 👨‍💻 Developed By

Shree  
3rd Year Electronics Engineering Student  
Aspiring Software Engineer | MERN | AWS | ML | DSA
