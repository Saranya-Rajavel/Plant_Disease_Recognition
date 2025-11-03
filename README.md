# 🌿 Plant Disease Detection using Deep Learning

This project is an **AI-powered system** that detects common **plant leaf diseases** using **Convolutional Neural Networks (CNN)**.  
The goal of this project is to help farmers **identify and treat plant diseases early**, ensuring healthier crops and improved yield.

---

## 🚀 Project Overview

The system takes an image of a plant leaf as input and predicts whether the leaf is **Healthy** or infected by a disease such as:
- **Early Blight**
- **Late Blight**
- **Leaf Mold or Rust**
- **Healthy**

The dataset was **manually created** by collecting images from **Google Images**, and the model was trained using **Google Colab**.  
Once trained, the model (`plant_disease_model.h5`) was integrated into a **Streamlit**-based web interface for real-time prediction.

---

## 🧠 Features

- 📸 Upload a plant leaf image for disease detection  
- 🧩 CNN model trained from scratch on a manually collected dataset  
- 📊 Displays prediction with confidence level  
- 💡 Provides disease information and preventive suggestions  
- 🎨 Modern and clean UI built using Streamlit  
- ⚙️ Runs locally on VS Code or deployable on Streamlit Cloud  

---

## 🧾 Dataset and Model

- 📁 **Dataset:** Custom dataset collected manually from Google Images  
  👉 [Download Dataset (Google Drive)](https://drive.google.com/drive/folders/1sqCTZYVZhfOdafvTCWhDYUj_2-lSJbnY?usp=sharing)

- 🧠 **Trained Model (.h5):**  
  👉 [Download Model File](https://drive.google.com/file/d/1LCh-VVTCQJmoN1206-SoHLW7dkm8KrIa/view?usp=sharing)

---

## 🛠️ Technologies Used

| Category | Technology |
|-----------|-------------|
| **Model Training** | TensorFlow, Keras, NumPy, Google Colab |
| **Frontend** | Streamlit |
| **Backend** | Python |
| **Visualization** | Matplotlib |
| **Image Processing** | Pillow (PIL) |

---

## 🧩 Folder Structure

Plant_Disease_Detection/
│
├── app.py                           
├── README.md                      
├── requirements.txt                
│
├── model/
│   └── plant_disease_model.h5      
│
├── dataset/                        
│   ├── Early Blight/                
│   ├── Late Blight/               
│   ├── Leaf Mold or Rust/         
│   └── Healthy/                     



