import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 🌿 Page Config
st.set_page_config(
    page_title="Plant Disease Detector 🌱",
    page_icon="🍃",
    layout="wide"
)

# 🌈 Custom CSS (Modern + Attractive)
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #e0f7fa 0%, #f1f8e9 100%);
        color: #333;
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 40px;
        margin-top: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    }
    h1 {
        text-align: center;
        color: #2e7d32;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4caf50, #81c784);
        border: none;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        transition: all 0.3s ease;
        padding: 10px 25px;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #388e3c, #66bb6a);
        transform: scale(1.05);
    }
    .result-box {
        background: rgba(46,125,50,0.1);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 2px solid #2e7d32;
        color: #2e7d32;
        font-weight: 600;
        box-shadow: 0 0 15px rgba(46,125,50,0.2);
    }
    footer {
        text-align: center;
        margin-top: 30px;
        font-size: 13px;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

# 🌱 Sidebar
st.sidebar.title("🌿 About Project")
st.sidebar.markdown("""
### 📘 Plant Disease Detection
Detects common plant leaf diseases using **Deep Learning (CNN)**.  
Upload an image of a plant leaf and get instant predictions with confidence levels.

---

**👩‍💻 Developer:** Saranya R  
**🧠 Technology:** TensorFlow | Streamlit | Python  
**🎯 Goal:** Early detection of plant diseases for sustainable farming 🌾  

---
📩 **Contact:** 727823tuam045@skct.edu.in  
""")

# 🧠 Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model/plant_disease_model.h5')

model = load_model()

# Class Names
class_names = ['Early Blight', 'Leaf Mold or Rust', 'Late Blight', 'Healthy']

# 🌿 Title
st.markdown("<h1>🍃 Smart Plant Disease Detection System</h1>", unsafe_allow_html=True)
st.write("Upload a **plant leaf image** below to detect the disease and get treatment suggestions instantly.")

# 📁 File Upload
uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="🌿 Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Analyzing the leaf..."):
        # Image preprocessing
        img = img.resize((128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    # 🌾 Display Results
    st.markdown(f"""
        <div class="result-box">
            ✅ Predicted Disease: <b>{predicted_class}</b><br>
            📊 Confidence: <b>{confidence:.2f}%</b>
        </div>
    """, unsafe_allow_html=True)

    # 🩺 Disease Info
    disease_info = {
        "Early Blight": "🦠 Caused by *Alternaria* fungus. Use fungicides and rotate crops.",
        "Leaf Mold or Rust": "🍂 Caused by fungal infection. Improve ventilation and use organic sprays.",
        "Late Blight": "💧 Caused by *Phytophthora infestans*. Avoid wet conditions and use resistant varieties.",
        "Healthy": "🌱 The plant looks healthy! Maintain regular watering and soil nutrients."
    }

    st.subheader("🩺 Disease Information")
    st.info(disease_info.get(predicted_class, "No information available."))

    # 📈 Confidence Graph
    st.subheader("📊 Model Confidence Levels")
    st.bar_chart(prediction[0])

else:
    st.warning("📤 Please upload a plant leaf image to begin detection.")

# 🌿 Footer
st.markdown("""
<footer>
    © 2025 Plant Disease Detection | Designed by <b>Saranya R</b> 💚
</footer>
""", unsafe_allow_html=True)
