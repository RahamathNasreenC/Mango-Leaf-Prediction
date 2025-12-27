# app_transfer.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# -----------------------
# 1️⃣ Load trained model
# -----------------------
model = tf.keras.models.load_model('models/leaf_model_transfer.h5')

# -----------------------
# 2️⃣ Load class names from training folder
# -----------------------
train_dir = 'data/train'
folders = sorted(os.listdir(train_dir))  # alphabetical order
class_names = folders  # matches model indices

st.title("🌿 Healthy vs Diseased Leaf Classifier (Transfer Learning)")
st.write("Upload a leaf image to classify if it’s healthy or diseased.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open image and convert to RGB
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Leaf", use_container_width=True)

        # Preprocess image exactly as training
        img = img.resize((128,128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # normalization

        # Predict
        prediction_prob = model.predict(img_array)[0][0]

        # Map prediction to label
        if prediction_prob > 0.5:
            label = class_names[1]  # index 1 = 'healthy' if sorted alphabetically
            confidence = prediction_prob
        else:
            label = class_names[0]  # index 0 = 'diseased'
            confidence = 1 - prediction_prob

        confidence_pct = confidence * 100

        # Display results
        st.markdown(f"### 🔍 Prediction: {label}")
        st.markdown(f"### 📊 Confidence: {confidence_pct:.2f}%")
        st.write("Raw model output (sigmoid probability):", prediction_prob)

    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")
