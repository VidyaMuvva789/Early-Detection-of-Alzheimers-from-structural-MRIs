import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import base64

# Custom CSS for styling
st.markdown("""
    <style>w
    body {
        background-color: #f5f5f5;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 {
        text-align: center;
        color: #2c3e50;
    }
    .button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
    .centered-image {
        display: flex;
        justify-content: center;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🧠 Alzheimer's Disease Detection")
st.write("Upload an MRI image to predict the stage of Alzheimer's disease.")

# Load Model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('Resnet101_best_weights.keras')
    return model

model = load_model()
IMG_HEIGHT, IMG_WIDTH = 128, 128
class_names = ['Mild', 'Moderate', 'Non-demented', 'Very mild']

# File Uploader
uploaded_file = st.file_uploader("Upload an MRI Image (JPG/PNG)", type=["jpg", "png"])

if uploaded_file is not None:
    # Resize image for display
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))  # Resize to fixed dimensions
    
    # Convert image to bytes and encode in base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    # Display image in center with controlled size
    st.markdown(
        f'<div class="centered-image"><img src="data:image/png;base64,{img_base64}" width="250"/></div>',
        unsafe_allow_html=True
    )

    # Preprocess for model input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class]

    # Display Result
    st.subheader(f"Prediction: **{predicted_class_name}**")
    st.write("Prediction Probabilities:")
    probabilities = {class_names[i]: f"{predictions[0][i]:.2f}" for i in range(len(class_names))}
    st.json(probabilities)
