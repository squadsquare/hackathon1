import streamlit as st
import tensorflow as tf
import numpy as np
import os
import urllib.request

# GitHub Model URL
MODEL_URL = "https://github.com/squadsquare/hackathon1/raw/main/trained_plant_disease_model.keras"
MODEL_PATH = "trained_plant_disease_model.keras"

# Function to download model from GitHub if missing
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... This may take a moment."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# TensorFlow Model Prediction Function
def model_prediction(test_image):
    download_model()  # Ensure model is downloaded before loading
    model = tf.keras.models.load_model(MODEL_PATH)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Custom CSS for UI Styling
st.markdown("""
    <style>
    .header { color: #2c3e50; font-size: 36px; text-align: center; padding: 20px; background-color: #ecf0f1; border-radius: 10px; }
    .about { color: #34495e; font-size: 18px; text-align: justify; }
    .result { font-size: 22px; font-weight: bold; color: #2ecc71; }
    .card { background-color: #fff; padding: 20px; margin: 10px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.markdown('<div class="header">PLANT DISEASE RECOGNITION SYSTEM</div>', unsafe_allow_html=True)
    st.image("home_page.jpg", use_container_width=True)
    st.markdown("""
    <p class="about">
    🌿 Welcome to the **Plant Disease Recognition System!**  
    Upload an image of a plant, and our AI model will analyze it to detect any disease signs. Let's protect our crops and ensure a healthy harvest!
    </p>
    """, unsafe_allow_html=True)

# About Page
elif app_mode == "About":
    st.markdown('<div class="header">About the Project</div>', unsafe_allow_html=True)
    st.markdown("""
    <p class="about">
    This project helps identify plant diseases using deep learning. Our dataset consists of 87K images across 38 disease classes.  
    - **Train set:** 70,295 images  
    - **Validation set:** 17,572 images  
    - **Test set:** 33 images  
    </p>
    """, unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.markdown('<div class="header">Disease Recognition</div>', unsafe_allow_html=True)
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image:
        st.image(test_image, width=300, use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Processing..."):
            result_index = model_prediction(test_image)

            # Disease classes
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]

            # Remedies
            remedy_dict = {
                'Apple___Apple_scab': 'Use fungicides like copper sulfate or sulfur-based products. Remove affected leaves.',
                'Apple___Black_rot': 'Remove infected fruits, leaves, and stems. Apply fungicides if necessary.',
                'Apple___Cedar_apple_rust': 'Remove affected leaves and fruits. Apply fungicides.',
                'Apple___healthy': 'No action needed. Keep the plants well-watered.',
                'Grape___Black_rot': 'Use copper-based fungicides and remove infected leaves and fruits.',
                'Grape___Esca_(Black_Measles)': 'Prune infected vines, ensure good air circulation, and apply fungicides.',
                'Orange___Haunglongbing_(Citrus_greening)': 'Use insecticides to control the vector (Asian citrus psyllid). Remove infected trees.',
                'Peach___Bacterial_spot': 'Apply copper-based bactericides. Remove infected fruit and leaves.',
                'Potato___Early_blight': 'Apply fungicides containing chlorothalonil or mancozeb. Rotate crops.',
                'Potato___Late_blight': 'Apply fungicides like copper or mefenoxam. Remove infected plants.',
                'Tomato___Bacterial_spot': 'Apply copper-based bactericides. Avoid overhead watering.',
                'Tomato___Early_blight': 'Apply fungicides like chlorothalonil. Remove affected leaves.',
                'Tomato___Late_blight': 'Apply fungicides containing copper or mefenoxam. Remove infected plants.',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Remove infected plants. Insecticides can control the vector (whiteflies).',
                'Tomato___Tomato_mosaic_virus': 'Remove and destroy infected plants. Use virus-free seeds.',
            }

            disease = class_name[result_index]
            remedy = remedy_dict.get(disease, "No remedy information available.")

            st.markdown(f"<p class='result'>**Prediction: {disease}**</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='result'>**Remedy:** {remedy}</p>", unsafe_allow_html=True)
