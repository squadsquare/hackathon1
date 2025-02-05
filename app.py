import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Remedy Dictionary (Same as before)
remedy_dict = {
    'Apple___Apple_scab': 'Use fungicides like copper sulfate or sulfur-based products. Remove affected leaves.',
    'Apple___Black_rot': 'Remove infected fruits, leaves, and stems. Apply fungicides if necessary.',
    'Apple___Cedar_apple_rust': 'Remove affected leaves and fruits. Apply fungicides to prevent further spread.',
    'Apple___healthy': 'No action needed. Keep the plants well-watered and healthy.',
    'Blueberry___healthy': 'No action needed. Ensure well-drained soil and proper sunlight for the plants.',
    'Cherry_(including_sour)___Powdery_mildew': 'Use fungicides like sulfur-based products. Remove affected leaves.',
    'Cherry_(including_sour)___healthy': 'No action needed. Keep the plants healthy and well-maintained.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Apply fungicides containing chlorothalonil or copper. Remove infected leaves.',
    'Corn_(maize)___Common_rust_': 'Use fungicides to treat, remove infected plants to limit spread.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Remove infected leaves and use fungicides like azoxystrobin.',
    'Corn_(maize)___healthy': 'No action needed. Ensure proper watering and sunlight for optimal growth.',
    'Grape___Black_rot': 'Use copper-based fungicides and remove infected leaves and fruits.',
    'Grape___Esca_(Black_Measles)': 'Prune infected vines, ensure good air circulation, and apply appropriate fungicides.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Remove affected leaves and apply fungicides like myclobutanil.',
    'Grape___healthy': 'No action needed. Ensure proper vineyard management and good airflow around the vines.',
    'Orange___Haunglongbing_(Citrus_greening)': 'Use insecticides to control the vector (Asian citrus psyllid). Remove infected trees to prevent spread.',
    'Peach___Bacterial_spot': 'Apply copper-based bactericides. Remove infected fruit and leaves.',
    'Peach___healthy': 'No action needed. Ensure the plant gets adequate sunlight and water.',
    'Pepper,_bell___Bacterial_spot': 'Apply copper-based bactericides. Avoid overhead watering to reduce the spread.',
    'Pepper,_bell___healthy': 'No action needed. Maintain healthy growing conditions with balanced watering and sunlight.',
    'Potato___Early_blight': 'Apply fungicides containing chlorothalonil or mancozeb. Rotate crops to avoid recurrence.',
    'Potato___Late_blight': 'Apply fungicides like copper or mefenoxam. Remove infected plants to limit spread.',
    'Potato___healthy': 'No action needed. Ensure proper irrigation and soil drainage.',
    'Raspberry___healthy': 'No action needed. Keep the plants well-watered and avoid overcrowding.',
    'Soybean___healthy': 'No action needed. Ensure soil health and proper irrigation.',
    'Squash___Powdery_mildew': 'Use fungicides like sulfur or potassium bicarbonate. Remove affected leaves.',
    'Strawberry___Leaf_scorch': 'Remove affected leaves. Apply fungicides if necessary.',
    'Strawberry___healthy': 'No action needed. Ensure proper care and avoid excessive watering.',
    'Tomato___Bacterial_spot': 'Apply copper-based bactericides. Avoid overhead watering to reduce infection.',
    'Tomato___Early_blight': 'Apply fungicides like chlorothalonil. Remove affected leaves to stop the spread.',
    'Tomato___Late_blight': 'Apply fungicides containing copper or mefenoxam. Remove infected plants immediately.',
    'Tomato___Leaf_Mold': 'Use fungicides like chlorothalonil or myclobutanil. Improve air circulation around plants.',
    'Tomato___Septoria_leaf_spot': 'Use fungicides like chlorothalonil and remove affected leaves.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Use miticides or insecticidal soaps to control mites.',
    'Tomato___Target_Spot': 'Use fungicides containing azoxystrobin or chlorothalonil. Remove infected leaves.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Remove infected plants immediately. Insecticides can control the vector (whiteflies).',
    'Tomato___Tomato_mosaic_virus': 'Remove and destroy infected plants. Use virus-free seeds and control aphid populations.',
    'Tomato___healthy': 'No action needed. Keep the plants well-maintained and properly irrigated.'
}

# TensorFlow Model Prediction
def predict_disease(image_path):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_batch = np.expand_dims(image_array, axis=0)  # Convert single image to batch
    predictions = model.predict(image_batch)
    return np.argmax(predictions)  # Return the index of the class with highest probability

# Custom CSS for UI styling
st.markdown(
    """
    <style>
        .sidebar .sidebar-content {
            background-color: #F1F8E9;
            padding: 20px;
        }
        .header {
            font-size: 24px;
            color: #2C6E49;
            font-weight: bold;
            text-align: center;
        }
        .subheader {
            font-size: 20px;
            color: #4CAF50;
            text-align: center;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            width: 200px;
            height: 50px;
        }
        .stImage {
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

# Streamlit Sidebar with a modern layout
st.sidebar.title("Plant Disease Recognition")
app_mode = st.sidebar.selectbox("Choose a Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.markdown('<h1 class="header">Welcome to the Plant Disease Recognition System!</h1>', unsafe_allow_html=True)
    st.image("home_page.jpg", use_column_width=True)
    st.markdown("""
    **Mission:** Protect crops by identifying plant diseases with AI. Upload an image of a plant leaf, and we’ll provide the diagnosis and a potential remedy.
    
    **How it Works:**
    1. Go to **Disease Recognition** to upload an image.
    2. Our AI model analyzes the image to identify potential diseases.
    3. View the result and suggested remedies to help your plant recover.

    **Why Choose Us?**
    - Quick and accurate diagnosis.
    - Easy-to-use interface for everyone.
    - Helps preserve the health of your plants!

    Click **Disease Recognition** to get started.
    """)

# About Page
elif app_mode == "About":
    st.markdown('<h1 class="subheader">About the Project</h1>', unsafe_allow_html=True)
    st.markdown("""
    This system leverages deep learning models to identify plant diseases from images. The model was trained on over 87,000 images of healthy and diseased plant leaves.
    **Key Features:**
    - Works with 38 different plant disease classes.
    - Provides suggestions for remedies based on disease detection.
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.markdown('<h1 class="subheader">Disease Recognition</h1>', unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Upload an Image of Plant Leaf", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.write("Processing...")

        # Save the uploaded image
        img = Image.open(uploaded_image)
        img.save("temp_image.jpg")

        # Prediction
        predicted_class_index = predict_disease("temp_image.jpg")
        
        # Class names (Mapping index to disease name)
        class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
                       'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
                       'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
                       'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
                       'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
                       'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
                       'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 
                       'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                       'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 
                       'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                       'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
                       'Tomato___healthy']
        
        predicted_class = class_names[predicted_class_index]
        
        # Display the result
        st.success(f"The plant is affected by: **{predicted_class}**")
        
        # Provide Remedy
        remedy = remedy_dict.get(predicted_class, "No remedy found. Please consult an expert.")
        st.markdown(f"**Suggested Remedy:** {remedy}")
