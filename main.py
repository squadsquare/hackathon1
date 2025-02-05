import streamlit as st
import tensorflow as tf
import numpy as np

# TensorFlow Model Prediction Function
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Custom CSS for the app
st.markdown("""
    <style>
    body {
        background-color: #f4f6f9;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 5px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>input {
        font-size: 18px;
        padding: 10px;
        border-radius: 5px;
        border: 2px solid #ddd;
    }
    .header {
        color: #2c3e50;
        font-size: 36px;
        text-align: center;
        padding: 20px;
        background-color: #ecf0f1;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .about {
        color: #34495e;
        font-size: 18px;
        text-align: justify;
    }
    .result {
        font-size: 22px;
        font-weight: bold;
        color: #2ecc71;
    }
    .card {
        background-color: #fff;
        padding: 20px;
        margin: 10px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.markdown('<div class="header">PLANT DISEASE RECOGNITION SYSTEM</div>', unsafe_allow_html=True)
    image_path = "home_page.jpg"  # Make sure the image path is correct
    st.image(image_path, use_column_width=True)
    st.markdown("""
    <p class="about">
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    </p>
    """, unsafe_allow_html=True)

# About Page
elif app_mode == "About":
    st.markdown('<div class="header">About the Project</div>', unsafe_allow_html=True)
    st.markdown("""
    <p class="about">
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo. This dataset consists of about 87K RGB images of healthy and diseased crop leaves, which are categorized into 38 different classes. The total dataset is divided into an 80/20 ratio for training and validation, preserving the directory structure. A new directory containing 33 test images is created later for prediction purposes.

    #### Content
    - 1. train (70,295 images)
    - 2. test (33 images)
    - 3. validation (17,572 images)
    </p>
    """, unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.markdown('<div class="header">Disease Recognition</div>', unsafe_allow_html=True)
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image:
        st.image(test_image, width=300, use_column_width=True)

    if st.button("Predict"):
        with st.spinner("please wait..."):
            st.write("Our Prediction")
            
            result_index = model_prediction(test_image)
            
            # Disease classes
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy']
            
            remedy_dict = {
                'Apple___Apple_scab': 'Use fungicides like copper sulfate or sulfur-based products. Remove affected leaves.',
                'Apple___Black_rot': 'Remove infected fruits, leaves, and stems. Apply fungicides if necessary.',
                'Apple___Cedar_apple_rust': 'Remove affected leaves and fruits. Apply fungicides to prevent further spread.',
                'Apple___healthy': 'No action needed. Keep the plants well-watered and healthy.',
                # Add other classes here...
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
            
            disease = class_name[result_index]
            remedy = remedy_dict.get(disease, "No remedy information available.")
            
            st.markdown(f"<p class='result'>The plant is diagnosed with: <strong>{disease}</strong></p>", unsafe_allow_html=True)
            st.markdown(f"<p class='result'>Suggested remedy: <strong>{remedy}</strong></p>", unsafe_allow_html=True)

