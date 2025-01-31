import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Function to make predictions
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Color Palette
primary_color = "#f875ed"  # Green
accent_color = "#F4A300"   # Vibrant Yellow
background_color = "linear-gradient(135deg,rgb(32, 76, 199),rgb(18, 144, 16))"  # Green-brown gradient
button_color = "linear-gradient(135deg,rgb(14, 67, 171),rgb(242, 144, 234))"  # Deep green button
header_color = "#88C0D0"  # Soft blue (for headers)

# Custom CSS for background gradient and text styling
st.markdown(
    f"""
    <style>
        .stApp {{
            background: {background_color};
            background-attachment: fixed;
            color: white;
        }}
        h1, h2, h3 {{
            color: {primary_color};
        }}
        div.stButton > button {{
        background-color: #245315; /* Deep Green */
        color: black !important; /* Black text */
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        font-weight: bold;
    }}
    div.stButton > button:hover {{
        background-color: #2D6A4F; /* Slightly lighter green */
        color: black !important;
    }}
    </style>
    """, unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("🌱 Plant Disease Detection System 🌿")
app_mode = st.sidebar.radio("Select Page", ["HOME", "DISEASE RECOGNITION"], index=0, label_visibility="collapsed")

# Displaying an image for the main page
img = Image.open("Diseases.png")
st.sidebar.image(img, use_container_width=True)

# Main Page
if app_mode == "HOME":
    st.title("Welcome to the Plant Disease Detection System")
    st.markdown(f"<h3 style='color:{primary_color}; text-align: center;'>For Sustainable Agriculture 🌍</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    ### 🌿 Promote Healthy Farming Practices
    Use this tool to detect plant diseases quickly. Upload images of your plants, and our model will predict the potential disease.
    """)

    st.image("https://plus.unsplash.com/premium_photo-1668096747228-c32252e8943f?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8OXx8cGxhbnQlMjB3YWxscGFwZXJ8ZW58MHx8MHx8fDA%3D", caption="Upload an image of your plant for disease detection.", use_container_width=True)

# Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    st.title("Plant Disease Recognition 🌱")
    st.markdown(f"<h3 style='color:{header_color}; text-align: center;'>Upload Your Plant Image Below 🌾</h3>", unsafe_allow_html=True)

    # File uploader to upload an image
    test_image = st.file_uploader("Choose an Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    if test_image:
        st.image(test_image, caption="Uploaded Image", use_container_width=True)
    
    # Predict button with a loading spinner
    if st.button("Predict Disease", help="Click to get disease prediction", key="predict_button"):
        if test_image:
            with st.spinner("Analyzing..."):
                result_index = model_prediction(test_image)
                # Reading Labels
                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
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
                    'Tomato___healthy'
                ]
                st.success(f"Prediction: {class_name[result_index]}")
        else:
            st.warning("Please upload an image first.")