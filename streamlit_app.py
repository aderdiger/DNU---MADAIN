# importing the necessary libraries
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TFSMLayer
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import time
import tempfile

# loading the trained model
# URL of the model file in the GitHub repository
model_url = 'https://raw.githubusercontent.com/mustafacanayter/MADAIN/main/run3/models/model_InceptionV3_Adam.h5'

# Download the model file from the URL
response = requests.get(model_url)


# # Check if the request was successful
# if response.status_code == 200:
#     # Load the model from the response content
#     model_content = response.content
#     model_file = BytesIO(model_content)
#     model = TFSMLayer(model_file, call_endpoint='serving_default')
#     print("Model loaded successfully!")
# else:
#     print("Failed to load the model from the URL.")

# # Check if the request was successful
# if response.status_code == 200:
#     # Create a temporary file to save the model content
#     with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#         temp_file.write(response.content)
#         temp_file_path = temp_file.name

#     # Load the model from the temporary file path
#     model = TFSMLayer(temp_file_path, call_endpoint='serving_default')
#     print("Model loaded successfully!")
# else:
#     print("Failed to load the model from the URL.")

if response.status_code == 200:
    # Create a temporary file to save the model content
    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name

    # Load the model from the temporary file path
    model = load_model(temp_file_path)
    print("Model loaded successfully!")
else:
    print("Failed to load the model from the URL.")

# defining necessary functions
# 
def preprocess_input(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.resize((299, 299))  # Resize the image to (299, 299)
    img = np.array(img)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def make_prediction(input_data):
    prediction = model.predict(input_data)
    return 'Cancerous' if prediction > 0.5 else 'Benign'

# setting up the app title and description
st.title('MADAIN: Skin Cancer Prediction App')
st.write('This app predicts whether a mole is likely to be cancerous or benign based on an uploaded image.')

# creating a file uploader for the user to input an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # preprocessing the input image
    input_data = preprocess_input(uploaded_file)
    
    # making a prediction
    prediction = make_prediction(input_data)
    
    # displaying the prediction
    st.write('Prediction:', prediction)