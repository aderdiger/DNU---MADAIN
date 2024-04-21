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
import json
import streamlit.components.v1 as components

components.html(
    """
    <script>
        function makePrediction() {
            // Get the uploaded file from the file input element
            const fileInput = document.getElementById('uploaded_file');
            const uploadedFile = fileInput.files[0];

            // Create a FormData object to send the file to the server
            const formData = new FormData();
            formData.append('uploaded_file', uploadedFile);

            // Make a POST request to the Streamlit backend
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Get the prediction from the response
                const prediction = data.prediction;

                // Display the prediction in the Streamlit app
                Streamlit.setComponentValue(`Prediction: ${prediction}`);
            })
            .catch(error => {
                console.error('Error:', error);
            });
        }
    </script>
    """,
    height=0,
)

# loading the trained model
# URL of the model file in the GitHub repository
model_url = 'https://raw.githubusercontent.com/mustafacanayter/MADAIN/main/run3/models/model_InceptionV3_Adam.h5'

# Download the model file from the URL
response = requests.get(model_url)

@st.cache(allow_output_mutation=True)
def predict_handler(uploaded_file):
    input_data = preprocess_input(uploaded_file)
    prediction = make_prediction(input_data)
    return json.dumps({"prediction": prediction})

from streamlit.server.server import Server

@Server.route("/predict", methods=["POST"])
def predict_route():
    uploaded_file = st.request.files.get("uploaded_file")
    if uploaded_file is not None:
        prediction = predict_handler(uploaded_file)
        return prediction
    return json.dumps({"error": "No file uploaded"})

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

def preprocess_input(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.resize((299, 299))  # Resize the image to (299, 299)
    img = np.array(img)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def make_prediction(input_data):
    prediction = model.predict(input_data)
    return float(prediction[0][0])

# setting up the app title and description
st.title('MADAIN: Skin Cancer Prediction App')
st.write('This app predicts whether a mole is likely to be cancerous or benign based on an uploaded image.')

# creating a file uploader for the user to input an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])