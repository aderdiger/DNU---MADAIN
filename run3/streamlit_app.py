# importing the necessary libraries
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# loading the trained model
model = load_model('models/model_InceptionV3_Adam.h5') 

# defining necessary functions
def preprocess_input(uploaded_file):
    img = Image.open(uploaded_file)
    # assuming model expects 224x224 images
    img = img.resize((224, 224))  
    # normalizing pixel values to [0, 1]
    img = np.array(img) / 255.0  
    # assuming model expects RGB images
    img = img.reshape(1, 224, 224, 3)  
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