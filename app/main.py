# Import necessary libraries
import os
import json
from PIL import Image  # Python Imaging Library for image manipulation
import numpy as np
import tensorflow as tf
import streamlit as st  # Web app framework

# Set working directory and model path
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"

# Load the pre-trained Keras model
model = tf.keras.models.load_model(model_path)

# Load the class names/labels from JSON file
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image from path
    img = Image.open(image_path)
    
    # Resize the image to the target size required by the model
    img = img.resize(target_size)
    
    # Convert the image to a numpy array for model input
    img_array = np.array(img)
    
    # Add batch dimension (model expects batch input)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values to range [0, 1]
    img_array = img_array.astype('float32') / 255.
    
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    # Preprocess the input image
    preprocessed_img = load_and_preprocess_image(image_path)
    
    # Get model predictions
    predictions = model.predict(preprocessed_img)
    
    # Get the index of the highest probability class
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Convert index to class name using the class_indices dictionary
    predicted_class_name = class_indices[str(predicted_class_index)]
    
    return predicted_class_name

# Begin Streamlit App interface
st.title('Plant Disease Classifier')

# Create a file uploader widget for images
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# Process the image if one is uploaded
if uploaded_image is not None:
    # Open the uploaded image
    image = Image.open(uploaded_image)
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    # Display the image in the first column
    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)
    
    # Add classification button and results in the second column
    with col2:
        if st.button('Classify'):
            # Get prediction when button is clicked
            prediction = predict_image_class(model, uploaded_image, class_indices)
            
            # Display the prediction result
            st.success(f'Prediction: {str(prediction)}')