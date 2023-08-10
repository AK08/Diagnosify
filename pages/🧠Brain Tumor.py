import os
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Get the absolute path to the model file
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
model_file_path = os.path.join(models_dir, 'BrainTumor10EpochsCategorical.h5')

# Load the model
model = load_model(model_file_path)

# Define the function to get class names
def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"

# Define the function to get the result from the model
def get_prediction(image):
    image = image.convert('RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)
    class_index = np.argmax(result)
    return class_index

# Main function to run the web application
def main():
    st.title("Brain Tumor Prediction")
    st.text("Upload an MRI scan image to check for brain tumor")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        value = get_prediction(image)
        result = get_className(value)
        
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.text(f"Prediction: {result}")

if __name__ == "__main__":
    main()
