import os
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from keras.models import load_model
from werkzeug.utils import secure_filename
from PIL import Image

# Get the absolute path to the model file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(current_dir, 'BrainTumor10EpochsCategorical.h5')

# Define the function to get class names
def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"

# Define the function to get the result from the model
def getResult(img):
    image = Image.open(img)
    image = image.convert('RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)
    class_index = np.argmax(result)
    return class_index

# Main function to run the web application
def main():
    st.title("Brain Tumor Image Classification")
    st.text("Upload an MRI scan image to check for brain tumor")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        file_path = os.path.join('uploads', secure_filename(uploaded_file.name))
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        value = getResult(file_path)
        result = get_className(value)
        st.text(f"Prediction: {result}")

if __name__ == "__main__":
    main()
