import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import os

def main():
    st.title("Skin Cancer Prediction")
    st.write("Upload an image to predict whether it's benign or malignant.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Read the uploaded image using OpenCV
        img = Image.open(uploaded_file)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Display the uploaded image
        st.image(img, caption='Uploaded Image', use_column_width=True)
        
        # Get the absolute path to the model file
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
        model_file_path = os.path.join(models_dir, 'cnn_model_skin1.h5')
        
        # Load the model
        model = load_model(model_file_path)
        
        # Make prediction
        prediction = model.predict(x)
        predicted_class = np.argmax(prediction)
        
        class_names = {0: "Benign", 1: "Malignant"}
        
        st.write("Predicted Class:", class_names[predicted_class])
        st.write("Confidence:", prediction[0][predicted_class])

if __name__ == '__main__':
    main()
