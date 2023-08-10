import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

def predict_label(img_path):
    test_image = Image.open(img_path).convert("L")
    test_image = test_image.resize((128, 128))
    test_image = np.array(test_image) / 255.0
    test_image = test_image.reshape(-1, 128, 128, 1)

    
    verbose_name = {
        0: "Non Demented",
        1: "Very Mild Demented",
        2: "Mild Demented",
        3: "Moderate Demented",
    }

    # Get the absolute path to the model file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_file_path = os.path.join(current_dir, 'alzheimer_cnn_model.h5')
    model = load_model(model_file_path)

    predict_x = model.predict(test_image)
    classes_x = np.argmax(predict_x, axis=1)

    return verbose_name[classes_x[0]]

def main():
    st.title("Alzheimer's Disease Prediction")
    st.write("Upload an MRI scan image for prediction")

    uploaded_file = st.file_uploader("Choose an MRI scan image...", type=["jpg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded MRI scan.', use_column_width=True)
        st.write("")
        st.write("Predicted Label:")
        
        prediction = predict_label(uploaded_file)
        st.write(prediction)

if __name__ == "__main__":
    main()
