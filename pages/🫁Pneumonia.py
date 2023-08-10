import os
import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model


# Get the absolute path to the model file
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
model_file_path = os.path.join(models_dir, 'chest_xray.h5')
model = load_model(model_file_path)

# Define class names
class_names = ["Normal", "Pneumonia"]

def get_prediction(image):
    image = image.convert('RGB')
    image = image.resize((224, 224))  # Resize the image to match the model's input shape
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)
    predicted_label = np.argmax(result)
    return class_names[predicted_label]

# Main function to run the web application
def main():
    st.title("Pneumonia Prediction")
    st.text("Upload a chest X-ray image to check for pneumonia")

    # Initialize session state dictionary
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.session_state.uploaded_image = image
        
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        prediction = get_prediction(image)
        st.text(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
