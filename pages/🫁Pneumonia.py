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
        st.write(f"##### Prediction: {prediction}")
        
        if prediction == "Pneumonia":
            # Draw a line under the prediction result
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            # Display detailed information about Pneumonia
            st.markdown("## Pneumonia Information")
            st.write("**📌 Common Name:** Pneumonia")
            st.write("**🌐 General Overview:** Pneumonia is an inflammatory condition affecting the air sacs in one or both lungs.")
            
            # Draw a line under the general overview section
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Symptoms")
            st.write("🫁 Symptoms of Pneumonia include:")
            st.markdown("- Cough, often with mucus.")
            st.markdown("- Fever, sweating, and chills.")
            st.markdown("- Shortness of breath and rapid breathing.")
            
            # Draw a line under the symptoms section
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Causes")
            st.write("🔍 Pneumonia can be caused by various factors:")
            st.markdown("- Bacterial, viral, or fungal infections.")
            st.markdown("- Inhaling irritants into the lungs.")
            
            # Draw a line under the causes section
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Diagnosis")
            st.write("🩺 Diagnosis involves:")
            st.markdown("- Chest X-ray, blood tests, and physical examination.")
            st.markdown("- Identification of the underlying cause of infection.")
            
            # Draw a line under the diagnosis section
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Treatment")
            st.write("💼 Treatment options depend on:")
            st.markdown("- The type of pneumonia and its severity.")
            st.markdown("- Antibiotics, antiviral medications, or antifungal drugs.")
            
            # Draw a line under the treatment section
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Prevention")
            st.write("🌴 Preventing pneumonia involves:")
            st.markdown("- Vaccination (for certain types of pneumonia).")
            st.markdown("- Good hygiene and avoiding smoking.")
            
            # Draw a line under the prevention section
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### More Information")
            st.write("📚 For more detailed information about Pneumonia, you can visit reputable medical websites or consult with a healthcare professional.")

if __name__ == "__main__":
    main()
