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

def main():
    st.title("Brain Tumor Prediction")
    st.text("Upload an MRI scan image to check for brain tumor")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        prediction = get_prediction(image)
        class_name = get_className(prediction)
        
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.text(f"##### Prediction: {class_name}")
        
        if class_name == "Yes Brain Tumor":  # Assuming the class name indicating brain tumor is "Yes Brain Tumor"
            # Draw a line under the prediction result
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            # Display detailed information about Brain Tumor
            st.markdown("## Brain Tumor Information")
            st.write("**📌 Common Name:** Brain Tumor")
            st.write("**🌐 General Overview:** A brain tumor is an abnormal growth of cells within the brain.")
            
            # Draw a line under the general overview section
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Symptoms")
            st.write("🧠 Symptoms of Brain Tumor include:")
            st.markdown("- Headaches.")
            st.markdown("- Seizures.")
            st.markdown("- Cognitive and memory problems.")
            
            # Draw a line under the symptoms section
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Diagnosis")
            st.write("🩺 Diagnosis involves:")
            st.markdown("- Imaging tests such as MRI and CT scans.")
            st.markdown("- Biopsy for confirming the tumor type.")
            
            # Draw a line under the diagnosis section
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Treatment")
            st.write("💼 Treatment options depend on:")
            st.markdown("- Tumor type, size, and location.")
            st.markdown("- They may include surgery, radiation therapy, chemotherapy, and targeted therapy.")
            
            # Draw a line under the treatment section
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Prognosis")
            st.write("⏳ Prognosis varies based on:")
            st.markdown("- Tumor type and stage.")
            st.markdown("- Treatment response.")
            
            # Draw a line under the prognosis section
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Living with a Brain Tumor")
            st.write("🤝 Managing life with a brain tumor:")
            st.markdown("- Regular medical checkups.")
            st.markdown("- Support groups and counseling.")
            
            # Draw a line under the living section
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### More Information")
            st.write("📚 For more detailed information about Brain Tumors, you can visit reputable medical websites or consult with a healthcare professional.")

if __name__ == "__main__":
    main()