import streamlit as st
from PIL import Image
import numpy as np
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

    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    model_file_path = os.path.join(models_dir, 'alzheimer_cnn_model.h5')
    
    if os.path.exists(model_file_path):
        model = load_model(model_file_path)
    else:
        st.write("Model file not found:", model_file_path)
        st.stop()

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
        prediction = predict_label(uploaded_file)
       
        st.write("##### Predicted Label: ",prediction)
        
        
        
       
        
        # Display information about Alzheimer's Disease if predicted as "Demented" or "Very Mild Demented"
        if prediction in ["Very Mild Demented", "Mild Demented", "Moderate Demented"]:
            
             # Draw a line under the predicted label
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            st.markdown("## Alzheimer's Disease Information")
            st.write("**📌 Common Name:** Alzheimer's Disease")
            st.write("**🌐 General Overview:** Alzheimer's Disease is a progressive neurodegenerative disorder that primarily affects memory and cognitive function.")
            
            # Draw a line under the general overview section
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Symptoms")
            st.write("🧠 Symptoms of Alzheimer's Disease can vary but often include:")
            st.markdown("- Memory loss.")
            st.markdown("- Difficulty in problem-solving and performing familiar tasks.")
            st.markdown("- Confusion, disorientation, and trouble understanding time and place.")
            st.markdown("- Trouble with language, communication, and visual perception.")
            st.markdown("- Changes in mood, behavior, and personality.")
            
            # Draw a line under the symptoms section
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Causes")
            st.write("🔍 The exact cause of Alzheimer's Disease is still under research.")
            st.markdown("- Genetic and environmental factors may play a role.")
            st.markdown("- Abnormal protein deposits, such as amyloid plaques and tau tangles, are observed in the brains of individuals with Alzheimer's.")
            
            # Draw a line under the causes section
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Diagnosis")
            st.write("🩺 Diagnosis involves a comprehensive assessment:")
            st.markdown("- Cognitive tests to evaluate memory, thinking, and problem-solving abilities.")
            st.markdown("- Brain imaging, such as MRI and PET scans, to detect brain changes.")
            st.markdown("- Medical history and physical examination.")
            
            # Draw a line under the diagnosis section
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Treatment")
            st.write("💼 Currently, there is no cure for Alzheimer's Disease, but treatments aim to:")
            st.markdown("- Manage symptoms and improve quality of life.")
            st.markdown("- Medications may help temporarily improve cognitive function and manage behavioral symptoms.")
            st.markdown("- Non-drug interventions, such as cognitive training and support, can also be beneficial.")
            
            # Draw a line under the treatment section
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Caregiving")
            st.write("🤝 Providing care and support for individuals with Alzheimer's Disease is essential.")
            st.markdown("- Creating a safe and supportive environment.")
            st.markdown("- Offering emotional and practical support.")
            st.markdown("- Caregivers may face challenges in balancing their own well-being.")
            
            # Draw a line under the caregiving section
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### More Information")
            st.write("📚 For more detailed information about Alzheimer's Disease, you can visit reputable medical websites or consult with a healthcare professional.")
        
if __name__ == "__main__":
    main()