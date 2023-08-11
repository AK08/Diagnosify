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
        # Read the uploaded image using PIL
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
        
        st.write("##### Predicted Class:", class_names[predicted_class])
        st.write("##### Confidence:", prediction[0][predicted_class])
        
        # Draw a line under the heading
        st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
        
        # Display information about skin cancer if predicted malignant
        if predicted_class == 1:
            st.markdown("## Skin Cancer Information")
            st.write("**📌 Common Name:** Malignant Skin Cancer")
            st.write("**🌐 General Overview:** Skin cancer is a condition in which the skin cells undergo abnormal growth and division.")
            st.write("**🌍 Frequency:** Skin cancer is one of the most common types of cancer worldwide.")
            
             # Draw a line under the heading
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
    
            
            st.markdown("### Symptoms")
            st.write("🔍 Common symptoms of skin cancer may include:")
            st.markdown("- Changes in skin appearance, such as the development of new moles or growth of existing moles.")
            st.markdown("- Appearance of a sore that does not heal.")
            st.markdown("- Changes in existing moles.")
            
             # Draw a line under the heading
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Causes")
            st.write("☀️ The primary cause of skin cancer is exposure to ultraviolet (UV) radiation from the sun or artificial sources like tanning beds.")
            st.markdown("- People with fair skin, a history of sunburns, a family history of skin cancer, and weakened immune systems are at a higher risk.")
            
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            
            st.markdown("### Precautions")
            st.write("🌴 To prevent skin cancer:")
            st.markdown("- Limit exposure to UV radiation by staying in the shade.")
            st.markdown("- Wear protective clothing.")
            st.markdown("- Apply sunscreen with at least SPF 30.")
            st.markdown("- Regular self-examinations of your skin can help in early detection.")
            
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Treatment")
            st.write("💼 Treatment options for skin cancer depend on:")
            st.markdown("- The type, size, and location of the cancer.")
            st.markdown("- They may include surgery, radiation therapy, chemotherapy, targeted therapy, and immunotherapy.")
            
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Medications")
            st.write("💊 In some cases, medications may be prescribed to treat certain types of skin cancer:")
            st.markdown("- Topical treatments, immunotherapy drugs, and targeted therapy drugs are used.")
            st.markdown("- The choice of medication depends on the specific diagnosis and healthcare provider's recommendations.")
            
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Specialists")
            st.write("👨‍⚕️ Dermatologists and oncologists are medical specialists who:")
            st.markdown("- Diagnose and treat skin cancer.")
            st.markdown("- Surgeons may perform excisions or other surgical procedures.")
            st.markdown("- Radiation oncologists administer radiation therapy.")
            
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Disease Type")
            st.write("🦠 Skin cancer is categorized into different types:")
            st.markdown("- Melanoma, basal cell carcinoma, and squamous cell carcinoma.")
            st.markdown("- Melanoma is the most aggressive type and can spread to other parts of the body if not treated early.")
            
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### More Information")
            st.write("📚 For more detailed information about skin cancer, you can visit reputable medical websites or consult with a healthcare professional.")
        
if __name__ == '__main__':
    main()