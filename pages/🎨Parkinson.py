import streamlit as st
from PIL import Image
import os
import joblib
import numpy as np
from skimage import feature

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path for the model
model_filename = "random_forest_spiral_model.pkl"
model_path = os.path.join(script_dir, model_filename)

# HOG feature extraction parameters
orientations = 9
pixels_per_cell = (10, 10)
cells_per_block = (2, 2)
transform_sqrt = True
block_norm = "L2"

def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((200, 200))
    image = np.array(image)
    image = image.astype(np.uint8)

    # Apply thresholding
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    features = feature.hog(image,
                           orientations=orientations,
                           pixels_per_cell=pixels_per_cell,
                           cells_per_block=cells_per_block,
                           transform_sqrt=transform_sqrt,
                           block_norm=block_norm)
    return features

def main():
    st.title("Parkinson's Disease Detector")
    st.success("🟢 **World's first publicly accessible Parkinson's Disease Detector** (based on spiral and wave drawings).")

    # Model selection dropdown
    model_option = st.selectbox("Select Model", ("Spiral", "Wave"))

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        

        # Preprocess the uploaded image
        img_array = np.array(image)
        img_features = preprocess_image(img_array)

        # Construct the absolute path for the selected model
        model_filename = "random_forest_spiral_model.pkl" if model_option == "Spiral" else "random_forest_wave_model.pkl"
        model_path = os.path.join(script_dir, model_filename)

        # Load the selected trained model
        model = joblib.load(model_path)
        
        # Predict using the model
        prediction = model.predict([img_features])

        
        if prediction[0]:
            st.markdown("## Prediction: Parkinson's Disease :warning:")
            st.write("Parkinson's disease is a neurodegenerative disorder that affects movement control. It is a progressive disorder of the nervous system that primarily affects motor function.")
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Common Name")
            st.write("**📌 Common Name:** Parkinson's Disease")
            
            st.markdown("### General Overview")
            st.write("🌐 Parkinson's Disease is a chronic and progressive neurological disorder that primarily affects motor function. It is characterized by the gradual loss of dopamine-producing cells in the brain, which leads to motor symptoms such as tremors, rigidity, and bradykinesia (slowness of movement). In addition to motor symptoms, Parkinson's Disease can also cause non-motor symptoms, including cognitive impairment, mood changes, and sleep disturbances.")
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Frequency")
            st.write("📊 Parkinson's Disease is the second most common neurodegenerative disorder after Alzheimer's Disease. It affects people worldwide, with a higher prevalence in older adults.")
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### General Information")
            st.write("📋 Here are some key points about Parkinson's Disease:")
            st.markdown("- It is named after Dr. James Parkinson, who first described its symptoms in 1817.")
            st.markdown("- The exact cause of Parkinson's Disease is not fully understood, but a combination of genetic and environmental factors is believed to play a role.")
            st.markdown("- It is characterized by the presence of protein aggregates known as Lewy bodies in brain cells.")
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Symptoms")
            st.write("🧠 The symptoms of Parkinson's Disease can vary and may include:")
            st.markdown("- Tremors: Involuntary shaking of hands, arms, legs, or jaw.")
            st.markdown("- Rigidity: Stiffness of muscles, making movement difficult.")
            st.markdown("- Bradykinesia: Slowness of movement and difficulty initiating movement.")
            st.markdown("- Postural instability: Impaired balance and coordination, leading to a stooped posture.")
            st.markdown("- Non-motor symptoms: Cognitive changes, depression, sleep disturbances, and more.")
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Causes")
            st.write("🔍 The exact cause of Parkinson's Disease is still under research.")
            st.markdown("- Genetic and environmental factors may play a role.")
            st.markdown("- Abnormal protein deposits, such as Lewy bodies, are observed in the brains of individuals with Parkinson's.")
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Precautions")
            st.write("🛡️ While there is no proven way to prevent Parkinson's Disease, certain lifestyle choices may reduce the risk or delay its onset.")
            st.markdown("- Regular exercise and a balanced diet can contribute to overall well-being.")
            st.markdown("- Avoid exposure to toxins and chemicals that may increase the risk.")
            st.markdown('- **Note:** Consult a healthcare professional for personalized advice.')
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Treatment")
            st.write("💼 Currently, there is no cure for Parkinson's Disease, but treatments aim to:")
            st.markdown("- Manage symptoms and improve quality of life.")
            st.markdown("- Medications may help regulate dopamine levels and alleviate motor symptoms.")
            st.markdown("- Deep brain stimulation (DBS) is a surgical procedure that can provide relief in some cases.")
            st.markdown('- **Note:** Treatment plans are tailored to each individual.')
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Medications")
            st.write("💊 Several medications are used to manage symptoms of Parkinson's Disease:")
            st.markdown("- Levodopa: Converts to dopamine and helps manage motor symptoms.")
            st.markdown("- Dopamine agonists: Stimulate dopamine receptors in the brain.")
            st.markdown("- MAO-B inhibitors: Increase dopamine levels by inhibiting an enzyme that breaks it down.")
            st.markdown('- **Note:** Medications should be taken under medical supervision.')
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Specialists")
            st.write("👩‍⚕️ Medical professionals involved in Parkinson's Disease care:")
            st.markdown("- Neurologist: Specializes in diagnosing and treating neurological disorders.")
            st.markdown("- Movement disorder specialist: Focuses on disorders affecting movement and coordination.")
            st.markdown("- Physical therapist: Provides exercises to improve mobility and strength.")
            st.markdown('- **Note:** Collaborate with a multidisciplinary team for comprehensive care.')
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Disease Type")
            st.write("📁 Parkinson's Disease is categorized as a neurodegenerative disorder.")
            st.markdown('- **Note:** It belongs to the group of movement disorders.')
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### More Information")
            st.write("📚 For more detailed information about Parkinson's Disease, you can visit reputable medical websites or consult with a healthcare professional.")
            st.write("🔗 [Parkinson's Foundation](https://www.parkinson.org/)")
            st.write("🔗 [National Institute on Aging](https://www.nia.nih.gov/)")
            
        else:
            st.markdown("## Prediction: Healthy :white_check_mark:")

if __name__ == "__main__":
    main()