import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the model
model_file_path = 'models/model.pkl'
model = pickle.load(open(model_file_path, 'rb'))

# Load the dataset
csv_file_path = 'dataset/diabetes.csv'
dataset = pd.read_csv(csv_file_path)


def main():
    st.title('Diabetes Predictor')

    # Select relevant columns
    dataset_X = dataset.iloc[:, [1, 2, 5, 7]].values

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = scaler.fit_transform(dataset_X)

    st.write("Enter the following information for prediction:")

    glucose_level = st.text_input("Glucose Level")
    insulin = st.text_input("Insulin")
    bmi = st.text_input("BMI")
    age = st.text_input("Age")

    if st.button("Predict"):
        # Convert inputs to a numpy array
        input_features = np.array([[float(glucose_level), float(insulin), float(bmi), float(age)]])

        # Make prediction
        prediction = model.predict(scaler.transform(input_features))

        if prediction == 1:
            output = "Diabetes"
            
            st.write("##### Predicted Label: ",output)
            # Draw a line under the prediction result
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            # Display detailed information about Diabetes
            st.markdown("## Diabetes Information")
            st.write("**📌 Common Name:** Diabetes")
            st.write("**🌐 General Overview:** Diabetes is a chronic condition that affects how your body processes glucose (sugar).")
            
            # Draw a line under the general overview section
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Symptoms")
            st.write("🔍 Symptoms of Diabetes include:")
            st.markdown("- Frequent urination.")
            st.markdown("- Excessive thirst.")
            st.markdown("- Unexplained weight loss.")
            st.markdown("- Fatigue.")
            st.markdown("- Slow-healing sores or frequent infections.")
            
            # Draw a line under the symptoms section
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Causes")
            st.write("🔍 The causes of Diabetes vary depending on the type:")
            st.markdown("- Type 1 Diabetes: Autoimmune reaction destroys insulin-producing cells.")
            st.markdown("- Type 2 Diabetes: Insulin resistance and inadequate insulin production.")
            
            # Draw a line under the causes section
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Management")
            st.write("💼 Managing Diabetes involves:")
            st.markdown("- Monitoring blood sugar levels.")
            st.markdown("- Healthy eating and balanced diet.")
            st.markdown("- Regular physical activity.")
            st.markdown("- Medications or insulin therapy as prescribed.")
            
            # Draw a line under the management section
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### Complications")
            st.write("⚠️ Untreated Diabetes can lead to various complications:")
            st.markdown("- Cardiovascular diseases.")
            st.markdown("- Kidney damage (nephropathy).")
            st.markdown("- Nerve damage (neuropathy) leading to tingling and numbness.")
            st.markdown("- Eye problems and potential blindness.")
            
            # Draw a line under the complications section
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            
            st.markdown("### More Information")
            st.write("📚 For more detailed information about Diabetes, you can visit reputable medical websites or consult with a healthcare professional.")
        else:
            output = "No Diabetes"
            
            st.write("##### Predicted Label: ",output)
        

if __name__ == '__main__':
    main()