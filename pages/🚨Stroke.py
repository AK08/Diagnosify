import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os


# Get the absolute path to the model file
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
model_file_path = os.path.join(models_dir, 'model.pkl')

# Load the model using pickle
with open(model_file_path, 'rb') as model_file:
    model = pickle.load(model_file)


# Function to make predictions
def predict_stroke(age, avg_glucose_level, bmi, gender_Male, hypertension_1, heart_disease_1,
                   ever_married_Yes, work_type_Never_worked, work_type_Private,
                   work_type_Self_employed, work_type_children, Residence_type_Urban,
                   smoking_status_formerly_smoked, smoking_status_never_smoked, smoking_status_smokes):
    
    input_features = [age, avg_glucose_level, bmi, gender_Male, hypertension_1, heart_disease_1,
                      ever_married_Yes, work_type_Never_worked, work_type_Private,
                      work_type_Self_employed, work_type_children, Residence_type_Urban,
                      smoking_status_formerly_smoked, smoking_status_never_smoked, smoking_status_smokes]

    features_value = [np.array(input_features)]
    features_name = ['age'	,'avg_glucose_level',	'bmi'	,'gender_Male'	,'hypertension_1',	'heart_disease_1','ever_married_Yes',	'work_type_Never_worked',	'work_type_Private',	'work_type_Self-employed',	'work_type_children'	,'Residence_type_Urban',	'smoking_status_formerly smoked','smoking_status_never smoked'	,'smoking_status_smokes']

    df = pd.DataFrame(features_value, columns=features_name)
    prediction = model.predict(df)[0]
    return prediction

# Streamlit UI
st.title('Stroke Event Prediction')
st.write('Enter the patient details to predict if a stroke event will occur.')

age = st.number_input('Age', min_value=0, max_value=150, value=30)
avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, value=80.0)
bmi = st.number_input('BMI', min_value=0.0, value=20.0)

gender_Male = st.selectbox('Gender', ['Male', 'Female'], index=1)
gender = 1 if gender_Male == 'Male' else 0

hypertension_1 = st.checkbox('Hypertension')
heart_disease_1 = st.checkbox('Heart Disease')
ever_married_Yes = st.checkbox('Ever Married')
work_type_Never_worked = st.checkbox('Never Worked')
work_type_Private = st.checkbox('Private Work')
work_type_Self_employed = st.checkbox('Self Employed')
work_type_children = st.checkbox('Children')
Residence_type_Urban = st.checkbox('Urban Residence')

smoking_status_formerly_smoked = st.checkbox('Formerly Smoked')
smoking_status_never_smoked = st.checkbox('Never Smoked')
smoking_status_smokes = st.checkbox('Smokes')


if st.button('Predict'):
    result = predict_stroke(age, avg_glucose_level, bmi, gender, hypertension_1, heart_disease_1,
                            ever_married_Yes, work_type_Never_worked, work_type_Private,
                            work_type_Self_employed, work_type_children, Residence_type_Urban,
                            smoking_status_formerly_smoked, smoking_status_never_smoked, smoking_status_smokes)
    if result:
        st.write('##### Prediction: Patient is likely to have a stroke event.')
        
        # Display detailed information about the disease
        st.markdown('### Stroke Event Information')
        st.write('A stroke occurs when there is a sudden interruption of blood supply to the brain. This can happen due to a blood clot or a burst blood vessel.')
        st.markdown('- Symptoms can include sudden numbness or weakness in the face, arm, or leg, confusion, trouble speaking, and more. 💡')
        st.markdown('- Immediate medical attention is crucial as stroke can cause lasting brain damage and disability. ⚠️')
        st.markdown('- Risk factors include high blood pressure, diabetes, smoking, and certain heart conditions. 🩺')
        st.markdown('For more detailed information about strokes, you can visit reputable medical websites or consult with a healthcare professional. 📚')
        
        st.markdown('### How to Recognize a Stroke')
        st.write("🚨 Recognizing the signs of a stroke can save lives. Remember the acronym FAST:")
        st.markdown("- **F:** Face drooping")
        st.markdown("- **A:** Arm weakness or numbness")
        st.markdown("- **S:** Speech difficulty")
        st.markdown("- **T:** Time to call emergency services")
        st.markdown("Acting FAST can help ensure timely medical intervention. 🏥")
        
        st.markdown('### Preventing Strokes')
        st.write("🛡️ Reducing the risk of stroke involves making healthy lifestyle choices:")
        st.markdown("- Maintain a balanced diet rich in fruits, vegetables, whole grains, lean proteins, and healthy fats.")
        st.markdown("- Engage in regular physical activity to keep your heart and blood vessels healthy.")
        st.markdown("- Manage chronic conditions like high blood pressure and diabetes.")
        st.markdown("- Avoid smoking and limit alcohol consumption.")
        st.markdown("Taking these steps can significantly lower your risk of stroke. 🏃‍♀️🥦")
        
        st.markdown('### Stroke Diagnosis and Treatment')
        st.write("🏨 If you suspect someone is having a stroke, seek medical help immediately. Diagnosing and treating a stroke promptly is crucial for better outcomes.")
        st.markdown("- Diagnostic tests include brain imaging, such as CT scans or MRIs, to determine the cause and extent of the stroke.")
        st.markdown("- Treatment may involve medication to dissolve blood clots or surgery to remove clots.")
        st.markdown("- Stroke rehabilitation focuses on restoring lost functions and improving quality of life.")
        st.markdown("Early intervention can prevent further damage and complications. 🏆")
        
        st.markdown('### Life After a Stroke')
        st.write("🌱 Recovery after a stroke is a journey that requires patience and support:")
        st.markdown("- Rehabilitation may involve physical therapy, occupational therapy, and speech therapy.")
        st.markdown("- Lifestyle changes may be necessary to reduce the risk of another stroke.")
        st.markdown("- Emotional support from healthcare professionals, family, and friends is essential.")
        st.markdown("- Each person's recovery journey is unique, and progress can vary.")
        st.markdown("With determination and appropriate care, many stroke survivors can regain independence. 💪")
        
    else:
        st.write('##### Prediction: Patient is unlikely to have a stroke event.')
        st.markdown("Your prediction indicates that the patient is unlikely to have a stroke event. Keep up the healthy lifestyle choices to maintain your well-being. 🌟")
