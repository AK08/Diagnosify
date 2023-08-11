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
        st.write('Prediction: Patient is likely to have a stroke event.')
    else:
        st.write('Prediction: Patient is unlikely to have a stroke event.')
