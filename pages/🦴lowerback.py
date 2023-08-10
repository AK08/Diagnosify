import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os


# Get the absolute path to the model file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(current_dir, 'lower_back.h5')
model = load_model(model_file_path)

st.write("""
# Lower Back Pain Prediction

Enter the following values to predict if you have lower back pain or not:
""")

# Create input fields for user to enter values
pelvic_incidence = st.number_input("Pelvic Incidence")
pelvic_tilt = st.number_input("Pelvic Tilt")
lumbar_lordosis_angle = st.number_input("Lumbar Lordosis Angle")
sacral_slope = st.number_input("Sacral Slope")
pelvic_radius = st.number_input("Pelvic Radius")
degree_spondylolisthesis = st.number_input("Degree Spondylolisthesis")
pelvic_slope = st.number_input("Pelvic Slope")
direct_tilt = st.number_input("Direct Tilt")
thoracic_slope = st.number_input("Thoracic Slope")
cervical_tilt = st.number_input("Cervical Tilt")
sacrum_angle = st.number_input("Sacrum Angle")
scoliosis_slope = st.number_input("Scoliosis Slope")

# Create an array of the entered values
input_data = np.array([
    [pelvic_incidence, pelvic_tilt, lumbar_lordosis_angle, sacral_slope,
    pelvic_radius, degree_spondylolisthesis, pelvic_slope, direct_tilt,
    thoracic_slope, cervical_tilt, sacrum_angle, scoliosis_slope]
])

# Make prediction
prediction = model.predict(input_data)
predicted_class = prediction[0][0]

# Define a threshold for class prediction
threshold = 0.5

# Display prediction result
if predicted_class >= threshold:
    st.write("Prediction: You have back pain.")
else:
    st.write("Prediction: You don't have back pain.")
