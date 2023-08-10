import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

model_file_path = 'C:\Alen\Work\Artistic\Github\Disease-Classification\models\model.pkl'
model = pickle.load(open(model_file_path, 'rb'))

csv_file_path = 'C:\Alen\Work\Artistic\Github\Disease-Classification\dataset\diabetes.csv'  # Provide the correct absolute path
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
            output = "You have Diabetes, please consult a Doctor."
        else:
            output = "You don't have Diabetes."
        st.write(output)

if __name__ == '__main__':
    main()
