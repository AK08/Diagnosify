from PIL import Image
import streamlit as st

st.set_page_config(
    page_title="Disease Prediction",
    page_icon="🩺🔍🏠",
)

st.write("# Welcome to Diagnosify! 👋")


# Display the image
image_path = "images/Medical research-amico.png"
image = Image.open(image_path)
resized_image = image.resize((500, 500))  # Adjust the size as needed
st.image(resized_image)

st.markdown(
    """
    In the conventional medical realm, disease prediction suffers from accuracy and
timeliness constraints due to manual interpretation and delayed interventions. This
necessitates a groundbreaking solution to revolutionize the field. Our project
addresses this by leveraging advanced machine learning techniques for predicting multiple diseases."""
)


st.markdown("### How it Works:")
st.markdown(
    """
- Upload an required scan image or enter the your data.
- Our advanced model will analyze the image and data using cutting-edge techniques.
- The system will provide a predict if the person has the particular disease.
"""
)

st.markdown("### Why Choose Us:")
st.markdown(
    """
- Highly Accurate: Our model has been trained on a vast dataset to ensure accurate predictions.
- Fast and Efficient: The prediction process is quick, providing you with results in just a few seconds.
- User-Friendly Interface: Our intuitive interface makes it easy for anyone to use, even without prior technical knowledge.
"""
)

st.markdown("### Benefits of Early Detection:")
st.markdown(
    """
- Early detection of brain tumors and diseases can significantly improve treatment outcomes.
- Timely diagnosis enables healthcare professionals to plan the best course of action for patients.
- Our system aids medical practitioners in making well-informed decisions, leading to better patient care.
"""
)


st.markdown("### Team Members")
st.markdown(
    """
Aaron Shaji, Alen Kottaram, Alin Biju and Siddarth K.M
"""
) 

# Display the image
image_path = "images/Team work-amico.png"
image = Image.open(image_path)
resized_image = image.resize((500, 500))  # Adjust the size as needed
st.image(resized_image)


