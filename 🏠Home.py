import streamlit as st

st.set_page_config(
    page_title="Disease Prediction",
    page_icon="🩺🔍🏠",
)

st.write("# Welcome to Disease Prediction! 👋")

st.markdown(
    """
    Our project aims to utilize the power of Deep Learning and Artificial Intelligence to help diagnose various brain diseases from MRI scan images. We understand the critical importance of early detection and accurate diagnosis in treating brain-related conditions, and that's why we have developed this intelligent system.
    """
)

st.markdown("### How it Works:")
st.markdown(
    """
- Upload an MRI scan image of the brain.
- Our advanced model will analyze the image using cutting-edge techniques.
- The system will provide a comprehensive diagnosis, including whether the person has a brain tumor and other brain diseases.
"""
)

st.markdown("### Why Choose Us:")
st.markdown(
    """
- Highly Accurate: Our model has been trained on a vast dataset of brain MRI scans to ensure accurate predictions.
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

st.markdown("### Data Privacy and Security:")
st.markdown(
    """
We understand the sensitivity of medical data and assure you that your uploaded images are handled with the utmost care and privacy. Your data is encrypted and not shared with any third parties.
"""
)

st.markdown("### Limitations:")
st.markdown(
    """
While our system provides valuable insights, it is essential to remember that it is not a substitute for professional medical advice. Always consult a qualified healthcare professional for a thorough diagnosis and treatment plan.
"""
)

st.markdown("### Get Started:")
st.markdown(
    """
Try out our Brain Disease Image Classification Project now! Upload an MRI scan image and receive instant results. We are committed to making a positive impact on healthcare and contributing to a healthier world.

Thank you for choosing our platform. Together, let's improve brain disease diagnosis and patient care!
"""
)
