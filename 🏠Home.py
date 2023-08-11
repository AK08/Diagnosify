from PIL import Image
import streamlit as st

# Set page configuration and styling
st.set_page_config(
    page_title="Brain Diagnosify",
    page_icon="🧠🔍🏠",
)
st.markdown(
    """
<style>
body {
    color: #333333;
    background-color: #f7f7f7;
}
h1, h2, h3 {
    color: #004d80;
}
</style>
""",
    unsafe_allow_html=True,
)

# Main content
st.markdown("<h1>Welcome to Diagnosify! 👋</h1>", unsafe_allow_html=True)




# Display the image
image_path = "images/brain chemistry-cuate.png"
image = Image.open(image_path)
resized_image = image.resize((500, 500))
st.image(resized_image)




st.markdown(
    """
Your brain is a complex and fascinating organ that plays a central role in your overall well-being. Welcome to Brain Diagnosify, where we focus on brain health and understanding various brain-related conditions.
"""
)

# How to Keep Your Brain Healthy
st.markdown("## How to Keep Your Brain Healthy 🌱")
healthy_brain_tips = [
    "Stay Active: Engage in regular physical activity to promote blood flow to the brain.",
    "Eat a Balanced Diet: Consume a diet rich in fruits, vegetables, whole grains, lean proteins, and healthy fats.",
    "Stay Mentally Active: Challenge your brain with puzzles, reading, learning new skills, and staying curious.",
    "Get Quality Sleep: Aim for 7-9 hours of restful sleep each night.",
    "Manage Stress: Practice relaxation techniques like meditation and deep breathing.",
]

st.markdown("🌟 Here are some tips to maintain a healthy brain:")
for tip in healthy_brain_tips:
    st.markdown(f"🔹 {tip}")

# Habits to Avoid for Brain Health
st.markdown("## Habits to Avoid for Brain Health 🚫")
unhealthy_habits = [
    "Excessive Sugar Intake: High sugar consumption can impair cognitive function over time.",
    "Sedentary Lifestyle: Lack of physical activity can negatively impact brain health.",
    "Smoking: Smoking is linked to cognitive decline and increased risk of brain-related disorders.",
    "Excessive Alcohol Consumption: Heavy drinking can damage brain cells and impair cognitive function.",
    "Poor Sleep Habits: Inadequate sleep can affect memory, concentration, and overall brain health.",
]

st.markdown("🚫 Avoid these habits for a healthier brain:")
for habit in unhealthy_habits:
    st.markdown(f"🔹 {habit}")

# Brain-Related Conditions
st.markdown("## Brain-Related Conditions 🫀🫁🧠")
brain_conditions = [
    "Alzheimer's Disease: A progressive neurodegenerative disorder affecting memory and cognitive function.",
    "Brain Tumors: Abnormal growth of cells within the brain, leading to various symptoms.",
    "Stroke: A sudden disruption of blood flow to the brain, resulting in damage to brain tissue.",
    "Depression and Anxiety: Mental health conditions that can impact brain function and overall well-being.",
]

st.markdown("🧠 Learn about these brain-related conditions:")
for condition in brain_conditions:
    st.markdown(f"🔹 {condition}")

# Benefits of Brain Health
st.markdown("## Benefits of Brain Health 🌈")
brain_health_benefits = [
    "Enhanced Cognitive Function: A healthy brain supports better memory, focus, and decision-making.",
    "Improved Mood: Brain health is closely linked to emotional well-being and mental health.",
    "Reduced Risk of Diseases: Healthy lifestyle choices can lower the risk of brain-related disorders.",
    "Higher Quality of Life: Maintaining brain health contributes to overall vitality and quality of life.",
]

st.markdown("🌟 Enjoy these benefits by maintaining brain health:")
for benefit in brain_health_benefits:
    st.markdown(f"🔹 {benefit}")

# About Our Team
st.markdown("## About Our Team 👥")
st.markdown(
    """
Meet the dedicated team behind Brain Diagnosify: Aaron Shaji, Alen Kottaram, Alin Biju, and Siddarth K.M. We are passionate about leveraging technology to improve brain health and provide accurate disease predictions.
"""
)

# Display the image
image_path = "images/Team work-amico.png"
image = Image.open(image_path)

st.image(image)
