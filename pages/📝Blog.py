import streamlit as st
from PIL import Image


# Set page configuration and styling
st.set_page_config(
    page_title="Brain Diagnosify Blog",
    page_icon="🧠📝",
    
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
st.title("Welcome to Brain Diagnosify Blog! 🧠📝")


st.write("Here you will find blog pages from the world's largest medical communities. Explore the blogs and acquaint yourself with knowlwedge ")

#Blog 1
st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
st.write("## Harvard Health Blog")
st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)

#blog page1
# Display the image
image_path = "images/blog1.jpg"
image = Image.open(image_path)

st.image(image)
st.write("### 12 ways to keep your brain young")
st.write("Every brain changes with age, and mental function changes along with it. Mental decline is common, and it's one of the most feared consequences of aging. But cognitive impairment is not inevitable. Here are 12 ways you can help maintain brain function.")
st.markdown("[Click here to read more](https://www.health.harvard.edu/mind-and-mood/12-ways-to-keep-your-brain-young)")
st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)

#blog page2
# Display the image
image_path = "images/blog2.jpg"
image = Image.open(image_path)

st.image(image)
st.write("### Why is music good for the brain?")
st.write("Can music really affect your well-being, learning, cognitive function, quality of life, and even happiness? A recent survey on music and brain health conducted by AARP revealed some interesting findings about the impact of music on cognitive and emotional well-being.")
st.markdown("[Click here to read more](https://www.health.harvard.edu/blog/why-is-music-good-for-the-brain-2020100721062)")
st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)



#Blog 2
st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
st.write("## Mayo Clinic Blog")
st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)

#blog page1
# Display the image
image_path = "images/blog3.jpg"
image = Image.open(image_path)

st.image(image)
st.write("### 6 tips to keep your brain healthy")
st.write("Changes to your body and brain are normal as you age. However, there are some things you can do to help slow any decline in memory and lower your risk of developing Alzheimer's disease or other dementias.")
st.markdown("[Click here to read more](https://www.mayoclinichealthsystem.org/hometown-health/speaking-of-health/5-tips-to-keep-your-brain-healthy)")
st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)

#blog page2
# Display the image
image_path = "images/blog4.jpg"
image = Image.open(image_path)

st.image(image)
st.write("### Dementia-related pain: What caregivers need to know")
st.write("Dementia isn't a specific disease. Instead, it describes a collection of symptoms that affect a person's thinking and social abilities enough to interfere with daily life. There are more than 55 million people worldwide living with dementia. Of these, 50% to 60% have Alzheimer's disease. Dementia is the seventh leading cause of death worldwide.")
st.markdown("[Click here to read more](https://www.mayoclinichealthsystem.org/hometown-health/speaking-of-health/dementia-related-pain-and-caregivers)")
st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)

st.write("## References for Dataset")
st.markdown("[Alzheimer Dataset](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset)")
st.markdown("[Stroke Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)")
st.markdown("[Parkinsons Dataset](https://www.kaggle.com/datasets/kmader/parkinsons-drawings)")
st.markdown("[Brain Tumor Git](https://github.com/aaronDev28/Brain-datasets/tree/main)")