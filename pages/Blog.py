import streamlit as st

# Define a dictionary of blogs (title: link)
blogs = {
    "View Blog": "https://www.health.harvard.edu/mind-and-mood/12-ways-to-keep-your-brain-young",
    "Brain Tumor Awareness": "https://en.wikipedia.org/wiki/Organ_(biology)",
    "Tips for a Healthy Brain": "https://en.wikipedia.org/wiki/Tissue_(biology)",
    # Add more blogs here
}

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

st.markdown(
    """
Read informative blogs written by medical professionals about brain health and related topics. Click on a blog title to read more!
"""
)



# Display blogs with clickable links
for title, link in blogs.items():
    st.markdown(f"🔹 [{title}]({link})")

# Note: Since Streamlit doesn't directly support opening external websites in a new tab,
# the user may need to manually open links in a new tab by right-clicking or using a keyboard shortcut.
