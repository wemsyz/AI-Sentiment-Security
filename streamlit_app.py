import streamlit as st

st.set_page_config(page_title="⚡ AI Sentiment & Security", layout="centered")

# Create navigation menu

nav = st.navigation({
    "Main": [
        st.Page("intro.py", title="🏠 Introduction"),
        st.Page("sentiment.py", title="🎬 Sentiment Analysis"),
        st.Page("phishing.py", title="🕵️‍♂️ Phishing Detection")
    ]
})

# Run selected pages
nav.run()