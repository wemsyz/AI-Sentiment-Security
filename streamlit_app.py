import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import joblib

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