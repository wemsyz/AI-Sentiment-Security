import streamlit as st
import pandas as pd
import numpy as np
import gdown
import os
import zipfile
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


# Folder paths
MODEL_DIR = "models/Notebooks"
ZIP_PATH = "models/Notebooks.zip"

# Create model folder
os.makedirs("models", exist_ok=True)

# Google Drive file ID of zip folder
file_id = "1DkU12ZX_3x3cBLHiXBOqDR5os4LZWZ5-"

# Generate direct download link
url = f"https://drive.google.com/uc?id=1DkU12ZX_3x3cBLHiXBOqDR5os4LZWZ5-"

# Download and extract files
if not os.path.exists(MODEL_DIR):
    st.info("Downloading fine-tuned sentiment model from Google Drive...")
    gdown.download(url, ZIP_PATH, quiet=False)

    st.info("Extracting model files...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall("models")

st.success("Model files ready!")

# Load model and tokenizer from extracted folder
st.info("Loading fine-tuned model... please wait...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

st.success("Model loaded successfully!")

st.subheader("🎬 IMDb Movie Review Sentiment Analysis")
st.markdown("Type a movie review below and let BERT **analyze** _its sentiment!_")
    

# Input text box
user_input = st.text_area("Enter Your Movie Review:", height=150, placeholder="Type here")

# Predict button
if st.button("Predict Sentiment"):
     with st.spinner("Analyzing sentiment..."):
        
        # Tokenize user input
        inputs = tokenizer(
            user_input,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        sentiment = "Positive 😀" if pred == 1 else "Negative 😞"
        confidence = probs[0][pred].item() * 100

        # Display result
        st.subheader("🎯 Sentiment Result")
        if pred == 1:
            st.success(f"✅ **Prediction:** {sentiment}\n**Confidence:** {confidence:.2f}%")
        else:
            st.error(f"❌ **Prediction:** {sentiment}\n**Confidence:** {confidence:.2f}%")

else:
    st.warning("⚠️ Please enter a movie review before analyzing.")

    st.markdown("---")
    st.caption("Model: `google-bert/bert-base-uncased` fine-tuned on IMDb dataset.")       
