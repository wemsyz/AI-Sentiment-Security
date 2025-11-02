import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


# Load model and tokenizer (Movie Review Sentiment Classifier)
MODEL_PATH = "google-bert/bert-base-uncased"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

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
