import streamlit as st

st.header("⚡ AI Sentiment & Security")
st.markdown("""
    ### 📘 **Capstone Project Overview**
    This capstone project demonstrates the power of **Machine Learning (ML)** and **Natural Language Processing (NLP)** through two real-world applications — **Movie Review Sentiment Analysis** and **Phishing Website Detection**.

    ---
    #### 🎬 **IMDb Movie Review Sentiment Analysis**
    The IMDb Sentiment Analyzer uses a fine-tuned **BERT (Bidirectional Encoder Representations from Transformers)** model to understand the emotions behind movie reviews.
    - Users can input a review (positive or negative), and the model predicts the sentiment with a confidence score.
    - This showcases the capability of **Large Language Models (LLMs)** to interpret human text with high accuracy.

    ---
    #### 🕵️‍♂️ **Phishing Website Detection**
    This section applies **supervised machine learning** to a cybersecurity problem — detecting malicious or phishing websites based on their attributes (e.g., URL length, SSL certificate, redirects).
    - A trained **XGBoost Classifier** analyzes user-provided website characteristics.
    - The system classifies the site as either **Legitimate** ✅ or **Phishing** 🚨.

    ---
    #### 🧠 **Tools and Technologies**
    - **Streamlit:** for the interactive user interface.
    - **Transformers (Hugging Face):** for BERT-based sentiment analysis.
    - **Scikit-learn / XGBoost:** for phishing detection.
    - **PyTorch:** for model inference and GPU acceleration.

    ---
    #### 🎯 **Project Goal**
    To combine NLP and traditional ML in a single, user-friendly web app — demonstrating practical AI applications that interpret emotions in text and safeguard users online.
 
    ---
    💬 *Developed with ❤️ using Streamlit.*
    """)
