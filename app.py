# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Cache model and word index loading for performance
@st.cache_resource
def load_resources():
    """Load model and word index with caching"""
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    model = load_model('simple_rnn_imdb.h5')
    return model, reverse_word_index

# Load resources
model, reverse_word_index = load_resources()

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    # Clip indices to ensure they're within valid range
    padded_review = np.clip(padded_review, 0, 9999)
    return padded_review


import streamlit as st

# Streamlit app UI
st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# Add sidebar with information
with st.sidebar:
    st.header("About")
    st.info(
        "This app uses a Recurrent Neural Network (RNN) trained on IMDB reviews "
        "to predict whether a movie review is positive or negative."
    )
    st.subheader("How it works:")
    st.markdown(
        """
        1. Enter your movie review
        2. Click "Classify" button
        3. Get sentiment prediction with confidence score
        """
    )

# User input
user_input = st.text_area('Movie Review', height=150, placeholder="Enter your movie review here...")

col1, col2 = st.columns([1, 1])

with col1:
    predict_button = st.button('🔍 Classify Review', use_container_width=True)

with col2:
    clear_button = st.button('🗑️ Clear', use_container_width=True)

if clear_button:
    st.rerun()

if predict_button:
    if not user_input.strip():
        st.warning("Please enter a movie review!")
    else:
        with st.spinner('Analyzing review...'):
            preprocessed_input = preprocess_text(user_input)
            prediction = model.predict(preprocessed_input, verbose=0)
            
        # Display results
        sentiment_score = prediction[0][0]
        sentiment = 'Positive 👍' if sentiment_score > 0.5 else 'Negative 👎'
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Sentiment", sentiment)
        
        with col2:
            st.metric("Confidence Score", f"{sentiment_score:.2%}")
        
        # Display prediction bar
        st.progress(sentiment_score, text=f"Positive Probability: {sentiment_score:.2%}")
        
        # Add color-coded message
        if sentiment_score > 0.7:
            st.success("🎉 This is a very positive review!")
        elif sentiment_score > 0.5:
            st.info("😊 This review is positive.")
        elif sentiment_score > 0.3:
            st.warning("😕 This review is negative.")
        else:
            st.error("😢 This is a very negative review!")
else:
    st.write("Please enter a movie review and click the Classify button.")
