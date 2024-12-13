import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string

# Load the saved model
model = load_model("sentiment_rnn_model.h5")  # Replace with the actual path of your saved model

# Constants
MAX_SEQUENCE_LENGTH = 100
LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f"[{string.punctuation}]", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Streamlit App
st.title("Tweet Sentiment Analyzer")
st.write("Enter a tweet to analyze its sentiment.")

# Input Text Box
tweet = st.text_area("Enter Tweet:")

# Analyze Button
if st.button("Analyze"):
    if tweet.strip() == "":
        st.write("Please enter a tweet to analyze.")
    else:
        # Preprocess the input tweet
        cleaned_tweet = preprocess_text(tweet)

        # Tokenization logic (replace with the exact logic used during training)
        words = cleaned_tweet.split()
        word_to_index = {  # Dummy mapping example, replace with actual mapping from training
            "good": 1,
            "bad": 2,
            "neutral": 3,
            "great": 4,
            "poor": 5
        }
        sequences = [[word_to_index.get(word, 0) for word in words]]  # Map words to indices
        padded_sequence = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  # Pad sequence

        # Predict sentiment
        prediction = model.predict(padded_sequence)
        sentiment = LABELS[np.argmax(prediction)]
        highest_score = np.max(prediction)

        # Display result
        st.write(f"### Sentiment: {sentiment}")
        st.write(f"#### Confidence Score: {highest_score:.2f}")

st.write("---")
st.write("This app uses an RNN-based sentiment analysis model trained on tweets.")
