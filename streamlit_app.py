import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model, tokenizer, and label encoder
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('dis_lstm.keras')

@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        return pickle.load(handle)

@st.cache_resource
def load_label_encoder():
    with open('label_encoder.pickle', 'rb') as handle:
        return pickle.load(handle)

model = load_model()
tokenizer = load_tokenizer()
label_encoder = load_label_encoder()

# Constants
MAX_LENGTH = 50

# Streamlit app
st.title("Text Classification with LSTM")
st.markdown("""
This app uses a Bidirectional LSTM model to classify text. Enter your text below, and the model will predict the label.
""")

# Text input
user_input = st.text_area("Enter your text:", placeholder="Type your text here...")

if st.button("Predict"):
    if user_input.strip():
        # Preprocess and tokenize input
        sequences = tokenizer.texts_to_sequences([user_input])
        padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

        # Predict
        predictions = model.predict(padded_sequences)
        predicted_label = label_encoder.inverse_transform([np.argmax(predictions)])

        # Display result
        st.write("### Prediction")
        st.write(f"Predicted Label: **{predicted_label[0]}**")
        st.write("### Confidence Scores")
        for label, confidence in zip(label_encoder.classes_, predictions[0]):
            st.write(f"{label}: {confidence:.2%}")
    else:
        st.warning("Please enter text for prediction.")
