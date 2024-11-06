import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# Define the dictionary for genre mapping
mydict = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 
          'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}

# Load your trained model (adjust the path as needed)
model = tf.keras.models.load_model("saved_models/audio_classification.keras")

def extract_features(audio_file):
    """Extract MFCC features from an audio file."""
    y, sr = librosa.load(audio_file, duration=30, sr=None)  # Load the audio file (you can set a duration)
    
    # Extract MFCC features and plot for visualization
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    
    return mfccs

# Streamlit interface
st.title("Music Genre Classification")
st.write("Upload an audio file (wav format) to classify the genre.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type="wav")

if uploaded_file is not None:
    # Display the audio file details
    st.audio(uploaded_file, format="audio/wav")

    # Extract features from the uploaded audio
    features = extract_features(uploaded_file)
    
    
    # Prepare input data for the model
    x = []
    x.append(features)
    x = np.array(x)  # Reshaping as expected by the model
    x = np.reshape(x, (x.shape[0], 10, 4, 1))
    
    # Predict the genre
    y_pre = model.predict(x)
    predicted_class = np.argmax(y_pre, axis=1)

    # Display the predicted genre
    predicted_genre = list(mydict.keys())[list(mydict.values()).index(predicted_class[0])]
    st.write(f"Predicted Genre: {predicted_genre}")

    # Display the prediction confidence
    confidence = np.max(y_pre)
    st.write(f"Confidence: {confidence:.2f}")
