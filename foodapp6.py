import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from playsound import playsound
import random
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time

# Load the trained model
MODEL_PATH = "model.h5"  # Path to your trained model

# Ensure the model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Error: Model file not found at {MODEL_PATH}")
else:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("Model loaded successfully!")

# Directory for sound files
SOUND_DIR = "C:\\Users\\saipr\\anaconda3\\DeepLearning\\sound_files"

# Function to play sound (improved)
def play_random_sound():
    sound_files = [f for f in os.listdir(SOUND_DIR) if f.endswith(('.wav', '.mp3'))]
    if not sound_files:
        st.warning("No sound files found in the directory!")
        return
    random_sound = random.choice(sound_files)
    sound_path = os.path.join(SOUND_DIR, random_sound)
    print(f"Playing sound from: {sound_path}")
    try:
        playsound(sound_path)
    except Exception as e:
        print(f"Error playing sound: {e}")

# Function to process and predict from a frame
def predict_from_frame(frame):
    # Resize and preprocess the frame
    img_resized = cv2.resize(frame, (224, 224))  # Resize to 224x224 as expected by MobileNetV2
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess image for MobileNetV2

    # Predict using the model
    prediction = model.predict(img_array)
    
    # Get the predicted class (0 or 1)
    predicted_class = 'Contaminates' if prediction[0] < 0.5 else 'Good'
    return predicted_class

# Streamlit App Definition
def app():
    st.title("Food Contamination Detection")
    st.write("This app uses a webcam feed to detect food contamination and plays an alert sound when contamination is detected.")

    # Initialize session state for video stream
    if "video_active" not in st.session_state:
        st.session_state.video_active = False

    # Buttons to control video feed
    if st.button('Start Video', key='start_video_button'):
        st.session_state.video_active = True

    if st.button('Stop Video', key='stop_video_button'):
        st.session_state.video_active = False

    # Placeholder for video feed
    placeholder = st.empty()

    # Custom VideoTransformer for live prediction
    class VideoTransformer(VideoTransformerBase):
        last_sound_time = 0  # Track the last time sound was played (for debouncing)

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")  # Convert frame to ndarray

            # Predict the class
            prediction = predict_from_frame(img)

            # Play sound immediately on contamination detection
            if prediction == 'Contaminates':
                current_time = time.time()
                # Optional: Ensure sound is played at most once every 20 seconds
                if current_time - self.last_sound_time > 20:
                    play_random_sound()
                    self.last_sound_time = current_time

            # Display prediction on the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f'Prediction: {prediction}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            return img

    # Start webcam feed only if video is active
    if st.session_state.video_active:
        with placeholder.container():
            st.write("Starting Webcam...")
            webrtc_streamer(
                key="webcam_feed",  # Unique key for the webcam feed
                video_transformer_factory=VideoTransformer,
                media_stream_constraints={"video": True, "audio": False},
            )
    else:
        placeholder.empty()  # Stop video feed

# Run the Streamlit app
if __name__ == "__main__":
    app()
