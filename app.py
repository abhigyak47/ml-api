from flask import Flask, request, jsonify, render_template, session
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Load the pre-trained model
model = load_model('heart_model.keras')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secure session key

# Function to process audio and extract MFCCs
n_fft = 2048
hop_length = 512

def process_audio(filename, duration=15):
    signal, sr = librosa.load(filename, sr=22050)
    time = librosa.get_duration(y=signal, sr=sr)

    # Ensure the audio is the desired length
    if round(time) < duration:
        signal = librosa.util.fix_length(signal, size=sr * duration, mode='wrap')

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

    # Pad or truncate the MFCCs to the desired length (646 time frames)
    target_length = 646
    if mfccs.shape[1] < target_length:
        padding = target_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, padding)), mode='constant')
    elif mfccs.shape[1] > target_length:
        mfccs = mfccs[:, :target_length]

    return mfccs

# Define route for inference and diagnosis
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save file temporarily
        filename = '/tmp/uploaded_audio.wav'
        file.save(filename)

        # Process audio to extract MFCCs
        mfccs = process_audio(filename)

        # Reshape the MFCCs to match the input shape of the model (1, 13, 646, 1)
        mfccs = np.expand_dims(mfccs, axis=-1)  # (13, 646, 1)
        mfccs = np.expand_dims(mfccs, axis=0)   # (1, 13, 646, 1)

        # Make prediction
        prediction = model.predict(mfccs)
        predicted_class = np.argmax(prediction)

        # Convert numpy int64 to regular Python int
        predicted_class = int(predicted_class)

        # Store prediction and diagnosis in session
        if predicted_class == 0:
            diagnosis = "Your heartbeat is classified as Normal. No irregularities detected."
        elif predicted_class == 1:
            diagnosis = "Your heartbeat shows signs of a Murmur. It's important to consult a doctor."
        elif predicted_class == 2:
            diagnosis = "Your heartbeat shows signs of an Extrastole. It may require further evaluation by a healthcare provider."
        else:
            diagnosis = "Unable to classify heartbeat."

        # Store both classification and diagnosis in session
        session['heartbeat_class'] = predicted_class
        session['diagnosis'] = diagnosis

        # Ensure the response is in standard Python types
        return jsonify({
            "predicted_class": predicted_class,
            "diagnosis": diagnosis
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to retrieve chat history and continue conversation
@app.route('/chat', methods=['GET'])
def chat():
    heartbeat_class = session.get('heartbeat_class', None)
    diagnosis = session.get('diagnosis', None)

    if heartbeat_class is not None:
        return jsonify({
            "previous_heartbeat_class": heartbeat_class,
            "previous_diagnosis": diagnosis
        })
    else:
        return jsonify({
            "message": "No previous heartbeat data found. Please upload a heartbeat file first."
        })

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
