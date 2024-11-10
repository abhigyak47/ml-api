from flask import Flask, request, jsonify
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('heart_model.keras')

# Initialize Flask app
app = Flask(__name__)

# Function to process audio and extract MFCCs
def process_audio(filename, duration=15):
    signal, sr = librosa.load(filename, sr=22050)
    time = librosa.get_duration(y=signal, sr=sr)

    # Ensure the audio is the desired length
    if round(time) < duration:
        signal = librosa.util.fix_length(signal, size=sr * duration, mode='wrap')

    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_fft=2048, hop_length=512, n_mfcc=13)
    return mfccs

# Define route for inference
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

        # Reshape data to match model input
        mfccs = np.expand_dims(mfccs, axis=-1)  # (time_frames, n_mfcc, 1)
        mfccs = np.expand_dims(mfccs, axis=0)   # (1, time_frames, n_mfcc, 1)

        # Make prediction
        prediction = model.predict(mfccs)
        predicted_class = np.argmax(prediction)

        return jsonify({"predicted_class": int(predicted_class)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

