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

        # Reshape the MFCCs to match the input shape of the model (1, 13, 646, 1)
        mfccs = np.expand_dims(mfccs, axis=-1)  # (13, 646, 1)
        mfccs = np.expand_dims(mfccs, axis=0)   # (1, 13, 646, 1)

        # Make prediction
        prediction = model.predict(mfccs)
        predicted_class = np.argmax(prediction)

        return jsonify({"predicted_class": int(predicted_class)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
