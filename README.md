# Heartbeat Classification API

This is a Flask application for classifying heartbeats using a CNN model. It takes an audio file as input and outputs one of three categories: normal, murmur, or extrastole.

## Setup Instructions

1. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Run the Flask server:

   ```
   python app.py
   ```

The server will start running and be accessible locally at `http://127.0.0.1:5000/predict`.

## Dependencies

The `requirements.txt` file includes all the necessary libraries for the project.
