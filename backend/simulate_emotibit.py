import random 
import time
from flask import Flask, jsonify
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

"""
Description: Simulating EDA, BPM, Temperature, Accelerometer, and Sp02 data from an imaginary EmotiBit.

"""
def simulate_emotibit_data():
    while True:
        # Simulate data for Emotobit:
        eda = round(random.uniform(0.5, 5.0), 2)
        heart_rate = random.randint(60,100)
        temperature = round(random.uniform(36.0, 37.5), 2)

        # Package data into a dictionary
        data = {
            "eda": eda,
            "heart_rate": heart_rate,
            "temperature": temperature
        }

        # Stream data into WebSocket
        socketio.emit('Emotibit', data)
        time.sleep(1) # Rate of data upate (per second)

# Route to trigger the simulation
@app.route('/start-emotibit')
def start_emotibit():
    socketio.start_background_task(simulate_emotibit_data)
    return jsonify({"message": "EmotiBit started!"})

        