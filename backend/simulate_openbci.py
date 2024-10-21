import random
import time
from flask import Flask, jsonify
from flask_socketio import SocketIO 

app = Flask(__name__)
socketio = SocketIO(app)

"""
Description: Simulate EGG 4 channels data to mimic OpenBCI headband.

"""
def simulate_open_bci():
    while True:
        # Generate EGG data for 4 channels
        egg_data = {
            # simulate data as 250 hz (same as OpenBCI headband)
            "channel 1": [random.uniform(-100, 100) for _ in range(250)], 
            "channel 2": [random.uniform(-100, 100) for _ in range(250)], 
            "channel 3": [random.uniform(-100, 100) for _ in range(250)],
            "channel 4": [random.uniform(-100, 100) for _ in range(250)]
        }

        # Stream EGG data into WebSocket
        socketio.emit('OpenBCI Headband', egg_data)
        time.sleep(1) # Data is updated per second

@app.route('/start_openbci')
def start_openbci():
    socketio.start_background_task(simulate_open_bci)
    return jsonify({"message": "OpenBCI Headband started!"})
