# app.py

from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to store training progress/state.
# In a real app, consider using a more robust state management approach.
training_state = {
    "training": False,
    "epoch": 0,
    "loss": None,
    "w": None,
    "b": None
}

def train_model():
    """
    Simulates training a simple linear regression model using dummy data.
    Updates the global training_state dictionary at each epoch.
    """
    global training_state
    # Create some dummy data for linear regression
    x_data = np.linspace(-10, 10, 100)
    # True relationship: y = 2x + 3, with added noise
    y_data = 2 * x_data + 3 + np.random.normal(scale=2, size=x_data.shape)

    # Initialize weights and bias
    w = 0.0
    b = 0.0
    learning_rate = 0.001
    num_epochs = 200

    training_state["training"] = True

    for epoch in range(1, num_epochs + 1):
        # Compute predictions and error
        y_pred = w * x_data + b
        error = y_data - y_pred
        loss = np.mean(error ** 2)

        # Compute gradients (Mean Squared Error loss derivatives)
        w_grad = -2 * np.mean(x_data * error)
        b_grad = -2 * np.mean(error)

        # Update parameters
        w -= learning_rate * w_grad
        b -= learning_rate * b_grad

        # Update the global training state for this epoch
        training_state["epoch"] = epoch
        training_state["loss"] = float(loss)
        training_state["w"] = float(w)
        training_state["b"] = float(b)

        # Simulate time taken per epoch (helps visualize training progress)
        time.sleep(0.1)

    # Mark training as finished
    training_state["training"] = False

@app.route('/')
def home():
    return "Welcome to the ML Visualization Project!"

@app.route('/api/train', methods=['POST'])
def start_training():
    """
    Starts the training process in a background thread.
    Returns an error if training is already in progress.
    """
    global training_state
    if training_state["training"]:
        return jsonify({"message": "Training already in progress."}), 400

    # Start the training loop in a new thread
    training_thread = threading.Thread(target=train_model)
    training_thread.start()

    return jsonify({"message": "Training started."})

@app.route('/api/status', methods=['GET'])
def training_status():
    """
    Returns the current state of the training process,
    including the current epoch, loss, weight (w), and bias (b).
    """
    global training_state
    return jsonify(training_state)

if __name__ == '__main__':
    app.run(debug=True)
