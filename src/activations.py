import numpy as np

def relu(z):
    """Apply Rectified Linear Unit element-wise.
        This is the chosen activation function for our simple
        neural network."""
    return np.maximum(0, z)

def sigmoid(z):
    """Apply Sigmoid element-wise.
        This is an alternative to the ReLU activation function for additional
        testing of our neural network."""
    return 1 / (1 + np.exp(-z))