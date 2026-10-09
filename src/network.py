import numpy as np

from src.activations import relu

class NeuralNetwork:
    def __init__(
        self,
        input_size = 784,
        hidden_size = 128,
        output_size = 10,
        seed = 42,
    ):
        rng = np.random.default_rng(seed=seed)

        # Weights from the input layer to the hidden layer.
        self.W1 = rng.normal(
            0.0,
            np.sqrt(2.0/input_size), # We use 2.0 here since the activation function, ReLU, removes half of the input.
            size=(input_size, hidden_size),
        ).astype(np.float32)

        self.b1 = np.zeros(hidden_size).astype(np.float32)

        self.W2 = rng.normal(
            0.0,
            np.sqrt(1.0/hidden_size),
            size=(hidden_size, output_size),
        ).astype(np.float32)

        self.b2 = np.zeros(output_size).astype(np.float32)

    def forward(self, X):
        """Calculate and return the output of the forward pass for a given input vector."""
        if X.ndim != 2 or X.shape[1] != self.W1.shape[0]:
            raise ValueError("Input shape is not valid.")

        # Input layer -> Hidden layer.
        Z1 = X @ self.W1 + self.b1
        A1 = relu(Z1)

        # Hidden layer -> Output layer.
        Z2 = A1 @ self.W2 + self.b2

        # Store the intermediate values for backpropegation.
        cache = {
            "X" : X,
            "Z1": Z1,
            "A1": A1,
        }

        return Z2, cache

