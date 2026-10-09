import matplotlib.pyplot as plt
import numpy as np

from src.data import load_mnist
from src.network import NeuralNetwork

def main():
    # Load the dataset.
    X_train, y_train, X_test, y_test = load_mnist()

    # Create the neural network.
    network = NeuralNetwork()

    # Start with a small batch of five images.
    X_batch = X_train[:5]

    # Run forward propagation.
    scores, cache = network.forward(X = X_batch)

    # Select the class with the highest score.
    predictions = np.argmax(scores, axis = 1)

    print("Input shape:", X_batch.shape)
    print("Hidden activations shape:", cache["A1"].shape)
    print("Output scores shape:", scores.shape)

    print("Predictions:", predictions)
    print("Correct labels:", y_train[:5])

if __name__ == "__main__":
    main()