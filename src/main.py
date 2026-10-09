import matplotlib.pyplot as plt
import numpy as np

from loss import softmax_and_grad
from src.data import load_mnist
from src.network import NeuralNetwork

def main():
    # Load the dataset.
    X_train, y_train, X_test, y_test = load_mnist()

    # Create the neural network.
    network = NeuralNetwork()

    # Start with a small batch of five images.
    X_batch = X_train[:5]
    y_batch = y_train[:5]

    # Run forward propagation.
    logits, cache = network.forward(X = X_batch)

    # Calculate probabilities, loss and output gradients.
    loss, probabilities, dlogits = softmax_and_grad(
        logits, y_batch
    )

    predictions = np.argmax(probabilities, axis=1)

    print("Logits shape:", logits.shape)
    print("Probabilities shape:", probabilities.shape)
    print("Gradient shape:", dlogits.shape)

    print("Probability sums:", probabilities.sum(axis=1))
    print("Initial loss:", loss)

    print("Correct labels:", y_batch)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()