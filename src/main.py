import matplotlib.pyplot as plt
import numpy as np

from backprop import backward_output, backward_hidden
from loss import softmax_and_grad
from src.data import load_mnist
from src.network import NeuralNetwork
from src.training import train_epoch
from src.evaluate import accuracy

def main():
    # Load the dataset.
    X_train, y_train, X_test, y_test = load_mnist()

    # Create the neural network.
    network = NeuralNetwork()

    # Training configuration.
    epochs = 10
    batch_size = 64
    learning_rate = 0.01

    # Train the network over multiple epochs.
    for epoch in range(1, epochs + 1):
        loss = train_epoch(
            network,
            X_train,
            y_train,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")

    test_accuracy = accuracy(network, X_test, y_test)

    print(f"Test accuracy: {test_accuracy:.2%}")

if __name__ == "__main__":
    main()