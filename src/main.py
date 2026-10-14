import matplotlib.pyplot as plt
import numpy as np

from backprop import backward_output, backward_hidden
from loss import softmax_and_grad
from src.data import load_mnist
from src.network import NeuralNetwork
from src.training import train_step

def main():
    # Load the dataset.
    X_train, y_train, X_test, y_test = load_mnist()

    # Create the neural network.
    network = NeuralNetwork()

    # Start with a small batch of five images.
    X_batch = X_train[:5]
    y_batch = y_train[:5]

    # Measure the loss before training.
    logits, _ = network.forward(X_batch)
    loss_before, _, _ = softmax_and_grad(logits, y_batch)

    # Perform one gradient descent step.
    train_step(network, X_batch, y_batch, learning_rate=0.01)

    # Measure the loss after training.
    logits, _ = network.forward(X_batch)
    loss_after, _, _ = softmax_and_grad(logits, y_batch)

    # Print the results.
    print("Loss before:", loss_before)
    print("Loss after: ", loss_after)
    print("Loss decreased:", loss_after < loss_before)

    # Basic validation.
    assert np.isfinite(loss_before), "Initial loss is not finite!"
    assert np.isfinite(loss_after), "Loss after training is not finite!"

    print("Training step completed successfully!")

if __name__ == "__main__":
    main()