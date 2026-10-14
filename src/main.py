import matplotlib.pyplot as plt
import numpy as np

from backprop import backwards_output
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

    # Backpropagation through the output layer.
    dW2, dA1, db2 = backwards_output(
        cache, dlogits, network.W2
    )

    print("Loss:", loss)
    print("dlogits shape:", dlogits.shape)
    print("dW2 shape:", dW2.shape)
    print("db2 shape:", db2.shape)
    print("dA1 shape:", dA1.shape)

    # Verify that the gradient shapes match their targets.
    assert dW2.shape == network.W2.shape
    assert db2.shape == network.b2.shape
    assert dA1.shape == cache["A1"].shape

    print("All gradient shape checks passed!")
    
if __name__ == "__main__":
    main()