import numpy as np
from src.activations import relu_derivative

def backward_output(cache, dlogits, W2):
    """
    Calculate gradients for the output layer.

    cache: intermediate values from the forward pass
    dlogits: gradient of the loss with respect to output logits
    W2: weights from the hidden layer to the output layer
    """
    A1 = cache["A1"]

    # Validation.
    if A1.ndim != 2 or dlogits.ndim != 2:
        raise ValueError("A1 and dlogits must be 2D arrays")

    if A1.shape[0] != dlogits.shape[0]:
        raise ValueError("A1 and dlogits must have the same batch size")

    if A1.shape[1] != W2.shape[0]:
        raise ValueError("A1 and W2 must have the same number of columns")

    if dlogits.shape[1] != W2.shape[1]:
        raise ValueError("dlogits and W2 must have the same number of columns")

    # Gradients for the output layer parameters.
    dW2 = A1.T @ dlogits
    db2 = np.sum(dlogits, axis=0)

    # Pass the gradient back to the hidden activations.
    dA1 = dlogits @ W2.T

    return dW2, db2, dA1

def backward_hidden(cache, dA1):
    """
    Calculate gradients for the hidden layer parameters.

    cache contains X and Z1 from the forward pass.
    dA1 is the gradient with respect to the hidden activations.
    """
    X = cache["X"]
    Z1 = cache["Z1"]

    if dA1.shape != Z1.shape:
        raise ValueError("dA1 and Z1 must have the same shape")

    # Pass the gradient backward through ReLU.
    dZ1 = dA1 * relu_derivative(Z1)

    # Calculate gradient for the first layer.
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    return dW1, db1