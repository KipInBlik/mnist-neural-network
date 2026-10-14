import numpy as np

def backwards_output(cache, dlogits, W2):
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

    if A1.shape[1] != W2.shape[1]:
        raise ValueError("A1 and W2 must have the same number of columns")

    if dlogits.shape[1] != W2.shape[1]:
        raise ValueError("dlogits and W2 must have the same number of columns")

    # Gradients for the output layer parameters.
    dW2 = A1.T @ dlogits
    db2 = np.sum(dlogits, axis=0, keepdims=True)

    # Pass the gradient back to the hidden activations.
    dA1 = dlogits @ W2.T

    return dW2, dA1, db2
