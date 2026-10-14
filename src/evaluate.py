import numpy as np


def accuracy(network, X, y, batch_size=256):
    correct = 0

    for start in range(0, len(X), batch_size):
        X_batch = X[start:start + batch_size]
        y_batch = y[start:start + batch_size]

        # Forward pass only: do not update the weights.
        logits, _ = network.forward(X_batch)

        # The predicted class is the class with the highest logit.
        predictions = np.argmax(logits, axis=1)

        correct += np.sum(predictions == y_batch)

    return correct / len(X)
