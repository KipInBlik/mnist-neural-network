import numpy as np

def softmax(logits):
    """Calculate the numeric softmax-probabilities for 2D logits."""
    if logits.ndim != 2:
        raise ValueError("Logits has to be a 2D array.")

    # Subtract the max value in the array of logits to avoid overflow
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)

    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def validate_inputs(logits, labels):
    """Verifies that logits and labels have a valid shape."""
    if logits.ndim != 2:
        raise ValueError("Logits has to be a 2D array.")

    labels = np.asarray(labels, dtype=np.int32)
    batch_size, num_classes = logits.shape

    if labels.shape != (batch_size, ):
        raise ValueError("Labels must contain one label per image.")

    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("Labels outside the valid class range.")

    return labels, batch_size

def cross_entropy_loss_and_grad(probabilities, labels):
    """Calculate the average cross entropy loss and gradient with respect to logits."""
    batch_size = probabilities.shape[0]
    row_indices = np.arange(batch_size)

    # 1. Calculate the loss
    correct_class_probs = probabilities[row_indices, labels]
    loss = -np.mean(np.log(correct_class_probs + 1e-15)) # to avoid log(0)

    # 2. Calculate the gradient (by using an elegant notation made possible by the combination of softmax and cross-entropy.
    dlogits = probabilities.copy()
    dlogits[row_indices, labels] -= 1
    dlogits /= batch_size

    return loss, dlogits

def softmax_and_grad(logits, labels):
    """Calculate the average cross-entropy loss and gradient with respect to logits."""
    labels, batch_size = validate_inputs(logits, labels)

    probabilities = softmax(logits=logits)
    loss, dlogits = cross_entropy_loss_and_grad(probabilities, labels)

    return loss, probabilities, dlogits

