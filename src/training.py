from src.backprop import backward_output, backward_hidden
from src.loss import softmax_and_grad

def train_step(network, X_batch, y_batch, learning_rate=0.01):
    # 1. Forward propagation.
    logits, cache = network.forward(X_batch)

    # 2. Calculate loss and gradient at the output.
    loss, probabilities, dlogits = softmax_and_grad(logits, labels=y_batch)

    # 3. Backpropagation through the output layer.
    dW2, db2, dA1 = backward_output(
        cache, dlogits, network.W2
    )

    # 4. Backpropagation through the hidden layer.
    dW1, db1 = backward_hidden(cache, dA1)

    # 5. Update all parameters.
    network.W1 -= learning_rate * dW1
    network.b1 -= learning_rate * db1
    network.W2 -= learning_rate * dW2
    network.b2 -= learning_rate * db2

    return loss



