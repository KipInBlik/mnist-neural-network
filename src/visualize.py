from src.data import load_mnist
from src.network import NeuralNetwork
from src.model_io import load_model
from src.visualization import show_predictions


def main():
    # Load the test data.
    _, _, X_test, y_test = load_mnist()

    # Create the network and restore its trained parameters.
    network = NeuralNetwork()
    load_model(network)

    # Visualize predictions without training.
    show_predictions(network, X_test, y_test)


if __name__ == "__main__":
    main()
