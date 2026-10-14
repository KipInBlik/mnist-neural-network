from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

def save_model(network, path=OUTPUT_DIR / "mnist_model.npz"):
    """Save the network model to a file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        path,
        W1=network.W1,
        b1=network.b1,
        W2=network.W2,
        b2=network.b2
    )


def load_model(network, path=OUTPUT_DIR / "mnist_model.npz"):
    """Load the network model from a file."""
    with np.load(path) as data:
        network.W1 = data["W1"].copy()
        network.b1 = data["b1"].copy()
        network.W2 = data["W2"].copy()
        network.b2 = data["b2"].copy()

    return network
