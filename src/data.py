
import gzip
import struct
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_images(filename):
    """Load MNIST images and normalize their pixel values."""
    path = DATA_DIR / filename

    with gzip.open(path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))

        if magic != 2051:
            raise ValueError(f"Invalid image file: {filename}")

        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows * cols)

    return images.astype(np.float32) / 255.0


def load_labels(filename):
    """Load MNIST labels."""
    path = DATA_DIR / filename

    with gzip.open(path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))

        if magic != 2049:
            raise ValueError(f"Invalid label file: {filename}")

        labels = np.frombuffer(f.read(), dtype=np.uint8).copy()

    if len(labels) != num_labels:
        raise ValueError(f"Invalid number of labels in {filename}")

    return labels


def load_mnist():
    """Load the complete MNIST training and test sets."""
    X_train = load_images("train-images-idx3-ubyte.gz")
    y_train = load_labels("train-labels-idx1-ubyte.gz")

    X_test = load_images("t10k-images-idx3-ubyte.gz")
    y_test = load_labels("t10k-labels-idx1-ubyte.gz")

    return X_train, y_train, X_test, y_test