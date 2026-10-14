from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

def show_predictions(network, X, y, num_images=20):
    # Select random test images.
    indices = np.random.choice(
        len(X),
        size=min(num_images, len(X)),
        replace=False
    )

    X_sample = X[indices]
    y_sample = y[indices]

    # Predict all selected images in one forward pass.
    logits, _ = network.forward(X_sample)
    predictions = np.argmax(logits, axis=1)

    # Create a grid of images.
    cols = 5
    rows = int(np.ceil(len(indices) / cols))

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(12, 2.5 * rows)
    )
    axes = np.asarray(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i >= len(indices):
            ax.axis("off")
            continue

        image = X_sample[i].reshape(28, 28)
        predicted = predictions[i]
        actual = y_sample[i]

        ax.imshow(image, cmap="gray")
        ax.set_title(
            f"Predicted: {predicted}\nActual: {actual}",
            color="green" if predicted == actual else "red"
        )
        ax.axis("off")

    fig.suptitle("MNIST: Neural Network Predictions")
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "predictions.png", dpi=150)
    plt.show()
