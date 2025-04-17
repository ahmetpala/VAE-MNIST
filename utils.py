import matplotlib.pyplot as plt
import yaml


def load_config():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def plot_sample_images(x_train, save_path):
    """Plot example MNIST input images."""
    plt.figure(figsize=(8, 6))
    for i, idx in enumerate([40, 50, 60, 70, 80, 90]):
        plt.subplot(2, 3, i + 1)
        plt.imshow(x_train[idx], cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
