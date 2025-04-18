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


def plot_loss(history, save_path):
    """Plot training and validation loss curves."""
    plt.plot(history['loss'], label="Train loss")
    plt.plot(history['val_loss'], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss change over epochs")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_latent_space(mu, y_test, save_path):
    """Plot 2D latent space."""
    # TODO: Check if mu is high dimensional (>2), if so, apply PCA
    plt.figure(figsize=(12, 8))
    plt.scatter(mu[:, 0], mu[:, 1], c=y_test, cmap='brg')
    plt.xlabel('mu[:, 0]')
    plt.ylabel('mu[:, 1]')
    plt.title("Latent space visualization")
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()
