import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
from keras.datasets import mnist

from generator import generate_samples
from model import build_vae
from utils import load_config


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


def main():

    config = load_config()

    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', type=int,
                        default=config["train"]["latent_dim"])
    parser.add_argument('--epochs', type=int,
                        default=config["train"]["epochs"])
    parser.add_argument('--batch_size', type=int,
                        default=config["train"]["batch_size"])
    args = parser.parse_args()

    encoder, decoder, vae = build_vae(args.latent_dim)

    figures_dir = config["paths"]["figures_dir"]
    Path(figures_dir).mkdir(exist_ok=True)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    plot_sample_images(x_train, os.path.join(figures_dir, 'example_plots.png'))

    print(encoder.summary())
    print(decoder.summary())
    print(vae.summary())

    history = vae.fit(
        x_train, None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2
    )

    plot_loss(history.history, os.path.join(figures_dir, 'loss.png'))

    decoder.save("decoder.keras")
    vae.save("vae.keras")

    mu, _, _ = encoder.predict(x_test, verbose=0)
    plot_latent_space(mu, y_test, os.path.join(
        figures_dir, 'latent_space.png'))

    generate_samples()


if __name__ == "__main__":
    main()
