import argparse
import os
from pathlib import Path

from keras.datasets import mnist

from generator import generate_samples
from model import build_vae
from utils import load_config, plot_latent_space, plot_loss, plot_sample_images


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    config = load_config()

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
