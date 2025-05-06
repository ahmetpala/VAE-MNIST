import os

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model


def generate_samples():
    """
    Generate and visualize 100 samples from a trained VAE decoder.
    The result is saved to 'figures/example_plots.png'.
    """
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)

    decoder = load_model("decoder.keras")

    n = 10  # 10x10 samples to be generated
    # TODO: Adjust the visualizations for different set of examples
    figure = np.zeros((28 * n, 28 * n, 1))

    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    # TODO: Add PCA for when we use latent_dimension > 2

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample, verbose=0)
            digit = x_decoded[0].reshape(28, 28, 1)
            figure[i * 28: (i + 1) * 28,
                   j * 28: (j + 1) * 28] = digit

    figure = figure.reshape((28 * n, 28 * n))

    plt.figure(figsize=(10, 10))
    plt.title("100 samples generated from trained decoder")
    plt.imshow(figure, cmap='gray')
    plt.axis("off")
    plt.savefig(os.path.join(figures_dir, "example_plots.png"),
                dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    generate_samples()
