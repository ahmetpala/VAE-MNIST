# Variational Autoencoder with LSTM and CNN on MNIST

This project builds a Variational Autoencoder (VAE) trained on the MNIST dataset.

- The **encoder** is based on an LSTM.
- The **decoder** uses Conv2DTranspose layers.
- Implemented with **Keras** and **TensorFlow**.

---

## Project structure

- `model.py` – Defines the VAE architecture with a custom loss layer.
- `generator.py` – Generates new MNIST-style digits using the trained decoder.
- `main.py` – Loads data, trains the VAE, saves outputs, and visualizes training results.
- `requirements.txt` – Lists minimal dependencies.
- `.pre-commit-config.yaml` – Defines formatting and linting rules.

---

## Model architecture

- **Encoder**: LSTM with 64 units → Dropout → two Dense layers for `μ` and `σ` → Sampling using the reparameterization trick.
- **Decoder**: Dense → Reshape → two Conv2DTranspose layers.
- **Loss**: Combines reconstruction loss (binary cross-entropy) with KL divergence.

---

## How to run

1. **Clone the repository:**

```bash
git clone https://github.com/ahmetpala/vae-lstm-cnn-mnist.git
cd vae-lstm-cnn-mnist
```

2. **Create a virtual environment (optional but recommended):**

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run the training and evaluation:**

```bash
python main.py
```

You can also specify the model parameters with different set of hyperparameters:

```bash
python main.py --latent_dim 4 --epochs 100 --batch_size 64
```

Default values:

- latent_dim = 2
- epochs = 100
- batch_size = 128

This will:
- Train the VAE for 100 epochs on MNIST
- Save model files: `vae.keras` and `decoder.keras`
- Save plots to the `figures/` folder:
  - `example_plots.png` – sample input images
  - `loss.png` – training and validation loss
  - `latent_space.png` – 2D latent space
  - `example_plots.png` (overwritten) – 100 generated samples from latent grid

---

## Code style

Pre-commit hooks are enabled for:

- `autopep8` (PEP8 formatting)
- `isort` (import sorting)
- `flake8` (linting)

To activate:

```bash
pip install pre-commit
pre-commit install
```

To manually run on all files:

```bash
pre-commit run --all-files
```

---

## Requirements

- Python 3.10+
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib

---

## Outputs

All visual outputs are saved under the `figures/` directory. Model files are saved in the project root.

---

## Author

- Ahmet Pala

---

## License

This project is open-source and intended for educational use.
