# Variational Autoencoder with LSTM and CNN on MNIST

This repository contains a Variational Autoencoder (VAE) model trained on the MNIST dataset. The encoder is based on an LSTM, while the decoder uses a CNN with transposed convolutions. The project is implemented using Keras and TensorFlow.

---

## Project Structure

- `model.py` – Defines the VAE architecture with a custom loss layer.
- `generator.py` – Generates new MNIST-style digits using the trained decoder.
- `main.py` – Loads data, trains the VAE, saves outputs, and visualizes training results.
- `requirements.txt` – Lists minimal dependencies.
- `.pre-commit-config.yaml` – Defines formatting and linting rules.

---

## Model Architecture

- **Encoder**: LSTM with 64 units → Dropout → two Dense layers for `μ` and `σ` → Sampling using the reparameterization trick.
- **Decoder**: Dense → Reshape → two Conv2DTranspose layers.
- **Loss**: Combines reconstruction loss (binary cross-entropy) with KL divergence.

---

## 🧪 How to Run

1. **Clone the repository:**

```bash
git clone https://github.com/ahmetpala/vae-lstm-cnn-mnist.git
cd vae-lstm-cnn-mnist
```

2. **Create a virtual environment (optional but recommended):**

```bash
python3 -m venv .venv
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

This will:
- Train the VAE for 3 epochs on MNIST
- Save model files: `vae.keras` and `decoder.keras`
- Save plots to the `figures/` folder:
  - `example_plots.png` – sample input images
  - `loss.png` – training and validation loss
  - `latent_space.png` – 2D latent space
  - `example_plots.png` (overwritten) – 100 generated samples from latent grid

---

## Code Style

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
