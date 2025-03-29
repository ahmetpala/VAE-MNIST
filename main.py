#### main.py ####

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from model import*
import os

figures_dir = "figures"
os.makedirs(figures_dir, exist_ok=True)


# Loading & Normalizing the MNIST Dataset
# Loading
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalization
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Parametrization of image sizes
img_width  = x_train.shape[1]
img_height = x_train.shape[2]

# Visualising Sample Images
plt.figure(1)
plt.subplot(231)
plt.imshow(x_train[40][:,:], cmap='gray')

plt.subplot(232)
plt.imshow(x_train[50][:,:], cmap='gray')

plt.subplot(233)
plt.imshow(x_train[60][:,:], cmap='gray')

plt.subplot(234)
plt.imshow(x_train[70][:,:], cmap='gray')

plt.subplot(235)
plt.imshow(x_train[80][:,:], cmap='gray')

plt.subplot(236)
plt.imshow(x_train[90][:,:], cmap='gray')
plt.savefig(os.path.join(figures_dir, 'example_plots.png'))
plt.show()

# Printing Encoder, Decoder and Overall VAE model summaries
print(encoder.summary()) # Encoder Model Summary
print(decoder.summary()) # Decoder Model Summary
print(vae.summary()) # VAE Model Summary


# Training the Built Variational Autoencoder

VAE_model = vae.fit(x_train, None, epochs = 3, batch_size = 128, validation_split = 0.2)
# Training for 30 epochs takes 1hrs and 7 minutes


# Visualizing the combined loss

plt.title("Loss Function Change per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss (Reconstruction + KL Divergence)")
plt.plot(np.arange(len(VAE_model.history['loss'])), VAE_model.history['loss'], label = "Train Loss")
plt.plot(np.arange(len(VAE_model.history['val_loss'])), VAE_model.history['val_loss'], label = "Validation Loss")
plt.legend()
plt.savefig(os.path.join(figures_dir, 'loss.png'))
plt.show()


## Saving and Loading VAE and Decoder

# Saving model and decoder separately
decoder.save('decoder.keras')
vae.save('vae.keras')


## Visualizing the latent space

mu, _, _ = encoder.predict(x_test)

#Plotting mu values
plt.figure(figsize = (15, 10))
plt.title('Latent Space of Encoder Output')
plt.scatter(mu[:, 0], mu[:, 1], c = y_test, cmap = 'brg')
plt.xlabel('mu[:, 0]')
plt.ylabel('mu[:, 1]')
plt.colorbar()
plt.savefig(os.path.join(figures_dir, 'latent_space.png'))
plt.show()

### Generation 100 Samples using Loaded Decoder (generator.py)
exec(open('generator.py').read())