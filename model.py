"""Defines the Variational Autoencoder (VAE) architecture
using LSTM encoder and CNN decoder."""

import tensorflow.keras.backend as K
from tensorflow.keras.layers import (LSTM, BatchNormalization, Conv2DTranspose,
                                     Dense, Dropout, Flatten, Input, Lambda,
                                     Layer, Reshape)
from tensorflow.keras.metrics import binary_crossentropy
from tensorflow.keras.models import Model


def sample_z(args):
    """Samples latent vector z using reparameterization trick."""
    z_mu, z_sigma = args
    eps = K.random_normal(shape=(K.shape(z_mu)[0], K.int_shape(z_mu)[1]))
    return z_mu + K.exp(z_sigma / 2) * eps


# TODO: Adjust the encoder for CNNs, make it optional
def build_encoder(latent_dim):
    """Creates the encoder model using an LSTM layer.
    Returns z_mu, z_sigma, and sampled z."""
    input_image = Input(shape=(28, 28), name='encoder_input')
    x = LSTM(64, activation='relu')(input_image)
    x = Dropout(0.2)(x)
    z_mu = Dense(latent_dim, name='latent_mu')(x)
    z_sigma = Dense(latent_dim, name='latent_sigma')(x)
    z = Lambda(sample_z, output_shape=(latent_dim,), name='z')([z_mu, z_sigma])
    return Model(input_image, [z_mu, z_sigma, z], name='Encoder')


def build_decoder(latent_dim):
    """Creates the decoder model using transposed convolution layers.
    Takes latent vector as input and outputs reconstructed image."""
    decoder_input = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(14 * 14 * 64, activation='relu')(decoder_input)
    x = BatchNormalization()(x)
    x = Reshape((14, 14, 64))(x)
    x = Conv2DTranspose(32, 3, activation='relu',
                        strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
    return Model(decoder_input, x, name='Decoder')


class CustomLayer(Layer):
    """Custom Keras layer that adds VAE loss (reconstruction + KL)."""

    def combined_loss(self, x, z_decoded, z_mu, z_sigma):
        """Computes total VAE loss: reconstruction + KL divergence."""
        x = Flatten()(x)
        z_decoded = Flatten()(z_decoded)

        # Reconstruction loss
        recon_loss = binary_crossentropy(x, z_decoded)

        # KL Divergence Loss
        kl_loss = -0.01 * 0.5 * K.mean(
            1 + z_sigma - K.square(z_mu) - K.exp(z_sigma), axis=-1
        )

        return K.mean(recon_loss + kl_loss)

    def call(self, inputs):
        """Adds VAE loss to model and returns input unchanged."""
        x, z_decoded, z_mu, z_sigma = inputs
        loss = self.combined_loss(x, z_decoded, z_mu, z_sigma)
        self.add_loss(loss)
        return x


def build_vae(latent_dim):
    """Builds the VAE model by connecting encoder, decoder, and loss layer.
    Returns encoder, decoder, and compiled VAE model."""
    encoder = build_encoder(latent_dim)
    decoder = build_decoder(latent_dim)
    input_image = encoder.input
    z_mu, z_sigma, z = encoder.output
    z_decoded = decoder(z)
    y = CustomLayer()([input_image, z_decoded, z_mu, z_sigma])
    vae = Model(input_image, y, name='VAE')
    vae.compile(optimizer='adam', loss=None)
    return encoder, decoder, vae
