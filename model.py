"""Defines the Variational Autoencoder (VAE) architecture
using LSTM encoder and CNN decoder."""

import tensorflow.keras.backend as K
from keras.layers import (LSTM, BatchNormalization, Conv2DTranspose, Dense,
                          Dropout, Flatten, Input, Lambda, Layer, Reshape)
from keras.metrics import binary_crossentropy
from keras.models import Model


def sample_z(args):
    z_mu, z_sigma = args
    eps = K.random_normal(shape=(K.shape(z_mu)[0], K.int_shape(z_mu)[1]))
    return z_mu + K.exp(z_sigma / 2) * eps


def build_encoder(latent_dim=2):
    """Builds the LSTM-based encoder model."""
    input_image = Input(shape=(28, 28), name='encoder_input')
    x = LSTM(64, activation='relu')(input_image)
    x = Dropout(0.2)(x)
    z_mu = Dense(latent_dim, name='latent_mu')(x)
    z_sigma = Dense(latent_dim, name='latent_sigma')(x)
    z = Lambda(sample_z, output_shape=(latent_dim,), name='z')([z_mu, z_sigma])
    return Model(input_image, [z_mu, z_sigma, z], name='Encoder')


def build_decoder(latent_dim=2):
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

    def combined_loss(self, x, z_decoded, z_mu, z_sigma):
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
        x, z_decoded, z_mu, z_sigma = inputs
        loss = self.combined_loss(x, z_decoded, z_mu, z_sigma)
        self.add_loss(loss)
        return x


def build_vae():
    encoder = build_encoder()
    decoder = build_decoder()
    input_image = encoder.input
    z_mu, z_sigma, z = encoder.output
    z_decoded = decoder(z)
    y = CustomLayer()([input_image, z_decoded, z_mu, z_sigma])
    vae = Model(input_image, y, name='VAE')
    vae.compile(optimizer='adam', loss=None)
    return encoder, decoder, vae


encoder, decoder, vae = build_vae()
