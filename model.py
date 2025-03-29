#### model.py ####

import keras
from keras.layers import Input, LSTM, Conv2DTranspose, Dense, Lambda, Reshape, Dropout, BatchNormalization
from keras.models import Model
from keras import backend as K
import tensorflow as tf
from tensorflow.keras import backend as K
#tf.compat.v1.enable_eager_execution()


img_width = 28
img_height = 28


### Defining Encoder (LSTM)

latent_dim = 2 # Latent dimension size
timesteps = 28 # There are 28 timesteps (the row number of the input image)


# Encoder - LSTM with 24 units and Dropout with 0.2 rate
input_image = Input(shape= (timesteps, 28, ), name = 'encoder_input') # For LSTM

# LSTM part
x = LSTM(64, activation = 'relu')(input_image)
x = Dropout(0.2)(x)

# Calculating Mean and Standard Deviation with two separate dense layers
z_mu = Dense(latent_dim, name = 'latent_mu')(x) # Mean
z_sigma = Dense(latent_dim, name = 'latent_sigma')(x) # Standard Deviation


## Reparametrization Trick by Gunderson and Huang

# Defining sampling function
def sample_z(args):
    z_mu, z_sigma = args
    eps = K.random_normal(shape=(K.shape(z_mu)[0], K.int_shape(z_mu)[1]))
    return z_mu + K.exp(z_sigma / 2) * eps

# Defining the labda custom layer
z = Lambda(sample_z, output_shape = (latent_dim, ), name = 'z')([z_mu, z_sigma])

# Final Encoder Model
encoder = Model(input_image, [z_mu, z_sigma, z], name = 'Encoder')


### Defining Decoder (CNN)

# Defining Decoder Input (the latent vector from encoder)
decoder_input = Input(shape = (latent_dim, ), name = 'Encoder_Output_Decoder_Input')

# First Dense Layer to increase the size of decoder input
x = Dense(14*14*64, activation = 'relu', name = 'Dense_Layer_1')(decoder_input)
x = BatchNormalization(name = 'Batchnorm_1')(x) # Batchnormalization

# Reshape the output to transform into tensor
x = Reshape((14, 14, 64), name = 'Reshape')(x)

# First Transposed CNN Layer with Batchnormalization
x = Conv2DTranspose(32, 3, activation = 'relu', strides = (2, 2), padding = 'same', name = 'Transposed_CNN_Layer_1')(x)
x = BatchNormalization(name = 'Batchnorm_2')(x) # Batchnormalization

# Second (Final) Transposed CNN Layer
x = Conv2DTranspose(1, 3, activation = 'sigmoid', padding = 'same', name = 'Transposed_CNN_Layer_2')(x)
# Note that the output size is equal to the input image size (28, 28, 1)
# 'sigmoid' is used for cross entropy loss

# Final Decoder Model
decoder = Model(decoder_input, x, name = 'Decoder')

## Applying Decoder to Encoder Output

z_decoded = decoder(z)


# Defining Custom Layer including both reconstruction and KL Divergence Loss
class CustomLayer(keras.layers.Layer):

    def combined_loss(self, x, z_decoded, z_mu, z_sigma):
        x = keras.layers.Flatten()(x)
        z_decoded = keras.layers.Flatten()(z_decoded)

        # Reconstruction loss
        recon_loss = keras.metrics.binary_crossentropy(x, z_decoded)

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


# Applying Custom Loss

y = CustomLayer()([input_image, z_decoded, z_mu, z_sigma])


# Overall Variational Autoencoder Model
vae = Model(input_image, y, name='VAE')

# Compile VAE
vae.compile(optimizer='adam', loss=None) # Choosing Adam optimizer
