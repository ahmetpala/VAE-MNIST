#### generator.py ####

import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

figures_dir = "figures"
os.makedirs(figures_dir, exist_ok=True)

# Loading decoder
loaded_decoder = keras.models.load_model('decoder.keras')


## Visualizing 100 sample inputs from particular intervals

n = 10  # generate 10*10 digits
figure = np.zeros((28 * n, 28 * n, 1))

# Specifying grids intervals from the latent space
grid_x = np.linspace(-4, 4, n)
grid_y = np.linspace(-4, 4, n)[::-1]

# Decoder for each sample
for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = loaded_decoder.predict(z_sample)
        digit = x_decoded[0].reshape(28, 28, 1)
        figure[i * 28: (i + 1) * 28,
               j * 28: (j + 1) * 28] = digit

plt.figure(figsize=(10, 10))
#Reshape for visualization
fig_shape = np.shape(figure)
figure = figure.reshape((fig_shape[0], fig_shape[1]))

plt.title('100 Samples Generated from Trained Model (Decoder)')
plt.imshow(figure, cmap ='gray')
plt.savefig(os.path.join(figures_dir, 'example_plots.png'))
plt.show()
