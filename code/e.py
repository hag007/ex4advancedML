'''This script demonstrates how to build a variational autoencoder with Keras.

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
import math
import random
import os
batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0


x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# instantiate VAE model
vae = Model(x, x_decoded_mean)

# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()


# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))


encoder = Model(x, [z_mean, z_log_var, z], name='encoder')
predictions = encoder.predict(x_test)

while True:
    r1 = int(math.floor(random.random() *len(y_test)))
    r2 = int(math.floor(random.random() *len(y_test)))
    if y_test[r1] != y_test[r2]:
        p1 = predictions[2][r1]
        p2 = predictions[2][r2]
        print("choson numbers are: {} {}".format(y_test[r1],y_test[r2]))
        f=open("//home//hag007//ex4py//e_digits", 'w+')
        f.write("choson digits are: {} {}".format(y_test[r1],y_test[r2]))
        f.close()
        break

zs = []
for i in range(10):
    zs.append([min(p1[0], p2[0])+(random.random() * abs(p1[0] - p2[0])),
    min(p1[1], p2[1])+random.random() * abs(p1[1] - p2[1])])


print(zs)
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(inputs=decoder_input, outputs= _x_decoded_mean)

z_sample = np.array(zs)
x_decoded = generator.predict(z_sample)
print(x_decoded)


# display reconstruction
for i, cur in enumerate(x_decoded):
    plt.imshow(cur.reshape(28, 28))
    plt.gray()
    if not os.path.exists("output"):
        os.mkdir("output")
    plt.savefig("output/e_{}.png".format(i))
