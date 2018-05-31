'''This script demonstrates how to build a variational autoencoder with Keras.

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import sys
from keras.layers import Input, Dense, Lambda, Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.utils import plot_model

batch_size = 1000
original_dim = 784
latent_dim = 2
intermediate_dim = 16
epochs = 50
epsilon_std = 1.0

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


################ Definition of Keras ConvNet architecture
input_shape = (28, 28, 1)
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
# shape info needed to build decoder model
shape = K.int_shape(x)

# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

x = Dense(intermediate_dim, activation='relu')(z)
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)
x = Reshape((shape[1], shape[2], shape[3]))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

vae = Model(inputs, outputs, name='vae')

###########################################################

# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(K.flatten(inputs),K.flatten(outputs))
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()


# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), input_shape[0],input_shape[1],input_shape[2]))

x_test = x_test.reshape((len(x_test),input_shape[0],input_shape[1],input_shape[2]))

print(x_train.shape)

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

print("#####")

test_digits_z = {}

encoder = Model(inputs, z, name='encoder')

predictions = encoder.predict(x_test)

for i in range(len(y_test)):
    if not test_digits_z.has_key(str(y_test[i])):
        test_digits_z[str(y_test[i])] = predictions[i]
    if len(test_digits_z.keys()) == 10:
        break


for k in test_digits_z.keys():
    sys.stdout.write("{}\t".format(k),)
print("")
for v in test_digits_z.values():
    sys.stdout.write("{}\t".format(v),)
print("")

xs = [v[0] for k, v in test_digits_z.iteritems()]
ys = [v[1] for k, v in test_digits_z.iteritems()]
ds = [k for k, v in test_digits_z.iteritems()]
plt.scatter(xs, ys)

for x, y, d in zip(xs, ys, ds):
    plt.annotate(
        d,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.savefig("../g_c.png")
