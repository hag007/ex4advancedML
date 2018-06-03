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
import math
import random
import os

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 16
epochs = 1
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


dense_1 = Dense(intermediate_dim, activation='relu')
_dense_1 = dense_1(z)
dense_2 = Dense(shape[1] * shape[2] * shape[3], activation='relu')
_dense_2 = dense_2(_dense_1)
reshape_1 = Reshape((shape[1], shape[2], shape[3]))
_reshape_1 = reshape_1(_dense_2)
conv_1 = Conv2D(8, (3, 3), activation='relu', padding='same')
_conv_1 = conv_1(_reshape_1)
up_1 = UpSampling2D((2, 2))
_up_1 = up_1(_conv_1)
conv_2 = Conv2D(16, (3, 3), activation='relu', padding='same')
_conv_2 = conv_2(_up_1)
up_2 = UpSampling2D((2, 2))
_up_2 = up_2(_conv_2)
out = Conv2D(1, (3, 3), activation='sigmoid', padding='same')
outputs = out(_up_2)

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

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

decoder_input = Input(shape=(latent_dim,))
_dense_1_decoded = dense_1(decoder_input)
_dense_2_decoded = dense_2(_dense_1_decoded)
_reshape_1_decoded = reshape_1(_dense_2_decoded)
_conv_1_decoded = conv_1(_reshape_1_decoded)
_up_1_decoded = up_1(_conv_1_decoded)
_conv_2_decoded = conv_2(_up_1_decoded)
_up_2_decoded = up_2(_conv_2_decoded)
_out_decoded = out(_up_2_decoded)

generator = Model(inputs=decoder_input, outputs= _out_decoded)
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

predictions = encoder.predict(x_test)
while True:
    r1 = int(math.floor(random.random() *len(y_test)))
    r2 = int(math.floor(random.random() *len(y_test)))
    if y_test[r1] != y_test[r2]:
        p1 = predictions[2][r1]
        p2 = predictions[2][r2]
        print("chosen numbers are: {} {}".format(y_test[r1],y_test[r2]))
        f=open("../g_e_digits", 'w+')
        f.write("chosen digits are: {} {}".format(y_test[r1],y_test[r2]))
        f.close()
        break

zs = []
for i in range(10):
    zs.append([min(p1[0], p2[0])+(random.random() * abs(p1[0] - p2[0])),
    min(p1[1], p2[1])+random.random() * abs(p1[1] - p2[1])])
print(zs)

z_sample = np.array(zs)
x_decoded = generator.predict(z_sample)

# display reconstruction
for i, cur in enumerate(x_decoded):
    plt.imshow(cur.reshape(28, 28))
    plt.gray()
    if not os.path.exists("output"):
        os.mkdir("output")
    plt.savefig("output/g_e_{}.png".format(i))
