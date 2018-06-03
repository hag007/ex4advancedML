'''This script demonstrates how to build a variational autoencoder with Keras.

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import sys
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.utils import plot_model

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0

k_constants = [0 for i in range(latent_dim)]
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
# z_log_var = Dense(latent_dim)(h)
z_log_var = Input(tensor=K.variable(k_constants))

# z_log_var = Dense(latent_dim, activation='relu')(z_log_var)

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
vae = Model([x, z_log_var], x_decoded_mean)

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

print("#####")

test_digits_mean = {}
test_digits_var = {}
test_digits_z = {}

encoder = Model([x, z_log_var], z, name='encoder')


predictions = encoder.predict(x_test)

xs = []
ys = []
ls = []
for i in range(len(y_test)):
    xs.append(predictions[i][0])
    ys.append(predictions[i][1])
    ls.append(y_test[i])
    if not test_digits_z.has_key(str(y_test[i])) and len(test_digits_z.keys()) != 10:
        test_digits_z[str(y_test[i])] = predictions[i]

for k in test_digits_z.keys():
    sys.stdout.write("{}\t".format(k),)
print("")
for v in test_digits_z.values():
    sys.stdout.write("{}\t".format(v),)
print("")



# xs = [v[0] for k, v in test_digits_z.iteritems()]
# ys = [v[1] for k, v in test_digits_z.iteritems()]
# ds = [k for k, v in test_digits_z.iteritems()]
plt.scatter(xs, ys, c=ls, cmap="gist_rainbow")

# for x, y, d in zip(xs, ys, ds):
#     plt.annotate(
#         d,
#         xy=(x, y), xytext=(-20, 20),
#         textcoords='offset points', ha='right', va='bottom',
#         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
#         arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
if not os.path.exists("output"):
    os.mkdir("output")
plt.savefig("output/f_c.png")




