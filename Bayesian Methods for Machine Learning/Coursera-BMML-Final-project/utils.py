import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import GPy
import GPyOpt
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Lambda, InputLayer, concatenate, Activation, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Deconv2D
from keras.losses import MSE
from keras.models import Model, Sequential
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.utils import np_utils
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import os




class CelebA:
    def __init__(self, path, sess, train=True, batch_size=32, height=218, width=178, channels=3, threads=1, file_type='.jpg'):
        image_filenames = [os.path.join(path, img) for img in os.listdir(path) if img.endswith(file_type)]
        if train:
            image_filenames = image_filenames[:-5000]
        else:
            image_filenames = image_filenames[-5000:]
        all_images = ops.convert_to_tensor(image_filenames, dtype=dtypes.string)
        input_queue = tf.train.slice_input_producer([image_filenames], shuffle=False)
        file_content = tf.read_file(input_queue[0])
        image = tf.image.decode_jpeg(file_content, channels=3)
        image.set_shape([height, width, channels])
        image_cropped = image[45:-45, 25:-25]
        image_cropped = tf.image.resize_images(image_cropped, (64, 64))
        batch = tf.train.batch([image_cropped], batch_size=batch_size, num_threads=threads)
        self.batch = tf.cast(batch, tf.float32)/256
        self.n_batches = len(image_filenames) // batch_size
        self.sess = sess
    
    def __iter__(self):
        return self
    
    def __next__(self):
        x = self.sess.run(self.batch)
        return x, x, None
    
    def next(self):
        return self.__next__()
    
def create_encoder(input_dims, base_filters=64, layers=4, latent=512):
    w = input_dims[0]//2**layers
    h = input_dims[1]//2**layers
    c = base_filters*2**(layers-1)
    encoder = Sequential()
    encoder.add(InputLayer(input_dims))
    for i in range(layers):
        encoder.add(Conv2D(filters=base_filters*2**i, kernel_size=(5, 5), strides=(2, 2), padding='same', bias=False))
        encoder.add(BatchNormalization(axis=3))
        encoder.add(Activation(K.relu))
    encoder.add(Reshape([w*h*c]))
    encoder.add(Dense(latent*2))
    return encoder

def create_decoder(output_dims, base_filters=64, layers=4, latent=512):
    w = output_dims[0]//2**layers
    h = output_dims[1]//2**layers
    c = base_filters*2**(layers-1)
    decoder = Sequential()
    decoder.add(InputLayer([latent]))
    decoder.add(Dense(w*h*c))
    decoder.add(Reshape([w, h, c]))
    for i in range(layers-1, 0, -1):
        decoder.add(Deconv2D(filters=base_filters*2**i, kernel_size=(5, 5), strides=(2, 2), padding='same', bias=False))
        decoder.add(BatchNormalization(axis=3))
        decoder.add(Activation(K.relu))
    decoder.add(Deconv2D(filters=3, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    return decoder

def sample(mean_log_var):
    mean, log_var = mean_log_var
    eps_shape = mean.get_shape()
    epsilon = K.random_normal(shape=eps_shape)
    z = epsilon*K.exp(log_var/2)+mean
    return z

def create_vae(batch_size, base_filters=64, latent=8,
               image_size=64, learning_rate=0.001,
               reconstruction_weight=1000, layers=4):
    '''
    Constructs VAE model with given parameters.
    :param batch_size: size of a batch (used for placeholder)
    :param base_filters: number of filters after first layer. Other layers will double this number
    :param latent: latent space dimension
    :param image_size: size of input image
    Returns compiled Keras model along with encoder and decoder
    '''
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    x = Input(batch_shape=(batch_size, image_size[0], image_size[1], 3))
    encoder = create_encoder([image_size[0], image_size[1], 3], base_filters=base_filters, latent=latent, layers=layers)
    decoder = create_decoder([image_size[0], image_size[1], 3], base_filters=base_filters, latent=latent, layers=layers)
    mean_log_var = encoder(x)
    mean_size = mean_log_var.shape[1]//2
    mean = Lambda(lambda h: h[:, :mean_size])(mean_log_var)
    log_var = Lambda(lambda h: h[:, mean_size:])(mean_log_var)
    z = Lambda(sample)([mean, log_var])
    reconstruction = decoder(z)
    loss_reconstruction = K.mean(metrics.mean_squared_error(x, reconstruction))
    loss_KL = - K.mean(0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=1))
    loss = reconstruction_weight*loss_reconstruction + loss_KL

    vae = Model(x, reconstruction)
    vae.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss=lambda x, y: loss)
    return vae, encoder, decoder