from typing import Tuple, Callable
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape, Activation, InputLayer
from keras.activations import relu, sigmoid
from keras.models import Model
from keras import metrics

class VAE:
    def __init__(self):
        self._input_image_shape = (64, 64, 3)
        self._encoded_output_dim = 32
        self._latent_vector_layer_name = "latent_vector_representation"
        self.model = self._build_and_compile()
        self.encoder = self._get_encoder_only(self.model)
        self.decoder = self._get_decoder_only(self.model)

    def _get_encoder_only(self, vae_model):
        encoder_model = Model(inputs=vae_model.input, outputs=vae_model.get_layer(self._latent_vector_layer_name).get_output_at(0))
        return encoder_model

    def _get_decoder_only(self, vae_model):
        latent_layer_index = -1
        for i, l in enumerate(vae_model.layers):
            if l.name == self._latent_vector_layer_name:
                latent_layer_index = i
                break

        new_input = Input(shape=(self._encoded_output_dim,), name="latent_input")
        x = new_input
        for layer in vae_model.layers[latent_layer_index + 1:]:
            x = layer(x)

        new_model = Model(inputs=new_input, outputs=x)
        return new_model

    def _sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self._encoded_output_dim), mean=0., stddev=1.)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def _build_encoder(self, input_layer):
        conv_1 = Conv2D(filters=32, kernel_size=4, strides=2, name="conv_1")(input_layer)
        act_1 = Activation(relu, name="conv_1_relu")(conv_1)
        conv_2 = Conv2D(filters=64, kernel_size=4, strides=2, name="conv_2")(act_1)
        act_2 = Activation(relu, name="conv_2_relu")(conv_2)
        conv_3 = Conv2D(filters=64, kernel_size=4, strides=2, name="conv_3")(act_2)
        act_3 = Activation(relu, name="conv_3_relu")(conv_3)
        conv_4 = Conv2D(filters=128, kernel_size=4, strides=2, name="conv_4")(act_3)
        act_4 = Activation(relu, name="conv_4_relu")(conv_4)

        z_in = Flatten()(act_4)
        z_mean = Dense(self._encoded_output_dim, name="z_mean")(z_in)
        z_log_var = Dense(self._encoded_output_dim, name="z_log_var")(z_in)

        latent_vector = Lambda(self._sampling, name=self._latent_vector_layer_name)([z_mean, z_log_var])

        return latent_vector, z_mean, z_log_var

    def _build_decoder(self, latent_vector_input_layer):
        dense = Dense(1024, name="decoder_input_dense")(latent_vector_input_layer)
        reshaped_dense = Reshape((1, 1, 1024), name="decoder_input_reshape")(dense)
        vae_d1 = Conv2DTranspose(filters=64, kernel_size=5, strides=2, name="conv2dtrans_1")(reshaped_dense)
        act_1 = Activation(relu, name="decoder_1_relu")(vae_d1)
        vae_d2 = Conv2DTranspose(filters=64, kernel_size=5, strides=2, name="conv2dtrans_2")(act_1)
        act_2 = Activation(relu, name="decoder_2_relu")(vae_d2)
        vae_d3 = Conv2DTranspose(filters=32, kernel_size=6, strides=2, name="conv2dtrans_3")(act_2)
        act_3 = Activation(relu, name="decoder_3_relu")(vae_d3)
        vae_d4 = Conv2DTranspose(filters=3, kernel_size=6, strides=2, name="conv2dtrans_4")(act_3)
        decoder_output = Activation(sigmoid, name="decoder_output_sigmoid")(vae_d4)

        return decoder_output

    def _build_and_compile(self):
        # Input
        input_image = Input(shape=self._input_image_shape, name="input_image")
        # Encoder
        latent_vector, z_mean, z_log_var = self._build_encoder(input_image)
        # Decoder
        reconstructed_image = self._build_decoder(latent_vector)
        # Complete VAE model (image -> encoder -> latent representation -> decoder -> reconstructed image)
        vae_model = Model(input_image, reconstructed_image)

        # Loss functions
        def vae_r_loss(y_true, y_pred):
            # Reconstruction loss
            return K.sum(K.abs(y_true - y_pred))

        def vae_kl_loss(y_true, y_pred):
            # KL loss
            return - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

        def vae_loss(y_true, y_pred):
            # Summarized (Reconstruction + KL) loss
            return vae_r_loss(y_true, y_pred) + vae_kl_loss(y_true, y_pred)

        vae_model.compile(optimizer="adam", loss=vae_loss, metrics=[vae_r_loss, vae_kl_loss])

        return vae_model
