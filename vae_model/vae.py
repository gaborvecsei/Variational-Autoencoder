import enum
from typing import Callable, Union, Optional

from keras import backend as K
from keras.layers import Input, Dense, Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model


class ReconstructionLoss(enum.Enum):
    MSE = "Mean Sqared Error"
    BINARY_CROSSENTROPY = "Binary Crossentropy"


class VAE:

    def __init__(self,
                 latent_dim: int = 32,
                 beta=1.0,
                 reconstruction_loss=ReconstructionLoss.BINARY_CROSSENTROPY,
                 optimizer: Optional[Union[Callable, str]] = None):
        self.latent_dim = latent_dim
        self._reconstruction_loss = reconstruction_loss
        self.beta = beta

        self.optimizer = "adam"
        if optimizer is not None:
            self.optimizer = optimizer

        self.encoder = None
        self.decoder = None
        self.vae = None

        self.build_architecture()

    def build_architecture(self):
        ### Encoder Part ###

        image_input = Input(shape=(64, 64, 3))
        conv_1 = Conv2D(filters=32, kernel_size=4, strides=2, activation="relu")(image_input)
        conv_2 = Conv2D(filters=64, kernel_size=4, strides=2, activation="relu")(conv_1)
        conv_3 = Conv2D(filters=64, kernel_size=4, strides=2, activation="relu")(conv_2)
        conv_4 = Conv2D(filters=128, kernel_size=4, strides=2, activation="relu")(conv_3)

        conv_4_flat = Flatten()(conv_4)

        z_mean = Dense(self.latent_dim)(conv_4_flat)
        z_log_var = Dense(self.latent_dim)(conv_4_flat)

        z_sampled_latent_vector = Lambda(self._sampling)([z_mean, z_log_var])

        ### Decoder part ###

        z_decoder_input = Input(shape=(self.latent_dim,))

        z_dense = Dense(64 * 5 * 5, activation="relu")(z_decoder_input)
        z_reshaped = Reshape((5, 5, 64))(z_dense)

        conv_tranp_1 = Conv2DTranspose(filters=64, kernel_size=5, strides=2, activation="relu")(z_reshaped)
        conv_tranp_2 = Conv2DTranspose(filters=32, kernel_size=6, strides=2, activation="relu")(conv_tranp_1)
        conv_tranp_3 = Conv2DTranspose(filters=3, kernel_size=6, strides=2, activation="sigmoid")(conv_tranp_2)

        ### Constructed Models ###

        vae_encoder = Model(image_input, z_mean)
        vae_sampler = Model(image_input, z_sampled_latent_vector)
        vae_decoder = Model(z_decoder_input, conv_tranp_3)
        vae = Model(image_input, vae_decoder(vae_sampler(image_input)))

        ### Custom Loss Functions ###

        def reconstruction(y_true, y_pred):
            if self._reconstruction_loss == ReconstructionLoss.MSE:
                r = K.square(y_true - y_pred)
            elif self._reconstruction_loss == ReconstructionLoss.BINARY_CROSSENTROPY:
                r = -(y_true * K.log(y_pred) + (1. - y_true) * K.log(1. - y_pred))
            else:
                raise ValueError("Reconstruction loss {0} is not known".format(self._reconstruction_loss))
            return K.sum(r, axis=(1, 2, 3))

        def kl_divergence(y_true=None, y_pred=None):
            return - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)

        def vae_loss(y_true, y_pred):
            r_loss = reconstruction(y_true, y_pred)
            kl_loss = kl_divergence()
            return K.mean(r_loss + self.beta * kl_loss)

        vae.compile(optimizer="adam", loss=vae_loss, metrics=[reconstruction, kl_divergence])

        self.vae = vae
        self.encoder = vae_encoder
        self.decoder = vae_decoder

    def _sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=1.)
        return z_mean + K.exp(z_log_var / 2) * epsilon
