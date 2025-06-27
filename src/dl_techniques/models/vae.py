# vae.py

"""
Keras-compliant implementation of a Variational Autoencoder (VAE) for images.

This module provides a fully Keras-compliant VAE that works with the standard
Keras compile/fit workflow. The custom training logic, which combines
reconstruction loss and KL divergence loss, is integrated into the model's
train_step and test_step methods.

Architecture Overview:
    1. Encoder: A convolutional network that outputs the parameters (mean and
       log-variance) of the latent distribution.
    2. Sampling Layer: Uses the reparameterization trick to sample from the
       latent distribution in a way that allows backpropagation.
    3. Decoder: A deconvolutional network that reconstructs the input image
       from a latent space sample.
"""

import os
import keras
from keras import ops
import tensorflow as tf
from typing import Optional, Tuple, Union, Dict, Any, List

# Make sure this import path is correct for your project structure
from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class Sampling(keras.layers.Layer):
    """Uses reparameterization trick to sample from a Normal distribution."""

    def call(self, inputs: Tuple[keras.KerasTensor, keras.KerasTensor]) -> keras.KerasTensor:
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

    def compute_output_shape(self, input_shape: Tuple[Tuple, Tuple]) -> Tuple:
        return input_shape[0]


@keras.saving.register_keras_serializable()
class VAE(keras.Model):
    """Keras-compliant Variational Autoencoder (VAE) model."""

    def __init__(
            self,
            latent_dim: int,
            encoder_filters: List[int] = [32, 64, 128],
            decoder_filters: List[int] = [128, 64, 32],
            kl_loss_weight: float = 1.0,
            input_shape: Optional[Tuple[int, int, int]] = None,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            use_batch_norm: bool = True,
            name: Optional[str] = "vae",
            **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.encoder_filters = encoder_filters.copy()
        self.decoder_filters = decoder_filters.copy()
        self.kl_loss_weight = kl_loss_weight
        self._input_shape_arg = input_shape
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_batch_norm = use_batch_norm

        self.encoder = None
        self.decoder = None
        self.sampling = Sampling(name="sampling")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        if self._input_shape_arg is not None:
            self.build((None,) + self._input_shape_arg)

    def _build_encoder(self, input_shape):
        """Builds the encoder as a standalone Keras model."""
        encoder_input = keras.layers.Input(shape=input_shape)
        x = encoder_input
        for filters in self.encoder_filters:
            x = keras.layers.Conv2D(
                filters, 3, strides=2, padding="same",
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
            )(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization()(x)
            x = keras.layers.LeakyReLU()(x)

        # The shape before flattening is the output shape of the last conv block
        shape_before_flatten = ops.shape(x)[1:]
        x = keras.layers.Flatten()(x)
        encoder_output = keras.layers.Dense(self.latent_dim * 2, name="encoder_dense_output")(x)

        encoder = keras.Model(encoder_input, [encoder_output, shape_before_flatten], name="encoder")
        logger.info("Encoder built successfully.")
        encoder.summary(print_fn=logger.info)
        return encoder

    def _build_decoder(self, shape_before_flatten):
        """Builds the decoder as a standalone Keras model."""
        latent_inputs = keras.layers.Input(shape=(self.latent_dim,))
        dense_units = int(ops.reduce_prod(shape_before_flatten))

        x = keras.layers.Dense(units=dense_units, activation="relu")(latent_inputs)
        x = keras.layers.Reshape(target_shape=shape_before_flatten)(x)

        for filters in self.decoder_filters:
            x = keras.layers.Conv2DTranspose(
                filters, 3, strides=2, padding="same",
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
            )(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization()(x)
            x = keras.layers.LeakyReLU()(x)

        decoder_outputs = keras.layers.Conv2DTranspose(
            filters=self._input_shape_arg[-1],
            kernel_size=3, padding="same", activation="sigmoid", name="decoder_output"
        )(x)

        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        logger.info("Decoder built successfully.")
        decoder.summary(print_fn=logger.info)
        return decoder

    def build(self, input_shape):
        """Build the encoder and decoder models."""
        if self.built:
            return

        super().build(input_shape)
        self._input_shape_arg = tuple(input_shape[1:])

        # Build encoder and get the shape_before_flatten
        # To get the shape, we must do a dummy pass through the conv layers
        temp_input = keras.layers.Input(shape=self._input_shape_arg)
        x = temp_input
        for filters in self.encoder_filters:
            x = keras.layers.Conv2D(filters, 3, strides=2, padding="same")(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization()(x)
            x = keras.layers.LeakyReLU()(x)
        shape_before_flatten = ops.shape(x)[1:]

        # Now build the real models
        self.encoder = self._build_encoder(self._input_shape_arg)
        self.decoder = self._build_decoder(shape_before_flatten)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> Dict[str, keras.KerasTensor]:
        encoder_output, _ = self.encoder(inputs, training=training)
        z_mean = encoder_output[:, :self.latent_dim]
        z_log_var = encoder_output[:, self.latent_dim:]
        z = self.sampling((z_mean, z_log_var))
        reconstruction = self.decoder(z, training=training)
        return {
            "reconstruction": reconstruction,
            "z_mean": z_mean,
            "z_log_var": z_log_var,
        }

    def train_step(self, data: Tuple[tf.Tensor, Any]) -> Dict[str, tf.Tensor]:
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        with tf.GradientTape() as tape:
            outputs = self(x, training=True)
            reconstruction_loss = ops.mean(
                ops.sum(keras.losses.mean_squared_error(x, outputs["reconstruction"]), axis=(1, 2))
            )
            kl_loss = -0.5 * ops.mean(
                ops.sum(
                    1 + outputs["z_log_var"] - ops.square(outputs["z_mean"]) - ops.exp(outputs["z_log_var"]),
                    axis=1,
                )
            )
            total_loss = reconstruction_loss + self.kl_loss_weight * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data: Tuple[tf.Tensor, Any]) -> Dict[str, tf.Tensor]:
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        outputs = self(x, training=False)
        reconstruction_loss = ops.mean(
            ops.sum(keras.losses.mean_squared_error(x, outputs["reconstruction"]), axis=(1, 2))
        )
        kl_loss = -0.5 * ops.mean(
            ops.sum(
                1 + outputs["z_log_var"] - ops.square(outputs["z_mean"]) - ops.exp(outputs["z_log_var"]),
                axis=1,
            )
        )
        total_loss = reconstruction_loss + self.kl_loss_weight * kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "encoder_filters": self.encoder_filters,
            "decoder_filters": self.decoder_filters,
            "kl_loss_weight": self.kl_loss_weight,
            "input_shape": self._input_shape_arg,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_batch_norm": self.use_batch_norm,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VAE":
        # Deserialization for initializers and regularizers
        config['kernel_initializer'] = keras.initializers.deserialize(config.get('kernel_initializer'))
        config['kernel_regularizer'] = keras.regularizers.deserialize(config.get('kernel_regularizer'))
        return cls(**config)


def create_vae(
        input_shape: Tuple[int, int, int],
        optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
        learning_rate: float = 0.001,
        **kwargs
) -> VAE:
    """Create and compile a VAE model."""
    model = VAE(input_shape=input_shape, **kwargs)

    if isinstance(optimizer, str):
        optimizer_instance = keras.optimizers.get(optimizer)
        optimizer_instance.learning_rate = learning_rate
    else:
        optimizer_instance = optimizer

    model.compile(optimizer=optimizer_instance)
    return model