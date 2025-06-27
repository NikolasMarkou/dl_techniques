# vae.py

"""
Keras-compliant implementation of a Variational Autoencoder (VAE) for images.

This version uses a simplified and more robust architecture where the VAE class
owns all layers directly, avoiding complex sub-model dependencies and build-cycle errors.

Features:
    - Full Keras 3 compliance with compile/fit workflow.
    - Custom train_step and test_step for VAE-specific loss.
    - Simplified and robust layer management.
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
            encoder_filters: List[int] = [32, 64],
            decoder_filters: List[int] = [64, 32],
            kl_loss_weight: float = 1.0,
            input_shape: Optional[Tuple[int, int, int]] = None,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            use_batch_norm: bool = True,
            name: Optional[str] = "vae",
            **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)
        # Store configuration
        self.latent_dim = latent_dim
        self.encoder_filters = encoder_filters
        self.decoder_filters = decoder_filters
        self.kl_loss_weight = kl_loss_weight
        self._input_shape_arg = input_shape
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_batch_norm = use_batch_norm

        # Layer containers
        self.encoder_layers = []
        self.decoder_layers = []

        # This will be built properly in the build method
        self.sampling = Sampling(name="sampling")
        self.flatten = keras.layers.Flatten()
        self.encoder_output_dense = keras.layers.Dense(self.latent_dim * 2)

        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        # If input_shape is provided at init, build the model
        if self._input_shape_arg is not None:
            self.build((None,) + self._input_shape_arg)

    def build(self, input_shape):
        if self.built:
            return

        super().build(input_shape)
        self._input_shape_arg = tuple(input_shape[1:])

        # --- Encoder Layers ---
        for filters in self.encoder_filters:
            self.encoder_layers.append(keras.layers.Conv2D(
                filters, 3, strides=2, padding="same",
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer
            ))
            if self.use_batch_norm:
                self.encoder_layers.append(keras.layers.BatchNormalization())
            self.encoder_layers.append(keras.layers.LeakyReLU())

        # --- Determine shape for Decoder ---
        # To do this, we create a symbolic input and pass it through the encoder conv layers
        dummy_input = keras.Input(shape=self._input_shape_arg)
        x = dummy_input
        for layer in self.encoder_layers:
            x = layer(x)
        shape_before_flatten = ops.shape(x)[1:]

        # --- Decoder Layers ---
        self.decoder_layers.append(keras.layers.Dense(
            units=int(ops.prod(shape_before_flatten)), activation="relu"
        ))
        self.decoder_layers.append(keras.layers.Reshape(target_shape=shape_before_flatten))
        for filters in self.decoder_filters:
            self.decoder_layers.append(keras.layers.Conv2DTranspose(
                filters, 3, strides=2, padding="same",
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer
            ))
            if self.use_batch_norm:
                self.decoder_layers.append(keras.layers.BatchNormalization())
            self.decoder_layers.append(keras.layers.LeakyReLU())

        # Final decoder output layer
        self.decoder_layers.append(keras.layers.Conv2DTranspose(
            filters=self._input_shape_arg[-1], kernel_size=3, padding="same", activation="sigmoid"
        ))
        logger.info("VAE layers built successfully.")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> Dict[str, keras.KerasTensor]:
        # --- Encoder Pass ---
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x, training=training)

        x = self.flatten(x)
        encoder_output = self.encoder_output_dense(x)

        z_mean = encoder_output[:, :self.latent_dim]
        z_log_var = encoder_output[:, self.latent_dim:]

        # --- Sampling ---
        z = self.sampling((z_mean, z_log_var))

        # --- Decoder Pass ---
        x = z
        for layer in self.decoder_layers:
            x = layer(x, training=training)
        reconstruction = x

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
                ops.sum(1 + outputs["z_log_var"] - ops.square(outputs["z_mean"]) - ops.exp(outputs["z_log_var"]),
                        axis=1)
            )
            total_loss = reconstruction_loss + self.kl_loss_weight * kl_loss
            # Add regularization losses if any
            total_loss += sum(self.losses)

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
            ops.sum(1 + outputs["z_log_var"] - ops.square(outputs["z_mean"]) - ops.exp(outputs["z_log_var"]), axis=1)
        )
        total_loss = reconstruction_loss + self.kl_loss_weight * kl_loss
        total_loss += sum(self.losses)

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