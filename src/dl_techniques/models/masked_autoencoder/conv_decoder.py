"""
A lightweight convolutional decoder for image reconstruction.

This decoder is designed for self-supervised learning frameworks like the
Masked Autoencoder (MAE). Following the asymmetric encoder-decoder design
philosophy, this component is intentionally lightweight to force the encoder
to learn semantic representations.

Architecture:
    It uses a series of upsampling blocks (Conv2DTranspose + Conv2D) to
    progressively recover spatial resolution from the latent representation.
    While modern Transformers often use Linear decoders, this Convolutional
    variant is ideal for hybrid architectures (e.g., ConvNeXt).
"""

import keras
from typing import Optional, Tuple, List, Dict, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ConvDecoder(keras.layers.Layer):
    """Convolutional decoder for MAE reconstruction.

    A lightweight decoder using transposed convolutions to reconstruct
    the original image from encoded features.

    Architecture:
        Input(shape=[batch, H', W', C_encoded])
               ↓
        For each decoder_dim:
            Conv2DTranspose (upsample 2x)
            → BatchNorm (optional)
            → Activation
            → Conv2D (refine)
            → BatchNorm (optional)
            → Activation
               ↓
        Conv2D (1x1, project to output_channels)
        → Final Activation (optional)
               ↓
        Output(shape=[batch, H, W, output_channels])

    Args:
        decoder_dims: List of integers, number of channels in each decoder layer.
        output_channels: Integer, number of output channels (typically 3 for RGB).
        kernel_size: Integer, kernel size for decoder convolutions. Defaults to 3.
        activation: String or callable, activation function. Defaults to "gelu".
        use_batch_norm: Boolean, whether to use batch normalization. Defaults to True.
        final_activation: String or None, activation for final layer. Defaults to None.
        **kwargs: Additional arguments for Layer base class.
    """

    def __init__(
        self,
        decoder_dims: List[int] = [512, 256, 128, 64],
        output_channels: int = 3,
        kernel_size: int = 3,
        activation: str = "gelu",
        use_batch_norm: bool = True,
        final_activation: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if not decoder_dims:
            raise ValueError("decoder_dims cannot be empty")
        if any(dim <= 0 for dim in decoder_dims):
            raise ValueError("All dimensions in decoder_dims must be positive")
        if output_channels <= 0:
            raise ValueError(f"output_channels must be positive, got {output_channels}")

        # Store configuration
        self.decoder_dims = decoder_dims
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.final_activation = final_activation

        # CREATE all sub-layers in __init__
        self.decoder_blocks = []

        for i, dim in enumerate(decoder_dims):
            # 1. Upsampling Layer
            upsample_conv = keras.layers.Conv2DTranspose(
                filters=dim,
                kernel_size=2,
                strides=2,
                padding="same",
                use_bias=not use_batch_norm,
                name=f"decoder_upsample_{i}"
            )

            # 2. Refinement Layer
            refine_conv = keras.layers.Conv2D(
                filters=dim,
                kernel_size=kernel_size,
                padding="same",
                use_bias=not use_batch_norm,
                name=f"decoder_refine_{i}"
            )

            # 3. Normalization Layers
            norm_upsample = None
            norm_refine = None
            if use_batch_norm:
                norm_upsample = keras.layers.BatchNormalization(name=f"decoder_bn_{i}")
                norm_refine = keras.layers.BatchNormalization(name=f"decoder_refine_bn_{i}")

            # 4. Activation Layers (Stateless, but good to instantiate for config)
            act_upsample = keras.layers.Activation(activation, name=f"decoder_act_{i}")
            act_refine = keras.layers.Activation(activation, name=f"decoder_refine_act_{i}")

            self.decoder_blocks.append({
                "upsample": upsample_conv,
                "norm_upsample": norm_upsample,
                "act_upsample": act_upsample,
                "refine": refine_conv,
                "norm_refine": norm_refine,
                "act_refine": act_refine
            })

        # Final projection to output channels
        self.final_conv = keras.layers.Conv2D(
            filters=output_channels,
            kernel_size=1,
            padding="same",
            name="decoder_output"
        )

        self.final_act = None
        if final_activation:
            self.final_act = keras.layers.Activation(
                final_activation,
                name="decoder_final_activation"
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build decoder sub-layers explicitly using the input shape.

        Args:
            input_shape: Shape tuple (batch, height, width, channels).
        """
        current_shape = input_shape

        for block in self.decoder_blocks:
            # Build Upsample
            block["upsample"].build(current_shape)
            current_shape = block["upsample"].compute_output_shape(current_shape)

            if block["norm_upsample"]:
                block["norm_upsample"].build(current_shape)

            # Activation doesn't change shape

            # Build Refine
            block["refine"].build(current_shape)
            current_shape = block["refine"].compute_output_shape(current_shape)

            if block["norm_refine"]:
                block["norm_refine"].build(current_shape)

        # Build Final Projection
        self.final_conv.build(current_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Decode features to reconstruct image.

        Args:
            inputs: Encoded features tensor.
            training: Boolean or None, whether in training mode.
        """
        x = inputs

        for block in self.decoder_blocks:
            # Upsample Phase
            x = block["upsample"](x)
            if block["norm_upsample"]:
                x = block["norm_upsample"](x, training=training)
            x = block["act_upsample"](x)

            # Refine Phase
            x = block["refine"](x)
            if block["norm_refine"]:
                x = block["norm_refine"](x, training=training)
            x = block["act_refine"](x)

        # Final Projection
        x = self.final_conv(x)
        if self.final_act:
            x = self.final_act(x)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "decoder_dims": self.decoder_dims,
            "output_channels": self.output_channels,
            "kernel_size": self.kernel_size,
            "activation": self.activation,
            "use_batch_norm": self.use_batch_norm,
            "final_activation": self.final_activation,
        })
        return config

# ---------------------------------------------------------------------
