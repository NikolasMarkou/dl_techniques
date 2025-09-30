import keras
from typing import Optional, Tuple, List, Dict, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ConvDecoder(keras.layers.Layer):
    """Convolutional decoder for MAE reconstruction.

    A lightweight decoder using transposed convolutions to reconstruct
    the original image from encoded features.

    **Architecture**:
    ```
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
    ```

    Args:
        decoder_dims: List of integers, number of channels in each decoder layer.
        output_channels: Integer, number of output channels (typically 3 for RGB).
        kernel_size: Integer, kernel size for decoder convolutions. Defaults to 3.
        activation: String or callable, activation function. Defaults to "gelu".
        use_batch_norm: Boolean, whether to use batch normalization. Defaults to True.
        final_activation: String or None, activation for final layer. Defaults to None.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`.

    Output shape:
        4D tensor with shape: `(batch_size, height*2^n, width*2^n, output_channels)`,
        where n is the number of decoder layers.

    Attributes:
        decoder_blocks: List of dictionaries containing decoder layer components.
        final_conv: Final projection layer to output channels.
        final_act: Final activation layer (if specified).

    Example:
        >>> decoder = ConvDecoder(
        ...     decoder_dims=[512, 256, 128, 64],
        ...     output_channels=3
        ... )
        >>> features = keras.random.normal((2, 7, 7, 768))
        >>> reconstructed = decoder(features)  # Shape: (2, 112, 112, 3)
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
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")

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
            # Transposed convolution for upsampling
            conv = keras.layers.Conv2DTranspose(
                filters=dim,
                kernel_size=2,
                strides=2,
                padding="same",
                use_bias=not use_batch_norm,
                name=f"decoder_upsample_{i}"
            )

            # Batch normalization
            if use_batch_norm:
                norm = keras.layers.BatchNormalization(name=f"decoder_bn_{i}")
            else:
                norm = None

            # Activation
            act = keras.layers.Activation(activation, name=f"decoder_act_{i}")

            # Refinement convolution
            refine = keras.layers.Conv2D(
                filters=dim,
                kernel_size=kernel_size,
                padding="same",
                use_bias=not use_batch_norm,
                name=f"decoder_refine_{i}"
            )

            # Refinement norm
            if use_batch_norm:
                refine_norm = keras.layers.BatchNormalization(
                    name=f"decoder_refine_bn_{i}"
                )
            else:
                refine_norm = None

            # Refinement activation
            refine_act = keras.layers.Activation(
                activation,
                name=f"decoder_refine_act_{i}"
            )

            self.decoder_blocks.append({
                "upsample": conv,
                "norm": norm,
                "activation": act,
                "refine": refine,
                "refine_norm": refine_norm,
                "refine_act": refine_act
            })

        # Final projection to output channels
        self.final_conv = keras.layers.Conv2D(
            filters=output_channels,
            kernel_size=1,
            padding="same",
            name="decoder_output"
        )

        if final_activation:
            self.final_act = keras.layers.Activation(
                final_activation,
                name="decoder_final_activation"
            )
        else:
            self.final_act = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build decoder sub-layers.

        Args:
            input_shape: Shape tuple (batch, height, width, channels).
        """
        # Build sub-layers in computational order
        current_shape = input_shape

        for block in self.decoder_blocks:
            # Build upsample layer
            block["upsample"].build(current_shape)
            current_shape = block["upsample"].compute_output_shape(current_shape)

            # Build norm if present
            if block["norm"]:
                block["norm"].build(current_shape)

            # Activation doesn't need building

            # Build refine conv
            block["refine"].build(current_shape)
            current_shape = block["refine"].compute_output_shape(current_shape)

            # Build refine norm if present
            if block["refine_norm"]:
                block["refine_norm"].build(current_shape)

        # Build final conv
        self.final_conv.build(current_shape)

        # Activation doesn't need building

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

        Returns:
            Reconstructed image tensor.
        """
        x = inputs

        for block in self.decoder_blocks:
            # Upsample
            x = block["upsample"](x)
            if block["norm"]:
                x = block["norm"](x, training=training)
            x = block["activation"](x)

            # Refine
            x = block["refine"](x)
            if block["refine_norm"]:
                x = block["refine_norm"](x, training=training)
            x = block["refine_act"](x)

        # Final projection
        x = self.final_conv(x)
        if self.final_act:
            x = self.final_act(x)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
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
