"""``ConvDecoder``, a lightweight convolutional decoder for reconstructing images from encoded features.

Each stage doubles spatial resolution with a Conv2DTranspose, then refines
with a Conv2D, unlike a transformer decoder's linear projection back to
pixels. Staying lightweight here pushes representational work onto the
encoder, matching the asymmetric encoder-decoder design MAE uses.

The caller sets `decoder_dims` to one entry per upsampling stage; total
upsampling is `2 ** len(decoder_dims)`.
"""

import keras
from typing import Optional, Tuple, List, Dict, Any, Sequence
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.masked_autoencoder.conv_decoder")
class ConvDecoder(keras.layers.Layer):
    """Reconstruct an image from encoded features via staged 2x upsampling.

    Architecture:

    .. code-block:: text

        features  [B, H', W', C_encoded]
           |
           v
        for each entry in decoder_dims:
           |
           +-> Conv2DTranspose (2x upsample) -> BatchNorm (optional) -> act
           |
           +-> Conv2D (refine) -> BatchNorm (optional) -> act
           |
           v
        Conv2D 1x1 -> output_channels
           |
        Activation (optional, 'final_activation' only)
           |
           v
        reconstruction  [B, H, W, output_channels]

    :param decoder_dims: Channel count for each decoder stage, one per 2x upsample.
    :param output_channels: Number of output channels, typically 3 for RGB.
    :param kernel_size: Kernel size for the refinement convolutions.
    :param activation: Activation used after every conv except the final projection.
    :param use_batch_norm: Apply BatchNormalization after each conv.
    :param final_activation: Activation applied after the final 1x1 projection, or `None`.
    :param kwargs: Passthrough to `keras.layers.Layer`.
    """

    def __init__(
        self,
        decoder_dims: Sequence[int] = (512, 256, 128, 64),
        output_channels: int = 3,
        kernel_size: int = 3,
        activation: str = "gelu",
        use_batch_norm: bool = True,
        final_activation: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if not decoder_dims:
            raise ValueError("decoder_dims cannot be empty")
        if any(dim <= 0 for dim in decoder_dims):
            raise ValueError("All dimensions in decoder_dims must be positive")
        if output_channels <= 0:
            raise ValueError(f"output_channels must be positive, got {output_channels}")

        # DECISION plan-2026-08-19T163559-499b6f0e/D-085: store as a list even though the default is a tuple.
        # get_config has always emitted a list, so keeping this conversion matches every saved config's JSON shape. See decisions.md.
        self.decoder_dims = list(decoder_dims)
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.activation = deserialize_activation(activation)
        self.use_batch_norm = use_batch_norm
        self.final_activation = deserialize_activation(final_activation)

        # DECISION plan-2026-08-19T163559-499b6f0e/D-026: keep six flat per-role lists, not a list of per-block dicts.
        # A Layer owner (unlike a Model) does not save a container nested two levels deep: the dict form left 11 of 51 tensors in the archive. See decisions.md and findings/audit-batch-6.md REF-9.
        self.upsample_convs: List[keras.layers.Layer] = []
        self.norm_upsamples: List[keras.layers.Layer] = []
        self.act_upsamples: List[keras.layers.Layer] = []
        self.refine_convs: List[keras.layers.Layer] = []
        self.norm_refines: List[keras.layers.Layer] = []
        self.act_refines: List[keras.layers.Layer] = []

        for i, dim in enumerate(decoder_dims):
            self.upsample_convs.append(keras.layers.Conv2DTranspose(
                filters=dim,
                kernel_size=2,
                strides=2,
                padding="same",
                use_bias=not use_batch_norm,
                name=f"decoder_upsample_{i}"
            ))

            self.refine_convs.append(keras.layers.Conv2D(
                filters=dim,
                kernel_size=kernel_size,
                padding="same",
                use_bias=not use_batch_norm,
                name=f"decoder_refine_{i}"
            ))

            # Both norm lists stay empty when use_batch_norm is False, so no None enters a tracked container.
            if use_batch_norm:
                self.norm_upsamples.append(
                    keras.layers.BatchNormalization(name=f"decoder_bn_{i}"))
                self.norm_refines.append(
                    keras.layers.BatchNormalization(name=f"decoder_refine_bn_{i}"))

            self.act_upsamples.append(
                keras.layers.Activation(activation, name=f"decoder_act_{i}"))
            self.act_refines.append(
                keras.layers.Activation(activation, name=f"decoder_refine_act_{i}"))

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

    @property
    def num_blocks(self) -> int:
        """Number of upsampling blocks, i.e. ``len(decoder_dims)``."""
        return len(self.upsample_convs)

    def decoder_block(self, index: int) -> Dict[str, Optional[keras.layers.Layer]]:
        """Return one decoder block as a role-keyed view over the flat per-role lists.

        A read-only view; never use it to store a layer, which would
        recreate the nested container the flat lists avoid.

        :param index: Block index in `[0, num_blocks)`.
        :return: Mapping with keys `upsample`, `norm_upsample`, `act_upsample`,
            `refine`, `norm_refine`, `act_refine`. The two `norm_*` values are
            `None` when `use_batch_norm` is False.
        """
        return {
            "upsample": self.upsample_convs[index],
            "norm_upsample": self.norm_upsamples[index] if self.use_batch_norm else None,
            "act_upsample": self.act_upsamples[index],
            "refine": self.refine_convs[index],
            "norm_refine": self.norm_refines[index] if self.use_batch_norm else None,
            "act_refine": self.act_refines[index],
        }

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every decoder sub-layer explicitly from the input shape.

        :param input_shape: Shape tuple `(batch, height, width, channels)`.
        """
        current_shape = input_shape

        for i in range(self.num_blocks):
            self.upsample_convs[i].build(current_shape)
            current_shape = self.upsample_convs[i].compute_output_shape(current_shape)

            if self.use_batch_norm:
                self.norm_upsamples[i].build(current_shape)

            self.refine_convs[i].build(current_shape)
            current_shape = self.refine_convs[i].compute_output_shape(current_shape)

            if self.use_batch_norm:
                self.norm_refines[i].build(current_shape)

        self.final_conv.build(current_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Decode features into a reconstructed image.

        :param inputs: Encoded feature tensor.
        :param training: Whether the batch norm layers run in training mode.
        :return: Reconstructed image.
        """
        x = inputs

        for i in range(self.num_blocks):
            x = self.upsample_convs[i](x)
            if self.use_batch_norm:
                x = self.norm_upsamples[i](x, training=training)
            x = self.act_upsamples[i](x)

            x = self.refine_convs[i](x)
            if self.use_batch_norm:
                x = self.norm_refines[i](x, training=training)
            x = self.act_refines[i](x)

        x = self.final_conv(x)
        if self.final_act:
            x = self.final_act(x)

        return x

    def compute_output_shape(self, input_shape):
        """Compute output shape: each decoder block upsamples 2x spatially."""
        batch_size, height, width, _ = input_shape
        num_upsamples = len(self.decoder_dims)
        if height is not None:
            height = height * (2 ** num_upsamples)
        if width is not None:
            width = width * (2 ** num_upsamples)
        return (batch_size, height, width, self.output_channels)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "decoder_dims": self.decoder_dims,
            "output_channels": self.output_channels,
            "kernel_size": self.kernel_size,
            "activation": serialize_activation(self.activation),
            "use_batch_norm": self.use_batch_norm,
            "final_activation": serialize_activation(self.final_activation),
        })
        return config

# ---------------------------------------------------------------------
