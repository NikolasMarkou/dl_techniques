"""
``ConvDecoder``, a lightweight convolutional decoder that reconstructs images from encoded features.

Defines :class:`ConvDecoder`, a Keras layer that takes a feature grid and returns an
image. Each stage doubles the spatial resolution with a ``Conv2DTranspose`` and then
refines with a ``Conv2D``, rather than projecting straight back to pixels the way a
transformer decoder does, and a 1x1 convolution makes the final channel projection.
Keeping the decoder small pushes representational work onto the encoder, the
asymmetric split MAE uses. The caller sets ``decoder_dims`` to one entry per
upsampling stage, so the total upsampling factor is ``2 ** len(decoder_dims)`` and the
input has to be a 4D feature grid.
"""

import keras
from typing import Optional, Tuple, List, Dict, Any, Sequence

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.masked_autoencoder.conv_decoder")
class ConvDecoder(keras.layers.Layer):
    """Reconstruct an image from encoded features via staged 2x upsampling.

    One block per entry of ``decoder_dims``, then a 1x1 projection to
    ``output_channels`` and an optional final activation. The sub-layers are held in
    six flat per-role lists; :meth:`decoder_block` gives a role-keyed view of one
    block.

    Architecture:

    .. code-block:: text

        features [B, H', W', C_encoded]
                 │
                 ▼
        ┌─────────────────────┐
        │ decoder block 0     │  2x upsample
        └─────────────────────┘
                 │
                 ▼
                ...              one block per decoder_dims entry
                 │
                 ▼
        ┌─────────────────────┐
        │ decoder block n-1   │  2x upsample
        └─────────────────────┘
                 │  [B, H, W, decoder_dims[-1]]
                 ▼
        ┌─────────────────────┐
        │ final_conv  1x1     │
        └─────────────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │ final_act           │  (final_activation only)
        └─────────────────────┘
                 │
                 ▼
        reconstruction [B, H, W, output_channels]

    H is H' * 2 ** len(decoder_dims), and W likewise.

    Decoder block:

    .. code-block:: text

        x
                 │
                 ▼
        ┌───────────────────────┐
        │ upsample  convT /2    │
        └───────────────────────┘
                 │
                 ▼
        ┌───────────────────────┐
        │ norm_upsample         │  (use_batch_norm only)
        └───────────────────────┘
                 │
                 ▼
        ┌───────────────────────┐
        │ act_upsample          │
        └───────────────────────┘
                 │
                 ▼
        ┌───────────────────────┐
        │ refine  conv k x k    │
        └───────────────────────┘
                 │
                 ▼
        ┌───────────────────────┐
        │ norm_refine           │  (use_batch_norm only)
        └───────────────────────┘
                 │
                 ▼
        ┌───────────────────────┐
        │ act_refine            │
        └───────────────────────┘

    With use_batch_norm on, both convolutions drop their bias.

    :param decoder_dims: Channel count for each decoder stage, one per 2x upsample.
        Must be non-empty with all entries positive.
    :param output_channels: Number of output channels, typically 3 for RGB. Must be
        positive.
    :param kernel_size: Kernel size for the refinement convolutions. The upsampling
        transposed convolutions are fixed at 2x2, stride 2.
    :param activation: Activation used after every conv except the final projection.
    :param use_batch_norm: Apply BatchNormalization after each conv. When True, the
        upsample and refine convolutions are built without bias.
    :param final_activation: Activation applied after the final 1x1 projection, or
        `None` to return the projection unchanged.
    :param **kwargs: Passthrough to `keras.layers.Layer`.

    :raises ValueError: If `decoder_dims` is empty or holds a non-positive entry, or
        if `output_channels` is not positive.

    Input shape:
        4D tensor `(batch, height, width, channels)`.

    Output shape:
        `(batch, height * 2 ** n, width * 2 ** n, output_channels)`, with
        `n = len(decoder_dims)`.
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

        # DECISION plan-2026-08-19T163559-499b6f0e/D-085: store as a list; get_config
        # has always emitted lists and a tuple changes that shape. See decisions.md.
        self.decoder_dims = list(decoder_dims)
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.activation = deserialize_activation(activation)
        self.use_batch_norm = use_batch_norm
        self.final_activation = deserialize_activation(final_activation)

        # DECISION plan-2026-08-19T163559-499b6f0e/D-026: six flat per-role lists, not
        # per-block dicts; a Layer owner dropped 11 of 51 tensors that way.
        # See decisions.md and findings/audit-batch-6.md REF-9.
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
        """Build every weight-bearing sub-layer, threading shapes through the blocks.

        The activation layers and `final_act` own no weights, so they are left to
        Keras.

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

        :param inputs: Encoded feature tensor `(batch, height, width, channels)`.
        :param training: Forwarded to the batch norm layers only.
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
        """Compute the output shape; each decoder block upsamples 2x spatially.

        :param input_shape: 4D shape tuple `(batch, height, width, channels)`.
        :return: `(batch, height * 2 ** n, width * 2 ** n, output_channels)`, with
            `None` kept for unknown spatial dims.
        """
        batch_size, height, width, _ = input_shape
        num_upsamples = len(self.decoder_dims)
        if height is not None:
            height = height * (2 ** num_upsamples)
        if width is not None:
            width = width * (2 ** num_upsamples)
        return (batch_size, height, width, self.output_channels)

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: Dict holding every constructor argument.
        """
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
