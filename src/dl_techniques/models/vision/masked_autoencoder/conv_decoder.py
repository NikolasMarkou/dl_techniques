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
from typing import Optional, Tuple, List, Dict, Any, Sequence
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)

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
        decoder_dims: Sequence[int] = (512, 256, 128, 64),
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
        # DECISION plan-2026-08-19T163559-499b6f0e/D-085: the DEFAULT is a
        # tuple (R-009 S1) and the STORED attribute is a list. Keeping the
        # store as `list(...)` is what makes the conversion invisible: it is
        # the type `get_config` has always emitted, so a saved config's JSON
        # shape and every `== [..]` assertion in the suites are unchanged.
        self.decoder_dims = list(decoder_dims)
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.activation = deserialize_activation(activation)
        self.use_batch_norm = use_batch_norm
        self.final_activation = deserialize_activation(final_activation)

        # CREATE all sub-layers in __init__.
        #
        # DECISION plan-2026-08-19T163559-499b6f0e/D-026
        # These are SIX FLAT, PARALLEL LISTS -- one list per role, indexed by
        # block -- and NOT the obvious `self.decoder_blocks = [{"upsample": ...,
        # "refine": ...}, ...]`. Do NOT "tidy" them back into a list of dicts (or
        # a list of lists): Keras 3.8 does not write a layer container nested two
        # or more levels deep to `model.weights.h5` when its owner is a
        # `keras.layers.Layer`. It DOES write the identical container when the
        # owner is a `keras.Model`, which is why the same shape is harmless in
        # `models/accunet/model.py:304` and `models/cliffordnet/model.py:352`.
        # MEASURED on this exact class: the list-of-dicts form put 11 of 51
        # tensors and 98,403 of 329,827 parameters into the archive -- every
        # decoder conv kernel and all 32 BatchNorm tensors silently absent, and
        # only 8 of 51 tensors surviving a perturb / save / reload comparison.
        # See decisions.md D-026 and REF-9 in findings/audit-batch-6.md.
        self.upsample_convs: List[keras.layers.Layer] = []
        self.norm_upsamples: List[keras.layers.Layer] = []
        self.act_upsamples: List[keras.layers.Layer] = []
        self.refine_convs: List[keras.layers.Layer] = []
        self.norm_refines: List[keras.layers.Layer] = []
        self.act_refines: List[keras.layers.Layer] = []

        for i, dim in enumerate(decoder_dims):
            # 1. Upsampling Layer
            self.upsample_convs.append(keras.layers.Conv2DTranspose(
                filters=dim,
                kernel_size=2,
                strides=2,
                padding="same",
                use_bias=not use_batch_norm,
                name=f"decoder_upsample_{i}"
            ))

            # 2. Refinement Layer
            self.refine_convs.append(keras.layers.Conv2D(
                filters=dim,
                kernel_size=kernel_size,
                padding="same",
                use_bias=not use_batch_norm,
                name=f"decoder_refine_{i}"
            ))

            # 3. Normalization Layers -- both lists stay EMPTY when
            #    `use_batch_norm` is False, so no `None` ever enters a tracked
            #    container.
            if use_batch_norm:
                self.norm_upsamples.append(
                    keras.layers.BatchNormalization(name=f"decoder_bn_{i}"))
                self.norm_refines.append(
                    keras.layers.BatchNormalization(name=f"decoder_refine_bn_{i}"))

            # 4. Activation Layers (Stateless, but good to instantiate for config)
            self.act_upsamples.append(
                keras.layers.Activation(activation, name=f"decoder_act_{i}"))
            self.act_refines.append(
                keras.layers.Activation(activation, name=f"decoder_refine_act_{i}"))

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

    @property
    def num_blocks(self) -> int:
        """Number of upsampling blocks, i.e. ``len(decoder_dims)``."""
        return len(self.upsample_convs)

    def decoder_block(self, index: int) -> Dict[str, Optional[keras.layers.Layer]]:
        """Return one decoder block as a role-keyed view over the flat lists.

        This is the read-only accessor that replaces the former
        ``self.decoder_blocks[index]`` dict. The layers themselves live in six
        flat, per-role lists (see the note in ``__init__``); this method only
        assembles a view of them and must never be used to STORE a layer, which
        would re-create the nested container the flat lists exist to avoid.

        Args:
            index: Block index in ``[0, num_blocks)``.

        Returns:
            Mapping with keys ``upsample``, ``norm_upsample``, ``act_upsample``,
            ``refine``, ``norm_refine``, ``act_refine``. The two ``norm_*``
            values are ``None`` when ``use_batch_norm`` is False.
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
        """Build decoder sub-layers explicitly using the input shape.

        Args:
            input_shape: Shape tuple (batch, height, width, channels).
        """
        current_shape = input_shape

        for i in range(self.num_blocks):
            # Build Upsample
            self.upsample_convs[i].build(current_shape)
            current_shape = self.upsample_convs[i].compute_output_shape(current_shape)

            if self.use_batch_norm:
                self.norm_upsamples[i].build(current_shape)

            # Activation doesn't change shape

            # Build Refine
            self.refine_convs[i].build(current_shape)
            current_shape = self.refine_convs[i].compute_output_shape(current_shape)

            if self.use_batch_norm:
                self.norm_refines[i].build(current_shape)

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

        for i in range(self.num_blocks):
            # Upsample Phase
            x = self.upsample_convs[i](x)
            if self.use_batch_norm:
                x = self.norm_upsamples[i](x, training=training)
            x = self.act_upsamples[i](x)

            # Refine Phase
            x = self.refine_convs[i](x)
            if self.use_batch_norm:
                x = self.norm_refines[i](x, training=training)
            x = self.act_refines[i](x)

        # Final Projection
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
