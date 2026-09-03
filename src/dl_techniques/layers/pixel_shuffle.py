"""
``PixelShuffle`` reduces the number of spatial tokens in a vision transformer
sequence by folding them into the channel dimension.

The input is a token sequence ``[CLS, spatial_tokens]``. The layer separates
the CLS token, reshapes the spatial tokens into a square grid, groups each
``scale_factor x scale_factor`` block into the channel dimension, and
re-flattens the result into a shorter, wider-channel sequence. The CLS token
is zero-padded to match the new channel width and re-attached. The operation
is lossless and fully differentiable.

Spatial tokens must form a perfect square and the grid side must be evenly
divisible by ``scale_factor``. ``scale_factor=1`` is the identity.
"""

import keras
from keras import ops
from typing import Optional, Tuple, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.pixel_shuffle")
class PixelShuffle(keras.layers.Layer):
    """Pixel shuffle for reducing spatial tokens in vision transformers.

    This layer performs a space-to-depth rearrangement on the spatial token
    portion of a sequence that starts with a CLS token. Given an input of
    shape ``[B, 1 + H*W, C]`` it separates the CLS token, reshapes the
    remaining tokens into a 2-D grid, groups ``scale_factor x scale_factor``
    spatial blocks into the channel dimension, and re-flattens back to a
    shorter sequence with wider channels:
    ``output = [B, 1 + (H/s)*(W/s), C*s^2]`` where ``s`` is the
    ``scale_factor``. The CLS token is zero-padded to match the new channel
    width. The operation is lossless and fully differentiable.

    Architecture:

    .. code-block:: text

        ┌───────────────────────────────────────┐
        │  Input [B, 1 + H*W, C]                │
        │  (CLS token + spatial tokens)         │
        └──────────────────┬────────────────────┘
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
        ┌──────────────┐   ┌───────────────────┐
        │  CLS token   │   │  Spatial tokens   │
        │  [B, 1, C]   │   │  [B, H*W, C]      │
        └──────┬───────┘   └────────┬──────────┘
               │                    │
               │                    ▼
               │           ┌───────────────────┐
               │           │  Reshape to grid  │
               │           │  [B, H, W, C]     │
               │           └────────┬──────────┘
               │                    │
               │                    ▼
               │           ┌───────────────────┐
               │           │  Space-to-depth   │
               │           │  [B, H/s, W/s,    │
               │           │   C*s^2]          │
               │           └────────┬──────────┘
               │                    │
               ▼                    ▼
        ┌──────────────┐   ┌───────────────────┐
        │  Pad to C*s^2│   │  Flatten grid     │
        └──────┬───────┘   └────────┬──────────┘
               │                    │
               └────────┬───────────┘
                        ▼
        ┌───────────────────────────────────────┐
        │  Concatenate along sequence axis      │
        │  Output [B, 1+(H/s)*(W/s), C*s^2]     │
        └───────────────────────────────────────┘

    :param scale_factor: Factor by which to reduce each spatial dimension.
        Must be a positive integer dividing the spatial side length.
    :type scale_factor: int
    :param validate_spatial_dims: Whether to validate spatial dimension
        compatibility at build time.
    :type validate_spatial_dims: bool
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any
    """

    def __init__(
        self,
        scale_factor: int = 2,
        validate_spatial_dims: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(scale_factor, int) or scale_factor <= 0:
            raise ValueError(
                f"scale_factor must be a positive integer, got {scale_factor}"
            )

        self.scale_factor = scale_factor
        self.validate_spatial_dims = validate_spatial_dims

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Validate input shape and build the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]"""
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected 3D input (batch, seq_len, channels), got shape {input_shape}"
            )

        if self.validate_spatial_dims and input_shape[1] is not None:
            seq_len = input_shape[1]
            spatial_len = seq_len - 1

            if spatial_len <= 0:
                raise ValueError(
                    f"Sequence length must be > 1 (need at least CLS + 1 spatial token), "
                    f"got {seq_len}"
                )

            h_float = spatial_len ** 0.5
            h = int(h_float)
            if h * h != spatial_len:
                raise ValueError(
                    f"Spatial tokens ({spatial_len}) must form a perfect square, "
                    f"got {spatial_len} tokens"
                )

            if h % self.scale_factor != 0:
                raise ValueError(
                    f"Spatial dimension ({h}) must be divisible by scale_factor "
                    f"({self.scale_factor})"
                )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply the pixel shuffle (space-to-depth) operation.

        :param inputs: Input tensor ``[batch, 1+H*W, C]``.
        :type inputs: keras.KerasTensor
        :param training: Training mode flag (unused).
        :type training: Optional[bool]
        :return: Shuffled tensor ``[batch, 1+(H/s)*(W/s), C*s^2]``.
        :rtype: keras.KerasTensor"""
        if self.scale_factor == 1:
            return inputs

        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        channels = inputs.shape[-1]

        cls_token = inputs[:, 0:1, :]
        spatial_tokens = inputs[:, 1:, :]

        # Spatial tokens are assumed to form a square grid.
        spatial_len = seq_len - 1
        h_float = ops.sqrt(ops.cast(spatial_len, "float32"))
        h = ops.cast(h_float, "int32")
        w = h

        spatial_tokens = ops.reshape(spatial_tokens, [batch_size, h, w, channels])

        new_h = h // self.scale_factor
        new_w = w // self.scale_factor
        new_c = channels * (self.scale_factor ** 2)

        shuffled = ops.reshape(spatial_tokens, [
            batch_size, new_h, self.scale_factor, new_w, self.scale_factor, channels
        ])
        shuffled = ops.transpose(shuffled, [0, 1, 3, 2, 4, 5])
        shuffled = ops.reshape(shuffled, [batch_size, new_h * new_w, new_c])

        padding_amount = new_c - channels
        paddings = [[0, 0], [0, 0], [0, padding_amount]]
        cls_token_expanded = ops.pad(cls_token, paddings)

        return ops.concatenate([cls_token_expanded, shuffled], axis=1)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]"""
        input_shape_list = list(input_shape)
        batch_size, seq_len, channels = input_shape_list

        if seq_len is None:
            new_seq_len = None
        else:
            spatial_len = seq_len - 1
            new_spatial_len = spatial_len // (self.scale_factor ** 2)
            new_seq_len = new_spatial_len + 1

        if channels is None:
            new_channels = None
        else:
            new_channels = channels * (self.scale_factor ** 2)

        return tuple([batch_size, new_seq_len, new_channels])

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]"""
        config = super().get_config()
        config.update({
            "scale_factor": self.scale_factor,
            "validate_spatial_dims": self.validate_spatial_dims,
        })
        return config

# ---------------------------------------------------------------------