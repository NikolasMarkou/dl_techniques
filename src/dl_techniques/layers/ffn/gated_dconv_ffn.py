"""Restormer's Gated-Dconv Feed-Forward Network, the
:class:`GatedDConvFeedForward` layer.

A transformer feed-forward block normally expands the channel width, applies a
pointwise non-linearity and projects back. GDFN changes that in two ways. It
adds a **depthwise 3x3 stage** on the expanded tensor, so the block carries a
local spatial prior instead of acting position-wise, and it makes the
non-linearity a **gate**: the expanded tensor is split in half, one half is
passed through GELU and multiplies the other half. The multiplicative
interaction is what lets the block suppress or pass information per position
and per channel, which is the property Restormer relies on to keep only the
detail worth propagating through a restoration decoder.

Two details of that recipe are load-bearing and invisible to a shape test, so
they are pinned in the code and guarded in
``tests/test_layers/test_ffn/test_gated_dconv_ffn.py``:

1. the hidden width is ``int(dim * ffn_expansion_factor)`` with Python
   **truncation**, not rounding -- at the four DocRes widths that is
   48 -> 127, 96 -> 255, 192 -> 510, 384 -> 1021, and ``round()`` would give
   128 and 511 for the first and third;
2. GELU is applied to the **first** half of the split and multiplies the
   second, in the **exact/erf** form rather than the tanh approximation.

References:
    - Zamir et al., 2022. Restormer: Efficient Transformer for High-Resolution
      Image Restoration. CVPR 2022. (https://arxiv.org/abs/2111.09881)
    - Zhang et al., 2024. DocRes: A Generalist Model Toward Unifying Document
      Image Restoration Tasks. CVPR 2024. (https://arxiv.org/abs/2405.04408)
      DocRes is the consumer in this repository; its backbone is an unmodified
      Restormer, so this layer is shared rather than buried in that model.
"""

import keras
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.ffn.gated_dconv_ffn")
class GatedDConvFeedForward(keras.layers.Layer):
    """
    Restormer's Gated-Dconv Feed-Forward Network (GDFN), in NHWC.

    The output width always equals the input width ``dim``; the expansion is
    internal only.

    Architecture:

    .. code-block:: text

        inputs [B, H, W, dim]
                │
                ▼
        Conv2D(2 * hidden, 1x1, use_bias)      # hidden = int(dim * factor)
                │  [B, H, W, 2 * hidden]
                ▼
        DepthwiseConv2D(3x3, same)             # fully depthwise over 2*hidden
                │  [B, H, W, 2 * hidden]
                ▼
        split(2, axis=-1) -> x1, x2            # FIRST half is the gate
                │  each [B, H, W, hidden]
                ▼
        gelu(x1, approximate=False) * x2       # exact/erf GELU, on x1
                │  [B, H, W, hidden]
                ▼
        Conv2D(dim, 1x1, use_bias)             # input width is hidden, not 2*hidden
                │
                ▼
        output [B, H, W, dim]

    :param dim: Number of input and output channels. Must be positive.
    :type dim: int
    :param ffn_expansion_factor: Multiplier for the hidden width. Must be
        positive. The hidden width is ``int(dim * ffn_expansion_factor)``,
        TRUNCATED. Defaults to ``2.66``, the value every Restormer and DocRes
        call site uses.
    :type ffn_expansion_factor: float
    :param use_bias: Whether the three convolutions carry a bias. Defaults to
        ``False``, which is the value every Restormer and DocRes call site
        uses.
    :type use_bias: bool
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar dim: The configured channel count.
    :vartype dim: int
    :ivar ffn_expansion_factor: The configured expansion factor.
    :vartype ffn_expansion_factor: float
    :ivar hidden_features: ``int(dim * ffn_expansion_factor)``, the gated
        width. The layer's widest tensor is twice this.
    :vartype hidden_features: int
    :ivar project_in: The ``1x1`` projection to ``2 * hidden_features``
        channels.
    :vartype project_in: keras.layers.Conv2D
    :ivar dwconv: The fully depthwise ``3x3`` convolution on
        ``2 * hidden_features`` channels.
    :vartype dwconv: keras.layers.DepthwiseConv2D
    :ivar project_out: The ``1x1`` projection from ``hidden_features`` back to
        ``dim`` channels.
    :vartype project_out: keras.layers.Conv2D

    :raises ValueError: If ``dim`` or ``ffn_expansion_factor`` is not
        positive, or if ``int(dim * ffn_expansion_factor)`` truncates to zero.
    :raises ValueError: From ``build()``, if the input is not rank 4 or if its
        trailing dimension does not equal ``dim``.

    Input shape:
        4D tensor with shape ``(batch_size, height, width, dim)``. ``height``
        and ``width`` may be ``None``; the layer reads them at runtime.

    Output shape:
        4D tensor with shape ``(batch_size, height, width, dim)``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.ffn.gated_dconv_ffn import (
            GatedDConvFeedForward,
        )

        x = keras.random.normal((2, 64, 64, 48))
        y = GatedDConvFeedForward(dim=48)(x)
        # y.shape == (2, 64, 64, 48); the hidden width inside is 127
    """

    def __init__(
            self,
            dim: int,
            ffn_expansion_factor: float = 2.66,
            use_bias: bool = False,
            **kwargs: Any
    ) -> None:
        """Validate the configuration and create the three sub-layers.

        :param dim: Number of input and output channels.
        :type dim: int
        :param ffn_expansion_factor: Multiplier for the hidden width.
        :type ffn_expansion_factor: float
        :param use_bias: Whether the convolutions carry a bias.
        :type use_bias: bool
        :param kwargs: Additional keyword arguments for the ``Layer`` base
            class.
        :type kwargs: Any

        :raises ValueError: If ``dim`` or ``ffn_expansion_factor`` is not
            positive, or if ``int(dim * ffn_expansion_factor)`` is zero.
        """
        super().__init__(**kwargs)

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if ffn_expansion_factor <= 0:
            raise ValueError(
                "ffn_expansion_factor must be positive, got "
                f"{ffn_expansion_factor}"
            )

        # Python `int()` TRUNCATES toward zero; upstream relies on that.
        # `round()` would give 128 at dim=48 and 511 at dim=192, i.e. a
        # silently wider layer than the reference. Measured in D-008:
        # 48 -> 127, 96 -> 255, 192 -> 510, 384 -> 1021.
        hidden_features = int(dim * ffn_expansion_factor)
        if hidden_features < 1:
            raise ValueError(
                f"int(dim * ffn_expansion_factor) truncates to "
                f"{hidden_features}, which is not a usable hidden width; got "
                f"dim={dim} and ffn_expansion_factor={ffn_expansion_factor}"
            )

        self.dim = dim
        self.ffn_expansion_factor = ffn_expansion_factor
        self.use_bias = use_bias
        self.hidden_features = hidden_features

        self.project_in = keras.layers.Conv2D(
            filters=2 * hidden_features,
            kernel_size=1,
            use_bias=use_bias,
            name="project_in",
        )

        # A Keras `DepthwiseConv2D` with the default `depth_multiplier=1` on
        # `2 * hidden_features` input channels is exactly the upstream
        # `Conv2d(2h, 2h, 3, groups=2h)`: its kernel is `(3, 3, 2h, 1)`, so
        # each output channel sees only its own input channel.
        # `test_gated_dconv_ffn.py::test_the_dwconv_is_fully_depthwise`
        # asserts both the weight count and that per-channel independence.
        self.dwconv = keras.layers.DepthwiseConv2D(
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=use_bias,
            name="dwconv",
        )

        # The input width here is `hidden_features`, NOT `2 * hidden_features`
        # -- the gating multiply has already halved the tensor.
        self.project_out = keras.layers.Conv2D(
            filters=dim,
            kernel_size=1,
            use_bias=use_bias,
            name="project_out",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the three convolutions.

        Every sub-layer that ``call()`` runs is built explicitly here, so all
        weight variables exist before weight restoration during model loading.

        :param input_shape: Shape tuple of the input tensor, expected to be
            ``(batch_size, height, width, dim)``.
        :type input_shape: tuple

        :raises ValueError: If ``input_shape`` is not rank 4, or if its
            trailing dimension does not equal ``dim``.
        """
        if self.built:
            return

        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape (batch, height, width, channels), "
                f"got {len(input_shape)}D: {input_shape}"
            )
        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            raise ValueError(
                f"Expected input channels ({input_shape[-1]}) to match "
                f"layer dim ({self.dim})"
            )

        batch, height, width = input_shape[0], input_shape[1], input_shape[2]

        self.project_in.build(input_shape)
        self.dwconv.build((batch, height, width, 2 * self.hidden_features))
        self.project_out.build((batch, height, width, self.hidden_features))

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply the gated depthwise feed-forward transform to a feature map.

        :param inputs: Input tensor of shape ``(batch, height, width, dim)``.
        :type inputs: keras.KerasTensor
        :param training: Standard Keras training flag. Unused -- the layer has
            no dropout and no normalisation statistics.
        :type training: Optional[bool]
        :return: Tensor of shape ``(batch, height, width, dim)``.
        :rtype: keras.KerasTensor
        """
        x = self.dwconv(self.project_in(inputs))

        # Split order matters: upstream `torch.chunk(x, 2, dim=1)` puts the
        # GATE in the first half and the value in the second.
        x1, x2 = keras.ops.split(x, 2, axis=-1)

        # `approximate=False` is pinned rather than left to the default.
        # D-008 measured the current Keras default AS False (exact/erf), and
        # also measured the two forms as 4.12e-04 apart -- far above this
        # layer's float32 parity bound. Pinning it means a future Keras
        # default flip cannot silently swap the activation underneath the
        # port. Do NOT "simplify" this to `keras.activations.gelu(x1)`.
        x = keras.activations.gelu(x1, approximate=False) * x2

        return self.project_out(x)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Return the output shape, which matches the input with ``dim`` channels.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: ``(batch, height, width, dim)``.
        :rtype: tuple
        """
        return tuple(input_shape[:-1]) + (self.dim,)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments needed to recreate this layer.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "ffn_expansion_factor": self.ffn_expansion_factor,
            "use_bias": self.use_bias,
        })
        return config

# ---------------------------------------------------------------------
