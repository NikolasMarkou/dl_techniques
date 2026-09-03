"""
An FNet block using Fourier Transforms for token mixing.

This layer constitutes a complete encoder block from the FNet architecture,
serving as a highly efficient drop-in replacement for a standard Transformer
encoder. It substitutes the computationally expensive self-attention mechanism
with a parameter-free 2D Discrete Fourier Transform for O(N log N) token mixing,
while retaining a standard position-wise FFN for channel mixing. The mixing
operation y = Real(FFT(FFT(x))) ensures every output element depends on every
input element, providing comprehensive global information exchange.

References:
    - Lee-Thorp et al. "FNet: Mixing Tokens with Fourier Transforms".
      https://arxiv.org/abs/2105.03824
"""

import keras
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .ffn.factory import FFN_REGISTRY
from .ffn import create_ffn_layer, FFNType
from .stochastic_depth import StochasticDepth
from .norms import create_normalization_layer, NormalizationType
from .attention.fnet_fourier_transform import FNetFourierTransform
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.fnet_encoder_block")
class FNetEncoderBlock(keras.layers.Layer):
    """
    Complete FNet encoder block with Fourier mixing and feed-forward components.

    Replaces self-attention with parameter-free Fourier-based token mixing,
    followed by a configurable feed-forward network. ``normalization_position``
    selects between the two residual arrangements:

    * ``'post'`` (default, and the original FNet/BERT arrangement) —
      [sublayer -> residual -> norm]: the residual is added first and the sum is
      then normalized, ``x = Norm(input + branch(input))``. This is what the
      diagram below draws.
    * ``'pre'`` — ``x = input + branch(Norm(input))``: the branch input is
      normalized and the residual stream is left untouched, so gradients reach
      the input through an unnormalized path. A pre-norm stack needs a final
      normalization after the last block, which :class:`FNet` adds.

    Both arrangements use the same two normalization layers and therefore the
    same weight tree; only the order of operations differs. Uses factory patterns
    for normalization and FFN layer creation, supporting all dl_techniques
    normalization and FFN types.

    Architecture:

    .. code-block:: text

        ┌───────────────────────────────────────────────┐
        │  Input [batch, seq_len, hidden_dim]           │
        └──────────────────┬────────────────────────────┘
                           ▼
        ┌───────────────────────────────────────────────┐
        │  FNet Fourier Transform                       │
        │  y = Real(FFT2D(x))  ── parameter-free        │
        └──────────────────┬────────────────────────────┘
                           ▼
        ┌───────────────────────────────────────────────┐
        │  Residual Add + Normalization                 │
        │  x = Norm(input + fourier_out)                │
        └──────────────────┬────────────────────────────┘
                           ▼
        ┌───────────────────────────────────────────────┐
        │  Feed-Forward Network (configurable type)     │
        └──────────────────┬────────────────────────────┘
                           ▼
        ┌───────────────────────────────────────────────┐
        │  Residual Add + Normalization                 │
        │  output = Norm(ffn_input + ffn_out)           │
        └──────────────────┬────────────────────────────┘
                           ▼
        ┌───────────────────────────────────────────────┐
        │  Output [batch, seq_len, hidden_dim]          │
        └───────────────────────────────────────────────┘

    :param intermediate_dim: Hidden size of feed-forward intermediate layer.
        Ignored if ffn_type requires different parameters.
    :type intermediate_dim: Optional[int]
    :param dropout_rate: Dropout probability for FFN. Defaults to 0.1.
    :type dropout_rate: float
    :param fourier_config: Optional dictionary of FNetFourierTransform
        configuration.
    :type fourier_config: Optional[Dict[str, Any]]
    :param normalization_type: Type of normalization to use. Defaults to
        'layer_norm'.
    :type normalization_type: NormalizationType
    :param normalization_kwargs: Optional normalization-specific parameters.
    :type normalization_kwargs: Optional[Dict[str, Any]]
    :param normalization_position: ``'post'`` (default) or ``'pre'``; see the
        class description. Anything else raises ``ValueError``.
    :type normalization_position: str
    :param ffn_type: Type of feed-forward network to use. Defaults to 'mlp'.
    :type ffn_type: FFNType
    :param ffn_kwargs: Optional FFN-specific parameters.
    :type ffn_kwargs: Optional[Dict[str, Any]]
    :param use_stochastic_depth: Whether to apply stochastic depth (drop-path)
        to each of the two residual branches. Defaults to False, in which case
        the block is bit-identical to one built without the parameter at all.
    :type use_stochastic_depth: bool
    :param stochastic_depth_rate: Drop-path probability used by both branches
        when ``use_stochastic_depth`` is True. Must be in ``[0, 1)``. Defaults
        to 0.1. Ignored when ``use_stochastic_depth`` is False.
    :type stochastic_depth_rate: float
    :param kwargs: Additional Layer base class arguments.
    """

    def __init__(
        self,
        intermediate_dim: Optional[int] = None,
        dropout_rate: float = 0.1,
        fourier_config: Optional[Dict[str, Any]] = None,
        normalization_type: NormalizationType = 'layer_norm',
        normalization_kwargs: Optional[Dict[str, Any]] = None,
        normalization_position: str = 'post',
        ffn_type: FFNType = 'mlp',
        ffn_kwargs: Optional[Dict[str, Any]] = None,
        use_stochastic_depth: bool = False,
        stochastic_depth_rate: float = 0.1,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if intermediate_dim is not None and intermediate_dim <= 0:
            raise ValueError(f"intermediate_dim must be positive, got {intermediate_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        if normalization_position not in ('pre', 'post'):
            raise ValueError(
                f"normalization_position must be 'pre' or 'post', "
                f"got {normalization_position!r}"
            )

        # Store configuration
        self.intermediate_dim = intermediate_dim
        self.dropout_rate = dropout_rate
        self.fourier_config = fourier_config or {}
        self.normalization_type = normalization_type
        self.normalization_kwargs = normalization_kwargs or {}
        self.normalization_position = normalization_position
        self.ffn_type = ffn_type
        self.ffn_kwargs = ffn_kwargs or {}
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_rate = stochastic_depth_rate
        self.supports_masking = True

        # DECISION plan-2026-08-19T163559-499b6f0e/D-081: `name` is explicit so it matches
        # the auto-name of the first instance; a caller-supplied `name` in `fourier_config` still wins.
        self.fourier_transform = FNetFourierTransform(
            **{"name": "f_net_fourier_transform", **self.fourier_config})

        # Two independent stochastic-depth instances, one per residual branch,
        # mirroring transformer.py's attention/ffn drop-path pair. Built only
        # when the flag is on, so the default stays bit-identical without it.
        self.fourier_stochastic_depth = None
        self.ffn_stochastic_depth = None
        if self.use_stochastic_depth:
            self.fourier_stochastic_depth = StochasticDepth(
                drop_path_rate=self.stochastic_depth_rate,
                name='fourier_stochastic_depth'
            )
            self.ffn_stochastic_depth = StochasticDepth(
                drop_path_rate=self.stochastic_depth_rate,
                name='ffn_stochastic_depth'
            )

        # Layer creation will be done in build() to ensure proper shape inference
        self.fourier_layer_norm = None
        self.ffn_layer = None
        self.output_layer_norm = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build encoder block and all sub-layers with proper shape inference.

        :param input_shape: Shape tuple of the input tensor (batch, seq_len,
            hidden_dim).
        :type input_shape: Tuple[Optional[int], ...]
        """
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input, got {len(input_shape)}D: {input_shape}")

        hidden_dim = input_shape[-1]
        if hidden_dim is None:
            raise ValueError(f"Hidden dimension must be known at build time, got {hidden_dim}")

        # Build Fourier transform layer
        self.fourier_transform.build(input_shape)

        # Create normalization layers using factory
        self.fourier_layer_norm = create_normalization_layer(
            normalization_type=self.normalization_type,
            name='fourier_layer_norm',
            **self.normalization_kwargs
        )

        self.output_layer_norm = create_normalization_layer(
            normalization_type=self.normalization_type,
            name='output_layer_norm',
            **self.normalization_kwargs
        )

        # Create FFN layer using factory
        # Prepare FFN parameters
        ffn_params = {
            'output_dim': hidden_dim,
            'name': 'ffn',
            **self.ffn_kwargs
        }

        # `hidden_dim` is the universal sizing knob; pass it to any FFN type
        # whose registry entry declares it as a required or optional parameter.
        if self.intermediate_dim is not None:
            ffn_info = FFN_REGISTRY.get(self.ffn_type)
            accepts_hidden_dim = ffn_info is not None and "hidden_dim" in (
                set(ffn_info["required_params"]) | set(ffn_info["optional_params"])
            )
            if accepts_hidden_dim:
                ffn_params.setdefault('hidden_dim', self.intermediate_dim)

        # Add dropout rate if not already specified
        if 'dropout_rate' not in ffn_params:
            ffn_params['dropout_rate'] = self.dropout_rate

        # A misconfigured `ffn_type` or parameter is a caller error, so raise
        # rather than silently falling back to a different FFN architecture.
        try:
            self.ffn_layer = create_ffn_layer(self.ffn_type, **ffn_params)
        except Exception as e:
            raise ValueError(
                f"Failed to create FFN layer of type '{self.ffn_type}' with "
                f"{sorted(ffn_params)}: {e}"
            ) from e

        # Build normalization layers
        self.fourier_layer_norm.build(input_shape)
        self.output_layer_norm.build(input_shape)

        # Build FFN layer
        self.ffn_layer.build(input_shape)

        # Build stochastic depth layers (weightless, but built for tree parity)
        if self.fourier_stochastic_depth is not None:
            self.fourier_stochastic_depth.build(input_shape)
        if self.ffn_stochastic_depth is not None:
            self.ffn_stochastic_depth.build(input_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through complete FNet encoder block.

        :param inputs: Input tensor of shape [batch, seq_len, hidden_dim].
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional mask tensor of shape [batch, seq_len].
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Boolean indicating training mode.
        :type training: Optional[bool]
        :return: Output tensor of same shape as input.
        :rtype: keras.KerasTensor
        """
        # Drop-path wraps each branch output before the residual add, never the
        # sum itself -- dropping the sum would delete the skip path.
        if self.normalization_position == 'pre':
            # DECISION plan-2026-08-14T233721-d4f9beb2/D-071: normalize the branch input
            # only (`x + branch(Norm(x))`) -- normalizing the sum reinstates a norm on the skip path. See decisions.md.
            fourier_output = self.fourier_transform(
                self.fourier_layer_norm(inputs, training=training),
                attention_mask=attention_mask,
                training=training,
            )
            if self.fourier_stochastic_depth is not None:
                fourier_output = self.fourier_stochastic_depth(
                    fourier_output, training=training
                )
            residual = inputs + fourier_output

            ffn_output = self.ffn_layer(
                self.output_layer_norm(residual, training=training),
                training=training,
            )
            if self.ffn_stochastic_depth is not None:
                ffn_output = self.ffn_stochastic_depth(
                    ffn_output, training=training
                )
            return residual + ffn_output

        # Post-norm (default, the original FNet/BERT arrangement): the residual
        # is added first and the sum is then normalized.
        fourier_output = self.fourier_transform(
            inputs, attention_mask=attention_mask, training=training
        )
        if self.fourier_stochastic_depth is not None:
            fourier_output = self.fourier_stochastic_depth(
                fourier_output, training=training
            )
        fourier_output = self.fourier_layer_norm(
            inputs + fourier_output, training=training
        )

        # Feed-forward network with residual connection
        ffn_output = self.ffn_layer(fourier_output, training=training)
        if self.ffn_stochastic_depth is not None:
            ffn_output = self.ffn_stochastic_depth(
                ffn_output, training=training
            )

        # Final normalization with residual connection
        return self.output_layer_norm(
            fourier_output + ffn_output, training=training
        )

    def compute_mask(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None
    ) -> Optional[keras.KerasTensor]:
        """Propagate the input mask unchanged.

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :param mask: Input mask.
        :type mask: Optional[keras.KerasTensor]
        :return: Unchanged mask.
        :rtype: Optional[keras.KerasTensor]
        """
        return mask

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Encoder block preserves input shape.

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape (same as input).
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return complete configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'intermediate_dim': self.intermediate_dim,
            'dropout_rate': self.dropout_rate,
            'fourier_config': self.fourier_config,
            'normalization_type': self.normalization_type,
            'normalization_kwargs': self.normalization_kwargs,
            'normalization_position': self.normalization_position,
            'ffn_type': self.ffn_type,
            'ffn_kwargs': self.ffn_kwargs,
            'use_stochastic_depth': self.use_stochastic_depth,
            'stochastic_depth_rate': self.stochastic_depth_rate,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'FNetEncoderBlock':
        """Create layer from configuration dictionary.

        :param config: Configuration dictionary.
        :type config: Dict[str, Any]
        :return: New layer instance.
        :rtype: FNetEncoderBlock
        """
        return cls(**config)

# ---------------------------------------------------------------------
