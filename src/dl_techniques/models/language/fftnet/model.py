"""
FFTNet vision encoder: a pure feature-extracting backbone that replaces self-attention
with adaptive spectral filtering, plus the block and mixer it is built from.

Self-attention mixes tokens by forming an `N x N` score matrix, which costs `O(N^2)`
in both time and memory and is the reason long sequences and high-resolution images
are expensive. The convolution theorem offers a different route to global mixing:
a pointwise multiplication in the frequency domain is a circular convolution in the
token domain, so multiplying by a length-`N` filter couples every token to every
other token at `O(N log N)` -- the cost of the transform -- and with `O(N)` parameters
instead of `O(N^2)` computation.

The catch is that a fixed filter makes the mixing input-independent, which is
precisely what attention buys and what a plain Fourier mixer gives up. FFTNet
recovers the adaptivity by conditioning the filter on the input. A global context
vector `c = mean(x, axis=tokens)` is passed through a small MLP to produce a
per-feature offset, which is added to a learned base filter:

`W = W_base + MLP(mean(x))`,  `y = IFFT(modReLU(FFT(x) * W))`

so the spectral gains applied to a given image depend on that image's summary
statistics. It is weaker than attention -- the modulation is one global vector, not a
per-token-pair score -- and that is the trade: global receptive field and log-linear
cost, in exchange for content dependence that is global rather than pairwise.

The nonlinearity is `modReLU`, which is what keeps a stack of these layers from
collapsing. Applying a real ReLU to a complex tensor is not meaningful, and applying
none at all would make consecutive spectral filters compose into a single linear
filter. `modReLU` acts on the magnitude only -- it shifts `|z|` by a learned
per-feature bias, rectifies, and rescales `z` by the ratio -- so the phase, which is
where the spatial arrangement of the signal lives, passes through untouched. The
magnitude used in the denominator is floored at `1e-8` so a zero-magnitude bin does
not produce a division by zero. The bias initializes at `-0.1`, a small negative
value, so the activation starts by suppressing low-magnitude bins rather than acting
as the identity.

**Which axis the FFT runs over is the thing to get right, and it was wrong once.**
`tf.signal.fft` transforms the INNERMOST axis. The token state is `(B, N, D)`, so
calling it directly transformed `D`, the feature axis, and the layer performed no
token mixing whatsoever -- the one thing the architecture exists to do. The sequence
axis is therefore transposed to the end for the transform and transposed back
afterwards. The shape of `W_base`, `(seq_len, embed_dim)`, is a gain per frequency
BIN per feature, and it is only meaningful when the bins index the token axis; that
shape is the check on this. Because `W_base` is sized by `seq_len`, the model is tied
to the token count it was built for -- a fixed image resolution, unlike attention.

**ACCEPTED RAW-TF EXCEPTION (production-map §L2-5 / H10).** ``FFTMixer.call`` uses
``tf.signal.fft`` / ``tf.signal.ifft`` on a complex64 tensor. This cannot migrate to
``keras.ops``: ``keras.ops`` exposes only a real/imag-tuple ``fft`` and has NO
``ifft``, so a backend-agnostic complex forward+inverse transform is not
expressible. The raw ``tf.signal`` path is a documented exception to the
keras.ops-only rule for the forward pass.

Structurally each block is a standard pre-norm transformer block with the mixer in
the attention slot: `x + FFTMixer(norm(x))` then `x + FFN(norm(x))`. Keeping the
residual-and-norm skeleton intact is deliberate -- it isolates the mixing mechanism as
the only variable, so a comparison against an attention baseline measures the mixer
rather than a differently-tuned block.

The `FFTNet` class is a pure encoder and holds no pooling or classification layer. It
embeds patches, prepends a CLS token, adds a learned positional embedding, runs the
block stack, and returns a dictionary of ``last_hidden_state``, ``cls_token`` and
``patch_features``. Returning all three unconditionally, rather than switching the
return type on a flag, gives downstream heads a stable interface: a classification
head reads the CLS token, a dense-prediction head reads the patch features, and
neither needs the encoder reconfigured. Heads are attached externally through
``create_fftnet_with_head``.

References:
    - Fein-Ashley, 2025. The FFT Strikes Back: An Efficient Alternative to
      Self-Attention. (https://arxiv.org/abs/2502.18394)
    - Lee-Thorp et al., 2021. FNet: Mixing Tokens with Fourier Transforms.
      (https://arxiv.org/abs/2105.03824)
    - Arjovsky et al., 2015. Unitary Evolution Recurrent Neural Networks.
      (https://arxiv.org/abs/1511.06464)
    - Rao et al., 2021. Global Filter Networks for Image Classification.
      (https://arxiv.org/abs/2107.00645)
    - Dosovitskiy et al., 2020. An Image is Worth 16x16 Words: Transformers for
      Image Recognition at Scale. (https://arxiv.org/abs/2010.11929)
"""

import keras
import tensorflow as tf
from typing import Optional, Dict, Any, Tuple, Literal

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.embedding.patch_embedding import PatchEmbedding2D

# ---------------------------------------------------------------------
# Core FFT Mixing Layer (As Described in Paper)
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class FFTMixer(keras.layers.Layer):
    """Adaptive spectral filtering: global token mixing at ``O(N log N)``.

    Replaces the ``O(N^2)`` self-attention score matrix with a pointwise
    multiplication in the frequency domain, which by the convolution theorem is
    a circular convolution over tokens and therefore couples every token to
    every other one. Adaptivity, which a fixed Fourier mixer gives up, is
    recovered by conditioning the filter on a global context vector:
    ``W = W_base + MLP(mean(x))``. The modulation is ONE global vector, not a
    per-token-pair score, so this is weaker than attention by construction and
    that is the trade.

    **Operation:**

    .. code-block:: text

        Input X [B, N, D]  (real)
              │
              ├──────────────────────────────┐
              ▼                              ▼
        ┌──────────────────────┐   ┌──────────────────────────┐
        │ cast complex64       │   │ c = mean(X, axis=tokens) │
        │ transpose → [B,D,N]  │   │   [B, D]                 │
        │ tf.signal.fft        │   │        ▼                 │
        │ transpose → [B,N,D]  │   │ Dense(mlp_hidden, gelu)  │
        │                      │   │        ▼                 │
        │ the transposes are   │   │ Dense(D)  →  ΔW  [B, D]  │
        │ NOT cosmetic; see    │   │        ▼                 │
        │ the axis note below  │   │ W = W_base + ΔW[:,None,:]│
        └──────────┬───────────┘   └────────────┬─────────────┘
                   │  F [B,N,D] complex         │  W [B,N,D]
                   └──────────────┬─────────────┘
                                  ▼
                          F̃ = F ⊙ complex(W)
                                  ▼
                ┌───────────────────────────────────────┐
                │  modReLU: magnitude only              │
                │    |z| + b → relu → scale z by ratio  │
                │    PHASE PASSES THROUGH UNTOUCHED     │
                └───────────────────┬───────────────────┘
                                    ▼
                ┌───────────────────────────────────────┐
                │ transpose → [B,D,N]                   │
                │ tf.signal.ifft                        │
                │ transpose → [B,N,D] → real() → cast   │
                └───────────────────┬───────────────────┘
                                    ▼
                              Dropout
                                    ▼
                        Output Y [B, N, D]  (real)

    **Which axis the FFT runs over (this was wrong once):**

    .. code-block:: text

        tf.signal.fft transforms the INNERMOST axis.

        WRONG:  fft(X)                  X is [B, N, D]
                                        → transforms D, the FEATURE axis
                                        → NO token mixing at all, i.e. the
                                          one thing this layer exists to do

        RIGHT:  transpose to [B, D, N] ► fft ► transpose back

        The check on this is W_base's shape, (seq_len, embed_dim):
        a gain per frequency BIN per feature, meaningful only when
        the bins index the TOKEN axis.

        Consequence: W_base is sized by seq_len, so the layer is
        TIED to the token count it was built for -- a fixed image
        resolution, unlike attention.

    **modReLU (why not a plain ReLU, and why not nothing):**

    .. code-block:: text

        plain relu(z) on a complex tensor  →  not meaningful
        no nonlinearity at all             →  consecutive spectral
                                              filters COLLAPSE into
                                              one linear filter

        modReLU(z) = z · relu(|z| + b) / max(|z|, 1e-8)
                         └─ magnitude ─┘   └─ floor guards a
                            shifted and       zero-magnitude bin
                            rectified

        b initializes at −0.1, a small NEGATIVE value, so the
        activation starts by SUPPRESSING low-magnitude bins
        rather than acting as the identity.

    :param embed_dim: Embedding dimension ``D``; preserved through the layer.
    :type embed_dim: int
    :param mlp_hidden_dim: Hidden width of the adaptive filter MLP that maps the
        global context vector to the per-feature filter offset. Defaults to 256.
    :type mlp_hidden_dim: int
    :param dropout_rate: Dropout probability applied to the real output.
        Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias_in_modrelu: Whether ``modrelu_bias`` is created in
        :meth:`build` and added inside modReLU. Defaults to True.
    :type use_bias_in_modrelu: bool
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    Input shape:
        3D tensor ``(batch_size, sequence_length, embed_dim)``. The sequence
        length must match the one the layer was built at, because ``W_base`` is
        sized by it.

    Output shape:
        3D tensor ``(batch_size, sequence_length, embed_dim)``.

    :ivar W_base: Learned base spectral filter of shape
        ``(seq_len, embed_dim)``, initialized to ones: a gain per frequency bin
        per feature.
    :vartype W_base: keras.Variable
    :ivar modrelu_bias: Per-feature magnitude bias of shape ``(embed_dim,)``,
        initialized to ``-0.1``. ``None`` when ``use_bias_in_modrelu`` is False.
    :vartype modrelu_bias: Optional[keras.Variable]
    :ivar filter_mlp: The two-layer MLP producing the filter offset.
    :vartype filter_mlp: keras.Sequential
    """

    def __init__(
            self,
            embed_dim: int,
            mlp_hidden_dim: int = 256,
            dropout_rate: float = 0.0,
            use_bias_in_modrelu: bool = True,
            **kwargs: Any
    ) -> None:
        """Initialize the mixer and create the filter MLP.

        :param embed_dim: Embedding dimension.
        :type embed_dim: int
        :param mlp_hidden_dim: Hidden width of the adaptive filter MLP.
        :type mlp_hidden_dim: int
        :param dropout_rate: Dropout probability.
        :type dropout_rate: float
        :param use_bias_in_modrelu: Whether modReLU carries a learnable bias.
        :type use_bias_in_modrelu: bool
        :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
        """
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.dropout_rate = dropout_rate
        self.use_bias_in_modrelu = use_bias_in_modrelu

        # Adaptive filter MLP: c -> ΔW
        self.filter_mlp = keras.Sequential([
            keras.layers.Dense(mlp_hidden_dim, activation='gelu', name='mlp_hidden'),
            keras.layers.Dense(embed_dim, name='mlp_out')
        ], name='filter_mlp')

        self.dropout = keras.layers.Dropout(dropout_rate)

        # Will be created in build()
        self.modrelu_bias = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the frequency-dependent parameters and build the filter MLP.

        :param input_shape: Shape ``(batch, seq_len, embed_dim)``. ``seq_len``
            fixes the first axis of ``W_base``, which is why the layer is tied
            to one token count.
        :type input_shape: Tuple[Optional[int], ...]
        """
        _, seq_len, embed_dim = input_shape

        # Base spectral filter W_base (initialized to ones)
        self.W_base = self.add_weight(
            name='W_base',
            shape=(seq_len, embed_dim),
            initializer=keras.initializers.Ones(),
            trainable=True,
            dtype="float32"
        )

        # modReLU bias (per feature, applies to magnitude)
        if self.use_bias_in_modrelu:
            self.modrelu_bias = self.add_weight(
                name='modrelu_bias',
                shape=(embed_dim,),
                initializer=keras.initializers.Constant(-0.1),
                trainable=True,
                dtype="float32"
            )

        # Build sub-layers
        self.filter_mlp.build((input_shape[0], embed_dim))

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass: FFT, adaptive filter, modReLU, inverse FFT.

        :param inputs: Input tensor of shape
            ``(batch, sequence_length, embed_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer is in training mode.
        :type training: Optional[bool]
        :return: Real output tensor of the same shape.
        :rtype: keras.KerasTensor
        """
        # 1. Fourier Transform along the TOKEN axis.
        #
        # ``tf.signal.fft`` transforms the INNERMOST axis. ``inputs`` is
        # (B, N, D), so calling it directly transformed D — the feature axis —
        # and the layer performed no token mixing at all, which is the one thing
        # this architecture exists to do. The sequence axis is therefore moved
        # to the end for the transform and moved back afterwards.
        #
        # ``W_base`` has shape (seq_len, embed_dim), i.e. a gain per frequency
        # BIN per feature; that shape is only meaningful when the bins index the
        # token axis. The repo's other Fourier layer does the same thing
        # explicitly — see layers/attention/fnet_fourier_transform.py:368-374.
        x_complex = keras.ops.cast(inputs, dtype="complex64")
        F = keras.ops.transpose(
            tf.signal.fft(keras.ops.transpose(x_complex, (0, 2, 1))), (0, 2, 1))

        # 2. Adaptive Spectral Filtering
        c = keras.ops.mean(inputs, axis=1)
        delta_W = self.filter_mlp(c)
        delta_W_expanded = keras.ops.expand_dims(delta_W, axis=1)
        W = keras.ops.cast(self.W_base, delta_W_expanded.dtype) + delta_W_expanded
        W_complex = keras.ops.cast(W, dtype="complex64")
        F_filtered = F * W_complex

        # 3. Nonlinear Activation: modReLU
        F_activated = self._apply_modrelu(F_filtered)

        # 4. Inverse Fourier Transform (same axis handling as the forward FFT)
        Y_complex = keras.ops.transpose(
            tf.signal.ifft(keras.ops.transpose(F_activated, (0, 2, 1))), (0, 2, 1))
        # ``real()`` of a complex64 tensor is float32; hand the caller the
        # layer's own compute dtype (a no-op under the float32 policy).
        Y = keras.ops.cast(keras.ops.real(Y_complex), self.compute_dtype)

        # 5. Apply dropout
        Y = self.dropout(Y, training=training)

        return Y

    def _apply_modrelu(self, z: keras.KerasTensor) -> keras.KerasTensor:
        """Apply modReLU to a complex tensor, rescaling magnitude and keeping phase.

        :param z: Complex64 tensor, the filtered spectrum.
        :type z: keras.KerasTensor
        :return: Complex64 tensor of the same shape, with magnitudes shifted by
            the learned bias and rectified and phases unchanged.
        :rtype: keras.KerasTensor
        """
        # DECISION plan-2026-08-19T163559-499b6f0e/D-054
        # ``magnitude`` is float32 by construction -- it is ``abs()`` of a
        # complex64 tensor and TensorFlow's complex ops have no half-precision
        # kernel. ``self.modrelu_bias`` is an ordinary autocast weight, so under
        # ``mixed_float16`` it arrived as float16 and this add raised
        # ``InvalidArgumentError: cannot compute AddV2``. The bias is lifted TO
        # the magnitude's dtype, never the magnitude cast DOWN to the bias's:
        # halving the magnitude would put the modReLU threshold and the
        # ``1e-8`` floor two lines below on different scales (float16 cannot
        # represent 1e-8 at all -- it is exactly 0.0). The sibling ``eps`` at
        # :252 already pins float32 for that reason. See decisions.md D-054.
        magnitude = keras.ops.abs(z)

        if self.modrelu_bias is not None:
            magnitude_biased = magnitude + keras.ops.cast(
                self.modrelu_bias, magnitude.dtype
            )
        else:
            magnitude_biased = magnitude

        magnitude_activated = keras.ops.relu(magnitude_biased)

        eps = keras.ops.convert_to_tensor(1e-8, dtype="float32")
        magnitude_safe = keras.ops.maximum(magnitude, eps)
        scale = magnitude_activated / magnitude_safe

        scale_complex = keras.ops.cast(scale, dtype="complex64")
        return z * scale_complex

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape, which is identical to the input shape.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'mlp_hidden_dim': self.mlp_hidden_dim,
            'dropout_rate': self.dropout_rate,
            'use_bias_in_modrelu': self.use_bias_in_modrelu,
        })
        return config


# ---------------------------------------------------------------------
# FFTNet Transformer Block
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class FFTNetBlock(keras.layers.Layer):
    """A pre-norm transformer block with :class:`FFTMixer` in the attention slot.

    The residual-and-norm skeleton is deliberately left intact: keeping
    everything except the mixer identical to a standard block is what makes a
    comparison against an attention baseline measure the MIXER rather than a
    differently-tuned block.

    **Block structure:**

    .. code-block:: text

        x ──┬─► norm1 ─► FFTMixer ─┐
            │                      ▼
            └───────────────────► (+)
                                   │
            ┌──────────────────────┘
            │
        x ──┬─► norm2 ─► FFN ──────┐
            │                      ▼
            └───────────────────► (+)
                                   ▼
                                Output

        pre-norm throughout: the skip path is an unmodified
        identity, and only the mixer differs from a standard
        attention block

    :param embed_dim: Embedding dimension, preserved through the block.
    :type embed_dim: int
    :param mlp_hidden_dim: Hidden width of the mixer's adaptive filter MLP.
        Defaults to 256.
    :type mlp_hidden_dim: int
    :param ffn_ratio: Expansion factor for the FFN hidden dimension, which is
        ``ffn_ratio * embed_dim``. Defaults to 4.
    :type ffn_ratio: int
    :param dropout_rate: Dropout probability, shared by the mixer and the FFN.
        Defaults to 0.0.
    :type dropout_rate: float
    :param ffn_type: FFN identifier passed to ``create_ffn_layer``. Defaults to
        ``'mlp'``.
    :type ffn_type: str
    :param normalization_type: Normalization identifier passed to
        ``create_normalization_layer``. Defaults to ``'layer_norm'``.
    :type normalization_type: str
    :param use_bias_in_modrelu: Whether the block's :class:`FFTMixer` uses a
        learnable bias in modReLU. Forwarded verbatim; defaults to
        :class:`FFTMixer`'s own default (``True``) so existing configurations
        are unchanged.
    :type use_bias_in_modrelu: bool
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    Input shape:
        3D tensor ``(batch_size, sequence_length, embed_dim)``.

    Output shape:
        3D tensor ``(batch_size, sequence_length, embed_dim)``.
    """

    def __init__(
            self,
            embed_dim: int,
            mlp_hidden_dim: int = 256,
            ffn_ratio: int = 4,
            dropout_rate: float = 0.0,
            ffn_type: str = 'mlp',
            normalization_type: str = 'layer_norm',
            use_bias_in_modrelu: bool = True,
            **kwargs: Any
    ) -> None:
        """Initialize the block and create its four sub-layers.

        :param embed_dim: Embedding dimension.
        :type embed_dim: int
        :param mlp_hidden_dim: Hidden width of the mixer's filter MLP.
        :type mlp_hidden_dim: int
        :param ffn_ratio: FFN expansion factor.
        :type ffn_ratio: int
        :param dropout_rate: Dropout probability.
        :type dropout_rate: float
        :param ffn_type: FFN identifier for the factory.
        :type ffn_type: str
        :param normalization_type: Normalization identifier for the factory.
        :type normalization_type: str
        :param use_bias_in_modrelu: Whether modReLU carries a learnable bias.
        :type use_bias_in_modrelu: bool
        :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
        """
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.ffn_ratio = ffn_ratio
        self.dropout_rate = dropout_rate
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type
        self.use_bias_in_modrelu = use_bias_in_modrelu

        # Create sub-layers using factories
        self.norm1 = create_normalization_layer(normalization_type, name='norm1')

        # DECISION plan-2026-08-22T035419-a11304c8/D-011
        # ``use_bias_in_modrelu`` MUST be forwarded here. It is a fully wired
        # ``FFTMixer`` knob -- it decides whether ``modrelu_bias`` is created in
        # ``FFTMixer.build`` and whether ``_apply_modrelu`` adds it -- but for as
        # long as this constructor omitted the keyword, ``FFTMixer``'s own
        # default was the only value ANY ``FFTNetBlock``/``FFTNet``/
        # ``create_fftnet_*`` caller could reach, and the knob was serialized at
        # the mixer level while being unreachable from every shipped entry
        # point. Do not "simplify" this back to a positional-only construction.
        # The default is pinned to ``FFTMixer``'s own (``True``) so no existing
        # config changes meaning. See D-011 in decisions.md.
        self.fft_mixer = FFTMixer(
            embed_dim=embed_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout_rate=dropout_rate,
            use_bias_in_modrelu=use_bias_in_modrelu,
            name='fft_mixer'
        )

        self.norm2 = create_normalization_layer(normalization_type, name='norm2')

        self.ffn = create_ffn_layer(
            ffn_type,
            hidden_dim=ffn_ratio * embed_dim,
            output_dim=embed_dim,
            dropout_rate=dropout_rate,
            name='ffn'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the two norms, the mixer and the FFN explicitly.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        """
        self.norm1.build(input_shape)
        self.fft_mixer.build(input_shape)
        self.norm2.build(input_shape)
        self.ffn.build(input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass: mixer residual, then FFN residual.

        :param inputs: Input tensor of shape
            ``(batch, sequence_length, embed_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer is in training mode.
        :type training: Optional[bool]
        :return: Output tensor of the same shape.
        :rtype: keras.KerasTensor
        """
        # First residual: FFT mixing
        x = inputs + self.fft_mixer(self.norm1(inputs), training=training)

        # Second residual: FFN
        x = x + self.ffn(self.norm2(x))

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape, which is identical to the input shape.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'mlp_hidden_dim': self.mlp_hidden_dim,
            'ffn_ratio': self.ffn_ratio,
            'dropout_rate': self.dropout_rate,
            'ffn_type': self.ffn_type,
            'normalization_type': self.normalization_type,
            'use_bias_in_modrelu': self.use_bias_in_modrelu,
        })
        return config


# ---------------------------------------------------------------------
# FFTNet Foundation Model
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class FFTNet(keras.Model):
    """FFTNet: a pure vision encoder built on adaptive spectral filtering.

    A ViT-shaped backbone in which every attention sublayer is an
    :class:`FFTMixer`. It embeds patches, prepends a CLS token, adds a learned
    positional embedding, runs ``num_layers`` :class:`FFTNetBlock` instances and
    normalizes. The model holds NO pooling and NO classification layer, and it
    returns all three of ``last_hidden_state``, ``cls_token`` and
    ``patch_features`` UNCONDITIONALLY rather than switching its return type on
    a flag: a classification head reads the CLS token, a dense-prediction head
    reads the patch features, and neither needs the encoder reconfigured. Heads
    attach externally through :func:`create_fftnet_with_head`.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  Input [B, image_size, image_size, 3]│
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  PatchEmbedding2D(patch_size)        │
        │  → [B, N, D],  N = (I / P)²          │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  prepend CLS token                   │
        │  ONE (1, 1, D) weight, tiled over B  │
        │  → [B, N+1, D]                       │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  + pos_embed  (1, N+1, D), LEARNED   │
        │  → Dropout                           │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  FFTNetBlock₁ (FFTMixer → FFN)       │
        └───────────────┬──────────────────────┘
                        ▼
                       ...
                        ▼
        ┌──────────────────────────────────────┐
        │  FFTNetBlockₙ                        │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  final normalization                 │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────────┐
        │  Output dict, ALL THREE always           │
        │    "last_hidden_state"  [B, N+1, D]      │
        │    "cls_token"          [B, D]    x[:,0] │
        │    "patch_features"     [B, N, D] x[:,1:]│
        └──────────────────────────────────────────┘

        FIXED RESOLUTION: each mixer's W_base is sized by N+1,
        so the encoder is tied to the token count it was built
        for. Unlike attention, it cannot be re-run at another
        image size.

    **Variants:**

    .. code-block:: text

        variant   embed_dim   layers   mlp_hidden   ffn_ratio   params
        tiny         384         4          96          4        --
        small        512         6         128          4        --
        base         768        12         256          4       ~76M
        large       1024        24         512          4      ~268M
        huge        1280        32         640          4      ~540M

    :param image_size: Input image size; images are assumed square. Must be
        positive and divisible by ``patch_size``. Defaults to 224.
    :type image_size: int
    :param patch_size: Edge length of each square patch. Must be positive.
        Defaults to 16.
    :type patch_size: int
    :param embed_dim: Embedding dimension. Must be positive. Defaults to 768.
    :type embed_dim: int
    :param num_layers: Number of :class:`FFTNetBlock` instances. Must be
        positive. Defaults to 12.
    :type num_layers: int
    :param mlp_hidden_dim: Hidden width of each mixer's adaptive filter MLP.
        Defaults to 256.
    :type mlp_hidden_dim: int
    :param ffn_ratio: FFN expansion factor. Defaults to 4.
    :type ffn_ratio: int
    :param dropout_rate: Dropout probability, in ``[0, 1]``. Defaults to 0.1.
    :type dropout_rate: float
    :param ffn_type: FFN identifier passed to ``create_ffn_layer``. Defaults to
        ``'mlp'``.
    :type ffn_type: str
    :param normalization_type: Normalization identifier passed to
        ``create_normalization_layer``. Defaults to ``'layer_norm'``.
    :type normalization_type: str
    :param use_bias_in_modrelu: Whether each block's :class:`FFTMixer` uses a
        learnable bias in modReLU. Defaults to True (:class:`FFTMixer`'s own
        default).
    :type use_bias_in_modrelu: bool
    :param kwargs: Additional keyword arguments for the ``Model`` base class.

    :raises ValueError: If ``image_size`` or ``patch_size`` is not positive, if
        ``image_size`` is not divisible by ``patch_size``, if ``embed_dim`` or
        ``num_layers`` is not positive, or if ``dropout_rate`` leaves ``[0, 1]``.

    Input shape:
        4D tensor ``(batch_size, image_size, image_size, 3)``.

    Output shape:
        A mapping with:

        - ``last_hidden_state``: ``(batch, num_patches + 1, embed_dim)``
        - ``cls_token``: ``(batch, embed_dim)``
        - ``patch_features``: ``(batch, num_patches, embed_dim)``

    :ivar num_patches: ``(image_size // patch_size) ** 2``.
    :vartype num_patches: int
    :ivar cls_token: Learnable ``(1, 1, embed_dim)`` token, created in ``build``.
    :vartype cls_token: keras.Variable
    :ivar pos_embed: Learnable ``(1, num_patches + 1, embed_dim)`` table,
        created in ``build``.
    :vartype pos_embed: keras.Variable
    :ivar blocks: The block stack.
    :vartype blocks: list[FFTNetBlock]

    Example:
        >>> # Create FFTNet-Base foundation model
        >>> model = FFTNet.from_variant("base")
        >>>
        >>> # Use as feature extractor
        >>> images = keras.random.normal((4, 224, 224, 3))
        >>> outputs = model(images)
        >>> print(outputs['cls_token'].shape)  # (4, 768)
        >>> print(outputs['last_hidden_state'].shape)  # (4, 197, 768)
    """

    # Model variant configurations matching paper Table 2
    MODEL_VARIANTS = {
        "base": {
            "embed_dim": 768,
            "num_layers": 12,
            "mlp_hidden_dim": 256,
            "ffn_ratio": 4,
            "description": "FFTNet-Base: ~76M parameters, suitable for most applications"
        },
        "large": {
            "embed_dim": 1024,
            "num_layers": 24,
            "mlp_hidden_dim": 512,
            "ffn_ratio": 4,
            "description": "FFTNet-Large: ~268M parameters, high performance"
        },
        "huge": {
            "embed_dim": 1280,
            "num_layers": 32,
            "mlp_hidden_dim": 640,
            "ffn_ratio": 4,
            "description": "FFTNet-Huge: ~540M parameters, maximum capacity"
        },
        "small": {
            "embed_dim": 512,
            "num_layers": 6,
            "mlp_hidden_dim": 128,
            "ffn_ratio": 4,
            "description": "FFTNet-Small: Lightweight for resource-constrained environments"
        },
        "tiny": {
            "embed_dim": 384,
            "num_layers": 4,
            "mlp_hidden_dim": 96,
            "ffn_ratio": 4,
            "description": "FFTNet-Tiny: Ultra-lightweight for mobile/edge deployment"
        },
    }

    # Default architecture constants
    DEFAULT_IMAGE_SIZE = 224
    DEFAULT_PATCH_SIZE = 16
    DEFAULT_DROPOUT = 0.1

    def __init__(
            self,
            image_size: int = DEFAULT_IMAGE_SIZE,
            patch_size: int = 16,
            embed_dim: int = 768,
            num_layers: int = 12,
            mlp_hidden_dim: int = 256,
            ffn_ratio: int = 4,
            dropout_rate: float = DEFAULT_DROPOUT,
            ffn_type: str = 'mlp',
            normalization_type: str = 'layer_norm',
            use_bias_in_modrelu: bool = True,
            **kwargs: Any
    ) -> None:
        """Initialize the encoder and build its architecture.

        :param image_size: Input image size (square).
        :type image_size: int
        :param patch_size: Square patch edge length.
        :type patch_size: int
        :param embed_dim: Embedding dimension.
        :type embed_dim: int
        :param num_layers: Number of blocks.
        :type num_layers: int
        :param mlp_hidden_dim: Hidden width of each mixer's filter MLP.
        :type mlp_hidden_dim: int
        :param ffn_ratio: FFN expansion factor.
        :type ffn_ratio: int
        :param dropout_rate: Dropout probability.
        :type dropout_rate: float
        :param ffn_type: FFN identifier for the factory.
        :type ffn_type: str
        :param normalization_type: Normalization identifier for the factory.
        :type normalization_type: str
        :param use_bias_in_modrelu: Whether modReLU carries a learnable bias.
        :type use_bias_in_modrelu: bool
        :param kwargs: Additional keyword arguments for ``keras.Model``.
        :raises ValueError: If any configuration value is invalid.
        """
        super().__init__(**kwargs)

        # Validate configuration
        self._validate_config(
            image_size, patch_size, embed_dim, num_layers, dropout_rate
        )

        # Store configuration
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.mlp_hidden_dim = mlp_hidden_dim
        self.ffn_ratio = ffn_ratio
        self.dropout_rate = dropout_rate
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type
        self.use_bias_in_modrelu = use_bias_in_modrelu

        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2

        # Build architecture
        self._build_architecture()

        logger.info(
            f"Created FFTNet foundation model: {self.num_layers} layers, "
            f"embed_dim={self.embed_dim}, patches={self.num_patches}"
        )

    def _validate_config(
            self,
            image_size: int,
            patch_size: int,
            embed_dim: int,
            num_layers: int,
            dropout_rate: float
    ) -> None:
        """Validate model configuration parameters.

        :param image_size: Input image size.
        :type image_size: int
        :param patch_size: Square patch edge length.
        :type patch_size: int
        :param embed_dim: Embedding dimension.
        :type embed_dim: int
        :param num_layers: Number of blocks.
        :type num_layers: int
        :param dropout_rate: Dropout probability.
        :type dropout_rate: float
        :raises ValueError: If any value is non-positive, if ``image_size`` is
            not divisible by ``patch_size``, or if ``dropout_rate`` leaves
            ``[0, 1]``.
        """
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by "
                f"patch_size ({patch_size})"
            )
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be between 0 and 1, got {dropout_rate}"
            )

    def _build_architecture(self) -> None:
        """Create the patch embedding, the block stack and the final norm.

        The CLS token and the positional table are WEIGHTS and are therefore
        created in :meth:`build` instead.
        """
        # Patch embedding
        self.patch_embed = PatchEmbedding2D(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            name='patch_embed'
        )

        # CLS token and positional embeddings will be created in build()
        self.cls_token = None
        self.pos_embed = None

        # Dropout after embeddings
        self.pos_drop = keras.layers.Dropout(self.dropout_rate)

        # Stack of FFTNet blocks
        self.blocks = [
            FFTNetBlock(
                embed_dim=self.embed_dim,
                mlp_hidden_dim=self.mlp_hidden_dim,
                ffn_ratio=self.ffn_ratio,
                dropout_rate=self.dropout_rate,
                ffn_type=self.ffn_type,
                normalization_type=self.normalization_type,
                use_bias_in_modrelu=self.use_bias_in_modrelu,
                name=f'block_{i}'
            ) for i in range(self.num_layers)
        ]

        # Final normalization
        self.norm = create_normalization_layer(self.normalization_type, name='norm')

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the CLS token and positional table, and build every sub-layer.

        :param input_shape: Shape of the input images,
            ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # CLS token: (1, 1, embed_dim)
        self.cls_token = self.add_weight(
            name='cls_token',
            shape=(1, 1, self.embed_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True
        )

        # Positional embeddings: (1, num_patches + 1, embed_dim)
        self.pos_embed = self.add_weight(
            name='pos_embed',
            shape=(1, self.num_patches + 1, self.embed_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True
        )

        # Explicitly build sublayers in forward order so their weights
        # materialize on .keras reload (lazy first-call build leaves the
        # patch-embed / block / norm weights unloadable on deserialization).
        self.patch_embed.build(input_shape)
        seq_shape = (input_shape[0], self.num_patches + 1, self.embed_dim)
        self.pos_drop.build(seq_shape)
        for block in self.blocks:
            block.build(seq_shape)
        self.norm.build(seq_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass of the FFTNet foundation model.

        :param inputs: Input images of shape
            ``(batch_size, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the model is in training mode.
        :type training: Optional[bool]
        :return: A dictionary with all three of the following keys, always:

            - ``last_hidden_state``: the final layer's full sequence, shape
              ``(batch, num_patches + 1, embed_dim)``.
            - ``cls_token``: the CLS features, shape ``(batch, embed_dim)``.
            - ``patch_features``: the patch features excluding CLS, shape
              ``(batch, num_patches, embed_dim)``.

        :rtype: Dict[str, keras.KerasTensor]
        """
        batch_size = keras.ops.shape(inputs)[0]

        # 1. Patch embedding
        x = self.patch_embed(inputs)  # (B, N, D)

        # 2. Prepend class token
        cls_tokens = keras.ops.tile(self.cls_token, [batch_size, 1, 1])  # (B, 1, D)
        x = keras.ops.concatenate([cls_tokens, x], axis=1)  # (B, N+1, D)

        # 3. Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x, training=training)

        # 4. Apply FFTNet blocks
        for block in self.blocks:
            x = block(x, training=training)

        # 5. Final normalization
        x = self.norm(x)

        # 6. Extract features
        cls_token_output = x[:, 0]  # (B, D)
        patch_features = x[:, 1:]  # (B, N, D)

        return {
            "last_hidden_state": x,
            "cls_token": cls_token_output,
            "patch_features": patch_features
        }

    @classmethod
    def from_variant(cls, variant: str, **kwargs: Any) -> "FFTNet":
        """Create an FFTNet model from a predefined variant.

        :param variant: One of ``"base"``, ``"large"``, ``"huge"``,
            ``"small"``, ``"tiny"``.
        :type variant: str
        :param kwargs: Additional arguments overriding the variant's defaults.
        :type kwargs: Any
        :return: An FFTNet model instance.
        :rtype: FFTNet
        :raises ValueError: If the variant is not recognized.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        description = config.pop("description", "")

        logger.info(f"Creating FFTNet-{variant.upper()} model")
        logger.info(f"Configuration: {description}")

        # Override defaults with kwargs
        config.update(kwargs)

        return cls(**config)

    def get_config(self) -> Dict[str, Any]:
        """Return the model configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "ffn_ratio": self.ffn_ratio,
            "dropout_rate": self.dropout_rate,
            "ffn_type": self.ffn_type,
            "normalization_type": self.normalization_type,
            "use_bias_in_modrelu": self.use_bias_in_modrelu,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FFTNet":
        """Create a model instance from its configuration.

        :param config: Configuration dictionary.
        :type config: Dict[str, Any]
        :return: FFTNet model instance.
        :rtype: FFTNet
        """
        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print the model summary with additional FFTNet-specific information.

        :param kwargs: Additional arguments passed to ``keras.Model.summary``.
        """
        super().summary(**kwargs)
        logger.info("FFTNet Foundation Model Configuration:")
        logger.info(f"  - Architecture: {self.num_layers} layers, {self.embed_dim} hidden size")
        logger.info(f"  - Image size: {self.image_size}×{self.image_size}, patch size: {self.patch_size}")
        logger.info(f"  - Number of patches: {self.num_patches}")
        logger.info(f"  - FFT mixer MLP: {self.mlp_hidden_dim} hidden dim")
        logger.info(f"  - Feed-forward: {self.ffn_type}, ratio={self.ffn_ratio}")
        logger.info(f"  - Normalization: {self.normalization_type}")
        logger.info(f"  - Dropout: {self.dropout_rate}")


# ---------------------------------------------------------------------
# Integration with Vision Task Heads
# ---------------------------------------------------------------------

def create_fftnet_with_head(
        fftnet_variant: str,
        task_type: Literal["classification", "detection", "segmentation"] = "classification",
        num_classes: Optional[int] = None,
        image_size: int = 224,
        patch_size: int = 16,
        fftnet_config_overrides: Optional[Dict[str, Any]] = None,
        head_config_overrides: Optional[Dict[str, Any]] = None,
) -> keras.Model:
    """Factory function to create a complete FFTNet model with a task head.

    This function demonstrates the intended integration pattern:
    1. Instantiate a foundational :class:`FFTNet` model.
    2. Create a task-specific head.
    3. Combine them into a single, end-to-end ``keras.Model``.

    **Head integration:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  keras.Input [B, I, I, 3]  "images"  │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  FFTNet encoder (from_variant)       │
        │  → last_hidden_state / cls_token /   │
        │    patch_features                    │
        └───────────────┬──────────────────────┘
                        │  classification reads cls_token
                        ▼
        ┌──────────────────────────────────────┐
        │  [Dropout] → Dense(num_classes)      │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Output {"logits": [B, num_classes]} │
        │  a DICT, not a bare tensor           │
        └──────────────────────────────────────┘

        detection / segmentation raise NotImplementedError:
        build the encoder directly and attach your own head,
        reading patch_features for dense prediction.

    :param fftnet_variant: The FFTNet variant to use (e.g. ``"base"``,
        ``"large"``).
    :type fftnet_variant: str
    :param task_type: The vision task: ``"classification"``, ``"detection"`` or
        ``"segmentation"``. Only classification is implemented. Defaults to
        ``"classification"``.
    :type task_type: Literal["classification", "detection", "segmentation"]
    :param num_classes: Number of classes; REQUIRED for classification.
    :type num_classes: Optional[int]
    :param image_size: Input image size. Defaults to 224.
    :type image_size: int
    :param patch_size: Patch size. Defaults to 16.
    :type patch_size: int
    :param fftnet_config_overrides: Optional dictionary overriding the chosen
        variant's encoder configuration.
    :type fftnet_config_overrides: Optional[Dict[str, Any]]
    :param head_config_overrides: Optional dictionary overriding the head
        configuration; ``dropout_rate`` is the recognized key.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :return: A complete ``keras.Model`` whose output is
        ``{"logits": (batch, num_classes)}``.
    :rtype: keras.Model
    :raises ValueError: If ``num_classes`` is omitted for classification, or if
        ``task_type`` is unrecognized.
    :raises NotImplementedError: If ``task_type`` is ``"detection"`` or
        ``"segmentation"``.

    Example:
        >>> # Create classification model
        >>> model = create_fftnet_with_head(
        ...     fftnet_variant="base",
        ...     task_type="classification",
        ...     num_classes=1000
        ... )
        >>> model.summary()
        >>>
        >>> # Create with custom configuration
        >>> model = create_fftnet_with_head(
        ...     fftnet_variant="large",
        ...     task_type="classification",
        ...     num_classes=100,
        ...     fftnet_config_overrides={"dropout_rate": 0.2, "ffn_type": "swiglu"}
        ... )
    """
    fftnet_config_overrides = fftnet_config_overrides or {}
    head_config_overrides = head_config_overrides or {}

    logger.info(f"Creating FFTNet-{fftnet_variant} with '{task_type}' head.")

    # 1. Create the foundational FFTNet model
    fftnet_encoder = FFTNet.from_variant(
        fftnet_variant,
        image_size=image_size,
        patch_size=patch_size,
        **fftnet_config_overrides
    )

    # 2. Create the task head based on task type
    if task_type == "classification":
        if num_classes is None:
            raise ValueError("num_classes must be provided for classification tasks")

        # Simple classification head
        head_dropout_rate = head_config_overrides.get("dropout_rate", 0.0)
        classification_head = keras.Sequential([
            keras.layers.Dropout(head_dropout_rate) if head_dropout_rate > 0 else keras.layers.Lambda(lambda x: x),
            keras.layers.Dense(
                num_classes,
                kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                name="classifier"
            )
        ], name="classification_head")

        # 3. Build the end-to-end model
        inputs = keras.Input(
            shape=(image_size, image_size, 3),
            name="images"
        )

        # Get features from encoder
        encoder_outputs = fftnet_encoder(inputs)

        # Use CLS token for classification
        logits = classification_head(encoder_outputs["cls_token"])

        # Create the final model
        model = keras.Model(
            inputs=inputs,
            outputs={"logits": logits},
            name=f"fftnet_{fftnet_variant}_classifier"
        )

    elif task_type == "detection":
        raise NotImplementedError(
            "Object detection heads are not yet implemented. "
            "Use the foundation FFTNet model with your custom detection head."
        )

    elif task_type == "segmentation":
        raise NotImplementedError(
            "Segmentation heads are not yet implemented. "
            "Use the foundation FFTNet model with your custom segmentation head."
        )

    else:
        raise ValueError(
            f"Unknown task_type '{task_type}'. "
            f"Available: 'classification', 'detection', 'segmentation'"
        )

    logger.info(f"Successfully created model with {model.count_params():,} parameters.")
    return model


# ---------------------------------------------------------------------
# Convenience Functions for Backward Compatibility
# ---------------------------------------------------------------------

def create_fftnet(
        variant: Literal["base", "large", "huge", "small", "tiny"] = "base",
        image_size: int = 224,
        patch_size: int = 16,
        **kwargs: Any
) -> FFTNet:
    """Create an FFTNet foundation model from a preset configuration.

    :param variant: Model variant: ``'base'``, ``'large'``, ``'huge'``,
        ``'small'`` or ``'tiny'``.
    :type variant: Literal["base", "large", "huge", "small", "tiny"]
    :param image_size: Input image size. Defaults to 224.
    :type image_size: int
    :param patch_size: Patch size. Defaults to 16.
    :type patch_size: int
    :param kwargs: Additional keyword arguments overriding the preset.
    :type kwargs: Any
    :return: Configured FFTNet foundation model.
    :rtype: FFTNet
    :raises ValueError: If the variant is not recognized.

    Example:
        >>> # Create base foundation model
        >>> model = create_fftnet('base')
        >>>
        >>> # Create large model with custom settings
        >>> model = create_fftnet(
        ...     'large',
        ...     dropout_rate=0.2,
        ...     ffn_type='swiglu'
        ... )
    """
    return FFTNet.from_variant(
        variant,
        image_size=image_size,
        patch_size=patch_size,
        **kwargs
    )

# ---------------------------------------------------------------------

def create_fftnet_classifier(
        variant: Literal["base", "large", "huge", "small", "tiny"] = "base",
        num_classes: int = 1000,
        image_size: int = 224,
        patch_size: int = 16,
        **kwargs: Any
) -> keras.Model:
    """Convenience function to create an FFTNet classification model.

    Note that ``kwargs`` are forwarded as ENCODER overrides
    (``fftnet_config_overrides``), not head overrides.

    :param variant: Model variant.
    :type variant: Literal["base", "large", "huge", "small", "tiny"]
    :param num_classes: Number of output classes. Defaults to 1000.
    :type num_classes: int
    :param image_size: Input image size. Defaults to 224.
    :type image_size: int
    :param patch_size: Patch size. Defaults to 16.
    :type patch_size: int
    :param kwargs: Additional encoder configuration overrides.
    :type kwargs: Any
    :return: Complete classification model whose output is
        ``{"logits": (batch, num_classes)}``.
    :rtype: keras.Model

    Example:
        >>> # Create ImageNet classifier
        >>> model = create_fftnet_classifier('base', num_classes=1000)
        >>>
        >>> # Create CIFAR-10 classifier
        >>> model = create_fftnet_classifier(
        ...     'small',
        ...     num_classes=10,
        ...     image_size=32,
        ...     dropout_rate=0.3
        ... )
    """
    return create_fftnet_with_head(
        fftnet_variant=variant,
        task_type="classification",
        num_classes=num_classes,
        image_size=image_size,
        patch_size=patch_size,
        fftnet_config_overrides=kwargs
    )

# ---------------------------------------------------------------------