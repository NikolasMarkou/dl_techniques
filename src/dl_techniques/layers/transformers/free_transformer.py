"""
Free Transformer Layer with integrated Variational Autoencoder components.

This module implements "The Free Transformer" architecture proposed by François Fleuret
(arXiv:2510.17558v1). It extends the standard decoder Transformer by conditioning its
generative process on random latent variables learned without supervision through a
variational procedure.

The Free Transformer addresses limitations of purely auto-regressive models by allowing
the model to make explicit latent decisions about the structure of the sequence being
generated, rather than implicitly inferring these decisions from already-generated tokens.

Architecture Overview:
---------------------
The Free Transformer splits a standard decoder into two halves and injects a latent
variable Z at the middle layer:

1. **First Half**: Standard causal transformer blocks (layers 0 to L/2-1)

2. **Injection Layer** (layer L/2):
   - During training: An encoder infers Z from the sequence
   - During inference: Z is sampled uniformly from a categorical distribution
   - Z is projected and added to the key/value inputs of this layer's attention

3. **Second Half**: Standard causal transformer blocks (layers L/2+1 to L-1)

Encoder Architecture:
--------------------
The encoder (only active during training or KV cache pre-filling) consists of:
- A learned constant query vector ζ (zeta) replicated across the sequence
- One non-causal transformer block receiving ζ as queries and the first-half
  output as keys/values
- A linear readout producing H bit logits per token
- A Binary Mapper that samples Z as a one-hot vector of dimension 2^H

Mathematical Foundation:
-----------------------
The model learns to maximize P(S) = ∫ P(S|Z=z)P(Z=z)dz through a VAE objective:

    Loss = CrossEntropy(S) + β * max(0, KL(Q(Z|S) || P(Z)) - κ)

where:
- Q(Z|S) is the encoder's posterior distribution
- P(Z) is a uniform prior over 2^H categories
- κ is a "free bits" threshold to prevent KL collapse
- β is a weighting coefficient

References:
-----------
- Fleuret, F. (2025). The Free Transformer. arXiv:2510.17558v1.
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv:1312.6114.
"""

import keras
import numpy as np
from keras import layers
from typing import Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from .transformer import TransformerLayer
from ..ffn.factory import create_ffn_layer, FFN_REGISTRY, FFNType
from ..norms import create_normalization_layer, NormalizationType
from ..attention.factory import create_attention_layer, AttentionType

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class BinaryMapper(keras.layers.Layer):
    """
    Samples one-hot vectors from bit logits with gradient pass-through.

    Converts ``H`` independent bit logits into a one-hot vector of
    dimension ``2^H`` via Bernoulli sampling, binary-to-integer
    conversion, and a straight-through gradient estimator. The gradient
    pass-through adds ``G - stop_gradient(G)`` to the one-hot output,
    providing gradients without altering forward values.

    **Cost warning -- the pass-through is O(B * T * 2^H).**
    Per eq. 8, ``G`` is a full ``(B, T, 2^H)`` tensor with one entry per
    category, so the training path materializes a second tensor the size of
    the one-hot output and performs a ``(B, T, H) x (H, 2^H)`` matmul against
    a constant ``(2^H, H)`` bit-pattern table built once in :meth:`build`.
    At the ``FreeTransformerLayer`` default ``num_latent_bits=16`` the table
    is ~4.2 MB (fp32) and the per-call matmul is ``B * T * 16 * 65536 * 2``
    FLOPs. MEASURED on an RTX 4070 (float32, ``tf.function``, training-path
    forward + backward, peak GPU memory via
    ``tf.config.experimental.get_memory_info``), scalar-broadcast -> this
    implementation:

    .. code-block:: text

        BinaryMapper alone, H=16
          (B,T)=(4,128)    2.6 ms ->   5.4 ms (2.1x)    135 MB ->  273 MB (2.0x)
          (B,T)=(8,256)    6.4 ms ->  19.4 ms (3.1x)    539 MB -> 1084 MB (2.0x)
          (B,T)=(16,512)  23.1 ms ->  77.1 ms (3.3x)   2152 MB -> 4301 MB (2.0x)
        FreeTransformerLayer end to end, hidden=256, num_latent_bits=16
          (B,T)=(4,128)    6.3 ms ->   9.2 ms (1.5x)    291 MB ->  563 MB (1.9x)
          (B,T)=(8,256)   17.2 ms ->  31.3 ms (1.8x)    896 MB -> 1984 MB (2.2x)

    Peak memory doubles because ``G`` is exactly the size of the one-hot
    output; note that the default was ALREADY ``2^H``-dominated before this
    change (the one-hot and ``post_sampler_fc`` both are), so this is a 2x on
    a cost that was large to begin with, not a new order of magnitude. Budget
    for it, or lower ``num_bits``: ``H=8`` costs 1/256th of the ``H=16``
    activation. The ``num_bits <= 20`` constructor cap remains the hard stop
    (a ``H=20`` table alone is ~84 MB).

    **Precision note.** ``G`` is a product of ``H`` probabilities, so its
    typical magnitude is ``2^-H``; at ``H=16`` that is ~1.5e-05, which is
    close to the float16 minimum normal (6.1e-05). Under a ``float16``
    compute policy the smaller entries flush to zero and their gradients
    vanish. This exposure is unchanged from the previous scalar
    implementation (the scalar was one entry of the same vector), but it is
    now spread across all ``2^H`` slots. Run this layer in ``float32`` or
    ``bfloat16`` if ``H`` is large.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Bit Logits (B, T, H)            │
        └────────────┬─────────────────────┘
                     ▼
        ┌──────────────────────────────────┐
        │  Sigmoid ─► Bernoulli sample     │
        │  ─► Binary-to-Integer            │
        │  ─► One-Hot (B, T, 2^H)          │
        └────────────┬─────────────────────┘
                     ▼
        ┌──────────────────────────────────┐
        │  [Training] Gradient pass-through│
        │  Z + G(Z) - stop_grad(G(Z))      │
        └────────────┬─────────────────────┘
                     ▼
        ┌──────────────────────────────────┐
        │  Output (B, T, 2^H)              │
        └──────────────────────────────────┘

    :param num_bits: Number of latent bits ``H``. Output has ``2^H`` dims.
    :type num_bits: int
    :param kwargs: Additional layer arguments.
    :type kwargs: Any
    """

    def __init__(
            self,
            num_bits: int,
            **kwargs: Any
    ):
        super().__init__(**kwargs)

        if not isinstance(num_bits, int) or num_bits <= 0:
            raise ValueError(
                f"num_bits must be a positive integer, got {num_bits}"
            )
        if num_bits > 20:
            raise ValueError(
                f"num_bits={num_bits} is too large (>20), would create 2^{num_bits} "
                f"= {2**num_bits} categories and consume excessive memory"
            )

        self.num_bits = num_bits
        self.num_categories = 2 ** num_bits
        self._bit_patterns = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Materialize the constant ``(2^H, H)`` bit-pattern table ONCE.

        ``U[d, h]`` is the ``h``-th bit of category index ``d``, 0-indexed to
        match :meth:`call`'s ``pow2 = [2**i]`` binary-to-integer convention.

        The table is kept as a HOST (numpy) array, deliberately neither a
        backend tensor nor a non-trainable weight:

        * A backend tensor built here is captured by whichever graph happens
          to be tracing at build time and then raises ``TypeError: ... is out
          of scope and cannot be used here`` in every OTHER graph. That is not
          hypothetical -- it is what ``test_graph_trace_training`` caught when
          this table was first written that way.
        * A non-trainable weight would be graph-safe, but it would change this
          layer's ``.keras`` checkpoint layout (and add ~4.2 MB of saved
          bytes at ``num_bits=16``) to store a value that is fully determined
          by ``num_bits`` and carries no state.

        :meth:`call` therefore converts this array to a tensor per call: a
        graph-mode trace folds it into one constant, and the eager path pays
        one host-to-device copy that the ``2^H``-wide matmul dwarfs.

        :param input_shape: Input shape ``(batch, seq, num_bits)``.
        """
        if self._bit_patterns is None:
            d = np.arange(self.num_categories, dtype=np.int64)[:, None]
            h = np.arange(self.num_bits, dtype=np.int64)[None, :]
            self._bit_patterns = ((d >> h) & 1).astype(self.compute_dtype)
        super().build(input_shape)

    def call(
            self,
            bit_logits: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass: sample one-hot vectors from bit logits.

        :param bit_logits: Logits tensor ``(B, T, num_bits)``.
        :type bit_logits: keras.KerasTensor
        :param training: Training flag; enables gradient pass-through.
            Evaluated with plain Python truthiness, so ``None`` and ``False``
            both disable the pass-through.

            .. warning::

               A genuinely **symbolic** (traced) ``training`` tensor is NOT
               supported and will RAISE. Inside a ``tf.function`` the branch
               below raises ``OperatorNotAllowedInGraphError`` ("Using a
               symbolic ``tf.Tensor`` as a Python ``bool`` is not allowed") --
               MEASURED, and deliberate: the alternative was the previous
               ``is True`` identity test, under which a tensor ``training``
               silently compared ``False`` and the eq. 8 pass-through was
               dropped with no error at all. ``fit``/``predict``/
               ``jit_compile=True`` all resolve ``training`` to a Python bool
               before tracing, so only a hand-written train step that threads a
               ``tf.Tensor`` into ``training=`` can reach this.
        :type training: Optional[bool]
        :return: One-hot tensor ``(B, T, 2^num_bits)``.
        :rtype: keras.KerasTensor
        """
        # Step 1: Compute bit probabilities
        # P(B_h = 1) = sigmoid(logit_h)
        probs = keras.ops.sigmoid(bit_logits)

        # Step 2: Sample bits using reparameterization trick
        # Sample uniform values and threshold by probability
        uniform_sample = keras.random.uniform(
            keras.ops.shape(probs),
            dtype=probs.dtype
        )
        sampled_bits = keras.ops.cast(
            uniform_sample < probs,
            dtype='int32'
        )

        # Step 3: Convert binary vector to integer index
        # index = Σ(B_h * 2^h) for h in [0, H-1]
        # Using einsum for efficient computation: (B, T, H) @ (H,) -> (B, T)
        pow2 = keras.ops.cast(keras.ops.array([2**i for i in range(self.num_bits)]), "int32")
        indices = keras.ops.einsum('bth,h->bt', sampled_bits, pow2)

        # Step 4: Create one-hot encoding
        # Shape: (batch_size, sequence_length, 2^H)
        z_one_hot = keras.ops.one_hot(
            indices,
            num_classes=self.num_categories,
            dtype=self.compute_dtype
        )

        # Step 5: Gradient pass-through (Equation 8 in paper)
        # DECISION plan-2026-07-31T132403-b3f540cb/D-020
        # Plain truthiness, NOT `if training is True:`. `tf.constant(True) is
        # True` is Python False, so the identity test made an EAGER tensor
        # `training` skip the pass-through silently. Do NOT restore it, and do
        # NOT try to rescue the symbolic case with `ops.where`/`tf.cond`: the
        # two branches here differ in the GRAPH they build, not just in their
        # values, so a blend would compute the whole (B,T,2^H) contraction on
        # every inference call. See D-020 and the `:param training:` warning.
        if training:
            # DECISION plan-2026-07-31T132403-b3f540cb/D-004
            # G_{t,d} = P(B_t = U(d)) for EVERY category d, per eq. 8 -- not
            # the single scalar of the SAMPLED category broadcast across all
            # 2^H slots. The scalar form (what this used to do) leaves the
            # forward value untouched, so nothing crashes, but it hands every
            # category an IDENTICAL gradient instead of routing one per
            # category through `post_sampler_fc`'s columns.
            #
            # Do NOT compute this by broadcasting bit_logits against the table
            # into a (B, T, 2^H, H) rank-4 intermediate and reducing over h:
            # at the shipped H=16 that is B*T*65536*16 elements (~50 MB for a
            # (2,6) probe). The contraction below is algebraically identical
            # and never leaves rank 3:
            #     log G_{t,d} = sum_h [U_dh log p_h + (1-U_dh) log(1-p_h)]
            #                 = (L_t @ U^T)[d] - sum_h softplus(L_th)
            # using log sigmoid(L) - log sigmoid(-L) == L exactly. VERIFIED
            # against a 60-digit mpmath brute-force oracle at H=3: <= 8.2e-15
            # relative (37 x float64 eps) over moderate, |L|<=30 and |L|=88
            # logits. See D-004; the two-matmul form
            # `U @ log p + (1-U) @ log(1-p)` is the documented fallback.
            logits = keras.ops.cast(bit_logits, self.compute_dtype)
            u_table = keras.ops.convert_to_tensor(self._bit_patterns)
            log_g = keras.ops.matmul(
                logits, keras.ops.transpose(u_table)
            ) - keras.ops.sum(
                keras.ops.softplus(logits), axis=-1, keepdims=True
            )
            g_td = keras.ops.exp(log_g)  # (B, T, 2^H)

            # DECISION plan-2026-07-31T132403-b3f540cb/D-013
            # Group the subtraction: `z + (G - sg(G))`, never `z + G - sg(G)`.
            # `G - sg(G)` is EXACTLY 0.0 in floating point, so the forward
            # value is bit-exact. The left-to-right form rounds `(1 + g) - g`
            # off the exact one-hot by 1 ulp for ~25 % of g values (MEASURED:
            # 1.11e-16 float64 / 5.96e-08 float32, hot slot only) -- a real,
            # pre-existing 1-ulp error in the old scalar code.
            z_one_hot = z_one_hot + (g_td - keras.ops.stop_gradient(g_td))

        return z_one_hot

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape: replace last dimension with 2^num_bits.

        :param input_shape: Input shape ``(batch, seq, num_bits)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape ``(batch, seq, 2^num_bits)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape[:-1] + (self.num_categories,)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'num_bits': self.num_bits
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class FreeTransformerLayer(TransformerLayer):
    """
    Transformer layer extended with the Free Transformer C-VAE architecture.

    When ``use_free_transformer=False``, behaves as a standard
    ``TransformerLayer``. When enabled, acts as the injection layer that
    conditions generation on a discrete latent variable ``Z`` sampled from
    ``2^H`` categories. During training an encoder infers ``Z`` from the
    sequence; during inference ``Z`` is sampled uniformly.

    The training loss is:
    ``L = CE(S) + beta * max(0, KL(Q(Z|S) || P(Z)) - kappa)``

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────────────────┐
        │  Input X (B, T, D)                         │
        └──────────────────┬─────────────────────────┘
                           ▼
        ┌────────────────────────────────────────────┐
        │  Causal Self-Attention + Residual          │
        │  ─► X_attn                                 │
        └──────────────────┬─────────────────────────┘
                           ▼
               ┌───────────┴───────────┐
               ▼                       ▼
        ┌────────────┐         ┌────────────┐
        │  Encoder   │         │  Uniform   │
        │ (training) │         │  Sample    │
        │  ─► Z      │         │  ─► Z      │
        └──────┬─────┘         └──────┬─────┘
               └───────────┬──────────┘
                           ▼
        ┌────────────────────────────────────────────┐
        │  R = Linear(Z)                             │
        │  Conditioned = X_attn + R                  │
        └──────────────────┬─────────────────────────┘
                           ▼
        ┌────────────────────────────────────────────┐
        │  FFN + Residual                            │
        └──────────────────┬─────────────────────────┘
                           ▼
        ┌────────────────────────────────────────────┐
        │  Output (B, T, D)                          │
        │  + bit_logits (training only)              │
        └────────────────────────────────────────────┘

    :param hidden_size: Hidden dimension of the layer.
    :type hidden_size: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param intermediate_size: FFN intermediate dimension.
    :type intermediate_size: int
    :param use_free_transformer: Enable the VAE mechanism. Default: False.
    :type use_free_transformer: bool
    :param num_latent_bits: Number of latent bits ``H`` (``2^H`` categories).
    :type num_latent_bits: int
    :param encoder_attention_type: Encoder attention type. Default: ``'multi_head'``.
    :type encoder_attention_type: AttentionType
    :param encoder_ffn_type: Encoder FFN type. Default: ``'swiglu'``.
    :type encoder_ffn_type: FFNType
    :param encoder_attention_args: Extra encoder attention arguments.
    :type encoder_attention_args: Optional[Dict[str, Any]]
    :param encoder_ffn_args: Extra encoder FFN arguments.
    :type encoder_ffn_args: Optional[Dict[str, Any]]
    :param encoder_normalization_type: Encoder normalization type.
    :type encoder_normalization_type: NormalizationType
    :param kwargs: All other arguments forwarded to ``TransformerLayer``.
    :type kwargs: Any
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            intermediate_size: int,
            use_free_transformer: bool = False,
            num_latent_bits: int = 16,
            encoder_attention_type: AttentionType = 'multi_head_cross',
            encoder_ffn_type: FFNType = 'swiglu',
            encoder_attention_args: Optional[Dict[str, Any]] = None,
            encoder_ffn_args: Optional[Dict[str, Any]] = None,
            encoder_normalization_type: NormalizationType = 'rms_norm',
            **kwargs: Any
    ):
        # Initialize base transformer layer
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            **kwargs
        )

        # Store Free Transformer configuration
        self.use_free_transformer = use_free_transformer
        self.num_latent_bits = num_latent_bits
        self.encoder_attention_type = encoder_attention_type
        self.encoder_ffn_type = encoder_ffn_type
        self.encoder_attention_args = encoder_attention_args or {}
        self.encoder_ffn_args = encoder_ffn_args or {}
        self.encoder_normalization_type = encoder_normalization_type

        # Only create encoder components if Free Transformer is enabled
        if not self.use_free_transformer:
            return

        # Validate configuration
        if num_latent_bits <= 0 or num_latent_bits > 20:
            raise ValueError(
                f"num_latent_bits must be between 1 and 20, got {num_latent_bits}"
            )

        self.num_latent_categories = 2 ** num_latent_bits

        # Zeta weight is created in build() via add_weight
        self.zeta = None

        # ---------------------------------------------------------------------
        # Encoder sublayers (created here; built in build())
        # ---------------------------------------------------------------------

        # The encoder is a cross-attention block (Q=zeta, K/V=sequence); it is
        # inherently non-causal, so no causal flag is needed.
        encoder_attn_args = (encoder_attention_args or {}).copy()

        self.encoder_attention = create_attention_layer(
            attention_type=self.encoder_attention_type,
            dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout_rate=self.attention_dropout_rate,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='encoder_attention',
            **encoder_attn_args
        )

        self.encoder_attention_norm = create_normalization_layer(
            normalization_type=self.encoder_normalization_type,
            name='encoder_attention_norm',
            **(self.attention_norm_args or {})
        )

        self.encoder_attention_dropout = layers.Dropout(
            rate=self.attention_dropout_rate,
            name='encoder_attention_dropout'
        )

        # DECISION plan-2026-07-30T081929-1645aa52/D-013
        # Pre-filter OUR OWN generic defaults to the ones this `encoder_ffn_type`
        # accepts, exactly as `gated_linear_attention_block.py` does at its own FFN
        # site. These are this layer's conveniences, not the caller's explicit
        # intent, so filtering them is correct -- and without it they reach
        # `create_ffn_layer`, which drops them silently.
        #
        # This was live at the encoder path's own default: `encoder_ffn_type`
        # defaults to 'swiglu', which does not accept `activation`, so the bundle
        # below handed swiglu an `activation` that was discarded without a word.
        # Measured across an instrumented 1634-test run, this was one of only two
        # (site, ffn_type) pairs in the repo that armed the drop. A strict factory
        # would break this site at 12 of 21 ffn_types.
        #
        # NOTE, as at the GLA site: this pre-filter deliberately does NOT cover the
        # caller's own `encoder_ffn_args`, which is merged AFTER it and reaches
        # `create_ffn_layer` verbatim -- that factory RAISES on a key the type does
        # not accept (D-023), which is exactly how a misspelled `encoder_ffn_args`
        # key becomes findable. Filtering it here would silently swallow the typo
        # again. The narrow `set(kwargs) - valid_param_names` predicate in
        # `layers/ffn/factory.py` is what tells an explicit caller key apart from
        # one of our defaults; the two are NOT indistinguishable.
        #
        # The dict-comprehension pre-filter below is a hand-rolled copy of
        # `layers/ffn/factory.py::assemble_ffn_config`, kept here (as at the GLA
        # site) because both already measure 0 dropped keys across all 21 registry
        # types, so migrating is pure de-duplication with no behaviour to fix.
        # Named as a follow-up at step 4, not an accepted permanent divergence.
        encoder_ffn_config = {
            'hidden_dim': self.intermediate_size,
            'output_dim': self.hidden_size,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
        }
        encoder_ffn_info = FFN_REGISTRY.get(self.encoder_ffn_type)
        if encoder_ffn_info is None:
            raise ValueError(
                f"Unknown encoder_ffn_type '{self.encoder_ffn_type}'. "
                f"Available: {sorted(FFN_REGISTRY)}."
            )
        valid_encoder_ffn_params = (
            set(encoder_ffn_info['required_params'])
            | set(encoder_ffn_info['optional_params'].keys())
        )
        encoder_ffn_config = {
            k: v for k, v in encoder_ffn_config.items()
            if k in valid_encoder_ffn_params
        }
        encoder_ffn_config.update(self.encoder_ffn_args)

        self.encoder_ffn = create_ffn_layer(
            ffn_type=self.encoder_ffn_type,
            name='encoder_ffn',
            **encoder_ffn_config
        )

        self.encoder_output_norm = create_normalization_layer(
            normalization_type=self.encoder_normalization_type,
            name='encoder_output_norm',
            **(self.ffn_norm_args or {})
        )

        self.encoder_ffn_dropout = layers.Dropout(
            rate=self.dropout_rate,
            name='encoder_ffn_dropout'
        )

        self.encoder_readout = layers.Dense(
            units=self.num_latent_bits,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='encoder_readout_fc',
            dtype=self.dtype
        )

        self.binary_mapper = BinaryMapper(
            num_bits=self.num_latent_bits,
            name='binary_mapper',
            dtype=self.dtype
        )

        self.post_sampler_fc = layers.Dense(
            units=self.hidden_size,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='post_sampler_fc',
            dtype=self.dtype
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build all sub-layers including encoder components if enabled.

        :param input_shape: Shape ``(batch, seq_len, hidden_size)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Build base transformer components
        if self.built:
            return

        super().build(input_shape)

        if not self.use_free_transformer:
            return

        # ---------------------------------------------------------------------
        # 1. Learned constant query vector ζ (zeta) for encoder
        # ---------------------------------------------------------------------
        # Shape: (1, 1, hidden_size) - will be tiled to (batch, seq_len, hidden_size)
        self.zeta = self.add_weight(
            name='zeta_query',
            shape=(1, 1, self.hidden_size),
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.dtype
        )

        # Build sublayers (created in __init__)
        self.encoder_attention.build(input_shape)
        self.encoder_attention_norm.build(input_shape)
        self.encoder_ffn.build(input_shape)
        self.encoder_output_norm.build(input_shape)
        self.encoder_readout.build(input_shape)
        # Deserialization hands `build()` a LIST, not a tuple, so the two
        # derived shapes below raised `TypeError: can only concatenate list
        # (not "tuple") to list` on every `.keras` load of a layer with
        # `use_free_transformer=True`. Normalize once.
        base_shape = tuple(input_shape)
        # Binary mapper input shape: (batch, seq, num_bits)
        self.binary_mapper.build(base_shape[:-1] + (self.num_latent_bits,))
        # Post-sampler input shape: (batch, seq, 2^H)
        self.post_sampler_fc.build(base_shape[:-1] + (self.num_latent_categories,))

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            layer_idx: int = 0,
            training: Optional[bool] = None
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """Forward pass of the Free Transformer layer.

        :param inputs: Input tensor ``(B, T, hidden_size)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional attention mask, a ``1 = attend`` keep
            predicate of rank 2 ``(B, S)``, rank 3 ``(B, T, S)`` or rank 4
            ``(B, heads, T, S)``. It is forwarded verbatim to the causal
            self-attention. The non-causal encoder cross-attention instead
            receives a **key-validity** reduction of it (``max`` over every
            query-side axis), so it honours padding without inheriting
            causality -- see the D-005 anchor in ``call``. Any other rank
            raises ``ValueError``.
        :type attention_mask: Optional[keras.KerasTensor]
        :param layer_idx: Layer index for differential attention.
        :type layer_idx: int
        :param training: Training mode flag. Selects the encoder path (``True``)
            or the uniform-sampling inference path (``None``/``False``), using
            plain Python truthiness.

            .. warning::

               A genuinely **symbolic** (traced) ``training`` tensor is NOT
               supported by this layer and will RAISE, rather than being
               handled. In graph mode it is refused TWICE, and the first
               refusal is not even this layer's: Keras' own ``Dropout.call``
               does ``if training and self.rate > 0`` and raises
               ``OperatorNotAllowedInGraphError`` at the attention-sub-block
               dropout above, before the encoder/inference branch is reached
               (MEASURED). The branch itself raises the same error for the same
               reason. This is deliberate -- the previous ``is True`` identity
               test made an EAGER tensor ``training=True`` run the INFERENCE
               path silently: no encoder sub-network, zero ``bit_logits``, no
               KL signal, and no error. ``fit``/``predict``/``jit_compile``
               resolve ``training`` to a Python bool before tracing, so only a
               hand-written train step can reach this.
        :type training: Optional[bool]
        :return: If ``use_free_transformer`` is False, the output ``(B, T, D)``.
            If True, ALWAYS the tuple ``(output, bit_logits)`` -- in both training
            and inference (the structure does not depend on ``training``). At
            inference ``bit_logits`` are zeros (the uniform prior). This matches
            ``compute_output_shape`` in every mode.
        :rtype: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]
        """
        # Standard transformer behavior when Free Transformer is disabled
        if not self.use_free_transformer:
            return super().call(
                inputs,
                attention_mask=attention_mask,
                layer_idx=layer_idx,
                training=training
            )

        # ---------------------------------------------------------------------
        # Free Transformer Forward Pass
        # ---------------------------------------------------------------------

        bit_logits = None  # Will be populated during training

        # DECISION plan-2026-07-31T042809-ddc92265/D-005
        # Derive the encoder's KEY-VALIDITY mask from the caller's mask.
        #
        # The encoder cross-attention (Q = learned zeta, K/V = the sequence S)
        # is DELIBERATELY non-causal -- that is the whole point of a posterior
        # Q(Z|S) that sees the entire sequence. So:
        #
        #   * do NOT keep passing ``attention_mask=None`` (the pre-fix
        #     behaviour): on a padded batch the posterior then pools over PAD
        #     keys and values, and because ``z_projected`` is added to
        #     ``attention_output`` BEFORE the FFN, PAD content reaches the
        #     layer output at every REAL position -- silently, finite, plausible.
        #   * do NOT forward the caller's rank-3/rank-4 mask VERBATIM either:
        #     that mask is typically causal, and the encoder would silently
        #     inherit causality it is designed not to have.
        #
        # What the encoder wants is only "which KEYS exist", so reduce every
        # query-side axis with ``max``: a key is valid iff SOME query may attend
        # to it. That maps a pure causal mask to all-ones (correct -- no key is
        # padding) and a causal+padding mask to the padding mask (correct). The
        # reduction is a union, so any genuinely per-query structure a caller
        # encoded in a rank-3 mask is intentionally discarded here.
        if attention_mask is None:
            encoder_key_mask = None
        else:
            mask_rank = len(attention_mask.shape)
            if mask_rank == 2:
                encoder_key_mask = attention_mask
            elif mask_rank in (3, 4):
                encoder_key_mask = keras.ops.max(
                    attention_mask, axis=tuple(range(1, mask_rank - 1))
                )
            else:
                raise ValueError(
                    f"attention_mask must have rank 2 (batch, keys), 3 "
                    f"(batch, queries, keys) or 4 (batch, heads, queries, keys) "
                    f"so the Free Transformer encoder can derive a key-validity "
                    f"mask from it; got rank {mask_rank} with shape "
                    f"{attention_mask.shape}."
                )

        # Step 1: Standard self-attention (first sub-layer)
        residual = inputs

        if self.normalization_position == 'pre':
            # Pre-norm: Normalize → Attention → Dropout → Add
            x = self.attention_norm(inputs, training=training)

            if self.attention_type == 'differential':
                attention_output = self.attention(
                    x,
                    attention_mask=attention_mask,
                    layer_idx=layer_idx,
                    training=training
                )
            else:
                attention_output = self.attention(
                    x,
                    attention_mask=attention_mask,
                    training=training
                )

            attention_output = self.dropout(attention_output, training=training)

            if self.attention_stochastic_depth is not None:
                attention_output = self.attention_stochastic_depth(
                    attention_output,
                    training=training
                )

            attention_output = attention_output + residual

        else:  # post-norm
            # Post-norm: Attention → Dropout → Add → Normalize
            if self.attention_type == 'differential':
                x = self.attention(
                    inputs,
                    attention_mask=attention_mask,
                    layer_idx=layer_idx,
                    training=training
                )
            else:
                x = self.attention(
                    inputs,
                    attention_mask=attention_mask,
                    training=training
                )

            x = self.dropout(x, training=training)

            if self.attention_stochastic_depth is not None:
                x = self.attention_stochastic_depth(x, training=training)

            attention_output = self.attention_norm(x + residual, training=training)

        # ---------------------------------------------------------------------
        # Step 2: Encoder path (training/prefill) or uniform sampling (inference)
        # ---------------------------------------------------------------------

        # DECISION plan-2026-07-31T132403-b3f540cb/D-020
        # Plain truthiness, NOT `if training is True:`. Do NOT restore the
        # identity test (an eager tensor `training=True` then silently runs the
        # inference path below -- no encoder, zero bit_logits, VAE never
        # trains), and do NOT replace this with `ops.where`/`tf.cond` to make a
        # symbolic `training` work: the two branches run structurally DIFFERENT
        # sub-networks (cross-attention + FFN + readout + sampling vs a bare
        # `keras.random.uniform`), so a blend pays for both on every call.
        # The symbolic case is documented as unsupported in `:param training:`.
        if training:
            # === Encoder Path (Training) ===
            # Run the non-causal encoder block to infer Z from the sequence

            # Tile learned query zeta to match sequence length
            # Shape: (1, 1, D) → (batch, seq_len, D)
            batch_size = keras.ops.shape(inputs)[0]
            seq_len = keras.ops.shape(inputs)[1]
            zeta_queries = keras.ops.tile(
                self.zeta,
                [batch_size, seq_len, 1]
            )

            # Encoder block: Pre-norm architecture (same as base layer)
            # Query: learned zeta, Keys/Values: attention_output
            encoder_residual = zeta_queries

            # Normalize queries
            zeta_norm = self.encoder_attention_norm(zeta_queries, training=training)

            # Cross-attention: Q = learned zeta queries, K/V = ``attention_output``
            # (the first-half sequence representation S). This makes the posterior
            # Q(Z|S) genuinely conditional on the sequence. ``encoder_attention`` is a
            # cross-attention layer ('multi_head_cross') taking ``kv_input`` separately.
            #
            # ``encoder_key_mask`` is the rank-2 key-validity predicate derived
            # above (see the D-005 anchor at the top of this branch): non-causal
            # over real tokens, PAD keys excluded. ``None`` when the caller
            # supplied no mask, which restores full attention exactly.
            encoder_attn_out = self.encoder_attention(
                zeta_norm,
                kv_input=attention_output,
                attention_mask=encoder_key_mask,
                training=training
            )
            encoder_attn_out = self.encoder_attention_dropout(
                encoder_attn_out,
                training=training
            )
            encoder_attn_out = encoder_attn_out + encoder_residual

            # Encoder FFN
            encoder_residual = encoder_attn_out
            encoder_x = self.encoder_output_norm(encoder_attn_out, training=training)
            encoder_x = self.encoder_ffn(encoder_x, training=training)
            encoder_x = self.encoder_ffn_dropout(encoder_x, training=training)
            encoder_output = encoder_x + encoder_residual

            # Readout: D → H bit logits
            bit_logits = self.encoder_readout(encoder_output, training=training)

            # Sample one-hot Z from bit logits with gradient pass-through
            z_one_hot = self.binary_mapper(bit_logits, training=training)

        else:
            # === Inference Path ===
            # Sample Z uniformly from categorical distribution
            batch_size = keras.ops.shape(inputs)[0]
            seq_len = keras.ops.shape(inputs)[1]

            # Sample random indices uniformly in [0, 2^H - 1]
            random_indices = keras.ops.cast(
                keras.random.uniform(
                    shape=(batch_size, seq_len),
                    minval=0,
                    maxval=self.num_latent_categories,
                    dtype='float32'
                ),
                dtype='int32'
            )

            # Convert to one-hot
            z_one_hot = keras.ops.one_hot(
                random_indices,
                num_classes=self.num_latent_categories,
                dtype=self.compute_dtype
            )

            # The uniform inference prior corresponds to per-bit Bernoulli(0.5),
            # i.e. zero bit-logits. Emitting it keeps the layer's output structure
            # training-independent (always (output, bit_logits) when the Free
            # Transformer is enabled), matching compute_output_shape.
            bit_logits = keras.ops.zeros(
                (batch_size, seq_len, self.num_latent_bits),
                dtype=self.compute_dtype
            )

        # ---------------------------------------------------------------------
        # Step 3: Condition on Z by injecting into keys/values
        # ---------------------------------------------------------------------
        # Project Z: 2^H → D
        z_projected = self.post_sampler_fc(z_one_hot, training=training)

        # Add to attention output to create conditioned representation
        # This creates the "R" tensor from the paper: R = Linear(Z)
        # The next operations will use attention_output + R as keys/values
        conditioned_kv = attention_output + z_projected

        # ---------------------------------------------------------------------
        # Step 4: FFN sub-layer (second sub-layer)
        # ---------------------------------------------------------------------
        # Note: According to Algorithm 2, the injection happens BEFORE the FFN.
        # The paper states: "block B/2 + 1 gets in_q=x, in_kv=x+r"
        # But since we're implementing this as a single layer, we interpret this
        # as: the attention already happened, now we just need to condition
        # the representation before FFN.
        #
        # For a more literal implementation, you would need to modify the
        # attention mechanism itself to accept separate query and key/value inputs.
        # For simplicity, we condition the representation directly.

        residual = conditioned_kv

        if self.normalization_position == 'pre':
            # Pre-norm: Normalize → FFN → Dropout → Add
            x = self.output_norm(conditioned_kv, training=training)
            x = self.ffn_layer(x, training=training)
            x = self.dropout(x, training=training)

            if self.ffn_stochastic_depth is not None:
                x = self.ffn_stochastic_depth(x, training=training)

            layer_output = x + residual

        else:  # post-norm
            # Post-norm: FFN → Dropout → Add → Normalize
            x = self.ffn_layer(conditioned_kv, training=training)
            x = self.dropout(x, training=training)

            if self.ffn_stochastic_depth is not None:
                x = self.ffn_stochastic_depth(x, training=training)

            layer_output = self.output_norm(x + residual, training=training)

        # ---------------------------------------------------------------------
        # Return (output, bit_logits)
        # ---------------------------------------------------------------------
        # The output STRUCTURE depends only on ``use_free_transformer`` (a
        # construction-time flag), never on ``training`` -- so it matches
        # ``compute_output_shape`` in both modes. During training ``bit_logits``
        # are the encoder readout (for the KL term); at inference they are the
        # uniform-prior zeros set above.
        return layer_output, bit_logits

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Union[Tuple[Optional[int], ...], Tuple[Tuple[Optional[int], ...], ...]]:
        """
        Compute output shape(s) of the layer.

        :param input_shape: Shape tuple (batch, sequence, hidden_size).
        :type input_shape: Tuple[Optional[int], ...]
        :return: Single shape tuple if use_free_transformer=False, or tuple of
            two shapes (output, bit_logits) if use_free_transformer=True. This
            mirrors ``call`` exactly: the free path always returns both outputs
            (at inference bit_logits is the uniform-prior zeros tensor).
        :rtype: Union[Tuple[Optional[int], ...], Tuple[Tuple[Optional[int], ...], ...]]
        """
        if not self.use_free_transformer:
            return input_shape

        # Return shapes for both outputs
        output_shape = input_shape
        bit_logits_shape = input_shape[:-1] + (self.num_latent_bits,)

        return output_shape, bit_logits_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Configuration dictionary with all parameters.
        """
        config = super().get_config()
        config.update({
            'use_free_transformer': self.use_free_transformer,
            'num_latent_bits': self.num_latent_bits,
            'encoder_attention_type': self.encoder_attention_type,
            'encoder_ffn_type': self.encoder_ffn_type,
            'encoder_attention_args': self.encoder_attention_args,
            'encoder_ffn_args': self.encoder_ffn_args,
            'encoder_normalization_type': self.encoder_normalization_type,
        })
        return config


# ---------------------------------------------------------------------
# Utility functions for computing VAE loss components
# ---------------------------------------------------------------------


def compute_kl_divergence_uniform_prior(
        bit_logits: keras.KerasTensor,
        num_bits: int,
        axis: int = -1
) -> keras.KerasTensor:
    """
    Compute KL divergence between encoder posterior Q(Z|S) and uniform prior P(Z).

    For the Free Transformer, the prior P(Z) is uniform over 2^H categories:
        P(Z_t = z) = 1 / 2^H  for all z ∈ {0, ..., 2^H - 1}

    The encoder outputs H independent bit logits, defining a factorized posterior:
        Q(Z_t = z | S) = ∏_{h=1}^H Q(B_h = U_h(z))
        where U(z) is the binary encoding of z.

    **Computation Flow**:
    ```
    Input: Bit Logits (B, T, H)
           ↓
    Sigmoid → p_h = σ(logit_h)
           ↓
    Clip probabilities: p_h ∈ [1e-7, 1-1e-7]
           ↓
    Binary Entropy: H(p_h) = -[p_h·log(p_h) + (1-p_h)·log(1-p_h)]
           ↓
    KL per bit: log(2) - H(p_h)
           ↓
    Sum over bits (axis=-1)
           ↓
    Output: KL Divergence (B, T)
    ```

    The KL divergence can be computed as (Equation 4 in paper):
        KL(Q(Z_t|S) || P(Z_t)) = H*log(2) + Σ_z Q(z|S) log Q(z|S)

    However, computing this sum over 2^H categories is expensive. Instead, we use
    the fact that for independent bits, the KL can be computed per-bit:
        KL = Σ_h KL(Q(B_h|S) || Uniform(B_h))

    For a Bernoulli with probability p:
        KL(Bernoulli(p) || Uniform) = H(1/2) - H(p)
                                     = log(2) - [p*log(p) + (1-p)*log(1-p)]

    :param bit_logits: Tensor of shape (batch, sequence, num_bits) containing
        logits for each independent bit.
    :type bit_logits: keras.KerasTensor
    :param num_bits: Number of bits H.
    :type num_bits: int
    :param axis: Axis along which to sum the KL (typically -1 for bits).
    :type axis: int
    :return: KL divergence tensor of shape (batch, sequence).
    :rtype: keras.KerasTensor
    """
    # Compute bit probabilities: p_h = sigmoid(logit_h)
    probs = keras.ops.sigmoid(bit_logits)

    # Clip probabilities for numerical stability
    probs = keras.ops.clip(probs, 1e-7, 1.0 - 1e-7)

    # Compute binary entropy: H(p) = -[p*log(p) + (1-p)*log(1-p)]
    entropy = -(
            probs * keras.ops.log(probs) +
            (1.0 - probs) * keras.ops.log(1.0 - probs)
    )

    # KL per bit: log(2) - H(p)
    log_2 = keras.ops.cast(keras.ops.log(2.0), bit_logits.dtype)
    kl_per_bit = log_2 - entropy

    # Sum over bits to get KL per token
    kl_divergence = keras.ops.sum(kl_per_bit, axis=axis)

    return kl_divergence


def compute_free_bits_kl_loss(
        bit_logits: keras.KerasTensor,
        num_bits: int,
        kappa: float = 0.5,
        reduction: str = 'mean'
) -> keras.KerasTensor:
    """
    Compute KL divergence loss with free bits thresholding.

    The free bits method (Kingma et al., 2016) prevents KL collapse by only
    penalizing KL divergence above a threshold κ (kappa):

        KL_loss = (1/T) * Σ_t max(0, KL(Q(Z_t|S) || P(Z_t)) - κ)

    where T is the sequence length.

    **Architecture Overview:**

    .. code-block:: text

        Input: Bit Logits (B, T, H)
               │
               ▼
        ┌─────────────────────────┐
        │  Per-Token KL → (B, T)  │
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │  max(0, KL_t - κ)       │
        │  (free bits threshold)  │
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │  Reduction (mean/sum)   │
        └────────────┬────────────┘
                     │
                     ▼
        Output: Scalar Loss

    :param bit_logits: Tensor of shape (batch, sequence, num_bits).
    :type bit_logits: keras.KerasTensor
    :param num_bits: Number of latent bits H.
    :type num_bits: int
    :param kappa: Free bits threshold in bits per token.
    :type kappa: float
    :param reduction: How to reduce the loss ('mean', 'sum', or 'none').
    :type reduction: str
    :return: Scalar loss tensor if reduction is 'mean' or 'sum',
        or tensor of shape (batch, sequence) if reduction is 'none'.
    :rtype: keras.KerasTensor
    """
    # Compute per-token KL divergence
    kl_per_token = compute_kl_divergence_uniform_prior(
        bit_logits,
        num_bits=num_bits,
        axis=-1
    )

    # Apply free bits threshold
    # Only penalize KL above threshold
    kl_above_threshold = keras.ops.maximum(
        0.0,
        kl_per_token - kappa
    )

    # Apply reduction
    if reduction == 'mean':
        return keras.ops.mean(kl_above_threshold)
    elif reduction == 'sum':
        return keras.ops.sum(kl_above_threshold)
    elif reduction == 'none':
        return kl_above_threshold
    else:
        raise ValueError(
            f"Invalid reduction '{reduction}'. "
            f"Expected 'mean', 'sum', or 'none'."
        )


# ---------------------------------------------------------------------