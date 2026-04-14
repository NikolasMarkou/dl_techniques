"""CliffordUNet DFlash-style diffusion draft model.

Combines the CliffordUNet backbone (hierarchical U-Net with Clifford-algebra
blocks) with the DFlash / DDTree speculative-decoding draft architecture
(Ringel et al., *Accelerating Speculative Decoding with Block Diffusion Draft
Trees*).

**How it differs from :class:`CliffordUNetLM`:**

* **Non-causal blocks.**  Uses :class:`CliffordNetBlock` (padding=same) instead
  of :class:`CausalCliffordNetBlock`.  A draft is a diffusion-style *denoiser*
  that predicts all masked positions in a block **in parallel**, so each block
  position must see the full block context.
* **Target hidden conditioning.**  Takes an auxiliary ``target_hidden`` input
  — concatenated hidden states pulled from a verifier model's intermediate
  layers — projected through ``Linear + RMSNorm`` and added to the noise
  embedding at the input (DFlash-style cross-feature injection).
* **Draft hidden size.**  Operates at the target model's hidden dimension
  (``target_hidden_size``) at the embedding and output projections so that
  the target's ``lm_head`` can directly decode the draft output to logits.
* **Block denoising interface.**  Accepts a ``noise_embedding`` tensor
  (the target's token embeddings with some positions replaced by the mask
  token) and returns the denoised hidden states for those positions.

Architecture:

.. code-block:: text

    noise_embedding [B, L, D_target]       target_hidden [B, L, K*D_target]
           │                                       │
           │                          target_fc (K*D_target → D_target)
           │                                 target_norm (RMSNorm)
           │                                       │
           └────────────── + ──────────────────────┘
                           │
                           ▼
                   input_proj (D_target → channels[0])
                           │
                           ▼     reshape (B, 1, L, channels[0])
             ┌────────────────────────────┐
             │ Encoder Stage 0 (D0)       │──► skip0
             └─────────────┬──────────────┘
                           │ CausalWindowPool k0 (+ projection to D1)
                           ▼
             ┌────────────────────────────┐
             │ Encoder Stage 1 (D1)       │──► skip1
             └─────────────┬──────────────┘
                           │ … pool-and-descend across all stages …
                           ▼
             ┌────────────────────────────┐
             │ Bottleneck (D_{n-1})       │
             └─────────────┬──────────────┘
                           │ MultiLinearUpsample
                           ▼
             ┌────────────────────────────┐
             │ + skip_{n-2}               │
             │ Decoder Stage (D_{n-2})    │
             └─────────────┬──────────────┘
                           │ … upsample + skip + decode …
             ┌────────────────────────────┐
             │ + skip0                    │
             │ Decoder Stage 0 (D0)      │
             └─────────────┬──────────────┘
                           │
                           ▼
                    squeeze, output_norm, output_proj
                           │
                           ▼
              hidden_states [B, L, D_target]
                           │
                           ▼
                target.lm_head → draft logits


**Target hidden conditioning is optional.**  If ``target_hidden`` is not
provided (or is ``None``), the model falls back to pure self-denoising on
the noise embedding alone.

**Trainable standalone.**  The block-denoising objective can be trained on
raw token streams without any verifier model: mask a random contiguous span
of tokens, embed the rest, and predict the masked span.  See
``train_cliffordunet_draft.py``.

References:
    Ringel, L., et al. (2026). Accelerating Speculative Decoding with
    Block Diffusion Draft Trees (DDTree / DFlash).

    Videau, M., et al. (2025). From Bytes to Ideas: Language Modeling
    with Autoregressive U-Nets.  arXiv:2506.14761.

    Brandstetter, J., et al. (2025). CliffordNet: All You Need is
    Geometric Algebra.  arXiv:2601.06793v2.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import keras
from keras import initializers, regularizers

from dl_techniques.utils.logger import logger
from dl_techniques.layers.geometric.clifford_block import (
    CliMode,
    CtxMode,
    CliffordNetBlock,
)
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.models.cliffordunet.lm import (
    CausalWindowPool,
    MultiLinearUpsample,
    _linear_drop_path_rates,
)

# ---------------------------------------------------------------------------

_DEFAULT_KERNEL_INIT = initializers.TruncatedNormal(stddev=0.02)


@keras.saving.register_keras_serializable(package="dl_techniques")
class CliffordUNetDraftModel(keras.Model):
    """CliffordUNet-based DFlash draft model for speculative decoding.

    :param target_hidden_size: Target verifier model hidden dimension ``D_target``.
        The draft operates at this dimension at its embedding and output
        projections so that the target's ``lm_head`` can decode the draft
        output directly.
    :param max_seq_length: Maximum sequence length (prefix + block) the draft
        will handle.  Used to size the positional embedding and to compute
        internal pooling padding.
    :param num_target_layers: Number of verifier hidden layers that will be
        concatenated into ``target_hidden``.  The input projection for the
        target conditioning is ``Linear(num_target_layers * target_hidden_size
        → target_hidden_size)``.  Set to ``0`` to disable cross-conditioning.
    :param channels: Per-stage channel dimensions of the CliffordUNet backbone.
    :param encoder_depths: Blocks per encoder (contracting) stage.
    :param decoder_depths: Blocks per decoder (expanding) stage
        (length = ``n_stages - 1``).
    :param pool_sizes: Pooling factors between adjacent stages
        (length = ``n_stages - 1``).
    :param shifts: Channel-shift offsets for sparse rolling geometric product.
    :param cli_mode: Algebraic components (``"inner"``, ``"wedge"``, ``"full"``).
    :param ctx_mode: Context calculation mode (``"diff"`` or ``"abs"``).
    :param use_global_context: Add global-average context branch inside blocks.
    :param layer_scale_init: Initial LayerScale gamma value.
    :param stochastic_depth_rate: Maximum DropPath rate (linear schedule).
    :param dropout_rate: Embedding and pre-output dropout rate.
    :param kernel_initializer: Kernel initializer.
    :param kernel_regularizer: Optional kernel regularizer.
    :param bias_regularizer: Optional bias regularizer.
    """

    LAYERNORM_EPSILON: float = 1e-6

    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        # ~25M trainable params (ex. target embeddings) — fast iteration
        "draft_nano": dict(
            channels=[256, 384, 512],
            encoder_depths=[2, 4, 2],
            decoder_depths=[2, 2],
            pool_sizes=[4, 4],
            shifts=[1, 2],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.05,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        # ~70M — balanced
        "draft_mini": dict(
            channels=[384, 512, 768],
            encoder_depths=[2, 4, 2],
            decoder_depths=[2, 2],
            pool_sizes=[4, 4],
            shifts=[1, 2, 4],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.1,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        # ~150M — stronger draft
        "draft_base": dict(
            channels=[512, 768, 1024, 1280],
            encoder_depths=[2, 4, 4, 2],
            decoder_depths=[2, 2, 2],
            pool_sizes=[4, 4, 2],
            shifts=[1, 2, 4, 8],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.15,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
    }

    def __init__(
        self,
        target_hidden_size: int,
        max_seq_length: int = 512,
        num_target_layers: int = 4,
        channels: Optional[List[int]] = None,
        encoder_depths: Optional[List[int]] = None,
        decoder_depths: Optional[List[int]] = None,
        pool_sizes: Optional[List[int]] = None,
        shifts: Optional[List[int]] = None,
        cli_mode: CliMode = "full",
        ctx_mode: CtxMode = "diff",
        use_global_context: bool = False,
        layer_scale_init: float = 1e-5,
        stochastic_depth_rate: float = 0.1,
        dropout_rate: float = 0.1,
        kernel_initializer: Any = "glorot_uniform",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Defaults
        channels = list(channels or [256, 384, 512])
        encoder_depths = list(encoder_depths or [2, 4, 2])
        decoder_depths = list(decoder_depths or [2, 2])
        pool_sizes = list(pool_sizes or [4, 4])
        shifts = list(shifts or [1, 2])

        n_stages = len(channels)
        if len(encoder_depths) != n_stages:
            raise ValueError(
                f"encoder_depths length ({len(encoder_depths)}) must match "
                f"channels length ({n_stages})"
            )
        if len(decoder_depths) != n_stages - 1:
            raise ValueError(
                f"decoder_depths length ({len(decoder_depths)}) must be "
                f"channels length - 1 ({n_stages - 1})"
            )
        if len(pool_sizes) != n_stages - 1:
            raise ValueError(
                f"pool_sizes length ({len(pool_sizes)}) must be "
                f"channels length - 1 ({n_stages - 1})"
            )
        if target_hidden_size <= 0:
            raise ValueError(
                f"target_hidden_size must be positive, got {target_hidden_size}"
            )
        if num_target_layers < 0:
            raise ValueError(
                f"num_target_layers must be non-negative, got {num_target_layers}"
            )

        # Store config for serialization
        self.target_hidden_size = target_hidden_size
        self.max_seq_length = max_seq_length
        self.num_target_layers = num_target_layers
        self.channels = channels
        self.encoder_depths = encoder_depths
        self.decoder_depths = decoder_depths
        self.pool_sizes = pool_sizes
        self.shifts = shifts
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.use_global_context = use_global_context
        self.layer_scale_init = layer_scale_init
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate
        self.kernel_initializer = (
            initializers.get(kernel_initializer)
            if kernel_initializer else None
        )
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self._n_stages = n_stages

        total_factor = math.prod(pool_sizes) if pool_sizes else 1
        self._total_pool_factor = total_factor
        self._internal_len = (
            ((max_seq_length + total_factor - 1) // total_factor)
            * total_factor
        )

        # --- Positional embedding at target hidden dim ---
        self.position_embedding = keras.layers.Embedding(
            self._internal_len,
            target_hidden_size,
            name="position_embedding",
        )

        # --- Target-hidden cross-feature projection (DFlash: fc + hidden_norm) ---
        if num_target_layers > 0:
            self.target_fc = keras.layers.Dense(
                target_hidden_size,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name="target_fc",
            )
            self.target_norm = RMSNorm(
                axis=-1, epsilon=1e-6, name="target_norm",
            )
        else:
            self.target_fc = None
            self.target_norm = None

        # --- Input norm + projection: D_target → channels[0] ---
        self.input_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON, name="input_norm",
        )
        self.input_dropout = keras.layers.Dropout(
            dropout_rate, name="input_dropout",
        )
        self.input_proj = keras.layers.Dense(
            channels[0],
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="input_proj",
        )

        # --- Stochastic depth schedule ---
        total_blocks = sum(encoder_depths) + sum(decoder_depths)
        drop_rates = _linear_drop_path_rates(total_blocks, stochastic_depth_rate)
        block_idx = 0

        _block_kw = dict(
            shifts=shifts,
            cli_mode=cli_mode,
            ctx_mode=ctx_mode,
            use_global_context=use_global_context,
            layer_scale_init=layer_scale_init,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )

        # --- Encoder (contracting) blocks — NON-causal ---
        self._encoder_blocks = []
        for i in range(n_stages):
            for j in range(encoder_depths[i]):
                blk = CliffordNetBlock(
                    channels=channels[i],
                    drop_path_rate=drop_rates[block_idx],
                    name=f"enc_s{i}_b{j}",
                    **_block_kw,
                )
                self._encoder_blocks.append(blk)
                block_idx += 1

        # --- Pool layers between encoder stages ---
        # Last-element-in-window pooling with a projection to the next-stage
        # channel count.  In a non-causal denoiser this is still a valid
        # downsampling scheme (equivalent to strided selection + a linear).
        self._pool_layers = [
            CausalWindowPool(
                d_out=channels[i + 1],
                pool_size=pool_sizes[i],
                kernel_initializer=kernel_initializer,
                name=f"pool_{i}",
            )
            for i in range(n_stages - 1)
        ]

        # --- Upsample layers (multi-linear, AU-Net style) ---
        self._upsample_layers = [
            MultiLinearUpsample(
                d_out=channels[i],
                factor=pool_sizes[i],
                kernel_initializer=kernel_initializer,
                name=f"upsample_{i}",
            )
            for i in range(n_stages - 1)
        ]

        # --- Decoder (expanding) blocks — NON-causal ---
        self._decoder_blocks = []
        for i in range(n_stages - 1):
            for j in range(decoder_depths[i]):
                blk = CliffordNetBlock(
                    channels=channels[i],
                    drop_path_rate=drop_rates[block_idx],
                    name=f"dec_s{i}_b{j}",
                    **_block_kw,
                )
                self._decoder_blocks.append(blk)
                block_idx += 1

        # --- Output head: channels[0] → target_hidden_size ---
        self.output_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON, name="output_norm",
        )
        self.output_dropout = (
            keras.layers.Dropout(dropout_rate, name="output_dropout")
            if dropout_rate > 0.0
            else None
        )
        self.output_proj = keras.layers.Dense(
            target_hidden_size,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="output_proj",
        )

        logger.info(
            f"Created CliffordUNetDraftModel "
            f"(D_target={target_hidden_size}, max_seq={max_seq_length}, "
            f"K_target_layers={num_target_layers}, stages={n_stages}, "
            f"channels={channels}, enc_depths={encoder_depths}, "
            f"dec_depths={decoder_depths}, pool_sizes={pool_sizes}, "
            f"shifts={shifts}, cli_mode={cli_mode}, "
            f"ctx_mode={ctx_mode}, internal_len={self._internal_len})"
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def call(
        self,
        inputs: Any,
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass.

        ``inputs`` may be either:

        * a single tensor ``noise_embedding`` of shape ``(B, L, D_target)``, or
        * a dict with keys ``"noise_embedding"`` and optionally
          ``"target_hidden"``.  ``target_hidden`` has shape
          ``(B, L, num_target_layers * D_target)``.

        :param inputs: Noise embedding (and optional target hidden).
        :param training: Whether in training mode.
        :return: Dict with ``"hidden_states"`` key:
            ``(B, L, D_target)``.  Feed this into the verifier's ``lm_head``
            to produce draft logits.
        """
        if isinstance(inputs, dict):
            noise_embedding = inputs["noise_embedding"]
            target_hidden = inputs.get("target_hidden", None)
        else:
            noise_embedding = inputs
            target_hidden = None

        seq_len = keras.ops.shape(noise_embedding)[1]

        # --- Positional embedding ---
        positions = keras.ops.arange(seq_len)
        pos_emb = self.position_embedding(positions)  # (L, D_target)

        # --- Target-hidden cross-feature injection (DFlash) ---
        x = noise_embedding + pos_emb
        if self.target_fc is not None and target_hidden is not None:
            t = self.target_fc(target_hidden)
            t = self.target_norm(t, training=training)
            x = x + t

        # --- Input norm + dropout + channel projection ---
        x = self.input_norm(x, training=training)
        x = self.input_dropout(x, training=training)
        x = self.input_proj(x)  # (B, L, channels[0])

        # Reshape to 4D for Clifford blocks
        x = keras.ops.expand_dims(x, axis=1)  # (B, 1, L, D)

        # Right-pad to internal length for pooling alignment.
        internal_len = self._internal_len
        pad_r = internal_len - seq_len
        x = keras.ops.pad(x, [[0, 0], [0, 0], [0, pad_r], [0, 0]])

        # --- Contracting path ---
        skips: List[keras.KerasTensor] = []
        enc_idx = 0
        for stage in range(self._n_stages):
            for _ in range(self.encoder_depths[stage]):
                x = self._encoder_blocks[enc_idx](x, training=training)
                enc_idx += 1
            skips.append(x)
            if stage < self._n_stages - 1:
                x = self._pool_layers[stage](x)

        # --- Expanding path ---
        for stage in reversed(range(self._n_stages - 1)):
            x = self._upsample_layers[stage](x)
            x = x + skips[stage]
            stage_start = sum(self.decoder_depths[:stage])
            for j in range(self.decoder_depths[stage]):
                x = self._decoder_blocks[stage_start + j](x, training=training)

        # Squeeze and crop to original length
        x = keras.ops.squeeze(x, axis=1)      # (B, internal_len, channels[0])
        x = x[:, :seq_len, :]                  # (B, L, channels[0])

        # --- Output head: project back to target hidden dim ---
        x = self.output_norm(x, training=training)
        if self.output_dropout is not None:
            x = self.output_dropout(x, training=training)
        hidden_states = self.output_proj(x)    # (B, L, D_target)

        return {"hidden_states": hidden_states}

    # ------------------------------------------------------------------
    # Shape / config / serialization
    # ------------------------------------------------------------------

    def compute_output_shape(
        self, input_shape: Any,
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        if isinstance(input_shape, dict):
            shape = input_shape["noise_embedding"]
        else:
            shape = input_shape
        return {"hidden_states": (shape[0], shape[1], self.target_hidden_size)}

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "target_hidden_size": self.target_hidden_size,
            "max_seq_length": self.max_seq_length,
            "num_target_layers": self.num_target_layers,
            "channels": self.channels,
            "encoder_depths": self.encoder_depths,
            "decoder_depths": self.decoder_depths,
            "pool_sizes": self.pool_sizes,
            "shifts": self.shifts,
            "cli_mode": self.cli_mode,
            "ctx_mode": self.ctx_mode,
            "use_global_context": self.use_global_context,
            "layer_scale_init": self.layer_scale_init,
            "stochastic_depth_rate": self.stochastic_depth_rate,
            "dropout_rate": self.dropout_rate,
            "kernel_initializer": (
                initializers.serialize(self.kernel_initializer)
                if self.kernel_initializer else None
            ),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CliffordUNetDraftModel":
        for key in ("kernel_regularizer", "bias_regularizer"):
            if config.get(key) and isinstance(config[key], dict):
                config[key] = regularizers.deserialize(config[key])
        return cls(**config)

    @classmethod
    def from_variant(
        cls,
        variant: str,
        target_hidden_size: int,
        max_seq_length: int = 512,
        num_target_layers: int = 4,
        **kwargs: Any,
    ) -> "CliffordUNetDraftModel":
        """Create a CliffordUNetDraftModel from a predefined variant.

        :param variant: One of ``"draft_nano"``, ``"draft_mini"``, ``"draft_base"``.
        :param target_hidden_size: Target verifier hidden dimension.
        :param max_seq_length: Maximum sequence length.
        :param num_target_layers: Number of verifier hidden layers in ``target_hidden``.
        :param kwargs: Override any default hyperparameter.
        :return: Configured :class:`CliffordUNetDraftModel` instance.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(cls.MODEL_VARIANTS.keys())}"
            )
        defaults = dict(cls.MODEL_VARIANTS[variant])
        defaults.update(kwargs)
        logger.info(f"Creating CliffordUNetDraftModel-{variant.upper()}")
        return cls(
            target_hidden_size=target_hidden_size,
            max_seq_length=max_seq_length,
            num_target_layers=num_target_layers,
            **defaults,
        )
