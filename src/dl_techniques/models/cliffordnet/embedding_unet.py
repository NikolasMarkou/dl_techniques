"""CliffordNet bidirectional U-Net embedding model (BERT-style packaging).

A non-causal, bidirectional sibling of :class:`CliffordNetLMUNet` packaged for
embedding / masked-language-model pretraining workflows. Mirrors the
hierarchical encoder/bottleneck/decoder structure but:

1. Uses **non-causal** :class:`CliffordNetBlock` /
   :class:`CliffordNetBlockDSv2` (bidirectional W-mixing).
2. Removes the causal upsample right-shift — plain
   ``UpSampling2D(nearest)`` is sufficient when no causality contract holds.
3. Drops the LM-head projection entirely.
4. Accepts BERT-style dict input ``{input_ids, attention_mask}`` and returns
   ``{last_hidden_state, pooled_output, attention_mask}`` — the contract
   consumed by :class:`MaskedLanguageModel`.
5. Exposes ``self.hidden_size`` (= ``base_channels``) for the
   :class:`MaskedLanguageModel` shape check (``mlm.py:174``).
6. Supports three pooling strategies: ``"mean"`` (mask-aware), ``"cls"``
   (first token), ``"max"`` (mask-aware via -inf sentinel).

Architecture::

    Input dict {input_ids: (B, T), attention_mask: (B, T)?}
         │
         ▼
    Token + Positional Embedding ─► LayerNorm ─► Dropout
         │
         ▼
    Reshape to (B, 1, T, base_channels)
         │
         ▼
    [right-pad along W to multiple of total_stride]
         │
         ▼
    Encoder / Bottleneck / Decoder (non-causal blocks)
    UpSampling2D(nearest) — NO causal right-shift
         │
         ▼
    [crop along W back to seq_len]
         │
         ▼
    Squeeze H ─► LayerNorm ─► Dropout ─► (sequence_output)
         │           │
         │           ├─► Pooling (mean | cls | max) ─► optional Pooler Dense (tanh)
         │           ▼
         │      pooled_output (B, hidden_size)
         ▼
    last_hidden_state (B, T, hidden_size)

The 5 variants (``nano/mini/base/large/xl``) reuse the same hyperparameter
ladder as :class:`CliffordNetLMUNet.MODEL_VARIANTS` for fair head-to-head
comparability with the causal LM-UNet.

References:
    Brandstetter, J., et al. (2025). CliffordNet: All You Need is Geometric
    Algebra. arXiv:2601.06793v2.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import keras
from keras import initializers, regularizers

from dl_techniques.layers.geometric.clifford_block import (
    CliMode,
    CtxMode,
    CliffordNetBlock,
    CliffordNetBlockDSv2,
)
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------------

_DEFAULT_KERNEL_INIT = initializers.TruncatedNormal(stddev=0.02)

PoolingStrategy = Literal["mean", "cls", "max"]


@keras.saving.register_keras_serializable()
class CliffordNetEmbedding(keras.Model):
    """CliffordNet bidirectional U-Net encoder for embedding / MLM workflows.

    Hierarchical encoder / bottleneck / decoder stack of non-causal Clifford
    blocks producing a per-token contextual embedding plus a pooled
    sequence-level representation. Channels grow with depth following
    ``channels_per_stage[i] = round(base_channels * (channel_multiplier ** i))``.
    The default ``channel_multiplier=1.5`` keeps the deepest stages tractable;
    pass ``2.0`` to recover a standard doubling U-Net.

    :param vocab_size: Vocabulary size (including special tokens).
    :param max_seq_length: Maximum sequence length for positional embeddings.
    :param base_channels: Channel count at level 0 (top of U). Equals
        the embedding dimensionality AND ``hidden_size``.
    :param channel_multiplier: Per-stage channel growth factor. Stage ``i``
        has ``round(base_channels * (channel_multiplier ** i))`` channels.
        Default ``1.5``; use ``2.0`` for classical U-Net doubling.
    :param stride_per_stage: List of strides applied between consecutive
        encoder levels. ``num_levels = len(stride_per_stage) + 1``. The
        product is the model's ``total_stride``.
    :param blocks_per_stage: List of length ``num_levels``: number of
        :class:`CliffordNetBlock` layers per encoder level (and the
        symmetrical decoder level).
    :param bottleneck_blocks: Number of :class:`CliffordNetBlock` layers
        at the deepest level.
    :param shifts: Channel-shift offsets for sparse rolling product. Must
        satisfy ``max(shifts) < base_channels`` (the smallest channel count).
    :param cli_mode: Algebraic components (``"inner"``, ``"wedge"``, ``"full"``).
    :param ctx_mode: Context calculation mode (``"diff"`` or ``"abs"``).
    :param use_global_context: Add global-context branch in blocks.
    :param layer_scale_init: Initial LayerScale gamma value.
    :param stochastic_depth_rate: Maximum DropPath rate (linear schedule
        across all blocks: encoder + bottleneck + decoder).
    :param dropout_rate: Embedding and pre-output dropout rate.
    :param pooling_strategy: How to derive ``pooled_output``. One of
        ``"mean"`` (mask-aware average), ``"cls"`` (first token), or
        ``"max"`` (mask-aware max via sentinel). Default ``"mean"``.
    :param pad_token_id: Padding token id (for symmetry with BERT — only
        persisted for downstream consumers; the encoder itself uses
        ``attention_mask`` directly).
    :param use_bias: Whether Dense/projection layers use bias.
    :param kernel_initializer: Kernel initializer for all dense layers.
    :param bias_initializer: Bias initializer for all dense layers.
    :param kernel_regularizer: Optional kernel regularizer.
    :param bias_regularizer: Optional bias regularizer.

    Example:
        .. code-block:: python

            model = CliffordNetEmbedding.from_variant("nano", vocab_size=100277)
            out = model({
                "input_ids": keras.ops.zeros((2, 128), dtype="int32"),
                "attention_mask": keras.ops.ones((2, 128), dtype="int32"),
            })
            print(out["last_hidden_state"].shape)  # (2, 128, 128)
            print(out["pooled_output"].shape)      # (2, 128)
    """

    LAYERNORM_EPSILON: float = 1e-6

    # Same ladder as CliffordNetLMUNet for head-to-head comparability with
    # the causal LM-UNet. See plan_2026-05-12_632605aa plan.md "Invariants".
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "nano": dict(
            base_channels=128,
            stride_per_stage=[2, 2],
            blocks_per_stage=[2, 2, 2],
            bottleneck_blocks=2,
            shifts=[1, 2],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.05,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "mini": dict(
            base_channels=192,
            stride_per_stage=[2, 2],
            blocks_per_stage=[2, 3, 3],
            bottleneck_blocks=2,
            shifts=[1, 2, 4],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.1,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "base": dict(
            base_channels=384,
            stride_per_stage=[2, 2, 2],
            blocks_per_stage=[3, 3, 3, 3],
            bottleneck_blocks=3,
            shifts=[1, 2, 4, 8, 16],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.15,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "large": dict(
            base_channels=512,
            stride_per_stage=[2, 2, 2],
            blocks_per_stage=[3, 4, 4, 4],
            bottleneck_blocks=4,
            shifts=[1, 2, 4, 8, 16],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.2,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "xl": dict(
            base_channels=768,
            stride_per_stage=[2, 2, 2],
            blocks_per_stage=[4, 4, 4, 4],
            bottleneck_blocks=4,
            shifts=[1, 2, 4, 8, 16],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.25,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
    }

    def __init__(
        self,
        vocab_size: int,
        max_seq_length: int = 512,
        base_channels: int = 64,
        channel_multiplier: float = 1.5,
        stride_per_stage: Optional[List[int]] = None,
        blocks_per_stage: Optional[List[int]] = None,
        bottleneck_blocks: int = 2,
        shifts: Optional[List[int]] = None,
        cli_mode: CliMode = "full",
        ctx_mode: CtxMode = "diff",
        use_global_context: bool = False,
        layer_scale_init: float = 1e-5,
        stochastic_depth_rate: float = 0.1,
        dropout_rate: float = 0.0,
        pooling_strategy: PoolingStrategy = "mean",
        pad_token_id: int = 0,
        use_bias: bool = True,
        kernel_initializer: Any = "glorot_uniform",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # --- Defaults / validation -------------------------------------
        if stride_per_stage is None:
            stride_per_stage = [2, 2]
        if blocks_per_stage is None:
            blocks_per_stage = [2, 2, 2]
        if shifts is None:
            shifts = [1, 2]

        num_levels = len(stride_per_stage) + 1
        if len(blocks_per_stage) != num_levels:
            raise ValueError(
                f"blocks_per_stage must have length len(stride_per_stage)+1 "
                f"(={num_levels}); got len(blocks_per_stage)="
                f"{len(blocks_per_stage)}"
            )
        if base_channels <= 0:
            raise ValueError(
                f"base_channels must be positive, got {base_channels!r}"
            )
        if channel_multiplier < 1.0:
            raise ValueError(
                f"channel_multiplier must be >= 1.0, got {channel_multiplier!r}"
            )
        if any(s < 1 for s in stride_per_stage):
            raise ValueError(
                f"stride_per_stage entries must be >= 1, got {stride_per_stage!r}"
            )
        if any(b < 1 for b in blocks_per_stage):
            raise ValueError(
                f"blocks_per_stage entries must be >= 1, got {blocks_per_stage!r}"
            )
        if bottleneck_blocks < 1:
            raise ValueError(
                f"bottleneck_blocks must be >= 1, got {bottleneck_blocks!r}"
            )
        if max(shifts) >= base_channels:
            raise ValueError(
                f"max(shifts)={max(shifts)} must be < base_channels="
                f"{base_channels}."
            )
        if pooling_strategy not in ("mean", "cls", "max"):
            raise ValueError(
                f"pooling_strategy must be one of 'mean'|'cls'|'max', "
                f"got {pooling_strategy!r}"
            )

        # --- Persisted hyperparameters ---------------------------------
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.base_channels = base_channels
        self.channel_multiplier = channel_multiplier
        self.stride_per_stage = list(stride_per_stage)
        self.blocks_per_stage = list(blocks_per_stage)
        self.bottleneck_blocks = bottleneck_blocks
        self.shifts = list(shifts)
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.use_global_context = use_global_context
        self.layer_scale_init = layer_scale_init
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate
        self.pooling_strategy = pooling_strategy
        self.pad_token_id = pad_token_id
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # --- Derived quantities ----------------------------------------
        self.num_levels = num_levels
        self.channels_per_stage: List[int] = [
            int(round(base_channels * (channel_multiplier ** i)))
            for i in range(num_levels)
        ]
        self.total_stride: int = 1
        for s in self.stride_per_stage:
            self.total_stride *= s

        # MaskedLanguageModel contract (mlm.py:174).
        self.hidden_size: int = base_channels

        # --- Embeddings ------------------------------------------------
        self.token_embedding = keras.layers.Embedding(
            vocab_size, base_channels, name="token_embedding",
        )
        self.position_embedding = keras.layers.Embedding(
            max_seq_length, base_channels, name="position_embedding",
        )
        self.embed_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON, name="embed_norm",
        )
        self.embed_dropout = keras.layers.Dropout(
            dropout_rate, name="embed_dropout",
        )

        # --- Encoder / bottleneck / decoder ----------------------------
        total_blocks = (
            sum(self.blocks_per_stage)
            + bottleneck_blocks
            + sum(self.blocks_per_stage[:-1])
        )
        drop_rates = linear_drop_path_rates(total_blocks, stochastic_depth_rate)
        dr_idx = 0

        _block_kw_common: Dict[str, Any] = dict(
            shifts=self.shifts,
            cli_mode=cli_mode,
            ctx_mode=ctx_mode,
            use_global_context=use_global_context,
            layer_scale_init=layer_scale_init,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )

        # Encoder: per-level blocks + downsampler (except the last level).
        self.encoder_blocks: List[List[CliffordNetBlock]] = []
        self.encoder_downsamplers: List[Optional[CliffordNetBlockDSv2]] = []
        for i in range(num_levels):
            level_channels = self.channels_per_stage[i]
            level_blocks: List[CliffordNetBlock] = []
            for j in range(self.blocks_per_stage[i]):
                level_blocks.append(
                    CliffordNetBlock(
                        channels=level_channels,
                        drop_path_rate=drop_rates[dr_idx],
                        name=f"enc_block_l{i}_b{j}",
                        **_block_kw_common,
                    )
                )
                dr_idx += 1
            self.encoder_blocks.append(level_blocks)

            if i < num_levels - 1:
                next_channels = self.channels_per_stage[i + 1]
                self.encoder_downsamplers.append(
                    CliffordNetBlockDSv2(
                        channels=level_channels,
                        out_channels=next_channels,
                        strides=self.stride_per_stage[i],
                        shifts=self.shifts,
                        cli_mode=cli_mode,
                        ctx_mode=ctx_mode,
                        use_global_context=use_global_context,
                        kernel_size=7,
                        # Pin avg/avg so behavior change vs causal version is
                        # exclusively non-causality (V2 default is "blur").
                        stream_pool="avg",
                        skip_pool="avg",
                        ctx_norm_type="bn",
                        ctx_activation="silu",
                        layer_scale_init=layer_scale_init,
                        drop_path_rate=0.0,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        name=f"enc_downsample_l{i}",
                    )
                )
            else:
                self.encoder_downsamplers.append(None)

        # Bottleneck.
        deepest_channels = self.channels_per_stage[-1]
        self.bottleneck_block_list: List[CliffordNetBlock] = []
        for k in range(bottleneck_blocks):
            self.bottleneck_block_list.append(
                CliffordNetBlock(
                    channels=deepest_channels,
                    drop_path_rate=drop_rates[dr_idx],
                    name=f"bottleneck_block_{k}",
                    **_block_kw_common,
                )
            )
            dr_idx += 1

        # Decoder: walk levels in reverse.
        self.decoder_upsamplers: List[Optional[keras.layers.UpSampling2D]] = []
        self.decoder_concats: List[Optional[keras.layers.Concatenate]] = []
        self.decoder_skip_projs: List[Optional[keras.layers.Conv2D]] = []
        self.decoder_blocks: List[List[CliffordNetBlock]] = []
        for i in range(num_levels):
            self.decoder_upsamplers.append(None)
            self.decoder_concats.append(None)
            self.decoder_skip_projs.append(None)
            self.decoder_blocks.append([])

        for i in reversed(range(num_levels - 1)):
            level_channels = self.channels_per_stage[i]
            self.decoder_upsamplers[i] = keras.layers.UpSampling2D(
                size=(1, self.stride_per_stage[i]),
                interpolation="nearest",
                name=f"dec_upsample_l{i}",
            )
            self.decoder_concats[i] = keras.layers.Concatenate(
                axis=-1, name=f"dec_concat_l{i}",
            )
            self.decoder_skip_projs[i] = keras.layers.Conv2D(
                filters=level_channels,
                kernel_size=1,
                padding="same",
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"dec_skip_proj_l{i}",
            )
            level_blocks_dec: List[CliffordNetBlock] = []
            for j in range(self.blocks_per_stage[i]):
                level_blocks_dec.append(
                    CliffordNetBlock(
                        channels=level_channels,
                        drop_path_rate=drop_rates[dr_idx],
                        name=f"dec_block_l{i}_b{j}",
                        **_block_kw_common,
                    )
                )
                dr_idx += 1
            self.decoder_blocks[i] = level_blocks_dec

        assert dr_idx == total_blocks, (
            f"drop-path index drift: dr_idx={dr_idx}, total_blocks={total_blocks}"
        )

        # --- Head ------------------------------------------------------
        self.head_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON, name="head_norm",
        )
        self.head_dropout = (
            keras.layers.Dropout(dropout_rate, name="head_dropout")
            if dropout_rate > 0.0
            else None
        )

        # BERT-style pooler dense (tanh) on the pooled feature. Skipped for
        # "max" pooling since max-pooled features are intentionally
        # un-projected (parallels e.g. SBERT max-strategy convention).
        # D-001 below records the trade-off.
        if pooling_strategy != "max":
            self.pooler_dense: Optional[keras.layers.Dense] = keras.layers.Dense(
                base_channels,
                activation="tanh",
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name="pooler",
            )
        else:
            self.pooler_dense = None

        logger.info(
            f"Created CliffordNetEmbedding (vocab_size={vocab_size}, "
            f"max_seq_length={max_seq_length}, base_channels={base_channels}, "
            f"channel_multiplier={channel_multiplier}, "
            f"channels_per_stage={self.channels_per_stage}, "
            f"stride_per_stage={self.stride_per_stage}, "
            f"blocks_per_stage={self.blocks_per_stage}, "
            f"bottleneck_blocks={bottleneck_blocks}, "
            f"total_stride={self.total_stride}, shifts={self.shifts}, "
            f"cli_mode={cli_mode}, ctx_mode={ctx_mode}, "
            f"pooling_strategy={pooling_strategy})"
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def call(
        self,
        inputs: Union[Dict[str, keras.KerasTensor], keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass.

        :param inputs: Either a raw token-id tensor of shape ``(B, T)`` or a
            dict with at least ``"input_ids"`` and optionally
            ``"attention_mask"`` (both ``(B, T)``).
        :param training: Training-mode flag.
        :return: Dict with keys ``"last_hidden_state"`` (``(B, T, hidden_size)``),
            ``"pooled_output"`` (``(B, hidden_size)``), and
            ``"attention_mask"`` (``(B, T)`` or ``None``).
        """
        ops = keras.ops

        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")
        else:
            input_ids = inputs
            attention_mask = None

        # --- Embeddings ------------------------------------------------
        seq_len = ops.shape(input_ids)[1]
        positions = ops.arange(0, seq_len, dtype="int32")
        tok = self.token_embedding(input_ids)              # (B, T, C0)
        pos = self.position_embedding(positions)           # (T, C0)
        x = tok + pos                                      # (B, T, C0)
        x = self.embed_norm(x)
        x = self.embed_dropout(x, training=training)

        # Reshape to (B, 1, T, C0) for the 4D blocks.
        x = ops.expand_dims(x, axis=1)

        # --- Right-pad along W to multiple of total_stride -------------
        total_stride = self.total_stride
        pad_w = (-seq_len) % total_stride
        x = ops.pad(x, [[0, 0], [0, 0], [0, pad_w], [0, 0]])

        # --- Encoder ---------------------------------------------------
        skip_features: List[Optional[keras.KerasTensor]] = [None] * self.num_levels
        for i in range(self.num_levels):
            for blk in self.encoder_blocks[i]:
                x = blk(x, training=training)
            skip_features[i] = x
            ds = self.encoder_downsamplers[i]
            if ds is not None:
                x = ds(x, training=training)

        # --- Bottleneck ------------------------------------------------
        for blk in self.bottleneck_block_list:
            x = blk(x, training=training)

        # --- Decoder ---------------------------------------------------
        # NOTE: no causal right-shift here — this model is bidirectional,
        # plain nearest-upsample is the correct semantics.
        for i in reversed(range(self.num_levels - 1)):
            up = self.decoder_upsamplers[i]
            x = up(x)
            skip = skip_features[i]
            x = self.decoder_concats[i]([x, skip])
            x = self.decoder_skip_projs[i](x)
            for blk in self.decoder_blocks[i]:
                x = blk(x, training=training)

        # --- Crop along W back to original seq_len --------------------
        x = x[:, :, :seq_len, :]

        # --- Squeeze H, head norm/dropout -----------------------------
        x = ops.squeeze(x, axis=1)             # (B, T, C0)
        x = self.head_norm(x)
        if self.head_dropout is not None:
            x = self.head_dropout(x, training=training)

        sequence_output = x  # (B, T, hidden_size)

        # --- Pooling ---------------------------------------------------
        pooled_output = self._pool(sequence_output, attention_mask)
        if self.pooler_dense is not None:
            pooled_output = self.pooler_dense(pooled_output)

        return {
            "last_hidden_state": sequence_output,
            "pooled_output": pooled_output,
            "attention_mask": attention_mask,
        }

    def _pool(
        self,
        sequence_output: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor],
    ) -> keras.KerasTensor:
        """Reduce ``(B, T, H)`` to ``(B, H)`` per ``self.pooling_strategy``."""
        ops = keras.ops
        if self.pooling_strategy == "cls":
            return sequence_output[:, 0, :]

        if self.pooling_strategy == "mean":
            if attention_mask is None:
                return ops.mean(sequence_output, axis=1)
            mask = ops.cast(attention_mask, sequence_output.dtype)
            mask = ops.expand_dims(mask, axis=-1)              # (B, T, 1)
            summed = ops.sum(sequence_output * mask, axis=1)   # (B, H)
            counts = ops.sum(mask, axis=1)                     # (B, 1)
            counts = ops.maximum(counts, 1.0)
            return summed / counts

        # "max"
        if attention_mask is None:
            return ops.max(sequence_output, axis=1)
        # Dtype-aware sentinel for masked-out positions (LESSONS).
        dtype = sequence_output.dtype
        sentinel = -1e9 if str(dtype) in ("float32", "float64") else -1e4
        mask = ops.cast(attention_mask, sequence_output.dtype)
        mask = ops.expand_dims(mask, axis=-1)                  # (B, T, 1)
        neg_inf = ops.cast(sentinel, sequence_output.dtype)
        masked = ops.where(mask > 0, sequence_output, neg_inf)
        return ops.max(masked, axis=1)

    def compute_output_shape(
        self,
        input_shape: Any,
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        # Accept either a raw `(B, T)` shape or a dict of shapes.
        if isinstance(input_shape, dict):
            shape = input_shape["input_ids"]
        else:
            shape = input_shape
        b, t = shape[0], shape[1]
        return {
            "last_hidden_state": (b, t, self.hidden_size),
            "pooled_output": (b, self.hidden_size),
            "attention_mask": (b, t),
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "max_seq_length": self.max_seq_length,
            "base_channels": self.base_channels,
            "channel_multiplier": self.channel_multiplier,
            "stride_per_stage": self.stride_per_stage,
            "blocks_per_stage": self.blocks_per_stage,
            "bottleneck_blocks": self.bottleneck_blocks,
            "shifts": self.shifts,
            "cli_mode": self.cli_mode,
            "ctx_mode": self.ctx_mode,
            "use_global_context": self.use_global_context,
            "layer_scale_init": self.layer_scale_init,
            "stochastic_depth_rate": self.stochastic_depth_rate,
            "dropout_rate": self.dropout_rate,
            "pooling_strategy": self.pooling_strategy,
            "pad_token_id": self.pad_token_id,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CliffordNetEmbedding":
        for key in ("kernel_regularizer", "bias_regularizer"):
            if config.get(key) and isinstance(config[key], dict):
                config[key] = regularizers.deserialize(config[key])
        return cls(**config)

    @staticmethod
    def _download_weights(
        variant: str,
        dataset: str = "default",
        cache_dir: Optional[str] = None,
    ) -> str:
        # DECISION plan_2026-05-12_632605aa/D-001: no public pretrained
        # weights are distributed for the CliffordNet embedding U-Net. We
        # explicitly raise NotImplementedError here (and use a narrow
        # try/except (IOError, OSError, ValueError) in `from_variant`) so
        # that "pretrained=True" can never silently produce a random-init
        # model. Mirrors plan_2026-05-07_9357982a/D-001 ghost.
        raise NotImplementedError(
            "No public pretrained weights are distributed for "
            f"CliffordNetEmbedding-{variant.upper()} ({dataset=}). "
            "Pass a local .keras path via pretrained=<path> or run "
            "src/train/cliffordnet/train_embeddings.py to pretrain."
        )

    @classmethod
    def from_variant(
        cls,
        variant: str,
        vocab_size: int,
        max_seq_length: int = 512,
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "default",
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> "CliffordNetEmbedding":
        """Create a CliffordNetEmbedding from a predefined variant.

        :param variant: One of ``"nano"``, ``"mini"``, ``"base"``, ``"large"``,
            ``"xl"``.
        :param vocab_size: Vocabulary size.
        :param max_seq_length: Maximum sequence length for positional embeddings.
        :param pretrained: If ``True`` attempt to download pretrained weights
            (currently always raises ``NotImplementedError`` — D-001). If a
            ``str``, treat as a local ``.keras`` path and load via
            ``model.load_weights``.
        :param weights_dataset: Dataset tag for pretrained weights.
        :param cache_dir: Local cache directory for pretrained weights.
        :param kwargs: Override any default hyperparameter from
            ``MODEL_VARIANTS[variant]``.
        :return: Configured :class:`CliffordNetEmbedding` instance.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(cls.MODEL_VARIANTS.keys())}"
            )
        defaults = dict(cls.MODEL_VARIANTS[variant])
        defaults.update(kwargs)
        logger.info(f"Creating CliffordNetEmbedding-{variant.upper()}")
        model = cls(
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            **defaults,
        )

        if pretrained is True:
            # DECISION plan_2026-05-12_632605aa/D-001: narrow exception scope
            # — only catch IO / config errors from the download path, never
            # swallow programming errors. Without these weights the only
            # honest behavior is to surface NotImplementedError.
            try:
                weights_path = cls._download_weights(
                    variant, dataset=weights_dataset, cache_dir=cache_dir,
                )
                model.load_weights(weights_path)
            except (IOError, OSError, ValueError) as exc:
                logger.error(
                    f"Failed to load pretrained weights for "
                    f"CliffordNetEmbedding-{variant.upper()}: {exc}"
                )
                raise
        elif isinstance(pretrained, str):
            logger.info(f"Loading local pretrained weights from {pretrained}")
            model.load_weights(pretrained)
        return model


# ---------------------------------------------------------------------------
# Module-level factories (mirror bert.create_bert API)
# ---------------------------------------------------------------------------


def create_cliffordnet_embedding(
    variant: str = "nano",
    vocab_size: Optional[int] = None,
    max_seq_length: int = 512,
    pretrained: Union[bool, str] = False,
    **kwargs: Any,
) -> CliffordNetEmbedding:
    """Factory for :class:`CliffordNetEmbedding`.

    :param variant: One of ``"nano"``, ``"mini"``, ``"base"``, ``"large"``, ``"xl"``.
    :param vocab_size: Vocabulary size — required.
    :param max_seq_length: Maximum sequence length.
    :param pretrained: ``True`` (raises NotImplementedError via D-001),
        a local path, or ``False``.
    :param kwargs: Override hyperparameters.
    """
    if vocab_size is None:
        raise ValueError("vocab_size is required.")
    return CliffordNetEmbedding.from_variant(
        variant=variant,
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        pretrained=pretrained,
        **kwargs,
    )


def create_cliffordnet_embedding_with_head(
    variant: str,
    task_config: Any,
    vocab_size: Optional[int] = None,
    max_seq_length: int = 512,
    pretrained: Union[bool, str] = False,
    encoder_config_overrides: Optional[Dict[str, Any]] = None,
    head_config_overrides: Optional[Dict[str, Any]] = None,
) -> keras.Model:
    """Build a :class:`CliffordNetEmbedding` encoder + NLP task head.

    Mirrors :func:`dl_techniques.models.bert.create_bert_with_head` in
    shape: returns a ``keras.Model`` whose inputs are the BERT-style dict
    and outputs are the NLP task head outputs.

    :param variant: Model size variant.
    :param task_config: ``NLPTaskConfig`` from ``dl_techniques.layers.heads.nlp``.
    :param vocab_size: Vocabulary size — required.
    :param max_seq_length: Maximum sequence length.
    :param pretrained: ``True`` / path / ``False``.
    :param encoder_config_overrides: Encoder hyperparameter overrides.
    :param head_config_overrides: Head hyperparameter overrides.
    :return: A ``keras.Model`` wrapping encoder + head.
    """
    from dl_techniques.layers.heads.nlp import create_nlp_head

    if vocab_size is None:
        raise ValueError("vocab_size is required.")
    encoder_kwargs = dict(encoder_config_overrides or {})
    encoder = CliffordNetEmbedding.from_variant(
        variant=variant,
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        pretrained=pretrained,
        **encoder_kwargs,
    )

    inputs = {
        "input_ids": keras.Input(
            shape=(max_seq_length,), dtype="int32", name="input_ids",
        ),
        "attention_mask": keras.Input(
            shape=(max_seq_length,), dtype="int32", name="attention_mask",
        ),
    }
    encoder_out = encoder(inputs)

    head_kwargs = dict(head_config_overrides or {})
    head = create_nlp_head(
        task_config=task_config,
        input_dim=encoder.hidden_size,
        **head_kwargs,
    )
    head_inputs = {
        "sequence_output": encoder_out["last_hidden_state"],
        "pooled_output": encoder_out["pooled_output"],
        "attention_mask": encoder_out["attention_mask"],
    }
    outputs = head(head_inputs)

    return keras.Model(inputs=inputs, outputs=outputs, name=f"cliffordnet_embedding_{variant}_with_head")
