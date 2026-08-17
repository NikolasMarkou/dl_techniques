"""
TiRex-style probabilistic forecaster: patch tokens processed by a stack of blocks
that mix LSTM recurrence with self-attention, decoded in one shot into
non-crossing quantiles under reversible per-instance normalization.

The design answers three problems at once, and each answer costs something worth
knowing about.

The first is *scale*. A forecaster trained across many series sees windows whose
level and amplitude differ by orders of magnitude; a network fed those raw learns
the scale instead of the shape. Every window is therefore z-scored along its own
time axis before anything else touches it and the prediction is mapped back
afterwards, `y = q * std + mean`. The statistics are per-series and per-feature,
computed over `axis=1` — never over the batch — so the model becomes indifferent to
level and scale without any leakage between examples. Missing data is folded into
the same step: NaNs are located, replaced by zeros, and the mean/variance sums are
divided by the *count of valid steps* rather than the window length, so a gap
neither poisons the statistics nor propagates a NaN forward. The validity mask is
not then discarded — it is concatenated onto the feature axis, doubling it, so the
encoder can tell an imputed zero from an observed zero. This is why the patch
embedding is constructed at `embed_dim * 2` and immediately projected back down to
`embed_dim`: the doubled width carries the mask, and shape threading in `build`
must mirror that doubling or the restored weights will not fit.

The second is *sequence length*. Point-wise attention over a long lookback is
quadratic in the number of timesteps and spends most of its capacity on
neighbouring samples that carry nearly identical information. Segmenting the
series into patches of `patch_size` and embedding each as one token cuts the
sequence by that factor and gives each token a local waveform rather than a scalar
— the same trade PatchTST makes.

The third is *inductive bias*. Attention has no notion of order beyond what a
positional encoding supplies, while an LSTM has order built in but struggles to
reach across a long context. A `mixed` block runs both in series, pre-normalized
and residual at each stage: `x = x + LSTM(norm(x))`, then `x = x + Attn(norm(x))`,
then `x = x + FFN(norm(x))`. Attention therefore operates on tokens that already
carry recurrent state, so the ordering information it needs is inside the values
rather than added to them, and no positional embedding exists anywhere in this
model. Per-block `block_types` let the stack be tuned from purely recurrent to
purely attentional; the `lstm` and `transformer` variants are the same block with
one of the two sub-layers omitted.

Attention follows the published TiRex by default and the windowed divergence is now
an opt-in, chosen through `attention_type`. The default `'multi_head'` is the
factory's key for standard full self-attention — there is no key spelled `'global'`
— so every patch token attends to every other, at `O(L^2)` in the number of patches.
Passing `attention_type='window'` restores the earlier behaviour: each token then
attends only within `attention_window_size` tokens (default 8) at `O(L*w)`, and
long-range coupling falls back onto the LSTM path and the stacking of windows across
depth. The knob exists because the two answers differ in kind, not degree — a
windowed stack cannot form a single long-range association at any depth cheaply, and
a global stack pays quadratically for one it may not need — and because a model whose
attention span is fixed in the source cannot be compared against the paper it cites.
`attention_window_size` stays wired through `attention_args` under both settings, and as
of 2026-08-17 (plan-2026-08-17T183311-79c63e38/D-011) it is `MixedSequentialBlock`, not the
attention factory, that scopes it. `create_attention_layer` used to filter keyword
arguments against the target type's own parameter list and drop the rest rather than
raising; it is now STRICT and raises on any key the target type does not declare, which
this model's unconditional `attention_args={'window_size': ...}` would otherwise trip at
its own `'multi_head'` default. The block treats `window_size` as a documented-conditional
key and removes it on the branches whose attention type does not accept it, so both paths
now behave as this docstring has always described:
`'multi_head'` genuinely ignores the knob, and `'window'` genuinely uses it. That second
half was itself broken until the same commit — the block's `'window'` branch was injecting
a `normalization='softmax'` key `WindowAttention` has no parameter for, which the old
silent drop hid. Every OTHER `attention_args` key still reaches the factory verbatim, so a
misspelled one is now a loud `ValueError` instead of a silent no-op. The remaining block
internals are fixed rather than exposed: RMSNorm, GeGLU feed-forward and Mish
activations throughout.

Decoding is one-shot and pooled. After the final normalization the patch axis is
collapsed by a mean — the whole encoded history becomes a single `(B, 1, embed_dim)`
summary — and the head projects that summary directly to
`prediction_length * num_quantiles` values, reshaped to `(B, H, Q)`. There is no
autoregressive loop, so horizon cost is constant and no error compounds across
steps; the price is that the head cannot condition step `h` on step `h-1`, and that
mean-pooling discards *which* patch a pattern came from. Pooling with
`keepdims=True` is load-bearing rather than cosmetic: the head flattens its input,
which requires a statically known sequence length, and a length of exactly 1
supplies one. Quantile crossing is structurally prevented instead of penalized —
the head emits `r`, then `Q_0 = r_0` and `Q_i = Q_{i-1} + softplus(r_i)`, so the
outputs are non-decreasing by construction and no loss term or post-hoc sort is
needed. For multivariate input the de-normalization uses the statistics of the
**last** feature, which is the model's standing convention for which column is the
target.

`predict_quantiles` maps user-requested levels onto the levels the model was
actually trained with, falling back to the nearest trained level and logging a
warning rather than interpolating: an interpolated 0.95 from a model that only
learned 0.9 would look like a calibrated quantile while being nothing of the kind.
The median is extracted as the point forecast because it is the minimizer of
absolute error under the quantile loss. When `use_layer_norm=False` the output
normalization becomes `keras.layers.Identity` rather than a `Lambda` identity —
lambdas serialize as pickled Python and do not survive a portable `.keras`
round-trip.

References:
    - Auer et al., 2025. TiRex: Zero-Shot Forecasting Across Long and Short
      Horizons with Enhanced In-Context Learning.
      (https://arxiv.org/abs/2505.23719)
    - Nie et al., 2023. A Time Series is Worth 64 Words: Long-term Forecasting
      with Transformers. ICLR 2023. (https://arxiv.org/abs/2211.14730)
    - Kim et al., 2022. Reversible Instance Normalization for Accurate Time-Series
      Forecasting against Distribution Shift. ICLR 2022.
    - Beltagy et al., 2020. Longformer: The Long-Document Transformer.
      (https://arxiv.org/abs/2004.05150)
    - Wen et al., 2017. A Multi-Horizon Quantile Recurrent Forecaster.
      (https://arxiv.org/abs/1711.11053)
"""

import keras
import numpy as np
from keras import ops
from typing import Optional, Union, List, Any, Tuple, Dict, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.time_series.forecast import Forecast, ForecastMixin
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.ffn.residual_block import ResidualBlock
from dl_techniques.layers.embedding.patch_embedding import PatchEmbedding1D
from dl_techniques.layers.time_series.quantile_head_fixed_io import QuantileHead
from dl_techniques.layers.time_series.mixed_sequential_block import MixedSequentialBlock

# ---------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------

BlockType = Literal['lstm', 'transformer', 'mixed']

# Default quantile levels for probabilistic forecasting.
# Canonical source list; also exposed as the class attr `TiRexCore.DEFAULT_QUANTILES`
# (which references this list). Kept module-level for backward-compat: external
# modules (e.g. model_extended.py) import this name directly.
DEFAULT_QUANTILES: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TiRexCore(keras.Model, ForecastMixin):
    """
    TiRex Core Model for Time Series Forecasting.

    **Intent**: Provide a Keras-3-canonical, serializable hybrid LSTM/Transformer
    forecaster that emits monotonic quantile predictions with reversible
    per-instance normalization, configurable per-block (lstm/transformer/mixed),
    and ForecastMixin-wired inference.

    This model implements a TiRex-inspired architecture using mixed sequential blocks
    (LSTM + Transformer) for probabilistic time series forecasting. The model follows
    modern Keras 3 patterns and utilizes factory systems for component creation.

    The architecture consists of:
    1. Input scaling and preprocessing
    2. Patch embedding for time series tokenization
    3. Sequential processing blocks (configurable LSTM/Transformer mix)
    4. Quantile prediction head for probabilistic outputs

    Args:
        patch_size: Integer, size of input patches for tokenization.
        embed_dim: Integer, embedding dimension for all model components.
        num_blocks: Integer, number of mixed sequential blocks.
        num_heads: Integer, number of attention heads for transformer components.
        lstm_units: Integer, LSTM units per block. If None, uses embed_dim.
        ff_dim: Integer, feed-forward dimension. If None, uses embed_dim * 4.
        block_types: List of BlockType strings, type for each block ('lstm', 'transformer', 'mixed').
        quantile_levels: List of floats, quantile levels to predict.
        prediction_length: Integer, length of prediction horizon.
        dropout_rate: Float, dropout rate for regularization.
        use_layer_norm: Boolean, whether to use layer normalization.
        use_normalization: Boolean, whether to apply reversible per-instance
            normalization to the inputs.
        attention_window_size: Integer, window width in tokens, used only when
            `attention_type='window'`. Wired through `attention_args`
            unconditionally; `MixedSequentialBlock` drops it on the attention
            types that do not accept it (see the module docstring), so on every
            other setting it is genuinely inert rather than merely tolerated.
        attention_type: String, attention factory key used by every block.
            Defaults to `'multi_head'` — full/global self-attention, `O(L^2)` in the
            number of patch tokens, matching the published TiRex. `'window'` selects
            the local-window variant at `O(L*attention_window_size)`. Any other key
            from `layers/attention/factory.py`'s registry is accepted and validated
            there — note that the factory is now STRICT about parameters it does
            not declare, so a type whose constructor rejects `dim`/`num_heads`
            fails loudly at construction instead of silently.
        **kwargs: Additional keyword arguments for the Model base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, features)`.
        Can also accept 2D tensor which will be expanded to 3D.

    Output shape:
        3D tensor with shape: `(batch_size, prediction_length, num_quantiles)`.

    Example:
        ```python
        # Create TiRex model for time series forecasting
        model = TiRexCore(
            patch_size=16,
            embed_dim=256,
            num_blocks=6,
            prediction_length=32,
            quantile_levels=[0.1, 0.5, 0.9]
        )

        # Mixed block types for different processing stages
        model = TiRexCore(
            patch_size=8,
            embed_dim=128,
            num_blocks=4,
            block_types=['lstm', 'mixed', 'transformer', 'mixed'],
            prediction_length=24
        )
        ```
    """

    # Default quantile levels for probabilistic forecasting (class-level attr).
    # References the single module-level source list (defined above the class)
    # so the value lives in exactly one place; model_extended.py imports the
    # module-level name, which remains a backward-compat alias for the same list.
    DEFAULT_QUANTILES: List[float] = DEFAULT_QUANTILES

    # Model variant configurations following ConvNeXt V2 pattern
    MODEL_VARIANTS = {
        "tiny": {
            "patch_size": 8,
            "embed_dim": 64,
            "num_blocks": 3,
            "num_heads": 4,
            "dropout_rate": 0.1
        },
        "small": {
            "patch_size": 12,
            "embed_dim": 128,
            "num_blocks": 6,
            "num_heads": 8,
            "dropout_rate": 0.1
        },
        "medium": {
            "patch_size": 16,
            "embed_dim": 256,
            "num_blocks": 8,
            "num_heads": 8,
            "dropout_rate": 0.1
        },
        "large": {
            "patch_size": 16,
            "embed_dim": 512,
            "num_blocks": 12,
            "num_heads": 16,
            "dropout_rate": 0.15
        }
    }

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 256,
        num_blocks: int = 6,
        num_heads: int = 8,
        lstm_units: Optional[int] = None,
        ff_dim: Optional[int] = None,
        block_types: Optional[List[BlockType]] = None,
        quantile_levels: List[float] = DEFAULT_QUANTILES,
        prediction_length: int = 32,
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True,
        use_normalization: bool = True,
        attention_window_size: int = 8,
        attention_type: str = 'multi_head',
        name: str = "TiRex",
        **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

        # Validate inputs
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if prediction_length <= 0:
            raise ValueError(f"prediction_length must be positive, got {prediction_length}")
        if attention_window_size <= 0:
            raise ValueError(f"attention_window_size must be positive, got {attention_window_size}")

        # Store configuration
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.lstm_units = lstm_units if lstm_units is not None else embed_dim
        self.ff_dim = ff_dim if ff_dim is not None else embed_dim * 4
        self.block_types = block_types if block_types is not None else ['mixed'] * num_blocks
        self.quantile_levels = quantile_levels
        self.prediction_length = prediction_length
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.use_normalization = use_normalization
        self.attention_window_size = attention_window_size
        # DECISION plan-2026-08-14T183218-f4c612aa/D-008
        # `attention_type` deliberately carries NO membership check here, unlike
        # every other argument above. `create_attention_layer` already raises
        # `ValueError: Unknown attention type '<value>'. Available types: [...]`
        # for an unregistered key, and it does so eagerly — the blocks below are
        # constructed in `__init__`, not lazily in `build`. A local whitelist would
        # either duplicate that raise or, worse, freeze this model to the two keys
        # anyone happened to test, locking out the other 29 registry entries for no
        # reason. Do NOT "fix" this by adding `if attention_type not in
        # ('multi_head', 'window')`. The one gap is an all-`'lstm'` `block_types`
        # stack, which builds no attention layer at all and so cannot validate the
        # key; there it is inert and round-trips unused.
        self.attention_type = attention_type

        if len(self.block_types) != num_blocks:
            raise ValueError(
                f"Length of block_types ({len(self.block_types)}) must match num_blocks ({num_blocks})"
            )

        # CREATE all sub-layers in __init__ (modern Keras 3 pattern)
        self.patch_embedding = PatchEmbedding1D(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim * 2,  # Include mask information
            name="patch_embedding"
        )

        self.input_projection = ResidualBlock(
            hidden_dim=self.embed_dim * 2,
            output_dim=self.embed_dim,
            dropout_rate=self.dropout_rate,
            activation="mish",
            name="input_projection"
        )

        # Create sequential processing blocks
        self.blocks = []
        for i, block_type in enumerate(self.block_types):
            # --- DIVERGENCE FROM TIREX: WINDOW ATTENTION INSTEAD OF GLOBAL ATTENTION ---
            # Kept as history, no longer as behaviour: this line hardcoded
            # `attention_type='window'`, so no caller could build the paper's global
            # attention. The divergence is now OPT-IN via the constructor argument
            # and the default is the paper's `'multi_head'` (the factory's key for
            # full self-attention; there is no key spelled `'global'`). Existing
            # windowed behaviour is one keyword away, and `window_size` stays wired
            # unconditionally.
            #
            # DECISION plan-2026-08-17T183311-79c63e38/D-011
            # That last clause used to be justified by the attention factory
            # filtering unknown kwargs against the target type's parameter list.
            # It no longer does: `create_attention_layer` RAISES on any key the
            # type does not declare. `MixedSequentialBlock` is now what scopes
            # `window_size` (its `_CONDITIONAL_ATTENTION_ARG_KEYS` allowlist).
            # Do NOT "fix" this by making the line below conditional on
            # `self.attention_type` -- that pushes registry knowledge into every
            # block consumer, which is exactly what the block-side repair
            # exists to avoid.
            block = MixedSequentialBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                lstm_units=self.lstm_units,
                ff_dim=self.ff_dim,
                block_type=block_type,
                dropout_rate=self.dropout_rate,
                use_layer_norm=self.use_layer_norm,
                normalization_type='rms_norm',
                attention_type=self.attention_type,
                ffn_type='geglu',
                activation='mish',
                attention_args={'window_size': self.attention_window_size},
                name=f"block_{i}"
            )
            # ---------------------------------------------
            self.blocks.append(block)

        # Output normalization using factory
        if self.use_layer_norm:
            self.output_norm = (
                create_normalization_layer(
                    normalization_type='rms_norm',
                    name="output_norm"
                )
            )
        else:
            # DEFECT #3 fix: keras.layers.Identity is a serializable Keras-3
            # drop-in for the old Lambda(lambda x: x), which serialized a
            # Python lambda (fragile / non-portable). Identity has build +
            # compute_output_shape and accepts the training kwarg.
            self.output_norm = keras.layers.Identity(name="output_norm")

        # Quantile prediction head
        self.quantile_head = QuantileHead(
            num_quantiles=len(self.quantile_levels),
            output_length=self.prediction_length,
            # Hardcode a safe low value, or dividing the global rate
            dropout_rate=min(self.dropout_rate, 0.1),
            enforce_monotonicity=True,
            use_bias=True,
            flatten_input=True,
            name="quantile_head"
        )

        logger.info(
            f"TiRex model initialized: {num_blocks} blocks, "
            f"embed_dim={embed_dim}, prediction_length={prediction_length}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build all sub-layers explicitly with threaded shapes.

        Explicit per-sublayer builds are REQUIRED (not optional): on
        ``.keras`` load, Keras replays the captured build config and restores
        weights BEFORE the first ``call``. If sub-layers are left unbuilt at
        restore time, the restored weights have nowhere to land and the first
        forward pass lazily re-initializes them, silently discarding the saved
        values. (See plan D-002 — the same failure mode bit DeepAR.)

        Shape threading mirrors ``call``: the raw input ``(B, T, F)`` is
        concatenated with its NaN-mask (doubling the feature axis to ``2F``)
        before patch embedding, then projected, processed through the blocks,
        mean-pooled over time, and projected to quantiles.

        Args:
            input_shape: Raw input shape ``(batch, seq_len, features)``. A 2D
                ``(batch, seq_len)`` shape is treated as ``(batch, seq_len, 1)``
                to match ``call``'s expand-dims path.
        """
        # Normalize a 2D input shape to 3D (mirrors call's expand_dims).
        if len(input_shape) == 2:
            input_shape = (input_shape[0], input_shape[1], 1)
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected 3D input (batch, seq_len, features), got "
                f"{len(input_shape)}D input with shape {input_shape}"
            )

        batch_size, seq_len, features = input_shape[0], input_shape[1], input_shape[2]

        # call() concatenates the NaN mask onto the feature axis -> 2 * features.
        masked_features = None if features is None else features * 2
        patch_input_shape = (batch_size, seq_len, masked_features)

        # 1. Patch embedding: (B, T, 2F) -> (B, num_patches, 2*embed_dim)
        self.patch_embedding.build(patch_input_shape)
        embedded_shape = self.patch_embedding.compute_output_shape(patch_input_shape)

        # 2. Input projection (ResidualBlock): -> (B, num_patches, embed_dim)
        self.input_projection.build(embedded_shape)
        projected_shape = self.input_projection.compute_output_shape(embedded_shape)

        # 3. Mixed sequential blocks (shape-preserving)
        current_shape = projected_shape
        for block in self.blocks:
            block.build(current_shape)
            current_shape = block.compute_output_shape(current_shape)

        # 4. Output normalization (rms_norm or Identity; shape-preserving)
        self.output_norm.build(current_shape)

        # 5. Quantile head: input is mean-pooled over time -> (B, 1, embed_dim)
        pooled_shape = (current_shape[0], 1, current_shape[2])
        self.quantile_head.build(pooled_shape)

        super().build(input_shape)

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape: ``(batch, prediction_length, num_quantiles)``.

        Matches the rank-3 ``[B, H, Q]`` quantile output of ``call``.

        Args:
            input_shape: Raw input shape ``(batch, seq_len, features)``.

        Returns:
            Output shape ``(batch, prediction_length, len(quantile_levels))``.
        """
        batch_size = input_shape[0]
        return (batch_size, self.prediction_length, len(self.quantile_levels))

    def call(
            self,
            inputs: Union[keras.KerasTensor, np.ndarray],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the TiRex model.

        Args:
            inputs: Input tensor of shape [batch_size, sequence_length, features] or
                   [batch_size, sequence_length] which will be expanded.
            training: Boolean, whether in training mode.

        Returns:
            Quantile predictions of shape [batch_size, prediction_length, num_quantiles].
        """
        # Ensure 3D input
        if len(inputs.shape) == 2:
            inputs = ops.expand_dims(inputs, axis=-1)

        # 1. HANDLE MASKING (before normalization to avoid NaN propagation)
        nan_mask = ops.logical_not(ops.isnan(inputs))
        nan_mask = ops.cast(nan_mask, dtype=inputs.dtype)
        # Replace NaN with 0 for safe stat computation
        clean_inputs = ops.where(ops.isnan(inputs), ops.zeros_like(inputs), inputs)

        # 2. CALCULATE STATISTICS & NORMALIZE
        if self.use_normalization:
            # Compute NaN-safe mean: sum of valid values / count of valid values
            valid_count = ops.maximum(ops.sum(nan_mask, axis=1, keepdims=True), 1e-7)
            mean = ops.sum(clean_inputs * nan_mask, axis=1, keepdims=True) / valid_count
            # Compute NaN-safe std
            sq_diff = ((clean_inputs - mean) * nan_mask) ** 2
            variance = ops.sum(sq_diff, axis=1, keepdims=True) / valid_count
            std = ops.sqrt(variance)
            std = ops.maximum(std, 1e-7)  # Prevent division by zero
            x = (clean_inputs - mean) / std
        else:
            x = clean_inputs
            mean = None
            std = None

        # 3. CONCATENATE DATA WITH MASK
        x_with_mask = ops.concatenate([x, nan_mask], axis=-1)

        # 4. ENCODE
        x_patches = self.patch_embedding(x_with_mask, training=training)
        x_embedded = self.input_projection(x_patches, training=training)

        # 5. PROCESS
        hidden_states = x_embedded
        for block in self.blocks:
            hidden_states = block(hidden_states, training=training)

        hidden_states = self.output_norm(hidden_states, training=training)
        mean_hidden_states = ops.mean(hidden_states, axis=1, keepdims=True)

        # 6. PREDICT (Normalized Space)
        # Shape: [batch, prediction_length, num_quantiles]
        quantile_predictions = self.quantile_head(mean_hidden_states, training=training)

        # 7. DENORMALIZE OUTPUT (Reversible Instance Normalization)
        if self.use_normalization:
            norm_mean, norm_std = self._get_target_stats(mean, std)
            # Broadcasting: (B, PredLen, Quantiles) * (B, 1, 1) + (B, 1, 1)
            quantile_predictions = (quantile_predictions * norm_std) + norm_mean

        return quantile_predictions

    @staticmethod
    def _get_target_stats(
        mean: keras.KerasTensor,
        std: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Extract normalization stats for the target (last) feature.

        For multivariate inputs, assumes the target is the last feature.
        Returns stats shaped (Batch, 1, 1) for broadcasting with quantile predictions.

        Args:
            mean: Mean tensor of shape (Batch, 1, Features).
            std: Std tensor of shape (Batch, 1, Features).

        Returns:
            Tuple of (norm_mean, norm_std), each shaped (Batch, 1, 1).
        """
        if mean.shape[-1] is not None and mean.shape[-1] > 1:
            # Select stats for the last feature -> (Batch, 1, 1)
            norm_mean = mean[:, :, -1:]
            norm_std = std[:, :, -1:]
        else:
            norm_mean = mean
            norm_std = std
        return norm_mean, norm_std

    def predict_quantiles(
            self,
            context: Union[np.ndarray, keras.utils.PyDataset],
            quantile_levels: Optional[List[float]] = None,
            batch_size: int = 32,
            **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate specific quantile and point forecasts for time series data.

        This method acts as a high-level wrapper around `model.predict()`. It handles
        the complexity of mapping user-requested quantile levels (e.g., 0.95) to the
        specific output indices of the model's neural network head. It also automatically
        extracts the median (0.5 quantile) to serve as a robust point forecast.

        **Shape Logic**:
        The raw model outputs a tensor of shape `(Batch, Time, Trained_Quantiles)`.
        This method slices the last dimension based on the requested `quantile_levels`.

        Args:
            context: Input data.
                - A Numpy array of shape `(batch_size, input_length, features)`.
                - Or a `keras.utils.PyDataset` / `tf.data.Dataset`.
            quantile_levels: List of floats between 0 and 1.
                The specific probabilities to extract (e.g., `[0.1, 0.5, 0.9]`).
                If None, returns all quantiles the model was trained with.
                If a requested quantile was not in the training set, the closest
                available trained quantile will be used (with a warning).
            batch_size: Integer, number of samples per batch during inference.
                Defaults to 32.
            **kwargs: Additional arguments passed directly to `model.predict()`,
                such as `verbose` or `callbacks`.

        Returns:
            A tuple `(quantile_preds, point_preds)`:
            1. **quantile_preds**: Numpy array of shape
               `(batch_size, prediction_length, num_requested_quantiles)`.
               Contains the predicted values for the requested probability levels.
            2. **point_preds**: Numpy array of shape
               `(batch_size, prediction_length)`.
               Contains the median prediction (0.5 quantile), used as the primary
               point forecast.

        Example:
            ```python
            # Train with [0.1, 0.5, 0.9]
            model = TiRexCore(...)

            # Request specific confidence intervals at inference
            # context shape: (100, 168, 1)
            q_preds, median = model.predict_quantiles(
                context,
                quantile_levels=[0.05, 0.5, 0.95] # 0.05/0.95 map to closest (0.1/0.9)
            )

            # q_preds shape: (100, 24, 3)
            # median shape:  (100, 24)
            ```
        """
        # ---------------------------------------------------------------------
        # 1. Setup and Validation
        # ---------------------------------------------------------------------
        # If no specific levels requested, return everything the model knows
        if quantile_levels is None:
            quantile_levels = self.quantile_levels

        # ---------------------------------------------------------------------
        # 2. Run Inference
        # ---------------------------------------------------------------------
        # Perform the forward pass.
        # Output Shape: [batch_size, prediction_length, num_trained_quantiles]
        raw_predictions = self.predict(context, batch_size=batch_size, **kwargs)

        # ---------------------------------------------------------------------
        # 3. Map Requested Quantiles to Model Output Indices
        # ---------------------------------------------------------------------
        # We need to find which index in the last dimension corresponds to
        # the requested quantiles (e.g., user asks for 0.5, we find index 2).
        quantile_indices = []
        trained_quantiles_arr = np.array(self.quantile_levels)

        for q in quantile_levels:
            # Case A: Exact match found
            if q in self.quantile_levels:
                idx = self.quantile_levels.index(q)
                quantile_indices.append(idx)
            # Case B: Approximation needed (User asks for 0.95, model has 0.9)
            else:
                # Find index of the smallest absolute difference
                closest_idx = int(np.argmin(np.abs(trained_quantiles_arr - q)))
                quantile_indices.append(closest_idx)

                logger.warning(
                    f"Requested quantile {q} not found in trained model "
                    f"{self.quantile_levels}. Using closest match: "
                    f"{self.quantile_levels[closest_idx]}"
                )

        # ---------------------------------------------------------------------
        # 4. Extract Quantile Predictions
        # ---------------------------------------------------------------------
        # Slice the raw predictions tensor.
        # We select all batches (:), all time steps (:), and specific quantile indices.
        # Result Shape: [batch_size, prediction_length, num_requested_quantiles]
        quantile_preds = raw_predictions[:, :, quantile_indices]

        # ---------------------------------------------------------------------
        # 5. Extract Point Forecast (Median)
        # ---------------------------------------------------------------------
        # The median (0.5) minimizes MAE and is the standard point forecast
        # for quantile regression models.
        if 0.5 in self.quantile_levels:
            median_idx = self.quantile_levels.index(0.5)
        else:
            # Fallback: Use the middle index if strict 0.5 is missing
            median_idx = len(self.quantile_levels) // 2
            logger.debug(
                f"Median (0.5) not found in quantiles. Using index {median_idx} "
                f"({self.quantile_levels[median_idx]}) as point forecast."
            )

        # Slice out the median to get a 2D array.
        # Result Shape: [batch_size, prediction_length]
        mean_preds = raw_predictions[:, :, median_idx]

        return quantile_preds, mean_preds

    def _forecast(
            self,
            x: Union[np.ndarray, keras.utils.PyDataset],
            quantile_levels: Optional[List[float]] = None,
            **kwargs: Any
    ) -> Forecast:
        """Produce a unified :class:`Forecast` by reusing ``predict_quantiles``.

        This is the ``ForecastMixin`` hook. It does NOT reimplement any quantile
        mapping; it delegates to the model's existing ``predict_quantiles`` and
        packs the result into the shared contract.

        Args:
            x: Context window, shape ``[B, input_length, F]`` (or a dataset).
            quantile_levels: Levels to extract; defaults to the model's
                configured ``self.quantile_levels``.
            **kwargs: Forwarded to ``predict_quantiles`` (e.g. ``batch_size``,
                ``verbose``).

        Returns:
            A :class:`Forecast` with ``point`` shape ``[B, H]`` and ``quantiles``
            shape ``[B, H, Q]``. TiRex flattens the target feature axis, so the
            shapes are intentionally passed through unchanged (no fabricated
            ``F`` axis); downstream metrics/helpers handle both ranks.
        """
        levels = quantile_levels if quantile_levels is not None else self.quantile_levels
        quantile_preds, point_preds = self.predict_quantiles(x, levels, **kwargs)
        return Forecast(
            point=np.asarray(point_preds),
            quantiles=np.asarray(quantile_preds),
            quantile_levels=list(levels),
        )

    @classmethod
    def from_variant(
        cls,
        variant: str,
        prediction_length: int = 32,
        quantile_levels: List[float] = DEFAULT_QUANTILES,
        **kwargs
    ) -> "TiRexCore":
        """
        Create a TiRex model from a predefined variant.

        Args:
            variant: String, one of "tiny", "small", "medium", "large"
            prediction_length: Integer, length of prediction horizon
            quantile_levels: List of quantile levels to predict
            **kwargs: Additional arguments passed to the constructor

        Returns:
            TiRexCore model instance

        Raises:
            ValueError: If variant is not recognized

        Example:
            >>> # Tiny model for quick experiments
            >>> model = TiRexCore.from_variant("tiny", prediction_length=24)
            >>> # Large model for production
            >>> model = TiRexCore.from_variant("large", prediction_length=48)
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()

        # Update config with kwargs (kwargs take precedence)
        config.update(kwargs)

        logger.info(f"Creating TiRex-{variant.upper()} model")

        return cls(
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            **config
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "num_blocks": self.num_blocks,
            "num_heads": self.num_heads,
            "lstm_units": self.lstm_units,
            "ff_dim": self.ff_dim,
            "block_types": self.block_types,
            "quantile_levels": self.quantile_levels,
            "prediction_length": self.prediction_length,
            "dropout_rate": self.dropout_rate,
            "use_layer_norm": self.use_layer_norm,
            "use_normalization": self.use_normalization,
            "attention_window_size": self.attention_window_size,
            "attention_type": self.attention_type,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TiRexCore":
        """Create model from configuration."""
        return cls(**config)

# ---------------------------------------------------------------------
# Factory Functions (following ConvNeXt V2 pattern)
# ---------------------------------------------------------------------


def create_tirex_model(
    input_length: int,
    prediction_length: int = 32,
    patch_size: int = 16,
    embed_dim: int = 256,
    num_blocks: int = 6,
    num_heads: int = 8,
    quantile_levels: List[float] = DEFAULT_QUANTILES,
    block_types: Optional[List[str]] = None,
    **kwargs
) -> TiRexCore:
    """
    Create a TiRex model with specified configuration.

    Args:
        input_length: Integer, length of input sequences.
        prediction_length: Integer, length of prediction horizon.
        patch_size: Integer, size of input patches.
        embed_dim: Integer, embedding dimension.
        num_blocks: Integer, number of sequential blocks.
        num_heads: Integer, number of attention heads.
        quantile_levels: List of quantile levels to predict.
        block_types: List of block types for each layer.
        **kwargs: Additional arguments for TiRexCore.

    Returns:
        TiRexCore model instance.
    """
    model = TiRexCore(
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        block_types=block_types,
        quantile_levels=quantile_levels,
        prediction_length=prediction_length,
        **kwargs
    )

    # Build the model with a dummy input to initialize weights and shapes
    dummy_input = np.zeros((1, input_length, 1), dtype='float32')
    _ = model(dummy_input)

    logger.info(
        f"Created TiRex model: input_length={input_length}, "
        f"prediction_length={prediction_length}, embed_dim={embed_dim}"
    )

    return model


def create_tirex_by_variant(
    variant: str = "medium",
    input_length: int = 128,
    prediction_length: int = 32,
    quantile_levels: List[float] = DEFAULT_QUANTILES,
    **kwargs
) -> TiRexCore:
    """
    Convenience function to create TiRex models from predefined variants.

    Args:
        variant: String, model variant ("tiny", "small", "medium", "large")
        input_length: Integer, length of input sequences
        prediction_length: Integer, length of prediction horizon
        quantile_levels: List of quantile levels to predict
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        TiRexCore model instance

    Example:
        >>> # Create TiRex-Small for quick experiments
        >>> model = create_tirex_by_variant("small", input_length=96, prediction_length=24)
        >>>
        >>> # Create TiRex-Large for production forecasting
        >>> model = create_tirex_by_variant("large", input_length=256, prediction_length=48)
    """
    model = TiRexCore.from_variant(
        variant,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
        **kwargs
    )

    # Build the model with a dummy input
    dummy_input = np.zeros((1, input_length, 1), dtype='float32')
    _ = model(dummy_input)

    logger.info(
        f"Created TiRex-{variant.upper()}: input_length={input_length}, "
        f"prediction_length={prediction_length}"
    )

    return model

# ---------------------------------------------------------------------
