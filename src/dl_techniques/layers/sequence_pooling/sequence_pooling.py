"""
A unified and configurable pooling layer for sequence data.

This layer serves as a bridge between sequence encoders (like Transformers or
LSTMs) and downstream tasks that require a fixed-size vector representation.
It transforms a sequence of vectors `(batch, seq_len, hidden_dim)` into a
single summary vector `(batch, output_dim)`. Its core design philosophy is
modularity, offering a wide array of pooling strategies that can be selected,
combined, and experimented with through a single, consistent interface.

Architecture:
    The layer operates on a two-stage principle: strategy execution followed by
    aggregation. First, it applies one or more user-defined pooling
    strategies to the input sequence. These strategies fall into several
    categories:
    -   **Positional:** Selects a vector from a specific position (e.g., the
        first `[CLS]` token).
    -   **Statistical:** Computes a summary statistic over the sequence
        dimension (e.g., mean, max).
    -   **Learnable:** Computes a weighted average of the sequence vectors,
        where the weights are learned during training.

    If multiple strategies are specified, their resulting vectors are combined
    in the second stage using an aggregation method, such as concatenation
    or a weighted sum, to produce the final output vector.

Foundational Mathematics and Concepts:
    The layer implements several key pooling concepts, each with a distinct
    theoretical motivation.

    1.  **Statistical and Positional Pooling:** These are simple, computationally
        efficient methods. Mean pooling (`mean`) averages all token vectors,
        capturing the overall semantic content. Max pooling (`max`) identifies
        the most salient features across the sequence, a technique popularized
        by early CNNs for NLP. Positional pooling (`cls`, `first`) relies on
        the model architecture (e.g., BERT) having learned to embed the
        summary of the entire sequence into a specific token's representation.

    2.  **Attention Pooling (`attention`):** This is a learnable, content-aware
        strategy based on the self-attention mechanism. It learns to assign
        an "importance" score to each element in the sequence and computes a
        weighted average. The process is as follows:
        -   First, each input vector `x_i` is passed through a non-linear
            transformation: `h_i = tanh(W*x_i + b)`.
        -   An unnormalized importance score `e_i` is computed by taking the
            dot product of `h_i` with a learnable context vector `u`:
            `e_i = h_i^T * u`. This context vector `u` can be interpreted as a
            learned query that represents "what is important".
        -   The scores are normalized into weights `a_i` using the softmax
            function: `a_i = softmax(e_i)`.
        -   The final representation is the weighted sum of the original input
            vectors: `v = sum(a_i * x_i)`.
        This allows the model to dynamically focus on the most relevant parts
        of the sequence for a given task.

    3.  **Weighted Pooling (`weighted`):** This provides a simpler, content-
        agnostic learnable pooling. It assigns a learnable scalar weight `p_i`
        to each *position* `i` in the sequence, up to a maximum length. These
        weights are normalized via softmax and used to compute a weighted
        average. Unlike attention, these weights are fixed after training and
        do not depend on the input content, making this method a middle ground
        between simple mean pooling and complex attention pooling.

References:
    -   Lin, Z. et al. (2017). "A Structured Self-attentive Sentence Embedding."
        This paper introduces the self-attentive pooling mechanism that forms
        the basis for the 'attention' strategy.
    -   Conneau, A. et al. (2017). "Supervised Learning of Universal Sentence
        Representations." This work demonstrated the effectiveness of simple
        pooling strategies like max-pooling over BiLSTM outputs for creating
        high-quality sentence embeddings.
    -   Zhang, T. et al. (2020). "Pooling Revisited: Your Receptive Field is
        Suboptimal." Provides a modern analysis comparing various pooling
        methods, highlighting that the optimal strategy is task-dependent.
"""

import keras
from keras import ops, initializers, regularizers
from typing import Optional, Union, Tuple, Dict, Any, Literal, List

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from .attention_pooling import AttentionPooling
from .weighted_pooling import WeightedPooling

# ---------------------------------------------------------------------
# Type definitions for enhanced type safety
# ---------------------------------------------------------------------

PoolingStrategy = Literal[
    # Positional pooling
    'cls', 'first', 'last', 'middle',
    # Statistical pooling
    'mean', 'max', 'min', 'sum',
    # Advanced statistical
    'mean_max', 'mean_std', 'mean_max_min',
    # Learnable pooling
    'attention', 'multi_head_attention', 'weighted',
    # Top-k pooling
    'top_k_mean', 'top_k_max',
    # Special
    'none', 'flatten'
]

AggregationMethod = Literal['concat', 'add', 'multiply', 'weighted_sum']

# ---------------------------------------------------------------------
# Main Configurable Pooling Layer
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SequencePooling(keras.layers.Layer):
    """Configurable pooling layer supporting multiple strategies for sequences.

    This layer provides a unified interface for positional, statistical,
    learnable, and top-k pooling strategies. One or more strategies can
    be applied simultaneously and their results aggregated via
    concatenation, addition, multiplication, or learned weighted sum.
    Supported strategies include ``cls``, ``first``, ``last``, ``middle``
    (positional); ``mean``, ``max``, ``min``, ``sum`` (statistical);
    ``mean_max``, ``mean_std``, ``mean_max_min`` (combined);
    ``attention``, ``multi_head_attention``, ``weighted`` (learnable);
    ``top_k_mean``, ``top_k_max`` (top-k); and ``none``, ``flatten``
    (special).

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input [B, seq_len, hidden_dim]  │
        └──────────────┬───────────────────┘
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
        ┌────────┐ ┌────────┐ ┌────────┐
        │Strat 1 │ │Strat 2 │ │Strat N │
        └───┬────┘ └───┬────┘ └───┬────┘
            │          │          │
            └──────────┼──────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  Aggregation                     │
        │  (concat/add/multiply/weighted)  │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  Output [B, output_dim]          │
        └──────────────────────────────────┘

    **Positional strategies and the keep-mask:**

    ``last`` returns the LAST KEPT position and ``middle`` the middle of the
    KEPT positions (with ``n`` kept positions, the ``n // 2``-th of them,
    0-based) — both derived from the mask, never from the padded sequence
    length. At ``mask=None`` or an all-keep mask they reduce exactly to
    ``inputs[:, -1, :]`` and ``inputs[:, seq_len // 2, :]``.

    ``cls`` and ``first`` return index 0 BY INTENT, mask or no mask: the caller
    asked for the token at index 0, so a mask covering position 0 does not
    redirect them. This asymmetry between the mask-aware pair and the
    by-intent pair is deliberate; it is pinned by
    ``tests/test_layers/test_sequence_pooling.py::TestPositionalModesIsolateMaskedPositions``.

    The ``last``/``middle`` quantifier above ("never a masked position") was
    MEASURED against this shipped implementation, not inferred: over EVERY
    non-empty keep-mask at ``seq_len = 1..9`` — 1013 masks — both modes
    returned exactly the expected kept token, 0 mismatches (``batch=1``,
    ``hidden_dim=4``, ``float32``, CPU, no ``exclude_positions``). This is the
    ONE home for that figure; do not restate it in a consumer module.

    A FULLY-MASKED row degenerates to index 0 for ``last`` and ``middle``,
    i.e. the returned token is itself masked. This is a deliberate, documented
    degeneration consistent with the rest of the package (no strategy here
    rescues a fully-masked row and none raises); it is not a rescue path.

    Args:
        strategy: Pooling strategy name or list of strategy names.
        exclude_positions: Positions to exclude from pooling. Honoured by EVERY
            strategy except ``none``/``flatten`` (which return the sequence, so
            there is no index to move and no reduction to exclude from). Unlike
            a keep-mask this is an EXPLICIT caller instruction, so it outranks a
            positional mode's default: ``cls``/``first`` with position 0
            excluded return the first NON-excluded position (they still ignore
            the keep-mask). If the exclusions leave no position kept, every
            positional mode degenerates to index 0 — the same documented,
            no-rescue degeneration as a fully-masked row.
        aggregation_method: How to combine multiple strategy outputs.
        attention_hidden_dim: Hidden dimension for attention pooling.
        attention_num_heads: Number of heads for multi-head attention.
        attention_dropout_rate: Dropout rate for attention mechanisms.
        weighted_max_seq_len: Maximum sequence length for weighted pooling.
        top_k: Number of top elements for top-k pooling.
        temperature: Temperature for softmax in learnable strategies.
        use_bias: Whether to use bias in learnable components.
        kernel_initializer: Initializer for kernels.
        bias_initializer: Initializer for biases.
        kernel_regularizer: Optional regularizer for kernels.
        bias_regularizer: Optional regularizer for biases.
    """

    def __init__(
        self,
        strategy: Union[PoolingStrategy, List[PoolingStrategy]] = 'mean',
        exclude_positions: Optional[List[int]] = None,
        aggregation_method: AggregationMethod = 'concat',
        attention_hidden_dim: int = 256,
        attention_num_heads: int = 1,
        attention_dropout_rate: float = 0.0,
        weighted_max_seq_len: int = 512,
        top_k: int = 10,
        temperature: float = 1.0,
        use_bias: bool = True,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Initialise the sequence pooling layer."""
        super().__init__(**kwargs)

        # Store ALL configuration (critical for get_config)
        self.strategy = strategy if isinstance(strategy, list) else [strategy]
        self.exclude_positions = exclude_positions or []
        self.aggregation_method = aggregation_method
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_num_heads = attention_num_heads
        self.attention_dropout_rate = attention_dropout_rate
        self.weighted_max_seq_len = weighted_max_seq_len
        self.top_k = top_k
        self.temperature = temperature
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer) if kernel_regularizer else None
        self.bias_regularizer = regularizers.get(bias_regularizer) if bias_regularizer else None

        # Create learnable components in __init__ (Golden Rule)
        self.learnable_components: Dict[str, keras.layers.Layer] = {}

        for strat in self.strategy:
            if strat in ['attention', 'multi_head_attention']:
                num_heads = self.attention_num_heads if strat == 'multi_head_attention' else 1
                self.learnable_components[strat] = AttentionPooling(
                    hidden_dim=self.attention_hidden_dim,
                    num_heads=num_heads,
                    dropout_rate=self.attention_dropout_rate,
                    use_bias=self.use_bias,
                    temperature=self.temperature,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'{strat}_pooling'
                )
            elif strat == 'weighted':
                self.learnable_components[strat] = WeightedPooling(
                    max_seq_len=self.weighted_max_seq_len,
                    dropout_rate=self.attention_dropout_rate,
                    temperature=self.temperature,
                    initializer='ones',
                    regularizer=self.kernel_regularizer,
                    name='weighted_pooling'
                )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and all learnable sub-layers.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # CRITICAL: Explicitly build all learnable components
        for component in self.learnable_components.values():
            component.build(input_shape)

        # Create aggregation weights for weighted sum
        if len(self.strategy) > 1 and self.aggregation_method == 'weighted_sum':
            self.aggregation_weights = self.add_weight(
                name='aggregation_weights',
                shape=(len(self.strategy),),
                initializer='ones',
                regularizer=self.kernel_regularizer,
                trainable=True
            )

        super().build(input_shape)

    def _apply_mask_and_exclusions(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None
    ) -> Tuple[keras.KerasTensor, Optional[keras.KerasTensor]]:
        """Fold ``exclude_positions`` into the keep-mask.

        This is the SINGLE place where the caller's keep-mask and the caller's
        ``exclude_positions`` are combined into ONE keep predicate. It does not
        touch ``inputs`` at all (the returned tensor is the input unchanged);
        the name is historical.

        :param inputs: Input sequence ``(batch, seq_len, hidden_dim)``, returned
            unchanged.
        :param mask: Optional keep-mask ``(batch, seq_len)``, 1/True = keep.
        :return: ``(inputs, combined_mask)``. ``combined_mask`` is ``None``
            exactly when ``mask`` is ``None`` AND ``exclude_positions`` is
            empty — the condition every caller's no-mask fast path keys on.
        """
        if self.exclude_positions:
            seq_len = ops.shape(inputs)[1]
            if mask is None:
                mask = ops.ones((ops.shape(inputs)[0], seq_len))

            # DECISION plan-2026-07-31T210633-b63a35aa/D-005: no `pos < seq_len`
            # Python guard. `indices != pos` is already a no-op for an
            # out-of-range `pos`, so the guard was redundant AND was a graph
            # trap: under a symbolic `seq_len` it is an `if` on a traced tensor,
            # which AutoGraph rewrites into a `tf.cond` (or raises). Positional
            # modes now reach this method, and they are required to be
            # symbolic-length-safe. Do NOT reinstate the guard.
            indices = ops.arange(seq_len)
            for pos in self.exclude_positions:
                exclusion = ops.cast(indices != pos, mask.dtype)
                mask = mask * exclusion

        return inputs, mask

    def _select_top_k(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor],
        seq_len: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, Optional[keras.KerasTensor]]:
        """Select the ``top_k`` highest-norm positions and report their validity.

        Shared by the ``top_k_mean`` and ``top_k_max`` branches, which must rank
        and exclude identically or the two strategies drift apart.

        :param inputs: Input sequence ``(batch, seq_len, hidden_dim)``.
        :param mask: Optional keep-mask ``(batch, seq_len)``, 1/True = keep, or
            ``None``.
        :param seq_len: Sequence length (a scalar tensor is fine; this path is
            graph-safe under a symbolic ``seq_len``).
        :return: ``(top_k_embeds, validity)`` where ``top_k_embeds`` is
            ``(batch, k, hidden_dim)`` gathered from the RAW ``inputs`` and
            ``validity`` is ``(batch, k)`` in ``inputs.dtype``, 1.0 for a
            selected position that is genuinely kept and 0.0 for one that is
            masked. ``validity`` is ``None`` exactly when ``mask is None``, in
            which case the caller must aggregate over all ``k`` positions
            unconditionally — that path is bit-identical to the pre-F-24 code.
        """
        # DECISION plan-2026-07-31T132403-b3f540cb/D-002: masked positions are
        # excluded in TWO places, and BOTH are required or the leak persists.
        # (i) RANKING - a masked position's norm is forced to a very negative
        #     sentinel so it always sorts LAST. Real norms are sums of squares
        #     and therefore >= 0, so any negative sentinel is strictly below
        #     every kept position; `-1e4` is used because it is finite under
        #     float16 (`float16(-1e9)` is `-inf`).
        # (ii) AGGREGATION - `k` is a BATCH-GLOBAL scalar while the kept count is
        #     PER-ROW, so a row with fewer kept positions than `k` is still
        #     forced to select masked ones. Their validity is gathered at the
        #     SAME indices and used to exclude them from the reduction.
        # Do NOT "simplify" this by clamping `k` to the batch-wide minimum kept
        # count (`ops.min(ops.sum(mask, axis=1))`): it is graph-safe but makes
        # one row's answer depend on the OTHER rows in its batch, and throws
        # away capacity for every row above the minimum.
        # Do NOT drop (i) and keep only (ii) either: without the sentinel a
        # masked position can outrank a kept one whose embedding is near zero.
        if mask is not None:
            mask_expanded = ops.expand_dims(ops.cast(mask, inputs.dtype), -1)
            masked_inputs = inputs * mask_expanded
        else:
            masked_inputs = inputs

        norms = ops.sum(masked_inputs ** 2, axis=-1)
        if mask is not None:
            norms = ops.where(
                ops.cast(mask, "bool"), norms, ops.cast(-1e4, norms.dtype)
            )

        k = ops.minimum(self.top_k, seq_len)
        _, top_k_indices = ops.top_k(norms, k=k)

        top_k_embeds = ops.take_along_axis(
            inputs,
            ops.expand_dims(top_k_indices, -1),
            axis=1
        )

        validity = None
        if mask is not None:
            validity = ops.take_along_axis(
                ops.cast(mask, inputs.dtype), top_k_indices, axis=1
            )

        return top_k_embeds, validity

    def _positional_index(
        self,
        mask: keras.KerasTensor,
        seq_len: keras.KerasTensor,
        *,
        mode: str
    ) -> keras.KerasTensor:
        """Resolve the per-row index selected by a MASK-AWARE positional mode.

        Shared by the ``cls``/``first``, ``middle`` and ``last`` branches, which
        must agree on what "a position exists" means or the modes drift apart
        under the same keep predicate.

        :param mask: COMBINED keep-mask ``(batch, seq_len)``, 1/True = keep —
            the caller's mask and ``exclude_positions`` already folded together
            by :meth:`_apply_mask_and_exclusions`. Must NOT be ``None``: each
            caller keeps a separate no-keep-predicate fast path so the
            ``mask=None``-and-no-exclusions numerics stay bit-identical to the
            pre-F-25 code.
        :param seq_len: Sequence length. A scalar tensor is fine; this path is
            graph-safe and jit-safe under a symbolic ``seq_len``.
        :param mode: ``'first'`` (the first kept position), ``'middle'`` (the
            middle of the KEPT positions) or ``'last'`` (the last KEPT
            position).
        :return: ``(batch,)`` ``int32`` index, safe to feed to
            ``ops.take_along_axis``. An EMPTY keep set yields index 0 — a
            DOCUMENTED DEGENERATION (the returned token is itself suppressed),
            not a rescue. This covers both routes into an empty set: a fully
            masked row, and an ``exclude_positions`` that removes every
            remaining kept position.
        :raises ValueError: If ``mode`` is not one of ``'first'``, ``'middle'``
            or ``'last'``.
        """
        # DECISION plan-2026-07-31T210633-b63a35aa/D-001: both modes resolve to
        # `where(keep-predicate, position, -1)` -> `max` -> `maximum(..., 0)`.
        # That shared shape is WHY one helper serves both; it is not a forced
        # unification of two unlike things.
        #
        # `middle` = the middle of the KEPT positions, NOT the geometric
        # midpoint of the padded sequence. The old `ops.shape(inputs)[1] // 2`
        # derived the index from the PADDED length inside a layer whose
        # contract is mask-awareness: under ORDINARY contiguous-prefix padding
        # it returned a PAD token in 19 of 42 measured (S, L) cells (exactly
        # when `L <= S // 2`). Do NOT "restore" a padded-length midpoint.
        #
        # `last` = the LAST KEPT index, NOT `sum(mask) - 1`. Those coincide for
        # a contiguous-prefix mask and for nothing else; with an interior gap
        # (e.g. keep = 110011) `sum(mask) - 1` is 3, which is MASKED, while the
        # last kept index is 5. Do NOT "simplify" back to a count.
        #
        # Do NOT use `ops.argmax` over a one-hot hit vector for `middle`: it
        # works, but it pins behaviour on argmax returning the FIRST maximum,
        # an unstated framework guarantee, and it does not share a shape with
        # `last`. Do NOT use `ops.tril`/`ops.triu` to build the prefix sums —
        # they are graph-mode traps on this stack.
        #
        # The `-1` sentinel is an INDEX, not a logit: it never reaches a
        # softmax and is compared only by integer `max`, so the fp16 concerns
        # that force `-1e4` elsewhere in this package do not apply here.
        # See decisions.md D-001.
        #
        # DECISION plan-2026-07-31T210633-b63a35aa/D-005: `first` was added here
        # rather than beside `inputs[:, 0, :]` so that ALL FOUR positional modes
        # resolve their index through ONE keep predicate. It is the MIRROR of
        # `last` (`where` -> `min` over a `seq_len` sentinel instead of `max`
        # over `-1`); the empty-set fallback needs an explicit `where` because
        # a `minimum(..., seq_len - 1)` clamp would return the LAST position
        # instead of the documented index 0. See decisions.md D-005.
        if mode not in ('first', 'middle', 'last'):
            raise ValueError(
                f"_positional_index: mode must be 'first', 'middle' or 'last', "
                f"got {mode!r}."
            )

        keep = ops.cast(mask, 'bool')
        keep_int = ops.cast(keep, 'int32')
        # `(batch, seq_len)` position grid, broadcast from `(1, seq_len)`.
        positions = ops.zeros_like(keep_int) + ops.expand_dims(
            ops.arange(seq_len, dtype='int32'), 0
        )

        if mode == 'first':
            sentinel = ops.cast(seq_len, 'int32')
            candidates = ops.where(keep, positions, sentinel)
            index = ops.min(candidates, axis=1)
            return ops.where(
                ops.equal(index, sentinel), ops.zeros_like(index), index
            )

        if mode == 'middle':
            cum = ops.cumsum(keep_int, axis=1)
            target = ops.expand_dims(cum[:, -1] // 2 + 1, -1)
            candidates = ops.where(
                ops.logical_and(keep, cum <= target), positions, -1
            )
        else:
            candidates = ops.where(keep, positions, -1)

        return ops.maximum(ops.max(candidates, axis=1), 0)

    def _apply_single_strategy(
        self,
        strategy: str,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply a single pooling strategy.

        Args:
            strategy: Name of the pooling strategy.
            inputs: Input sequence tensor.
            mask: Optional boolean mask.
            training: Whether in training mode.

        Returns:
            Pooled tensor.
        """
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]

        # DECISION plan-2026-07-31T210633-b63a35aa/D-005: `exclude_positions` is
        # HONOURED by the four positional modes. The old gate listed all four
        # alongside `none`/`flatten` and skipped `_apply_mask_and_exclusions`
        # entirely, which made `exclude_positions` a measured `0.000e+00` NO-OP
        # for `cls`/`first`/`last`/`middle` — a silently inert public config key.
        #
        # `none`/`flatten` remain excluded: they return the sequence itself, so
        # there is no index to move and no reduction to exclude a position from.
        #
        # `cls`/`first` fold in the exclusions but NOT the caller's mask
        # (`_apply_mask_and_exclusions(inputs, None)`). The two inputs are not
        # interchangeable: a keep-MASK is inferred padding and does not outrank
        # "give me index 0", while `exclude_positions` is an EXPLICIT caller
        # instruction and does. `test_cls_and_first_return_index_zero_regardless
        # _of_mask` pins the mask half; do NOT pass `mask` here.
        #
        # Do NOT implement this by ZEROING `inputs`: that would change
        # `none`/`flatten`, and would make `cls` return a zero VECTOR instead of
        # a different INDEX. The exclusions must reach `_positional_index` as an
        # additional keep TERM. See decisions.md D-005.
        if strategy in ['cls', 'first']:
            _, keep = self._apply_mask_and_exclusions(inputs, None)
            if keep is None:
                # No exclusions: bit-identical to the pre-H-02 code.
                return inputs[:, 0, :]
            selected = self._positional_index(keep, seq_len, mode='first')
            return ops.take_along_axis(
                inputs,
                ops.expand_dims(ops.expand_dims(selected, -1), -1),
                axis=1
            )[:, 0, :]

        if strategy not in ['none', 'flatten']:
            inputs, mask = self._apply_mask_and_exclusions(inputs, mask)

        if strategy in ['last', 'middle']:
            if mask is None:
                # No keep-predicate FAST PATH (no mask AND no exclusions), kept
                # separate so those numerics stay bit-identical to the pre-F-25
                # code (I1_UNMASKED_GOLDEN).
                if strategy == 'last':
                    return inputs[:, -1, :]
                return inputs[:, seq_len // 2, :]

            selected = self._positional_index(mask, seq_len, mode=strategy)
            return ops.take_along_axis(
                inputs,
                ops.expand_dims(ops.expand_dims(selected, -1), -1),
                axis=1
            )[:, 0, :]

        # Statistical strategies
        elif strategy == 'mean':
            if mask is not None:
                mask_expanded = ops.expand_dims(ops.cast(mask, inputs.dtype), -1)
                masked_inputs = inputs * mask_expanded
                sum_pooled = ops.sum(masked_inputs, axis=1)
                lengths = ops.sum(mask_expanded, axis=1)
                lengths = ops.maximum(lengths, 1.0)
                return sum_pooled / lengths
            else:
                return ops.mean(inputs, axis=1)

        elif strategy == 'max':
            if mask is not None:
                mask_expanded = ops.expand_dims(ops.cast(mask, inputs.dtype), -1)
                masked_inputs = inputs + (1.0 - mask_expanded) * (-1e9)
                return ops.max(masked_inputs, axis=1)
            else:
                return ops.max(inputs, axis=1)

        elif strategy == 'min':
            if mask is not None:
                mask_expanded = ops.expand_dims(ops.cast(mask, inputs.dtype), -1)
                masked_inputs = inputs + (1.0 - mask_expanded) * 1e9
                return ops.min(masked_inputs, axis=1)
            else:
                return ops.min(inputs, axis=1)

        elif strategy == 'sum':
            if mask is not None:
                mask_expanded = ops.expand_dims(ops.cast(mask, inputs.dtype), -1)
                masked_inputs = inputs * mask_expanded
                return ops.sum(masked_inputs, axis=1)
            else:
                return ops.sum(inputs, axis=1)

        # Combined statistical strategies
        elif strategy == 'mean_max':
            mean_pool = self._apply_single_strategy('mean', inputs, mask, training)
            max_pool = self._apply_single_strategy('max', inputs, mask, training)
            return ops.concatenate([mean_pool, max_pool], axis=-1)

        elif strategy == 'mean_std':
            mean_pool = self._apply_single_strategy('mean', inputs, mask, training)
            if mask is not None:
                mask_expanded = ops.expand_dims(ops.cast(mask, inputs.dtype), -1)
                masked_inputs = inputs * mask_expanded
                variance = ops.sum(
                    (masked_inputs - ops.expand_dims(mean_pool, 1)) ** 2 * mask_expanded,
                    axis=1
                )
                lengths = ops.sum(mask_expanded, axis=1)
                lengths = ops.maximum(lengths, 1.0)
                std_pool = ops.sqrt(variance / lengths + 1e-6)
            else:
                std_pool = ops.std(inputs, axis=1)
            return ops.concatenate([mean_pool, std_pool], axis=-1)

        elif strategy == 'mean_max_min':
            mean_pool = self._apply_single_strategy('mean', inputs, mask, training)
            max_pool = self._apply_single_strategy('max', inputs, mask, training)
            min_pool = self._apply_single_strategy('min', inputs, mask, training)
            return ops.concatenate([mean_pool, max_pool, min_pool], axis=-1)

        # Learnable strategies
        elif strategy in ['attention', 'multi_head_attention']:
            return self.learnable_components[strategy](
                inputs, mask=mask, training=training
            )

        elif strategy == 'weighted':
            return self.learnable_components[strategy](
                inputs, mask=mask, training=training
            )

        # Top-k strategies
        elif strategy == 'top_k_mean':
            top_k_embeds, validity = self._select_top_k(inputs, mask, seq_len)
            if validity is None:
                return ops.mean(top_k_embeds, axis=1)
            # Divide by the number of VALID selected positions, never by `k`.
            # A fully-masked row floors the denominator at 1 and returns zeros
            # rather than dividing by zero; it is deliberately NOT rescued.
            validity_expanded = ops.expand_dims(validity, -1)
            valid_sum = ops.sum(top_k_embeds * validity_expanded, axis=1)
            valid_count = ops.maximum(
                ops.sum(validity_expanded, axis=1),
                ops.cast(1.0, top_k_embeds.dtype)
            )
            return valid_sum / valid_count

        elif strategy == 'top_k_max':
            top_k_embeds, validity = self._select_top_k(inputs, mask, seq_len)
            if validity is None:
                return ops.max(top_k_embeds, axis=1)
            # Bias the INVALID SELECTED embeddings below every real value before
            # the max. This mirrors the `max` strategy above, but uses the
            # `ops.where` form with the finite `-1e4` sentinel rather than
            # `+ (1 - mask) * -1e9`: the additive form is `0 * -inf = NaN` once
            # `-1e9` underflows to `-inf` under `mixed_float16`.
            # A fully-masked row therefore returns the sentinel itself; it is
            # deliberately NOT rescued (the `max` strategy already returns
            # `-1e9` in the same situation).
            validity_expanded = ops.expand_dims(validity, -1)
            biased = ops.where(
                ops.cast(validity_expanded, "bool"),
                top_k_embeds,
                ops.cast(-1e4, top_k_embeds.dtype)
            )
            return ops.max(biased, axis=1)

        # Special strategies
        elif strategy == 'none':
            return inputs

        elif strategy == 'flatten':
            return ops.reshape(inputs, (batch_size, -1))

        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")

    def call(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply the configured pooling strategies.

        Args:
            inputs: Input tensor ``(batch, seq_len, hidden_dim)``.
            mask: Optional boolean mask ``(batch, seq_len)``.
            training: Whether in training mode.

        Returns:
            Pooled features tensor.
        """
        # Apply each strategy
        outputs = []
        for strat in self.strategy:
            output = self._apply_single_strategy(strat, inputs, mask, training)
            outputs.append(output)

        # Return single output if only one strategy
        if len(outputs) == 1:
            return outputs[0]

        # Handle different aggregation methods
        if self.aggregation_method == 'concat':
            if any(s == 'none' for s in self.strategy):
                raise ValueError("Cannot concatenate 'none' strategy with others")
            return ops.concatenate(outputs, axis=-1)

        elif self.aggregation_method == 'add':
            result = outputs[0]
            for output in outputs[1:]:
                result = result + output
            return result

        elif self.aggregation_method == 'multiply':
            result = outputs[0]
            for output in outputs[1:]:
                result = result * output
            return result

        elif self.aggregation_method == 'weighted_sum':
            weights = ops.softmax(self.aggregation_weights)
            result = outputs[0] * weights[0]
            for i, output in enumerate(outputs[1:], 1):
                result = result + output * weights[i]
            return result

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape based on pooling strategy.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple.
        """
        batch_size = input_shape[0]
        hidden_dim = input_shape[-1]

        # Handle single strategy
        if len(self.strategy) == 1:
            strat = self.strategy[0]
            if strat == 'none':
                return input_shape
            elif strat == 'flatten':
                seq_len = input_shape[1]
                return (batch_size, seq_len * hidden_dim if seq_len else None)
            elif strat in ['mean_max', 'mean_std']:
                return (batch_size, hidden_dim * 2)
            elif strat == 'mean_max_min':
                return (batch_size, hidden_dim * 3)
            else:
                return (batch_size, hidden_dim)

        # Handle multiple strategies
        if self.aggregation_method == 'concat':
            total_dim = 0
            for strat in self.strategy:
                if strat in ['mean_max', 'mean_std']:
                    total_dim += hidden_dim * 2
                elif strat == 'mean_max_min':
                    total_dim += hidden_dim * 3
                else:
                    total_dim += hidden_dim
            return (batch_size, total_dim)
        else:
            # add, multiply, weighted_sum preserve dimension
            return (batch_size, hidden_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration for serialization.

        Returns:
            Dictionary containing all constructor parameters.
        """
        config = super().get_config()
        config.update({
            'strategy': self.strategy,
            'exclude_positions': self.exclude_positions,
            'aggregation_method': self.aggregation_method,
            'attention_hidden_dim': self.attention_hidden_dim,
            'attention_num_heads': self.attention_num_heads,
            'attention_dropout_rate': self.attention_dropout_rate,
            'weighted_max_seq_len': self.weighted_max_seq_len,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------