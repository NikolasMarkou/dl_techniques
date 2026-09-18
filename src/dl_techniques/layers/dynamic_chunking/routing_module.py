"""Routing module for H-Net dynamic chunking.

``RoutingModule`` reads a full-resolution sequence ``(B, L, D)`` and decides, for
every position, whether a new chunk begins there. It scores a position by the
cosine dissimilarity between two learned projections of adjacent hidden states,

    p_{t+1} = clip((1 - cos(q(h_t), k(h_{t+1}))) / 2, 0, 1)

so a position whose predecessor points in a different direction gets a high
boundary probability. Both projections start at the identity, so an untrained
layer computes raw adjacent cosine similarity. Position 0 is forced to
``p = 1.0``, and the hard decision is the strict comparison ``p > 0.5``.

Only the padded branch of the reference implementation is ported; the packed
``cu_seqlens`` branch and the ``inference_params`` cache path are absent. The
last axis of the input must equal ``d_model``. Of the three outputs, only
``selected_probs`` carries a gradient back to the projections; the boolean
``boundary_mask`` is a hard decision with no gradient.

References:
    - Hwang et al., 2025. Dynamic Chunking for End-to-End Hierarchical Sequence
      Modeling. (https://arxiv.org/abs/2507.07955)
"""

from typing import Any, Dict, Optional, Tuple

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

#: Denominator floor used by ``torch.nn.functional.normalize`` (dc.py:88-89).
#: The clamp form max(||x||, eps) maps an all-zero row to zero, not to NaN.
NORMALIZE_EPS = 1e-12

#: Forced boundary probability of position 0 (dc.py:95-96).
FORCED_FIRST_BOUNDARY_PROB = 1.0

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.dynamic_chunking.routing_module")
class RoutingModule(keras.layers.Layer):
    """Detect chunk boundaries from the similarity of adjacent hidden states.

    The layer projects each position and its successor through two separate
    weight matrices, normalizes both, and turns their cosine similarity into a
    boundary probability. Position 0 is prepended at probability 1.0 so every
    row starts a chunk. The probability feeds three outputs: the two-class
    distribution, the hard boolean mask, and the probability of whichever class
    won.

    Architecture:

    .. code-block:: text

                 hidden_states [B, L, D]
                                │
                   ┌────────────┴────────────┐
                   ▼                         ▼
         ┌───────────────────┐     ┌───────────────────┐
         │ q_proj(h[:, :-1]) │     │ k_proj(h[:, 1:])  │
         │ identity init     │     │ identity init     │
         └─────────┬─────────┘     └─────────┬─────────┘
                   ▼                         ▼
         ┌───────────────────┐     ┌───────────────────┐
         │ L2 normalize      │     │ L2 normalize      │
         └─────────┬─────────┘     └─────────┬─────────┘
                   └────────────┬────────────┘
                                ▼
                      ┌───────────────────┐
                      │ cosine similarity │
                      └─────────┬─────────┘
                             [B, L-1]
                                ▼
                      ┌───────────────────┐
                      │ p = (1 - cos) / 2 │
                      │  clipped to [0, 1]│
                      └─────────┬─────────┘
                                ▼
                      ┌───────────────────┐
                      │ prepend 1.0 at    │
                      │  position 0       │
                      └─────────┬─────────┘
                             p [B, L]
                                │
                   ┌────────────┴────────────┐
                   ▼                         ▼
         ┌───────────────────┐     ┌───────────────────┐
         │ stack [1-p, p]    │     │ p > 0.5           │
         └─────────┬─────────┘     └─────────┬─────────┘
                   ▼                         │
        boundary_prob [B, L, 2]              │
                                    is_boundary [B, L]
                                             │
                                   ┌─────────┴──────────┐
                                   ▼                    ▼
                          ┌────────────────┐   ┌────────────────┐
                          │ and mask       │   │ where(p, 1 - p)│
                          │ (optional)     │   │                │
                          └───────┬────────┘   └───────┬────────┘
                                  ▼                    ▼
                 boundary_mask [B, L] selected_probs [B, L, 1]

    ``mask`` enters only at the ``and``, so the other two outputs are unmasked.

    The threshold is the strict ``p > 0.5``. The reference takes
    ``argmax([1 - p, p])`` (``dc.py:104-106``), which resolves a tie to the first
    index, so a position at exactly ``p = 0.5`` is not a boundary. That tie is
    reachable bit-exactly: any two orthogonal adjacent hidden states give
    ``cos = 0`` and hence ``p = 0.5``.

    :param d_model: Hidden width ``D``. Both projections are ``(D, D)``, and the
        last axis of the input must equal it.
    :type d_model: int
    :param kwargs: Additional :class:`keras.layers.Layer` arguments.

    :raises ValueError: If ``d_model`` is not a positive integer.

    Example:
        >>> import keras, numpy as np
        >>> layer = RoutingModule(d_model=8)
        >>> h = keras.ops.convert_to_tensor(np.random.randn(2, 16, 8).astype("float32"))
        >>> mask = keras.ops.ones((2, 16), dtype="bool")
        >>> boundary_prob, boundary_mask, selected_probs = layer(h, mask=mask)
        >>> boundary_prob.shape, boundary_mask.shape, selected_probs.shape
        ((2, 16, 2), (2, 16), (2, 16, 1))
    """

    def __init__(self, d_model: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if not isinstance(d_model, int) or isinstance(d_model, bool):
            raise ValueError(f"d_model must be an int, got {type(d_model).__name__}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        self.d_model = d_model

        # Identity init makes the untrained layer compute raw adjacent cosine
        # similarity (dc.py:53-59). Use an initializer, not a post-build assign.
        self.q_proj = keras.layers.Dense(
            d_model,
            use_bias=False,
            kernel_initializer=keras.initializers.Identity(),
            name="q_proj",
        )
        self.k_proj = keras.layers.Dense(
            d_model,
            use_bias=False,
            kernel_initializer=keras.initializers.Identity(),
            name="k_proj",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Materialize both projections.

        The identity weights come from ``keras.initializers.Identity``, so there
        is nothing to assign here. An ``.assign()`` after ``add_weight`` or
        ``build`` would be discarded whenever this layer is first reached from a
        parent's ``call()``, because Keras 3 runs that build pass inside a
        ``StatelessScope`` that records the write and drops it.

        :param input_shape: Shape of ``hidden_states``, ``(B, L, D)``.
        :type input_shape: tuple

        :raises ValueError: If the input is not rank 3, or its last axis is not
            ``d_model``.
        """
        if len(input_shape) != 3:
            raise ValueError(
                f"RoutingModule expects rank-3 (batch, seq_len, d_model) input, "
                f"got shape {input_shape}"
            )
        if input_shape[-1] is not None and int(input_shape[-1]) != self.d_model:
            raise ValueError(
                f"Input last axis must equal d_model={self.d_model}, "
                f"got {input_shape[-1]} (shape {input_shape})"
            )

        self.q_proj.build(input_shape)
        self.k_proj.build(input_shape)

        super().build(input_shape)

    def _normalize(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """L2-normalize along the last axis with PyTorch's clamped denominator.

        ``torch.nn.functional.normalize`` divides by ``max(||x||_2, eps)``. The
        clamp, rather than an added epsilon, maps an all-zero row to exactly
        zero instead of to NaN, and padded positions do produce all-zero rows
        once an upstream layer masks them.

        :param x: Tensor whose last axis is normalized.
        :type x: keras.KerasTensor
        :return: Tensor of the same shape and dtype.
        :rtype: keras.KerasTensor
        """
        norm = keras.ops.sqrt(
            keras.ops.sum(keras.ops.square(x), axis=-1, keepdims=True)
        )
        eps = keras.ops.cast(NORMALIZE_EPS, x.dtype)
        return x / keras.ops.maximum(norm, eps)

    def call(
        self,
        hidden_states: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Compute boundary probabilities, the hard boundary mask and its probability.

        :param hidden_states: ``(B, L, D)`` full-resolution hidden states.
        :type hidden_states: keras.KerasTensor
        :param mask: ``(B, L)`` validity mask, where ``True`` or nonzero marks a
            real token. Padded positions are forced to non-boundary. ``None``
            means every position is valid.
        :type mask: keras.KerasTensor or None
        :param training: Unused. Accepted so the layer composes with parents
            that pass it through.
        :type training: bool or None
        :return: ``(boundary_prob (B, L, 2), boundary_mask (B, L) bool,
            selected_probs (B, L, 1))``.
        :rtype: tuple
        """
        del training

        # q reads h_t, k reads h_{t+1} (dc.py:86-90). Swapping them is
        # bit-identical under the identity init; a non-identity test guards it.
        q_proj = self._normalize(self.q_proj(hidden_states[:, :-1]))
        k_proj = self._normalize(self.k_proj(hidden_states[:, 1:]))
        cos_sim = keras.ops.sum(q_proj * k_proj, axis=-1)

        # Keeps (1 - cos) / 2 inside [0, 1] when rounding pushes it out
        # (dc.py:92).
        one = keras.ops.cast(1.0, cos_sim.dtype)
        two = keras.ops.cast(2.0, cos_sim.dtype)
        boundary_prob = keras.ops.clip(
            (one - cos_sim) / two,
            keras.ops.cast(0.0, cos_sim.dtype),
            one,
        )

        # Position 0 of every row is forced to 1.0 (dc.py:95-96).
        batch_size = keras.ops.shape(hidden_states)[0]
        pad = keras.ops.full(
            (batch_size, 1),
            FORCED_FIRST_BOUNDARY_PROB,
            dtype=boundary_prob.dtype,
        )
        boundary_prob = keras.ops.concatenate([pad, boundary_prob], axis=1)

        # The 2-class distribution [1 - p, p] (dc.py:102).
        boundary_prob_2class = keras.ops.stack(
            [one - boundary_prob, boundary_prob], axis=-1
        )

        # DECISION D-012: keep the strict p > 0.5; argmax tie behaviour is
        # backend-defined and p >= 0.5 is a different layer. See decisions.md.
        half = keras.ops.cast(0.5, boundary_prob.dtype)
        is_boundary = keras.ops.greater(boundary_prob, half)

        # The probability of the winning class, read before masking, so a padded
        # position still reports its own winner (dc.py:130-132).
        selected_probs = keras.ops.expand_dims(
            keras.ops.where(is_boundary, boundary_prob, one - boundary_prob),
            axis=-1,
        )

        # No invalid token may be selected (dc.py:107-109).
        boundary_mask = is_boundary
        if mask is not None:
            boundary_mask = keras.ops.logical_and(
                boundary_mask, keras.ops.cast(mask, "bool")
            )

        return boundary_prob_2class, boundary_mask, selected_probs

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], ...]:
        """Return the shapes of the three outputs.

        :param input_shape: ``(B, L, D)``.
        :type input_shape: tuple
        :return: ``((B, L, 2), (B, L), (B, L, 1))``.
        :rtype: tuple
        """
        batch_size, seq_len = input_shape[0], input_shape[1]
        return (
            (batch_size, seq_len, 2),
            (batch_size, seq_len),
            (batch_size, seq_len, 1),
        )

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument.

        :return: Serializable configuration dictionary.
        :rtype: dict
        """
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config

# ---------------------------------------------------------------------
