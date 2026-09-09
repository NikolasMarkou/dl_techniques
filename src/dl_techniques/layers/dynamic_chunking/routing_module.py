"""H-Net's routing module: learned adjacent-token boundary detection.

``RoutingModule`` is the first of H-Net's three dynamic-chunking layers. It reads
a full-resolution sequence ``(B, L, D)`` and decides, for every position, whether
a new chunk begins there. The decision is a cosine dissimilarity between two
learned projections of *adjacent* hidden states::

    p_{t+1} = clip((1 - cos(q(h_t), k(h_{t+1}))) / 2, 0, 1)

so a position whose predecessor points in a different direction gets a high
boundary probability. Position 0 is forced to ``p = 1.0``: every sequence starts
a chunk. Both projections are initialised to the identity, which makes the layer
compute *raw* adjacent cosine similarity at step 0 and lets it learn away from
that as training proceeds.

The layer is a faithful port of the padded/masked branch of
``RoutingModule.forward`` in the reference implementation
(``hnet/modules/dc.py:69-138``, vendored for the test suite under
``tests/test_layers/test_dynamic_chunking/_reference/``). The packed
(``cu_seqlens``) branch and the ``inference_params`` cache path are deliberately
not ported -- see the plan decision D-007 recorded in
``plans/plan-2026-09-09T042752-6d66ac56/decisions.md``.

Only ``selected_probs`` is differentiable with respect to the projections; the
boolean ``boundary_mask`` is a hard decision with no gradient, and the downstream
stack recovers a gradient path through the straight-through residual gate and
through the ratio loss, not through the mask.

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

# ---------------------------------------------------------------------

#: ``torch.nn.functional.normalize`` default epsilon (``dc.py:88-89``). The
#: denominator is ``max(||x||_2, eps)``, NOT ``||x||_2 + eps`` -- the clamp form
#: sends an all-zero row (which a padding mask produces) to exactly zero rather
#: than to NaN.
NORMALIZE_EPS = 1e-12

#: ``PAD_PROB`` -- the forced boundary probability of position 0 (``dc.py:95-96``).
FORCED_FIRST_BOUNDARY_PROB = 1.0


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.dynamic_chunking.routing_module")
class RoutingModule(keras.layers.Layer):
    """Boundary-detection head of H-Net's dynamic chunking.

    Architecture:

    .. code-block:: text

        hidden_states (B, L, D)          mask (B, L)
              |                                |
              +---------------+                |
              |               |                |
        h[:, :-1]          h[:, 1:]            |
              |               |                |
          q_proj (D->D)   k_proj (D->D)        |
          identity init   identity init        |
              |               |                |
          L2 normalize    L2 normalize         |
              \\             /                 |
               cosine similarity (B, L-1)      |
                      |                        |
             p = clip((1 - cos)/2, 0, 1)       |
                      |                        |
            left-pad position 0 with 1.0       |
                      |                        |
                  p  (B, L)                    |
                 /        \\                   |
       [1-p, p] (B,L,2)   p > 0.5 ------- logical_and
                 |                             |
          boundary_prob             boundary_mask (B, L)
                 |
          selected_probs (B, L, 1) = p where boundary else 1-p

    The threshold is **strictly** ``p > 0.5``. The reference takes
    ``argmax([1 - p, p])`` (``dc.py:104-106``) and ``argmax`` resolves a tie to
    the first index, so a position at exactly ``p = 0.5`` is NOT a boundary. That
    tie is reachable bit-exactly -- any two orthogonal adjacent hidden states
    give ``cos = 0`` and hence ``p = 0.5``.

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

        # The identity initialisation is load-bearing, not cosmetic: it makes the
        # untrained layer compute raw adjacent cosine similarity, which is the
        # signal the whole chunking mechanism bootstraps from (`dc.py:53-59`,
        # where the reference copies `torch.eye(d_model)` into both weights and
        # flags them `_no_reinit`). It is expressed as an INITIALIZER rather than
        # as a post-build `.assign()` on purpose -- see the note in `build`.
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

        The identity weights are produced by ``keras.initializers.Identity``, so
        there is nothing to assign here. Writing them with ``.assign()`` after
        ``add_weight``/``build`` would be silently discarded whenever this layer
        is first reached from a parent's ``call()``: Keras 3 runs that build pass
        inside a ``StatelessScope``, which records the assignment and throws it
        away, leaving the projections at their initializer value in every real
        model while every direct-``build`` unit test still passes.

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
        clamp, rather than an added epsilon, is what maps an all-zero row to
        exactly zero instead of to NaN, and padded positions do produce all-zero
        rows once an upstream layer masks them.

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
        :param mask: ``(B, L)`` validity mask, ``True``/nonzero = a real token.
            Padded positions are forced to non-boundary. ``None`` means every
            position is valid.
        :type mask: keras.KerasTensor or None
        :param training: Unused -- this layer has no training-mode behaviour. It
            is accepted so the layer composes with parents that pass it through.
        :type training: bool or None
        :return: ``(boundary_prob (B, L, 2), boundary_mask (B, L) bool,
            selected_probs (B, L, 1))``.
        :rtype: tuple
        """
        del training  # no dropout, no norm statistics: nothing is mode-dependent

        # dc.py:86-90 -- q reads h_t, k reads h_{t+1}. The direction is the whole
        # meaning of the layer and a mirrored version of it is bit-identical
        # under the identity init, so it is guarded by a NON-identity-projection
        # test rather than by a parity test alone.
        q_proj = self._normalize(self.q_proj(hidden_states[:, :-1]))
        k_proj = self._normalize(self.k_proj(hidden_states[:, 1:]))
        cos_sim = keras.ops.sum(q_proj * k_proj, axis=-1)  # (B, L-1)

        # dc.py:92 -- a no-op absent precision issues, kept because precision
        # issues are exactly what it is there for.
        one = keras.ops.cast(1.0, cos_sim.dtype)
        two = keras.ops.cast(2.0, cos_sim.dtype)
        boundary_prob = keras.ops.clip(
            (one - cos_sim) / two,
            keras.ops.cast(0.0, cos_sim.dtype),
            one,
        )

        # dc.py:95-96 -- position 0 of every row is forced to PAD_PROB = 1.0.
        batch_size = keras.ops.shape(hidden_states)[0]
        pad = keras.ops.full(
            (batch_size, 1),
            FORCED_FIRST_BOUNDARY_PROB,
            dtype=boundary_prob.dtype,
        )
        boundary_prob = keras.ops.concatenate([pad, boundary_prob], axis=1)  # (B, L)

        # dc.py:102 -- the 2-class distribution [1 - p, p].
        boundary_prob_2class = keras.ops.stack(
            [one - boundary_prob, boundary_prob], axis=-1
        )  # (B, L, 2)

        # DECISION plan-2026-09-09T042752-6d66ac56/D-012: the hard decision is
        # written as the STRICT comparison `p > 0.5`, not as
        # `keras.ops.argmax(boundary_prob_2class, axis=-1)`, even though the
        # reference is literally an argmax (`dc.py:104-106`). Do NOT "restore
        # fidelity" by swapping in argmax, and do NOT relax this to `>=`:
        #   * `tf.math.argmax` documents that "in case of ties the identity of
        #     the return value is not guaranteed", so the tie behaviour this
        #     layer depends on is not a promise the backend makes -- whereas
        #     `p > 0.5` breaks the tie low by construction, on every backend;
        #   * `p >= 0.5` is a DIFFERENT layer. `argmax([1-p, p])` returns the
        #     first maximal index, so exactly `p = 0.5` is NOT a boundary, and
        #     `p = 0.5` is reachable bit-exactly whenever two adjacent hidden
        #     states are orthogonal (`cos = 0`).
        # `p > 0.5` and `p > 1 - p` agree everywhere under round-to-nearest, so
        # this is the argmax's realised predicate, not an approximation of it.
        half = keras.ops.cast(0.5, boundary_prob.dtype)
        is_boundary = keras.ops.greater(boundary_prob, half)  # (B, L) bool

        # dc.py:130-132 -- the probability of whichever class WON, gathered from
        # the UNMASKED decision: a padded position still reports its own winning
        # class even though its boundary_mask entry is about to be forced False.
        selected_probs = keras.ops.expand_dims(
            keras.ops.where(is_boundary, boundary_prob, one - boundary_prob),
            axis=-1,
        )  # (B, L, 1)

        # dc.py:107-109 -- no invalid token may be selected.
        boundary_mask = is_boundary
        if mask is not None:
            boundary_mask = keras.ops.logical_and(
                boundary_mask, keras.ops.cast(mask, "bool")
            )

        return boundary_prob_2class, boundary_mask, selected_probs

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], ...]:
        """Shapes of the three outputs, from stored config alone.

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
