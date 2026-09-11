"""Additively embed a per-sample present/absent modality flag into a token
sequence with one of two learned ``(1, 1, dim)`` vectors.

This is OmniPoint's input-state embedding (plan.md D-005): unlike
:class:`~dl_techniques.layers.embedding.mask_token.MaskTokenApply` (which
*replaces* a per-position token with a learned vector, keyed on a per-position
boolean mask), this layer *adds* one of two learned vectors to every token of
a sample, keyed on a single per-sample boolean flag. The two operations have
different semantics AND different gradient behavior: a replaced position gets
zero gradient into the original token; an added embedding does not, and a
per-sample decision applies uniformly across the whole sequence rather than
per position. See ``decisions.md`` D-005 for why this is a new layer rather
than a reuse of ``MaskTokenApply``.

Architecture:

.. code-block:: text

    tokens X  [B, N, D]          flag F  [B] or [B, 1]  (bool)
        │                              │
        │                    expand -> [B, 1, 1] (broadcastable)
        │                              │
        │      present_embedding [1,1,D]   absent_embedding [1,1,D]
        │                (both weights)
        │                              │
        └───────────► where(F, present_embedding, absent_embedding) ◄────┘
                              │
                              ▼
                    selected  [B, 1, D]  (broadcasts over N)
                              │
                              ▼
                       Y = X + selected   [B, N, D]

Mathematics:
    Let ``X in R^{B x N x D}``, ``f in {0,1}^{B}`` and ``p, a in R^{1 x 1 x D}``
    be the learnable present/absent vectors. With ``f' = reshape(f, (B, 1, 1))``
    broadcast over ``N`` and ``D``::

        Y = X + where(f', p, a)   in R^{B x N x D}
"""

from typing import Any, Dict, Optional, Tuple

import keras

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.embedding.state_indicator_embedding")
class StateIndicatorEmbedding(keras.layers.Layer):
    """Add a learned present/absent vector to every token of a sample.

    Call the layer on a pair: ``layer((tokens, flag))``, where ``tokens`` is
    ``(B, N, D)`` and ``flag`` is a boolean (or 0/1) tensor of shape ``(B,)``
    or ``(B, 1)``. ``True`` (or nonzero) selects ``present_embedding`` for
    that sample; ``False`` (or zero) selects ``absent_embedding``. Every
    token of a given sample receives the same selected vector, additively;
    the sequence length and feature dimension are unchanged.

    :param initializer: Initializer for both learned vectors. The default
        string ``"truncated_normal"`` is replaced by
        ``TruncatedNormal(stddev=0.02)``, matching
        :class:`~dl_techniques.layers.embedding.mask_token.MaskTokenApply`'s
        DINO/ViT/iBOT-style token initialization convention.
    :type initializer: str or keras.initializers.Initializer
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar initializer: The resolved initializer object.
    :vartype initializer: keras.initializers.Initializer
    :ivar present_embedding: Learned ``(1, 1, dim)`` vector added when
        ``flag`` is ``True``. ``None`` until ``build()`` runs.
    :vartype present_embedding: keras.Variable or None
    :ivar absent_embedding: Learned ``(1, 1, dim)`` vector added when
        ``flag`` is ``False``. ``None`` until ``build()`` runs.
    :vartype absent_embedding: keras.Variable or None

    :raises ValueError: From ``build()``, if the input is not a pair of
        shapes, if the token sequence is not rank 3, or if its last
        dimension is not statically known.

    Input shape:
        Tuple or list of two tensors:

        - ``tokens``: 3D ``(batch_size, sequence_length, dim)``.
        - ``flag``: 1D ``(batch_size,)`` or 2D ``(batch_size, 1)``, boolean
          (or 0/1-valued).

    Output shape:
        3D tensor with shape ``(batch_size, sequence_length, dim)``, the same
        shape as ``tokens``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.embedding.state_indicator_embedding import (
            StateIndicatorEmbedding,
        )

        x = keras.random.normal((4, 196, 384))
        present = keras.ops.convert_to_tensor([True, False, True, False])
        y = StateIndicatorEmbedding()((x, present))
        y.shape  # (4, 196, 384)
    """

    def __init__(
            self,
            initializer: Any = "truncated_normal",
            **kwargs: Any,
    ) -> None:
        """Resolve the initializer. No weight is created here.

        :param initializer: Initializer for both learned vectors. The bare
            string default is upgraded to ``TruncatedNormal(stddev=0.02)``.
        :type initializer: str or keras.initializers.Initializer
        :param kwargs: Additional keyword arguments for the ``Layer`` base
            class.
        :type kwargs: Any
        """
        super().__init__(**kwargs)
        if initializer == "truncated_normal":
            initializer = keras.initializers.TruncatedNormal(stddev=0.02)
        self.initializer = keras.initializers.get(initializer)
        self.present_embedding = None
        self.absent_embedding = None

    def build(self, input_shape: Tuple) -> None:
        """Create the two ``(1, 1, dim)`` learned vectors.

        :param input_shape: A list or tuple of two shapes, in the order
            ``[tokens_shape, flag_shape]``.
        :type input_shape: list of tuple
        :raises ValueError: If the layer did not receive exactly two inputs,
            if the tokens are not rank 3, or if their last axis is ``None``.
        """
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                "StateIndicatorEmbedding expects two inputs (tokens, flag); "
                f"got input_shape={input_shape}"
            )
        tokens_shape = input_shape[0]
        if len(tokens_shape) != 3:
            raise ValueError(
                "StateIndicatorEmbedding expects 3D tokens (batch, seq, dim); "
                f"got tokens shape={tokens_shape}"
            )
        dim = tokens_shape[-1]
        if dim is None:
            raise ValueError(
                "StateIndicatorEmbedding requires a static feature dimension "
                "(tokens shape[-1] must be known)."
            )
        # Weights are created here, in build, after super().__init__ -- the
        # same reason MaskTokenApply defers its own weight creation, so a
        # host Functional Model never runs add_weight before its own
        # super().__init__(inputs=, outputs=).
        #
        # DECISION plan-2026-09-11T050223-1b47bcf6/D-014: two SEPARATE
        # initializer instances, not `self.initializer` passed twice.
        # MEASURED: calling the same `keras.initializers.TruncatedNormal`
        # instance's `__call__` twice with the same shape returns BIT-IDENTICAL
        # values (its RNG state does not advance across separate `__call__`
        # invocations on one instance) -- passing `self.initializer` to both
        # `add_weight` calls below would silently initialize
        # `present_embedding` and `absent_embedding` to the exact same vector,
        # making the two states indistinguishable until gradients (which
        # start symmetric, since both feed the same downstream computation)
        # eventually break the tie. Re-`keras.initializers.get`-ing a fresh
        # instance from the SAME config for the second weight avoids this
        # while keeping both initializers configured identically. See
        # decisions.md.
        self.present_embedding = self.add_weight(
            shape=(1, 1, dim),
            initializer=self.initializer,
            trainable=True,
            name="present_embedding",
        )
        absent_initializer = keras.initializers.get(
            keras.initializers.serialize(self.initializer)
        )
        self.absent_embedding = self.add_weight(
            shape=(1, 1, dim),
            initializer=absent_initializer,
            trainable=True,
            name="absent_embedding",
        )
        super().build(input_shape)

    def call(
            self,
            inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Add the flag-selected vector to every token of each sample.

        :param inputs: The pair ``(tokens, flag)``, shapes ``(B, N, D)`` and
            ``(B,)`` or ``(B, 1)``.
        :type inputs: tuple of keras.KerasTensor
        :param training: Unused; present for interface consistency with
            other layers in this package.
        :type training: Optional[bool]
        :return: ``tokens`` plus the selected embedding, broadcast over the
            sequence axis, shape ``(B, N, D)``.
        :rtype: keras.KerasTensor
        """
        tokens, flag = inputs
        flag = keras.ops.cast(flag, "bool")
        # Reshape (B,) or (B, 1) -> (B, 1, 1) so it broadcasts against both
        # the (1, 1, D) embeddings and the (B, N, D) token sequence.
        flag = keras.ops.reshape(flag, (-1, 1, 1))
        selected = keras.ops.where(flag, self.present_embedding, self.absent_embedding)
        return tokens + selected

    def compute_output_shape(
            self,
            input_shape: Tuple,
    ) -> Tuple[Optional[int], ...]:
        """Return the tokens shape, unchanged.

        :param input_shape: The pair of input shapes.
        :type input_shape: list of tuple
        :return: ``input_shape[0]`` as a tuple.
        :rtype: tuple
        """
        return tuple(input_shape[0])

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: The base ``Layer`` config plus the serialized
            ``initializer``.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            "initializer": keras.initializers.serialize(self.initializer),
        })
        return config

# ---------------------------------------------------------------------
