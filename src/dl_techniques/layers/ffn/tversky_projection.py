"""
A Dense-like projection scored by Tversky's contrast model of similarity.

A Dense layer scores an input against each of its output units with a dot
product, which is symmetric. This layer scores it with a differentiable form
of Tversky's contrast model instead, which is not. Asymmetry is the point: it
lets the layer say that `a` resembles `b` more than `b` resembles `a`.

**This layer takes rank-2 input only**, shape `(batch_size, input_dim)`.
`build()` raises `ValueError` on any other rank. Wrap it in `TimeDistributed`
to apply it per token or per pixel.

Three weight groups
-------------------
- `prototypes`, shape `(units, input_dim)`. One per output unit. This is the
  analogue of a Dense kernel.
- `feature_bank`, shape `(num_features, input_dim)`. A learned set of feature
  directions, shared by inputs and prototypes.
- `theta`, `alpha`, `beta`. Three learnable scalars.

A vector "has" feature `i` when its dot product with `feature_bank[i]` is
positive. That dot product is also the feature's salience for that vector.
This is the bridge from continuous vectors to Tversky's set logic, and it is
what makes the whole thing differentiable.

The score
---------
For input `a` and prototype `p`, with `A` and `B` the feature sets they have:

    S(a, p) = theta * f(A n B) - alpha * f(A - B) - beta * f(B - A)

`f(A n B)` sums a per-feature combination of the two saliences over the
features both have. `intersection_reduction` picks the combination: the
product, the minimum, or the mean.

Both reduction names are checked by `create_ffn_layer('tversky', ...)`, which
raises `ValueError` on a bad one. Constructing the class directly skips that
check: a bad name survives `__init__` and `get_config()` and surfaces as a
`NotImplementedError` from `call()`.

`f(A - B)` and `f(B - A)` depend on `difference_reduction`, and the two
settings do not measure the same thing:

- `'ignorematch'` is the literal set difference. `f(A - B)` sums `a`'s
  salience over the features `a` has and `p` does not.
- `'subtractmatch'`, the DEFAULT, does not look at one-sided features at all.
  It sums the salience GAP over the features both have: `f(A - B)` sums
  `sal_a - sal_p` over the common features where `a` scores higher.

Read the class docstring's block-internals diagram for the exact expressions.
All three `f(.)` terms are non-negative; the sign of the total comes from the
three learnable scalars, which are unconstrained.

References:
    - Tversky, A. (1977). Features of similarity. Psychological Review.
    - Doumbouya, M. K. B., et al. (2025). Tversky Neural Networks. arXiv.
"""

import keras
from typing import Optional, Union, Tuple, Dict, Any, Literal

from dl_techniques.initializers.clone import clone_initializer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TverskyProjectionLayer(keras.layers.Layer):
    """
    Projection layer scored by a differentiable Tversky similarity model.

    Instead of a dot product, this layer scores the input against each of its
    ``units`` learned prototypes with Tversky's contrast model:
    ``S(a, b) = theta * f(A n B) - alpha * f(A - B) - beta * f(B - A)``.
    A vector has feature ``i`` when its dot product with ``feature_bank[i]``
    is positive, and that dot product is the feature's salience. Prototypes,
    feature bank and the three scalars are all learned.

    .. warning::
        This layer takes **rank-2** input only, shape
        ``(batch_size, input_dim)``. ``build()`` raises ``ValueError`` for
        every other rank, including rank 3. Wrap it in ``TimeDistributed`` to
        apply it per token or per pixel.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────────────┐
        │  Input [B, D]   (rank 2 ONLY)          │
        └──────────────┬─────────────────────────┘
                       ▼
        ┌────────────────────────────────────────┐
        │  Feature salience, two matmuls         │
        │  input_dots = Input @ feat_bank^T      │
        │                          -> [B, NF]    │
        │  proto_dots = prototypes @ feat_bank^T │
        │                          -> [U, NF]    │
        └──────────────┬─────────────────────────┘
                       ▼
        ┌────────────────────────────────────────┐
        │  Broadcast to [B, U, NF] and reduce    │
        │  over NF: f(AnB), f(A-B), f(B-A)       │
        │                          -> [B, U]     │
        └──────────────┬─────────────────────────┘
                       ▼
        ┌────────────────────────────────────────┐
        │  Tversky contrast, scalar weights      │
        │  S = θ*f(AnB) - α*f(A-B) - β*f(B-A)    │
        └──────────────┬─────────────────────────┘
                       ▼
        ┌────────────────────────────────────────┐
        │  Output [B, U]                         │
        └────────────────────────────────────────┘

        B = batch, D = input_dim, U = units,
        NF = num_features.

    **Intersection and difference reduction (block internals):**

    .. code-block:: text

        a = input_dots[b, :]   one input row,     [NF]
        p = proto_dots[u, :]   one prototype row, [NF]
        in_A = a > 0           the input HAS the feature
        in_B = p > 0           the prototype HAS the feature
        both = in_A AND in_B

        intersection_reduction, summed over `both`:
            'product'  f(A n B) = sum a * p
            'min'      f(A n B) = sum min(a, p)
            'mean'     f(A n B) = sum (a + p) / 2

        difference_reduction = 'ignorematch':
            f(A - B) = sum over (in_A AND NOT in_B) of a
            f(B - A) = sum over (in_B AND NOT in_A) of p

        difference_reduction = 'subtractmatch'  (DEFAULT):
            f(A - B) = sum over (both AND a > p) of (a - p)
            f(B - A) = sum over (both AND p > a) of (p - a)

        S[b, u] = θ*f(AnB) - α*f(A-B) - β*f(B-A)

        The two difference leaves read DIFFERENT feature sets.
        'ignorematch' scores features only one side has.
        'subtractmatch' ignores those and scores the gap on
        features both sides have. Every f(.) is a sum of
        positive terms, so all three are >= 0; θ, α and β are
        unconstrained, so S can have either sign.

    :param units: Dimensionality of the output space (number of prototypes). Must be positive.
    :type units: int
    :param num_features: Size of the learnable feature universe. Must be positive.
    :type num_features: int
    :param intersection_reduction: How the two saliences are combined on a
        shared feature. One of ``'product'``, ``'min'``, ``'mean'``. NOT
        checked by ``__init__``; a wrong value constructs and serializes fine
        and only raises ``NotImplementedError`` when ``call()`` runs. Build
        the layer through ``create_ffn_layer('tversky', ...)`` to get a
        ``ValueError`` up front instead. Defaults to ``'product'``.
    :type intersection_reduction: str
    :param difference_reduction: Which features the two difference terms
        measure. One of ``'ignorematch'``, ``'subtractmatch'``. Same rule as
        above: unchecked here, checked by the factory. Defaults to
        ``'subtractmatch'``.
    :type difference_reduction: str
    :param prototype_initializer: Initializer for the prototype matrix.
        Defaults to ``'glorot_uniform'``.
    :type prototype_initializer: str or keras.initializers.Initializer
    :param feature_initializer: Initializer for the feature bank. Defaults to
        ``'glorot_uniform'``.
    :type feature_initializer: str or keras.initializers.Initializer
    :param contrast_initializer: Initializer for the three scalars ``theta``,
        ``alpha`` and ``beta``. Each scalar is drawn from its own clone of
        it, so a random initializer gives three different starting values;
        under the ``'ones'`` default all three are 1.0 either way. Defaults
        to ``'ones'``.
    :type contrast_initializer: str or keras.initializers.Initializer
    :param kwargs: Additional arguments for Layer base class (``name``,
        ``dtype``, and so on).
    :type kwargs: Any

    :ivar units: The stored output width, i.e. the number of prototypes.
    :vartype units: int
    :ivar num_features: The stored feature-bank size.
    :vartype num_features: int
    :ivar intersection_reduction: The stored reduction name, unvalidated.
    :vartype intersection_reduction: str
    :ivar difference_reduction: The stored reduction name, unvalidated.
    :vartype difference_reduction: str
    :ivar prototype_initializer: The resolved prototype initializer.
    :vartype prototype_initializer: keras.initializers.Initializer
    :ivar feature_initializer: The resolved feature-bank initializer.
    :vartype feature_initializer: keras.initializers.Initializer
    :ivar contrast_initializer: The resolved initializer for the scalars. It
        is the source the three per-scalar clones are rebuilt from, and is
        not handed to ``add_weight`` itself.
    :vartype contrast_initializer: keras.initializers.Initializer
    :ivar prototypes: Weight of shape ``(units, input_dim)``. ``None`` until
        ``build()`` runs.
    :vartype prototypes: Optional[keras.Variable]
    :ivar feature_bank: Weight of shape ``(num_features, input_dim)``.
        ``None`` until ``build()`` runs.
    :vartype feature_bank: Optional[keras.Variable]
    :ivar theta: Scalar weight on the common-feature term. ``None`` until
        ``build()`` runs.
    :vartype theta: Optional[keras.Variable]
    :ivar alpha: Scalar weight on ``f(A - B)``. ``None`` until ``build()``.
    :vartype alpha: Optional[keras.Variable]
    :ivar beta: Scalar weight on ``f(B - A)``. ``None`` until ``build()``.
    :vartype beta: Optional[keras.Variable]

    :raises ValueError: If ``units`` is not positive.
    :raises ValueError: If ``num_features`` is not positive.
    :raises ValueError: If the input to ``build()`` is not rank 2.
    :raises ValueError: If the last input dimension is ``None`` at build time.
    :raises NotImplementedError: From ``call()``, if either reduction name is
        not one of the values listed above.

    Input shape:
        2D tensor of shape ``(batch_size, input_dim)``. No other rank works.

    Output shape:
        2D tensor of shape ``(batch_size, units)``.

    Example:
        .. code-block:: python

            layer = TverskyProjectionLayer(units=10, num_features=32)
            y = layer(keras.random.normal((4, 16)))
            y.shape                 # (4, 10)

            # For a sequence, apply it per token.
            td = keras.layers.TimeDistributed(
                TverskyProjectionLayer(units=10, num_features=32)
            )
            td(keras.random.normal((4, 7, 16))).shape   # (4, 7, 10)

    Note:
        The intermediate tensor has shape ``(batch, units, num_features)``.
        Memory grows with the product of all three, so a large
        ``num_features`` is expensive at a large batch size.
    """

    def __init__(
        self,
        units: int,
        num_features: int,
        intersection_reduction: Literal['product', 'min', 'mean'] = 'product',
        difference_reduction: Literal['ignorematch', 'subtractmatch'] = 'subtractmatch',
        prototype_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        feature_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        contrast_initializer: Union[str, keras.initializers.Initializer] = 'ones',
        **kwargs: Any
    ) -> None:
        """
        Validate the two sizes and store the configuration.

        Every argument is documented on the class. No weight exists yet; all
        five weight attributes are set to ``None`` and created in ``build()``.
        The two reduction names are NOT validated here. A wrong one is caught
        by ``validate_ffn_config`` when the layer is built through the
        factory, and otherwise only by ``call()``.

        :raises ValueError: If ``units`` or ``num_features`` is not positive.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if units <= 0:
            raise ValueError(f"`units` must be positive, got {units}")
        if num_features <= 0:
            raise ValueError(f"`num_features` must be positive, got {num_features}")

        # Store ALL configuration for serialization
        self.units = units
        self.num_features = num_features
        self.intersection_reduction = intersection_reduction
        self.difference_reduction = difference_reduction
        self.prototype_initializer = keras.initializers.get(prototype_initializer)
        self.feature_initializer = keras.initializers.get(feature_initializer)
        self.contrast_initializer = keras.initializers.get(contrast_initializer)

        # Initialize weight attributes - they will be created in build()
        self.prototypes = None
        self.feature_bank = None
        self.theta = None
        self.alpha = None
        self.beta = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the prototypes, the feature bank and the three scalars.

        Both weight matrices are ``(something, input_dim)``, so the last
        dimension has to be known here.

        :param input_shape: Shape of the input tensor. Must have exactly two
            entries.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not rank 2.
        :raises ValueError: If the last entry of ``input_shape`` is ``None``.
        """
        if self.built:
            return

        # The set-operation broadcasting below is rank-2 only. Fail loud rather
        # than silently produce mismatched-rank broadcasts for higher-rank inputs.
        if len(input_shape) != 2:
            raise ValueError(
                "`TverskyProjectionLayer` operates on rank-2 inputs "
                f"(batch_size, input_dim). Got rank-{len(input_shape)} input "
                f"with shape {input_shape}. Wrap the layer in `TimeDistributed` "
                "to apply it per-token on higher-rank tensors."
            )

        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError(
                "The last dimension of the input to `TverskyProjectionLayer` "
                "must be defined. Found `None`."
            )

        # Create the learnable prototype bank
        self.prototypes = self.add_weight(
            name='prototypes',
            shape=(self.units, input_dim),
            initializer=self.prototype_initializer,
            trainable=True,
        )

        # Create the learnable feature universe
        self.feature_bank = self.add_weight(
            name='feature_bank',
            shape=(self.num_features, input_dim),
            initializer=self.feature_initializer,
            trainable=True,
        )

        # Create the Tversky contrast model scalar parameters.
        # Each takes its OWN clone of contrast_initializer; see the rule and
        # the mechanism at glu_ffn.py, decisions.md D-008. The three are all
        # shape (), so no configuration separates them: with one shared
        # instance a random contrast_initializer gave theta == alpha == beta
        # (MEASURED max|delta| = 0.0 with an unseeded RandomNormal()).
        # prototypes and feature_bank are NOT part of this: they already
        # take two different initializer objects (MEASURED 1.1362).
        self.theta = self.add_weight(
            name='theta',
            shape=(),
            initializer=clone_initializer(self.contrast_initializer),
            trainable=True,
        )
        self.alpha = self.add_weight(
            name='alpha',
            shape=(),
            initializer=clone_initializer(self.contrast_initializer),
            trainable=True,
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(),
            initializer=clone_initializer(self.contrast_initializer),
            trainable=True,
        )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """
        Score every input row against every prototype.

        Runs the two matmuls, broadcasts to ``(batch, units, num_features)``,
        reduces over the feature axis and applies the contrast formula. The
        block-internals diagram on the class gives the exact expressions.

        :param inputs: Rank-2 tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Unused. Accepted so the call signature matches the
            other FFN layers; this layer behaves the same either way.
        :type training: Optional[bool]
        :return: Similarity scores of shape ``(batch_size, units)``.
        :rtype: keras.KerasTensor
        :raises NotImplementedError: If ``intersection_reduction`` is not
            'product', 'min' or 'mean', or if ``difference_reduction`` is not
            'ignorematch' or 'subtractmatch'.
        """
        # Compute dot products to get feature presence scores.
        # inputs shape: (batch_size, input_dim)
        # feature_bank shape: (num_features, input_dim)
        # -> input_dots shape: (batch_size, num_features)
        input_dots = keras.ops.matmul(inputs, keras.ops.transpose(self.feature_bank))

        # prototypes shape: (units, input_dim)
        # -> proto_dots shape: (units, num_features)
        proto_dots = keras.ops.matmul(self.prototypes, keras.ops.transpose(self.feature_bank))

        # Create boolean masks for set operations (feature is present if dot > 0).
        input_mask = input_dots > 0
        proto_mask = proto_dots > 0

        # Reshape for broadcasting:
        # (batch, 1, num_features) vs (1, units, num_features)
        # -> results will have shape: (batch, units, num_features)
        input_dots_b = keras.ops.expand_dims(input_dots, axis=1)
        input_mask_b = keras.ops.expand_dims(input_mask, axis=1)
        proto_dots_b = keras.ops.expand_dims(proto_dots, axis=0)
        proto_mask_b = keras.ops.expand_dims(proto_mask, axis=0)

        # Calculate f(A ∩ B): common features measure.
        common_mask = keras.ops.logical_and(input_mask_b, proto_mask_b)

        if self.intersection_reduction == 'product':
            intersection_scores = input_dots_b * proto_dots_b
        elif self.intersection_reduction == 'min':
            intersection_scores = keras.ops.minimum(input_dots_b, proto_dots_b)
        elif self.intersection_reduction == 'mean':
            intersection_scores = (input_dots_b + proto_dots_b) / 2.0
        else:
            raise NotImplementedError(
                f"Intersection reduction '{self.intersection_reduction}' not implemented."
            )
        f_intersection = keras.ops.sum(
            keras.ops.where(common_mask, intersection_scores, 0.0), axis=-1
        )

        # Calculate f(A - B) and f(B - A): distinctive features measures.
        if self.difference_reduction == 'ignorematch':
            input_distinct_mask = keras.ops.logical_and(input_mask_b, keras.ops.logical_not(proto_mask_b))
            f_input_distinctive = keras.ops.sum(
                keras.ops.where(input_distinct_mask, input_dots_b, 0.0), axis=-1
            )
            proto_distinct_mask = keras.ops.logical_and(proto_mask_b, keras.ops.logical_not(input_mask_b))
            f_proto_distinctive = keras.ops.sum(
                keras.ops.where(proto_distinct_mask, proto_dots_b, 0.0), axis=-1
            )
        elif self.difference_reduction == 'subtractmatch':
            subtract_mask_A = keras.ops.logical_and(common_mask, input_dots_b > proto_dots_b)
            f_input_distinctive = keras.ops.sum(
                keras.ops.where(subtract_mask_A, input_dots_b - proto_dots_b, 0.0), axis=-1
            )
            subtract_mask_B = keras.ops.logical_and(common_mask, proto_dots_b > input_dots_b)
            f_proto_distinctive = keras.ops.sum(
                keras.ops.where(subtract_mask_B, proto_dots_b - input_dots_b, 0.0), axis=-1
            )
        else:
            raise NotImplementedError(
                f"Difference reduction '{self.difference_reduction}' not implemented."
            )

        # Apply Tversky's contrast model formula.
        similarity = (
            self.theta * f_intersection
            - self.alpha * f_input_distinctive
            - self.beta * f_proto_distinctive
        )
        return similarity

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Replace the last dimension of ``input_shape`` with ``units``.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape with the last entry set to ``units``. For the
            rank-2 input this layer accepts that is
            ``(batch_size, units)``.
        :rtype: Tuple[Optional[int], ...]
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return everything ``__init__`` needs to rebuild this layer.

        :return: The base ``Layer`` config plus the two sizes, the two
            reduction names and the three serialized initializers.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'units': self.units,
            'num_features': self.num_features,
            'intersection_reduction': self.intersection_reduction,
            'difference_reduction': self.difference_reduction,
            'prototype_initializer': keras.initializers.serialize(self.prototype_initializer),
            'feature_initializer': keras.initializers.serialize(self.feature_initializer),
            'contrast_initializer': keras.initializers.serialize(self.contrast_initializer),
        })
        return config

# ---------------------------------------------------------------------
