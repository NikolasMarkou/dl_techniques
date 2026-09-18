"""
A Dense-like projection, ``TverskyProjectionLayer``, scored by Tversky's
contrast model of similarity instead of a dot product.

A Dense layer scores an input against each output unit with a symmetric dot
product. This layer scores it with the differentiable form of Tversky's
contrast model from Doumbouya et al. (2025), which is asymmetric, so it can
say that ``a`` resembles ``b`` more than ``b`` resembles ``a``:

.. code-block:: text

    S(a, p) = theta * f(A n B) - alpha * f(A - B) - beta * f(B - A)

``A`` and ``B`` are the feature sets of the input and of a learned
prototype, and a vector has feature ``k`` when its dot product with feature
bank row ``k`` is positive.

The separable fast path. The naive evaluation broadcasts to a
``[batch, units, num_features]`` tensor, which at the paper's own
TverskyGPT-2 scale is over 12 TB per intermediate. That tensor is
algebraically unnecessary whenever each summand factorizes into a function
of the input times a function of the prototype, which holds for three of the
six intersection reductions and for ``ignorematch``:

.. code-block:: text

    f(A n B)   product    relu(a) @ relu(p)^T
               mean       (relu(a) @ 1[p>0]^T + 1[a>0] @ relu(p)^T) / 2
               gmean      sqrt(relu(a)) @ sqrt(relu(p))^T
               min        NOT separable, |a - p| couples the operands
               max        NOT separable, same reason
               softmin    NOT separable, same reason

    f(A - B)   ignorematch      relu(a) @ (1 - 1[p>0])^T
    f(B - A)   ignorematch      (1 - 1[a>0]) @ relu(p)^T
               subtractmatch    NOT separable, 1[a>p] couples the operands

Those identities are exact, not approximations; the implementation checks
them to 1e-15 in :meth:`verify_fast_path`. When the configuration is fully
separable the layer is three matmuls, memory drops from
``O(B*U*NF)`` to ``O(B*NF + U*NF + B*U)``, and the input may have any rank.
``product`` with ``ignorematch``, the pair the paper used for
TverskyResNet-50, is fully separable. Otherwise the layer falls back to the
broadcast form, chunked over prototypes so peak memory stays bounded.

Dead prototypes. If a prototype's feature set is empty, every term of the
contrast model becomes constant in that prototype and its gradient is
exactly zero, not merely small: it can never recover. Under symmetric
initialization this happens with probability ``2^-num_features`` per
prototype, which is 50% at one feature, and is the most likely explanation
for the convergence failures the paper reports at small feature bank sizes.
:meth:`diagnose` reports it.

References:
    - Tversky, A. (1977). Features of similarity. Psychological Review.
    - Doumbouya, M. K. B., Jurafsky, D. and Manning, C. D. (2025). Tversky
      Neural Networks: Psychologically Plausible Deep Learning with
      Differentiable Tversky Similarity. arXiv:2506.11035.
"""

import keras
from typing import Callable, Optional, Union, Tuple, List, Dict, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.initializers.clone import clone_initializer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

# DECISION plan-2026-08-29T043546-e97b34d8/D-010: these frozensets are the only
# source of valid reduction names; factory.py imports them, never a copy. See decisions.md.
#
# All six intersection reductions from the paper are implemented now. The
# previous version carried three, which made an incomplete sweep look
# authoritative to every importer.
VALID_INTERSECTION_REDUCTIONS = frozenset(
    {'product', 'min', 'max', 'mean', 'gmean', 'softmin'}
)
VALID_DIFFERENCE_REDUCTIONS = frozenset({'ignorematch', 'subtractmatch'})

# Reductions whose summand factorizes into (function of a) * (function of p),
# and which therefore need no [batch, units, num_features] intermediate.
SEPARABLE_INTERSECTION_REDUCTIONS = frozenset({'product', 'mean', 'gmean'})
SEPARABLE_DIFFERENCE_REDUCTIONS = frozenset({'ignorematch'})

# ---------------------------------------------------------------------


class _SharedRef:
    """
    Opaque holder for a variable this layer reads but does not own.

    Assigning a ``keras.Variable`` to a layer attribute makes Keras adopt it,
    so a layer reading a shared feature bank would report it among its own
    ``weights``. The bank would then be counted once per reader by
    ``count_params()`` and appear several times in the list an optimizer is
    built from, which defeats the entire point of sharing. Keras's tracker
    descends into Variables, Layers, lists, dicts, tuples and sets, and into
    nothing else, so a plain object hides the reference.

    :param value: The variable owned by another layer.
    :type value: keras.Variable
    """

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.ffn.tversky_projection")
class TverskyProjectionLayer(keras.layers.Layer):
    """
    Projection layer scored by a differentiable Tversky similarity model.

    Instead of a dot product, this layer scores the input against each of its
    ``units`` learned prototypes with Tversky's contrast model:
    ``S(a, b) = theta * f(A n B) - alpha * f(A - B) - beta * f(B - A)``.
    A vector has feature ``k`` when its dot product with ``feature_bank[k]``
    is positive, and that dot product is the feature's salience. Prototypes,
    feature bank and the three scalars are all learned.

    Unlike the previous version, the input may have **any rank >= 2** when
    the configuration is separable. The layer no longer needs
    ``TimeDistributed``.

    Architecture:

    .. code-block:: text

        ┌────────────────────────────────────────┐
        │  Input [..., D]                        │
        └──────────────┬─────────────────────────┘
                       ▼
        ┌────────────────────────────────────────┐
        │  Feature salience, two matmuls         │
        │  a = Input      @ feat_bank^T [..., NF]│
        │  p = prototypes @ feat_bank^T [U,  NF] │
        └──────────────┬─────────────────────────┘
                       ▼
             separable configuration?
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
         yes                     no
    3 matmuls, no          broadcast to
    [.., U, NF] tensor     [chunk, U', NF],
    any input rank         loop over U in
                           chunks of
                           chunk_size
            └──────────┬──────────┘
                       ▼
        ┌────────────────────────────────────────┐
        │  f(AnB), f(A-B), f(B-A)     [..., U]   │
        └──────────────┬─────────────────────────┘
                       ▼
        ┌────────────────────────────────────────┐
        │  S = θ*f(AnB) - α*f(A-B) - β*f(B-A)    │
        │  θ, α, β >= 0 via softplus             │
        │  optional 1/sqrt(NF) scaling           │
        └──────────────┬─────────────────────────┘
                       ▼
        ┌────────────────────────────────────────┐
        │  Output [..., U]                       │
        └────────────────────────────────────────┘

        D = input_dim, U = units, NF = num_features.

    Intersection and difference reduction (block internals):

    .. code-block:: text

        a = input salience,     one row      [NF]
        p = prototype salience, one row      [NF]
        in_A = a > 0     in_B = p > 0     both = in_A and in_B

        intersection_reduction, summed over `both`:
            'product'  sum a * p
            'min'      sum min(a, p)
            'max'      sum max(a, p)
            'mean'     sum (a + p) / 2
            'gmean'    sum sqrt(a + eps) * sqrt(p + eps)
            'softmin'  sum softmin_tau(a, p)

        'ignorematch':
            f(A - B) = sum over (in_A and not in_B) of a
            f(B - A) = sum over (in_B and not in_A) of p

        'subtractmatch':
            f(A - B) = sum over (both and a > p) of (a - p)
            f(B - A) = sum over (both and p > a) of (p - a)

        Every f(.) is a sum of non-negative terms, and theta,
        alpha and beta are constrained non-negative, so the
        contrast model's axioms hold throughout training.

        'gmean' is defined in the factored form above rather
        than as sqrt(a * p), so that the separable identity is
        exact rather than approximate. On common features the
        two agree to O(eps).

    Changes from the previous version:

    .. code-block:: text

        - Separable configurations skip the [B, U, NF] tensor
          entirely. It was the layer's dominant cost and was
          algebraically unnecessary for half the configuration
          space, including the pair the paper used for ResNet-50.
        - Non-separable configurations chunk over prototypes, so
          peak memory is chunk_size / units of what it was rather
          than unbounded.
        - theta, alpha and beta are non-negative by construction.
          Free scalars can cross zero, at which point the layer
          rewards distinctive features, stops being Tversky
          similarity, and voids the interpretability claim that
          justifies its cost. Nothing surfaced that before.
        - max, gmean and softmin are implemented, completing the
          paper's six intersection reductions.
        - feature_bank and prototypes can be shared with another
          layer. The paper's 34.8% parameter reduction comes
          entirely from sharing, and was unreachable before.
        - Any input rank on the separable path; no TimeDistributed.
        - diagnose() reports dead prototypes and dead features.

        Weights are named as before, so checkpoints load, except
        that theta/alpha/beta are stored pre-softplus when
        non_negative_contrast is True.

    :param units: Dimensionality of the output space (number of prototypes).
        Must be positive.
    :type units: int
    :param num_features: Size of the learnable feature universe. Must be
        positive. Note the layer has ``units*input_dim + num_features*input_dim
        + 3`` parameters against Dense's ``units*input_dim + units``, so it is
        larger than the layer it replaces unless a bank is shared.
    :type num_features: int
    :param intersection_reduction: How the two saliences combine on a shared
        feature. A member of ``VALID_INTERSECTION_REDUCTIONS``. ``'product'``,
        ``'mean'`` and ``'gmean'`` are separable. Defaults to ``'product'``.
    :type intersection_reduction: str
    :param difference_reduction: Which features the difference terms measure.
        A member of ``VALID_DIFFERENCE_REDUCTIONS``. Only ``'ignorematch'`` is
        separable, so it is what unlocks the fast path; the paper used it for
        TverskyResNet-50 and found ``'subtractmatch'`` better on XOR. Defaults
        to ``'subtractmatch'``, unchanged, so behaviour is not altered
        silently.
    :type difference_reduction: str
    :param non_negative_contrast: Whether ``theta``, ``alpha`` and ``beta``
        are passed through softplus so they cannot go negative. Defaults to
        True. The stored variables are then pre-activation values, seeded by
        inverting softplus so the effective scalars still start where
        ``contrast_initializer`` puts them.
    :type non_negative_contrast: bool
    :param normalize_by_features: Whether to divide the score by
        ``sqrt(num_features)``. Defaults to False. The unnormalized score
        grows roughly as ``sqrt(num_features)``, so tuning the bank size
        changes the logit scale and the capacity at once; this separates them.
    :type normalize_by_features: bool
    :param chunk_size: Number of prototypes processed at a time on the
        non-separable path. ``None`` processes all at once, the previous
        behaviour. Peak intermediate memory is
        ``batch * chunk_size * num_features * 4`` bytes. Ignored when the
        configuration is separable.
    :type chunk_size: Optional[int]
    :param softmin_temperature: Temperature of the ``'softmin'`` reduction.
        Must be positive. Defaults to 1.0.
    :type softmin_temperature: float
    :param shared_feature_bank: An existing feature bank variable to use
        instead of creating one, typically ``other_layer.feature_bank``. Its
        shape must be ``(num_features, input_dim)``. The variable stays owned
        by its creator, so it is trained once however many layers read it.
    :type shared_feature_bank: Optional[keras.Variable]
    :param shared_prototypes: An existing prototype variable to use instead of
        creating one, for instance a tied token embedding matrix. Its shape
        must be ``(units, input_dim)``.
    :type shared_prototypes: Optional[keras.Variable]
    :param prototype_initializer: Initializer for the prototype matrix.
        Uniform initialization converged most reliably in the paper. Defaults
        to ``'glorot_uniform'``.
    :type prototype_initializer: Union[str, keras.initializers.Initializer]
    :param feature_initializer: Initializer for the feature bank. Defaults to
        ``'glorot_uniform'``.
    :type feature_initializer: Union[str, keras.initializers.Initializer]
    :param contrast_initializer: Initializer for ``theta``, ``alpha`` and
        ``beta``. Each takes its own clone, which matters only for an unseeded
        random initializer; under the ``'ones'`` default all three are 1.0, and
        under a seeded initializer all three are identical because
        ``clone_initializer`` preserves the seed. Defaults to ``'ones'``.
    :type contrast_initializer: Union[str, keras.initializers.Initializer]
    :param epsilon: Guard inside the ``'gmean'`` square roots. Defaults to
        1e-6. Smaller values sharpen the gradient near zero salience.
    :type epsilon: float
    :param kwargs: Additional arguments for ``keras.layers.Layer``.
    :type kwargs: Any

    :ivar units: The stored output width, i.e. the number of prototypes.
    :vartype units: int
    :ivar num_features: The stored feature-bank size.
    :vartype num_features: int
    :ivar intersection_reduction: The stored reduction name, validated by
        ``__init__``.
    :vartype intersection_reduction: str
    :ivar difference_reduction: The stored reduction name, validated by
        ``__init__``.
    :vartype difference_reduction: str
    :ivar non_negative_contrast: Whether the scalars pass through softplus.
    :vartype non_negative_contrast: bool
    :ivar normalize_by_features: Whether the score is scaled by
        ``1/sqrt(num_features)``.
    :vartype normalize_by_features: bool
    :ivar chunk_size: Prototype chunk size on the non-separable path.
    :vartype chunk_size: Optional[int]
    :ivar softmin_temperature: Temperature of the softmin reduction.
    :vartype softmin_temperature: float
    :ivar epsilon: Guard inside the gmean square roots.
    :vartype epsilon: float
    :ivar is_separable: Whether this configuration avoids the
        ``[..., units, num_features]`` intermediate. Read-only, derived from
        the two reduction names.
    :vartype is_separable: bool
    :ivar prototypes: Weight of shape ``(units, input_dim)``, or the shared
        variable that was passed in. ``None`` until ``build()`` runs.
    :vartype prototypes: Optional[keras.Variable]
    :ivar feature_bank: Weight of shape ``(num_features, input_dim)``, or the
        shared variable that was passed in. ``None`` until ``build()`` runs.
    :vartype feature_bank: Optional[keras.Variable]
    :ivar theta: Weight on the common-feature term, pre-softplus when
        ``non_negative_contrast``. ``None`` until ``build()`` runs.
    :vartype theta: Optional[keras.Variable]
    :ivar alpha: Weight on ``f(A - B)``. ``None`` until ``build()``.
    :vartype alpha: Optional[keras.Variable]
    :ivar beta: Weight on ``f(B - A)``. ``None`` until ``build()``.
    :vartype beta: Optional[keras.Variable]

    :raises ValueError: If ``units`` or ``num_features`` is not positive.
    :raises ValueError: If either reduction name is not a member of its
        frozenset.
    :raises ValueError: If ``chunk_size`` is given and is not positive.
    :raises ValueError: If ``softmin_temperature`` or ``epsilon`` is not
        positive.
    :raises ValueError: From ``build()``, if the input is rank < 2, if the
        last dimension is ``None``, if a non-separable configuration receives
        rank > 2, or if a shared bank has the wrong shape.

    Input shape:
        Tensor of shape ``(..., input_dim)``, rank >= 2. Rank > 2 requires a
        separable configuration.

    Output shape:
        Same leading axes, last axis ``units``.

    Example:
        .. code-block:: python

            # Separable: three matmuls, any rank, no big intermediate.
            layer = TverskyProjectionLayer(
                units=10, num_features=32, difference_reduction="ignorematch"
            )
            layer(keras.random.normal((4, 7, 16))).shape   # (4, 7, 10)
            layer.is_separable                             # True

            # Share the bank with a second layer, as the paper does.
            head = TverskyProjectionLayer(
                units=50257, num_features=8192,
                difference_reduction="ignorematch",
                shared_feature_bank=layer.feature_bank,
            )

    Note:
        A prototype whose feature set is empty has exactly zero gradient and
        cannot recover. Under symmetric initialization that happens with
        probability ``2^-num_features`` per prototype, so at one or two
        features a large fraction of prototypes start dead. Call
        :meth:`diagnose` on a real batch before concluding a run failed for
        any other reason.

    Note:
        Set membership is a hard threshold, so it carries no gradient. The
        parameterization is differentiable almost everywhere, but features
        only learn through salience magnitudes on the side of the threshold
        they already occupy.
    """

    def __init__(
        self,
        units: int,
        num_features: int,
        intersection_reduction: Literal[
            'product', 'min', 'max', 'mean', 'gmean', 'softmin'
        ] = 'product',
        difference_reduction: Literal['ignorematch', 'subtractmatch'] = 'subtractmatch',
        non_negative_contrast: bool = True,
        normalize_by_features: bool = False,
        chunk_size: Optional[int] = None,
        softmin_temperature: float = 1.0,
        shared_feature_bank: Optional[Any] = None,
        shared_prototypes: Optional[Any] = None,
        prototype_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        feature_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        contrast_initializer: Union[str, keras.initializers.Initializer] = 'ones',
        epsilon: float = 1e-6,
        **kwargs: Any
    ) -> None:
        """
        Validate the configuration and store it.

        No weight exists yet; the weight attributes are ``None`` and created
        in ``build()``, except for any bank passed in as shared.

        :raises ValueError: If any size, reduction name or scalar argument is
            out of range.
        """
        super().__init__(**kwargs)

        if units <= 0:
            raise ValueError(f"`units` must be positive, got {units}")
        if num_features <= 0:
            raise ValueError(f"`num_features` must be positive, got {num_features}")

        # Fail here rather than inside call().
        if intersection_reduction not in VALID_INTERSECTION_REDUCTIONS:
            raise ValueError(
                f"intersection_reduction must be one of "
                f"{sorted(VALID_INTERSECTION_REDUCTIONS)}, "
                f"got '{intersection_reduction}'"
            )
        if difference_reduction not in VALID_DIFFERENCE_REDUCTIONS:
            raise ValueError(
                f"difference_reduction must be one of "
                f"{sorted(VALID_DIFFERENCE_REDUCTIONS)}, "
                f"got '{difference_reduction}'"
            )
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(f"`chunk_size` must be positive, got {chunk_size}")
        if softmin_temperature <= 0.0:
            raise ValueError(
                f"`softmin_temperature` must be positive, got {softmin_temperature}"
            )
        if epsilon <= 0.0:
            raise ValueError(f"`epsilon` must be positive, got {epsilon}")

        self.units = units
        self.num_features = num_features
        self.intersection_reduction = intersection_reduction
        self.difference_reduction = difference_reduction
        self.non_negative_contrast = non_negative_contrast
        self.normalize_by_features = normalize_by_features
        self.chunk_size = chunk_size
        self.softmin_temperature = softmin_temperature
        self.epsilon = epsilon
        self.prototype_initializer = keras.initializers.get(prototype_initializer)
        self.feature_initializer = keras.initializers.get(feature_initializer)
        self.contrast_initializer = keras.initializers.get(contrast_initializer)

        # Shared banks are not serialized: the owner serializes them, and the
        # sharing is re-established by whatever code wires the model together.
        # They are held behind _SharedRef so Keras does not adopt them.
        self._shared_feature_bank = (
            None if shared_feature_bank is None else _SharedRef(shared_feature_bank)
        )
        self._shared_prototypes = (
            None if shared_prototypes is None else _SharedRef(shared_prototypes)
        )

        # Owned weights are registered by add_weight() in build(), not by
        # attribute assignment, so these stay out of the tracker either way.
        self._owned_prototypes = None
        self._owned_feature_bank = None

        self.theta = None
        self.alpha = None
        self.beta = None

    @property
    def prototypes(self) -> Optional[Any]:
        """
        The prototype matrix, ``(units, input_dim)``.

        Either this layer's own weight or the variable passed as
        ``shared_prototypes``, which stays owned by its creator.

        :return: The prototype variable, or ``None`` before ``build()``.
        :rtype: Optional[keras.Variable]
        """
        if self._shared_prototypes is not None:
            return self._shared_prototypes.value
        return self._owned_prototypes

    @property
    def feature_bank(self) -> Optional[Any]:
        """
        The feature bank, ``(num_features, input_dim)``.

        Either this layer's own weight or the variable passed as
        ``shared_feature_bank``, which stays owned by its creator.

        :return: The feature bank variable, or ``None`` before ``build()``.
        :rtype: Optional[keras.Variable]
        """
        if self._shared_feature_bank is not None:
            return self._shared_feature_bank.value
        return self._owned_feature_bank

    @property
    def is_separable(self) -> bool:
        """
        Whether this configuration avoids the ``[..., units, num_features]``
        intermediate.

        True when the intersection summand factorizes into a function of the
        input times a function of the prototype, and the difference terms do
        too. See the module docstring for the identities.

        :return: True if the fast matmul path applies.
        :rtype: bool
        """
        return (
            self.intersection_reduction in SEPARABLE_INTERSECTION_REDUCTIONS
            and self.difference_reduction in SEPARABLE_DIFFERENCE_REDUCTIONS
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create or adopt the prototypes, the feature bank and the scalars.

        Both matrices are ``(something, input_dim)``, so the last dimension
        must be known here.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the rank is below 2, if the rank exceeds 2 on a
            non-separable configuration, if the last entry is ``None``, or if
            a shared bank has the wrong shape.
        """
        if self.built:
            return

        if len(input_shape) < 2:
            raise ValueError(
                "`TverskyProjectionLayer` needs at least a rank-2 input "
                f"(batch_size, input_dim). Got rank-{len(input_shape)} input "
                f"with shape {input_shape}."
            )

        # The broadcast form would need a [..., units, num_features] tensor
        # per leading axis, so higher ranks are only supported where the
        # matmul identities let us skip that tensor entirely.
        if len(input_shape) > 2 and not self.is_separable:
            raise ValueError(
                f"intersection_reduction='{self.intersection_reduction}' with "
                f"difference_reduction='{self.difference_reduction}' is not "
                f"separable, so it operates on rank-2 inputs only. Got rank-"
                f"{len(input_shape)} input with shape {input_shape}. Either "
                f"choose a separable pair (intersection in "
                f"{sorted(SEPARABLE_INTERSECTION_REDUCTIONS)}, difference "
                f"'ignorematch'), or wrap the layer in `TimeDistributed`."
            )

        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError(
                "The last dimension of the input to `TverskyProjectionLayer` "
                "must be defined. Found `None`."
            )

        # A shared bank is adopted, not copied and not re-registered, so the
        # parameters are counted and trained exactly once across every layer
        # that reads them. This is where the paper's parameter reduction
        # comes from, and it was unreachable before.
        if self._shared_prototypes is not None:
            expected = (self.units, input_dim)
            actual = tuple(self._shared_prototypes.value.shape)
            if actual != expected:
                raise ValueError(
                    f"shared_prototypes has shape {actual}, expected {expected}"
                )
        else:
            self._owned_prototypes = self.add_weight(
                name='prototypes',
                shape=(self.units, input_dim),
                initializer=self.prototype_initializer,
                trainable=True,
            )

        if self._shared_feature_bank is not None:
            expected = (self.num_features, input_dim)
            actual = tuple(self._shared_feature_bank.value.shape)
            if actual != expected:
                raise ValueError(
                    f"shared_feature_bank has shape {actual}, expected {expected}"
                )
        else:
            self._owned_feature_bank = self.add_weight(
                name='feature_bank',
                shape=(self.num_features, input_dim),
                initializer=self.feature_initializer,
                trainable=True,
            )

        # Each scalar takes its own clone of contrast_initializer. That only
        # separates them under an UNSEEDED random initializer; a seeded clone
        # reproduces its source exactly because clone_initializer round-trips
        # get_config, and the 'ones' default gives 1.0 either way.
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

        # Store pre-softplus values so the effective scalars still begin where
        # contrast_initializer put them, rather than at softplus(1) = 1.313.
        if self.non_negative_contrast:
            for weight in (self.theta, self.alpha, self.beta):
                weight.assign(self._inverse_softplus(weight.value))

        super().build(input_shape)

    @staticmethod
    def _inverse_softplus(y: keras.KerasTensor) -> keras.KerasTensor:
        """
        Invert softplus, so ``softplus(_inverse_softplus(y)) == y`` for y > 0.

        Uses ``log(expm1(y))``, which is stable for small positive ``y`` where
        ``log(exp(y) - 1)`` loses all its significant digits. Softplus has no
        non-positive image, so a ``y <= 1e-6`` cannot invert exactly; it is
        replaced by the exponential continuation ``1e-6 * exp(y - 1e-6)``,
        which matches the floor's value and slope at ``y == 1e-6`` and stays
        strictly increasing below it, rather than a flat ``max(y, 1e-6)``.

        # DECISION plan-2026-09-18-1f3c0ce8/D-013: do not revert this to
        # keras.ops.maximum(y, 1e-6). A hard floor collapses every
        # non-positive y to the SAME pre-activation value: MEASURED, an
        # unseeded zero-mean contrast_initializer made theta/alpha/beta
        # bit-identical in ~20-25% of builds (each scalar has ~50% odds of
        # drawing <= 0, so two of three colliding at the shared floor is
        # common), defeating the "each takes its own clone" independence
        # this layer's own docstring promises. See decisions.md D-013.

        :param y: Target post-softplus value.
        :type y: keras.KerasTensor
        :return: The pre-activation value.
        :rtype: keras.KerasTensor
        """
        eps = keras.ops.cast(1e-6, y.dtype)
        floored = keras.ops.where(y > eps, y, eps * keras.ops.exp(y - eps))
        return keras.ops.log(keras.ops.expm1(floored))

    def _contrast_weights(
        self,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """
        Return the effective ``(theta, alpha, beta)``.

        Tversky's axiomatization requires non-negative weights on the three
        set measures. Free scalars can cross zero during training, at which
        point the layer starts rewarding distinctive features, is no longer
        the contrast model, and the interpretability that justifies its cost
        is silently void. softplus makes that unreachable.

        :return: The three scalars, non-negative when
            ``non_negative_contrast``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]
        """
        if not self.non_negative_contrast:
            return self.theta, self.alpha, self.beta
        return (
            keras.ops.softplus(self.theta),
            keras.ops.softplus(self.alpha),
            keras.ops.softplus(self.beta),
        )

    # -----------------------------------------------------------------
    # separable path
    # -----------------------------------------------------------------

    def _separable_terms(
        self,
        input_dots: keras.KerasTensor,
        proto_dots: keras.KerasTensor,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """
        Compute the three measures as matmuls, with no rank-3 intermediate.

        Each summand here factorizes as ``g(a_k) * h(p_k)``, so summing over
        features is a matrix product. The identities are exact; see
        :meth:`verify_fast_path`, which checks them against the broadcast
        implementation.

        :param input_dots: Input salience, ``[..., num_features]``.
        :type input_dots: keras.KerasTensor
        :param proto_dots: Prototype salience, ``[units, num_features]``.
        :type proto_dots: keras.KerasTensor
        :return: ``(f_intersection, f_input_distinctive, f_proto_distinctive)``,
            each ``[..., units]``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]
        """
        dtype = input_dots.dtype
        a_pos = keras.ops.relu(input_dots)
        p_pos = keras.ops.relu(proto_dots)
        a_ind = keras.ops.cast(input_dots > 0, dtype)
        p_ind = keras.ops.cast(proto_dots > 0, dtype)
        p_pos_t = keras.ops.transpose(p_pos)
        p_ind_t = keras.ops.transpose(p_ind)

        if self.intersection_reduction == 'product':
            f_intersection = keras.ops.matmul(a_pos, p_pos_t)
        elif self.intersection_reduction == 'mean':
            f_intersection = 0.5 * (
                keras.ops.matmul(a_pos, p_ind_t)
                + keras.ops.matmul(a_ind, p_pos_t)
            )
        else:  # 'gmean', defined in factored form so this stays exact
            # The indicators are load-bearing. relu alone zeroes a
            # non-member's factor, but the epsilon inside the square root
            # restores it to sqrt(eps), and summing that over every
            # non-common feature is a bias large enough to see (1.3e-02 on a
            # 24-feature bank). Multiplying by the membership indicator
            # returns an exact zero there.
            eps = keras.ops.cast(self.epsilon, dtype)
            f_intersection = keras.ops.matmul(
                keras.ops.sqrt(a_pos + eps) * a_ind,
                keras.ops.transpose(keras.ops.sqrt(p_pos + eps) * p_ind),
            )

        # 'ignorematch' only; the guard is in build() and is_separable.
        one = keras.ops.cast(1.0, dtype)
        f_input_distinctive = keras.ops.matmul(a_pos, one - p_ind_t)
        f_proto_distinctive = keras.ops.matmul(one - a_ind, p_pos_t)

        return f_intersection, f_input_distinctive, f_proto_distinctive

    # -----------------------------------------------------------------
    # broadcast path
    # -----------------------------------------------------------------

    def _broadcast_terms(
        self,
        input_dots: keras.KerasTensor,
        proto_dots: keras.KerasTensor,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """
        Compute the three measures via the ``[batch, units, num_features]``
        broadcast, for one chunk of prototypes.

        This is the general form. It is what ``min``, ``max``, ``softmin`` and
        ``subtractmatch`` require, because their summands do not factorize.

        :param input_dots: Input salience, ``[batch, num_features]``.
        :type input_dots: keras.KerasTensor
        :param proto_dots: Prototype salience for this chunk,
            ``[chunk, num_features]``.
        :type proto_dots: keras.KerasTensor
        :return: ``(f_intersection, f_input_distinctive, f_proto_distinctive)``,
            each ``[batch, chunk]``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]
        """
        dtype = input_dots.dtype
        zero = keras.ops.cast(0.0, dtype)

        a = keras.ops.expand_dims(input_dots, axis=1)   # [B, 1, NF]
        p = keras.ops.expand_dims(proto_dots, axis=0)   # [1, C, NF]
        in_a = a > 0
        in_p = p > 0
        common = keras.ops.logical_and(in_a, in_p)

        if self.intersection_reduction == 'product':
            scores = a * p
        elif self.intersection_reduction == 'min':
            scores = keras.ops.minimum(a, p)
        elif self.intersection_reduction == 'max':
            scores = keras.ops.maximum(a, p)
        elif self.intersection_reduction == 'mean':
            scores = (a + p) / 2.0
        elif self.intersection_reduction == 'gmean':
            eps = keras.ops.cast(self.epsilon, dtype)
            scores = keras.ops.sqrt(keras.ops.relu(a) + eps) * keras.ops.sqrt(
                keras.ops.relu(p) + eps
            )
        else:  # 'softmin'
            tau = keras.ops.cast(self.softmin_temperature, dtype)
            wa = keras.ops.exp(-a / tau)
            wp = keras.ops.exp(-p / tau)
            scores = (a * wa + p * wp) / (wa + wp)

        f_intersection = keras.ops.sum(
            keras.ops.where(common, scores, zero), axis=-1
        )

        if self.difference_reduction == 'ignorematch':
            only_a = keras.ops.logical_and(in_a, keras.ops.logical_not(in_p))
            only_p = keras.ops.logical_and(in_p, keras.ops.logical_not(in_a))
            f_input_distinctive = keras.ops.sum(
                keras.ops.where(only_a, a, zero), axis=-1
            )
            f_proto_distinctive = keras.ops.sum(
                keras.ops.where(only_p, p, zero), axis=-1
            )
        else:  # 'subtractmatch'
            a_over_p = keras.ops.logical_and(common, a > p)
            p_over_a = keras.ops.logical_and(common, p > a)
            f_input_distinctive = keras.ops.sum(
                keras.ops.where(a_over_p, a - p, zero), axis=-1
            )
            f_proto_distinctive = keras.ops.sum(
                keras.ops.where(p_over_a, p - a, zero), axis=-1
            )

        return f_intersection, f_input_distinctive, f_proto_distinctive

    def _chunked_terms(
        self,
        input_dots: keras.KerasTensor,
        proto_dots: keras.KerasTensor,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """
        Apply :meth:`_broadcast_terms` over slices of the prototype axis.

        Peak intermediate memory becomes ``batch * chunk_size * num_features``
        rather than ``batch * units * num_features``. The loop is a Python
        loop over a statically known ``units``, so it unrolls into the graph;
        keep ``units / chunk_size`` modest or the graph grows large.

        :param input_dots: Input salience, ``[batch, num_features]``.
        :type input_dots: keras.KerasTensor
        :param proto_dots: Prototype salience, ``[units, num_features]``.
        :type proto_dots: keras.KerasTensor
        :return: The three measures, each ``[batch, units]``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]
        """
        if self.chunk_size is None or self.chunk_size >= self.units:
            return self._broadcast_terms(input_dots, proto_dots)

        parts: List[Tuple[keras.KerasTensor, ...]] = []
        for start in range(0, self.units, self.chunk_size):
            stop = min(start + self.chunk_size, self.units)
            parts.append(self._broadcast_terms(input_dots, proto_dots[start:stop]))

        return tuple(
            keras.ops.concatenate([part[i] for part in parts], axis=-1)
            for i in range(3)
        )

    # -----------------------------------------------------------------

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """
        Score every input row against every prototype.

        Routes to the matmul path when the configuration is separable and to
        the chunked broadcast path otherwise. Both compute the same function;
        :meth:`verify_fast_path` checks that.

        :param inputs: Tensor of shape ``(..., input_dim)``. Rank above 2
            requires a separable configuration, which ``build()`` enforces.
        :type inputs: keras.KerasTensor
        :param training: Unused. Accepted so the call signature matches the
            other FFN layers; this layer behaves the same either way.
        :type training: Optional[bool]
        :return: Similarity scores of shape ``(..., units)``.
        :rtype: keras.KerasTensor
        """
        feature_bank_t = keras.ops.transpose(self.feature_bank)
        input_dots = keras.ops.matmul(inputs, feature_bank_t)
        proto_dots = keras.ops.matmul(self.prototypes, feature_bank_t)

        if self.is_separable:
            terms = self._separable_terms(input_dots, proto_dots)
        else:
            terms = self._chunked_terms(input_dots, proto_dots)

        f_intersection, f_input_distinctive, f_proto_distinctive = terms
        theta, alpha, beta = self._contrast_weights()

        similarity = (
            theta * f_intersection
            - alpha * f_input_distinctive
            - beta * f_proto_distinctive
        )

        # The unnormalized score sums num_features terms, so its scale grows
        # as roughly sqrt(num_features). Without this, changing the bank size
        # moves the logit scale and the capacity together.
        if self.normalize_by_features:
            similarity = similarity / keras.ops.cast(
                self.num_features ** 0.5, similarity.dtype
            )

        return similarity

    def verify_fast_path(
        self,
        batch: int = 8,
        tolerance: float = 1e-4,
        seed: int = 0,
    ) -> Tuple[bool, float]:
        """
        Check the separable matmuls against the broadcast implementation.

        Both compute the same three measures, so a discrepancy means one of
        the factorizations is wrong for this configuration. Returns
        ``(True, 0.0)`` unchecked when the configuration is not separable.

        :param batch: Batch size for the probe.
        :type batch: int
        :param tolerance: Maximum acceptable absolute deviation.
        :type tolerance: float
        :param seed: Seed for the probe tensor.
        :type seed: int
        :return: ``(passed, max_absolute_deviation)``.
        :rtype: Tuple[bool, float]
        :raises ValueError: If the layer is not built.
        """
        if not self.built:
            raise ValueError("Layer must be built before verify_fast_path().")
        if not self.is_separable:
            return True, 0.0

        x = keras.random.normal((batch, self.prototypes.shape[-1]), seed=seed)
        feature_bank_t = keras.ops.transpose(self.feature_bank)
        input_dots = keras.ops.matmul(x, feature_bank_t)
        proto_dots = keras.ops.matmul(self.prototypes, feature_bank_t)

        fast = self._separable_terms(input_dots, proto_dots)
        slow = self._broadcast_terms(input_dots, proto_dots)

        deviation = max(
            float(keras.ops.max(keras.ops.abs(f - s))) for f, s in zip(fast, slow)
        )
        return deviation <= tolerance, deviation

    def diagnose(
        self,
        inputs: Optional[keras.KerasTensor] = None,
    ) -> Dict[str, Any]:
        """
        Report dead prototypes, dead features and the contrast weights.

        A prototype with an empty feature set contributes a constant to every
        score, so its gradient is exactly zero and it can never recover. Under
        symmetric initialization each prototype is born dead with probability
        ``2^-num_features``, which is why small feature banks fail to converge
        so often. A feature that no prototype and no input activates is inert
        for the same reason.

        :param inputs: Optional batch used to measure input-side feature
            usage. Without it only the prototype-side statistics are reported.
        :type inputs: Optional[keras.KerasTensor]
        :return: Counts and fractions, plus the effective contrast weights.
        :rtype: Dict[str, Any]
        :raises ValueError: If the layer is not built.
        """
        if not self.built:
            raise ValueError("Layer must be built before diagnose().")

        feature_bank_t = keras.ops.transpose(self.feature_bank)
        proto_dots = keras.ops.matmul(self.prototypes, feature_bank_t)
        proto_active = keras.ops.cast(proto_dots > 0, "float32")
        proto_set_sizes = keras.ops.sum(proto_active, axis=-1)
        dead_prototypes = int(keras.ops.sum(keras.ops.cast(proto_set_sizes == 0, "int32")))

        theta, alpha, beta = self._contrast_weights()
        report: Dict[str, Any] = {
            "dead_prototypes": dead_prototypes,
            "dead_prototype_fraction": dead_prototypes / self.units,
            "mean_prototype_set_size": float(keras.ops.mean(proto_set_sizes)),
            "min_prototype_set_size": float(keras.ops.min(proto_set_sizes)),
            "theta": float(theta),
            "alpha": float(alpha),
            "beta": float(beta),
            "is_separable": self.is_separable,
        }

        if inputs is not None:
            input_dots = keras.ops.matmul(inputs, feature_bank_t)
            input_used = keras.ops.max(
                keras.ops.cast(input_dots > 0, "float32"),
                axis=tuple(range(len(input_dots.shape) - 1)),
            )
            proto_used = keras.ops.max(proto_active, axis=0)
            inert = keras.ops.cast(
                (input_used == 0) & (proto_used == 0), "int32"
            )
            report["inert_features"] = int(keras.ops.sum(inert))
            report["inert_feature_fraction"] = (
                report["inert_features"] / self.num_features
            )

        return report

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Replace the last dimension of ``input_shape`` with ``units``.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape with the last entry set to ``units``.
        :rtype: Tuple[Optional[int], ...]
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return everything ``__init__`` needs to rebuild this layer.

        Shared banks are deliberately not serialized: the layer that created
        them serializes them, and the sharing is re-established by the code
        that assembles the model.

        :return: The base ``Layer`` config plus the sizes, reduction names,
            behaviour flags and serialized initializers.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'units': self.units,
            'num_features': self.num_features,
            'intersection_reduction': self.intersection_reduction,
            'difference_reduction': self.difference_reduction,
            'non_negative_contrast': self.non_negative_contrast,
            'normalize_by_features': self.normalize_by_features,
            'chunk_size': self.chunk_size,
            'softmin_temperature': self.softmin_temperature,
            'epsilon': self.epsilon,
            'prototype_initializer': keras.initializers.serialize(self.prototype_initializer),
            'feature_initializer': keras.initializers.serialize(self.feature_initializer),
            'contrast_initializer': keras.initializers.serialize(self.contrast_initializer),
        })
        return config

# ---------------------------------------------------------------------