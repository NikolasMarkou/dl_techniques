"""
Shared cluster-axis machinery for the mixture / clustering layers.

This module holds the axis-handling code that ``GMMLayer`` and ``KMeansLayer``
previously carried as verbatim duplicates. Both layers share the exact same
notion of a *cluster axis*: one or more input axes whose dimensions are flattened
into a single feature vector, over which prototypes (mixture components or
centroids) are defined, and which are collapsed to a single prototype axis in the
layer's output.

Two classes are provided:

1.  ``_ClusterAxisMixin`` -- a stateless, ``__init__``-free mixin holding the six
    shared methods. It owns no attributes; it *reads* attributes that the
    concrete layer (or ``BaseMixtureLayer``) is responsible for setting.

2.  ``BaseMixtureLayer`` -- the abstract Keras ``Layer`` seat that composes the
    mixin with ``keras.layers.Layer``, declares ``call`` abstract, initializes
    the four build-derived placeholders the mixin reads, and provides
    ``_init_cluster_axis`` for the shared constructor-side axis intake.

Two module-level helpers hold the prototype-initializer handling that the two
concrete layers also carried verbatim: :func:`resolve_initializer_arg`
(``__init__``-time, preserves the ``'orthonormal'`` string) and
:func:`resolve_prototype_initializer` (``build()``-time, resolves it to a
concrete initializer with a documented fallback).

Note on the pre-build / post-build cluster-axis split (load-bearing):

-   ``self._cluster_axis_arg`` is the ORIGINAL constructor value, negative axes
    preserved, never mutated. It is what ``get_config()`` serializes and what
    ``compute_output_shape()`` normalizes locally, because that method may run
    BEFORE ``build()`` during functional-API tracing.
-   ``self.cluster_axis`` is the ``build()``-mutated form: positive and sorted.
    It is what ``build()`` and ``call()`` read.

These two must not be conflated. See the inline ``DECISION`` comments below.

Note on ``RBFLayer``: it is deliberately NOT a member of this hierarchy. It has
no ``cluster_axis`` concept and is left untouched.
"""

import keras
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List, Any, Tuple, Union, Literal

from ...utils.logger import logger
from ...initializers.orthonormal_initializer import OrthonormalInitializer

# ---------------------------------------------------------------------

# Type aliases shared by ``GMMLayer`` and ``KMeansLayer``, which import them from here.
# ``RBFLayer`` deliberately does NOT use ``OutputMode``: it is outside this hierarchy and
# its legal set is the disjoint ``{'basis', 'normalized'}``. See the D-003 amendment in
# ``factory.py`` -- the two vocabularies must stay two.
OutputMode = Literal['assignments', 'mixture']
Axis = Union[int, List[int]]

# ---------------------------------------------------------------------


def resolve_initializer_arg(
    value: Union[str, keras.initializers.Initializer]
) -> Union[str, keras.initializers.Initializer]:
    """Resolve a prototype-initializer constructor argument, preserving ``'orthonormal'``.

    Shared by ``GMMLayer.__init__`` (``mean_initializer``) and
    ``KMeansLayer.__init__`` (``centroid_initializer``). The return value is what
    ``get_config()`` serializes and what :func:`resolve_prototype_initializer`
    consumes at ``build()``, so the string passthrough must survive both.

    :param value: An initializer alias string or an ``Initializer`` instance.
    :type value: Union[str, keras.initializers.Initializer]
    :return: ``value`` unchanged when it is the string ``'orthonormal'`` (any
        case); otherwise ``keras.initializers.get(value)``.
    :rtype: Union[str, keras.initializers.Initializer]
    :raises ValueError: Propagated from ``keras.initializers.get`` for an
        unknown alias.
    """
    # DECISION plan_2026-06-08_57a975d1/D-002: do NOT replace this with a bare
    # keras.initializers.get(value). 'orthonormal' is not a registered keras alias
    # (OrthonormalInitializer registers as Custom>OrthonormalInitializer), so
    # get('orthonormal') raises. Keep the string and let build() resolve it via
    # resolve_prototype_initializer (which handles both the string and an
    # Initializer instance). See D-001.
    if isinstance(value, str) and value.lower() == 'orthonormal':
        return value
    return keras.initializers.get(value)


def resolve_prototype_initializer(
    value: Union[str, keras.initializers.Initializer],
    count: int,
    count_name: str,
    feature_dims: int,
    seed: Optional[int],
) -> keras.initializers.Initializer:
    """Turn a stored prototype-initializer argument into a concrete initializer.

    Shared by ``GMMLayer._initialize_means`` and
    ``KMeansLayer._initialize_centroids``. Non-orthonormal values pass straight
    through; an orthonormal request is honoured only when the prototype matrix
    is not wider than it is tall, since ``OrthonormalInitializer`` cannot produce
    ``count`` mutually orthonormal rows in ``feature_dims`` dimensions.

    :param value: The stored initializer, as produced by
        :func:`resolve_initializer_arg`.
    :type value: Union[str, keras.initializers.Initializer]
    :param count: Number of prototype rows (``n_components`` / ``n_clusters``).
    :type count: int
    :param count_name: The caller's public name for ``count``; used verbatim in
        the fallback warning so each layer reports its own vocabulary.
    :type count_name: str
    :param feature_dims: Width of the prototype matrix.
    :type feature_dims: int
    :param seed: Random seed forwarded to the constructed initializer.
    :type seed: Optional[int]
    :return: ``OrthonormalInitializer(seed=seed)`` when orthonormal is requested
        and fits; ``keras.initializers.GlorotNormal(seed=seed)`` (with a logged
        warning) when it is requested but does not fit; otherwise ``value``
        unchanged.
    :rtype: keras.initializers.Initializer
    """
    # DECISION plan-2026-08-26T061816-c515641a/D-013: detect orthonormal with isinstance,
    # not a `__class__.__name__ == 'OrthonormalInitializer'` string sniff. MEASURED to agree
    # with the sniff on the string alias, an OrthonormalInitializer instance, GlorotNormal
    # and HeOrthonormalInitializer; it differs only on a SUBCLASS, which the sniff rejected
    # and isinstance accepts. Do NOT restore the sniff: it silently gives a subclass the
    # non-orthonormal path, skipping the count <= feature_dims feasibility check that is the
    # only thing standing between an over-wide request and a failed init.
    is_orthonormal = (
        isinstance(value, OrthonormalInitializer)
        or (isinstance(value, str) and value.lower() == 'orthonormal')
    )
    if not is_orthonormal:
        return value

    if count <= feature_dims:
        return OrthonormalInitializer(seed=seed)

    logger.warning(
        f"{count_name} ({count}) > feature_dims ({feature_dims}), "
        "falling back to glorot_normal initializer"
    )
    return keras.initializers.GlorotNormal(seed=seed)

# ---------------------------------------------------------------------


class _ClusterAxisMixin:
    """Stateless mixin providing shared cluster-axis geometry.

    This class deliberately defines no ``__init__`` and owns no state. It is a
    pure behavior bundle, mixed in ahead of ``keras.layers.Layer`` in the MRO so
    that its methods win over the base ``Layer`` implementations
    (notably ``compute_output_shape``) without perturbing ``Layer.__init__``.

    **Interface contract.** Every method here reads attributes it does not set.
    A host class MUST provide:

    :ivar output_mode: ``'assignments'`` or ``'mixture'``; set in ``__init__``.
    :ivar _cluster_axis_arg: ``List[int]``, the as-passed constructor axes with
        negative values preserved; set in ``__init__``, never mutated.
    :ivar cluster_axis: ``List[int]``, normalized to positive and sorted by
        ``_setup_cluster_axes()`` during ``build()``.
    :ivar input_rank: ``int``, set in ``build()``.
    :ivar feature_dims: ``int``, set in ``build()``.
    :ivar non_feature_dims: ``List[int]``, set in ``build()``.
    :ivar original_shape: ``List[int]``, set in ``build()``.
    :ivar _n_prototypes: ``int`` property; the per-layer prototype count
        (see ``BaseMixtureLayer._n_prototypes``).

    Failure mode: reading any of the ``build()``-derived attributes before
    ``build()`` has run raises ``TypeError`` on the ``None`` placeholder. The one
    method safe to call pre-build is ``compute_output_shape``, which is written
    against ``_cluster_axis_arg`` precisely for that reason.
    """

    def _setup_cluster_axes(self) -> None:
        """Setup and validate cluster axes.

        :raises ValueError: If any cluster axis resolves to the batch axis, or
            if any cluster axis is out of range for ``self.input_rank``.
        """
        # DECISION plan_2026-06-14_7384c2e3/D-003: re-derive from the ORIGINAL constructor
        # value (_cluster_axis_arg), not in-place on self.cluster_axis. This makes build()
        # idempotent -- a second build() re-normalizes from the stable source instead of
        # double-shifting an already-positive axis (which would corrupt cluster_axis).
        self.cluster_axis = [
            axis if axis >= 0 else self.input_rank + axis
            for axis in self._cluster_axis_arg
        ]

        # DECISION plan-2026-08-26T061816-c515641a/D-007: axis 0 is the BATCH axis and can
        # never be a cluster axis. Checked HERE -- after the negative-to-positive
        # normalization above and BEFORE the generic range check below -- because
        # cluster_axis=-3 on a rank-3 input normalizes to 0, and a guard written against
        # the raw constructor value would pass it. Ordered first so axis 0 gets the
        # batch-axis message rather than the shape/range one. Removing this guard restores
        # the measured pre-fix behaviour: on a STATIC-batch model it builds and runs while
        # fitting prototypes ACROSS samples, so perturbing one sample changed another
        # sample's output by 0.787 max abs -- silent cross-sample leakage, not an error.
        if 0 in self.cluster_axis:
            raise ValueError(
                f"cluster_axis resolves to the batch axis (axis 0): "
                f"cluster_axis={self._cluster_axis_arg} normalizes to "
                f"{sorted(self.cluster_axis)} on a rank-{self.input_rank} input. "
                f"Clustering over the batch axis breaks batch independence -- "
                f"prototypes are fitted across samples, so one sample's values leak "
                f"into another sample's output. Use a non-batch axis "
                f"(1..{self.input_rank - 1}, or the negative aliases "
                f"-1..-{self.input_rank - 1})."
            )

        if not all(0 <= axis < self.input_rank for axis in self.cluster_axis):
            raise ValueError(
                f"Invalid cluster_axis: {self.cluster_axis} for input rank {self.input_rank}"
            )

        self.cluster_axis.sort()

    def _compute_feature_dims(self, input_shape: Tuple[Optional[int], ...]) -> int:
        """Compute total feature dimensions.

        :param input_shape: Input tensor shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Product of dimensions along cluster axes.
        :rtype: int
        :raises ValueError: If input shape is invalid.
        """
        try:
            return int(np.prod([input_shape[axis] for axis in self.cluster_axis]))
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid input shape {input_shape} for cluster axes {self.cluster_axis}"
            ) from e

    def _compute_non_feature_dims(self) -> List[int]:
        """Compute non-feature dimensions.

        :return: List of axes not used for clustering.
        :rtype: List[int]
        """
        return [i for i in range(self.input_rank) if i not in self.cluster_axis]

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute shape of layer output.

        :param input_shape: Shape of input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output tensor shape.
        :rtype: Tuple[Optional[int], ...]
        """
        if self.output_mode == 'assignments':
            # Normalize axes LOCALLY from the original constructor value + input rank
            # rather than reading self.cluster_axis (which build() mutates negative->
            # positive and which may not be normalized yet during functional-API tracing,
            # when compute_output_shape is called BEFORE build). Mirrors _setup_cluster_axes.
            rank = len(input_shape)
            axes = sorted(
                ax if ax >= 0 else rank + ax for ax in self._cluster_axis_arg
            )
            output_shape = list(input_shape)

            if len(axes) > 1:
                # Reverse order so popping does not shift the remaining indices.
                for axis in reversed(axes[1:]):
                    output_shape.pop(axis)
                output_shape[axes[0]] = self._n_prototypes
            else:
                output_shape[axes[0]] = self._n_prototypes

            return tuple(output_shape)

        return tuple(input_shape)

    def _reshape_for_clustering(
        self, inputs: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Reshape input tensor for clustering operations.

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :return: ``(flat, leading)`` -- ``flat`` has shape
            ``(batch * non_feature_dims, feature_dims)``; ``leading`` is a 1-D
            ``int32`` tensor holding the length of every non-feature axis, in the
            post-transpose order, to be handed to ``_reshape_output``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # Optimize for common case of single axis at end
        if len(self.cluster_axis) == 1 and self.cluster_axis[0] == self.input_rank - 1:
            laid_out = inputs
        else:
            laid_out = keras.ops.transpose(
                inputs, self.non_feature_dims + self.cluster_axis
            )

        # DECISION plan-2026-08-26T061816-c515641a/D-004: read the non-feature axis
        # lengths off the CONCRETE tensor here, AFTER the transpose, and hand them to
        # _reshape_output -- do NOT let that method rebuild them from
        # self.original_shape. original_shape is captured at build() time, so for a
        # layer built against keras.Input(shape=(None, C)) it holds a Python None,
        # and a later concrete call raises ("Can't convert Python sequence with mixed
        # types to Tensor" under model(x); "Failed to convert elements of
        # [-1, None, K] to Tensor" under model.predict(x)). That is ordinary fit /
        # predict usage for any variable-length sequence model. Both forward paths
        # need this: the last-axis fast path skips the transpose on the way IN but
        # its output side is reconstructed identically. See decisions.md D-004.
        leading = keras.ops.stack(
            [
                keras.ops.cast(keras.ops.convert_to_tensor(dim), "int32")
                for dim in keras.ops.shape(laid_out)[: len(self.non_feature_dims)]
            ]
        )
        return keras.ops.reshape(laid_out, [-1, self.feature_dims]), leading

    def _reshape_output(
        self, output: keras.KerasTensor, leading: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Reshape clustering output to match desired output shape.

        :param output: Output tensor from clustering.
        :type output: keras.KerasTensor
        :param leading: 1-D ``int32`` tensor of non-feature axis lengths, as
            returned by ``_reshape_for_clustering`` for this same call.
        :type leading: keras.KerasTensor
        :return: Reshaped output tensor.
        :rtype: keras.KerasTensor
        """
        # DECISION plan-2026-07-20T160907-7de371a1/D-001: this method INVERTS the forward
        # layout move made by _reshape_for_clustering, and that inversion is a TRANSPOSE,
        # not a reshape.
        #
        # The layout contract: _reshape_for_clustering transposes by
        # `perm = non_feature_dims + cluster_axis` and then collapses, so the buffer
        # arriving here is laid out (non_feature_dims..., W) -- W is the prototype count K
        # for 'assignments', or feature_dims for 'mixture'. The DECLARED output order is
        # the ORIGINAL axis order. Those two differ for every cluster_axis except the
        # last-axis fast path. `keras.ops.reshape` is layout-preserving in row-major order;
        # it does NOT reorder axes, so a bare reshape stamps a correct SHAPE onto a
        # wrongly-ordered buffer and scrambles the values whenever K != non_feature_dims.
        #
        # Do NOT replace either transpose below with a bare reshape, and do NOT "simplify"
        # by merging the two branches: they compute genuinely different permutations
        # (insert ONE collapsed axis vs. restore EVERY original cluster axis) and have one
        # call site each. Both perms provably degenerate to the identity on the fast path
        # (cluster_axis == [input_rank-1]), which is what makes backward compatibility
        # structural rather than merely tested. See decisions.md D-001 and D-005.
        n_non_feature = len(self.non_feature_dims)

        def _target(static_tail: List[int]) -> keras.KerasTensor:
            """Undo the ``[-1, W]`` collapse: the measured leading dims, then the tail.

            ``leading`` already carries every non-feature length concretely (see
            D-004 above), so no ``-1`` placeholder and no static lookup is needed.
            """
            return keras.ops.concatenate(
                [leading, keras.ops.convert_to_tensor(static_tail, dtype="int32")],
                axis=0,
            )

        if self.output_mode == 'assignments':
            # Buffer is (non_feature_dims..., K); K sits at source index n_non_feature.
            # Target order places K where cluster_axis[0] sat in the original axis order,
            # i.e. after the `p` non-feature axes that precede it.
            output = keras.ops.reshape(output, _target([self._n_prototypes]))
            p = sum(1 for axis in self.non_feature_dims if axis < self.cluster_axis[0])
            perm = (
                list(range(p))
                + [n_non_feature]
                + list(range(p, n_non_feature))
            )
            return keras.ops.transpose(output, perm)

        # output_mode == 'mixture': the buffer is (non_feature_dims..., cluster_axis...)
        # once uncollapsed per-axis, so restoring the original order is exactly the
        # inverse of the forward `non_feature_dims + cluster_axis` permutation.
        forward_perm = self.non_feature_dims + self.cluster_axis
        output = keras.ops.reshape(
            output,
            _target([self.original_shape[axis] for axis in self.cluster_axis]),
        )
        inv_perm = sorted(range(len(forward_perm)), key=lambda j: forward_perm[j])
        return keras.ops.transpose(output, inv_perm)

# ---------------------------------------------------------------------


class BaseMixtureLayer(_ClusterAxisMixin, keras.layers.Layer, ABC):
    """Abstract base for cluster-axis-aware mixture / clustering layers.

    Deliberately thin. It exists to be the seat where ``_ClusterAxisMixin`` is
    composed with ``keras.layers.Layer`` and where the mixin's ``self.*`` reads
    are declared. It contributes exactly three things:

    1.  MRO placement -- the mixin precedes ``keras.layers.Layer`` so that
        ``compute_output_shape`` resolves to the mixin's implementation.
    2.  The four shared ``build()``-derived attribute placeholders.
    3.  The abstract contract: ``call`` and the ``_n_prototypes`` property.

    This class is intentionally **not** decorated with
    ``@keras.saving.register_keras_serializable()``. Only concrete, instantiable
    layers are registered; registering an ABC would put an unconstructible entry
    in the Keras custom-object registry. Concrete subclasses carry the decorator.

    Subclasses are responsible for setting ``output_mode`` in their own
    ``__init__`` (their constructor signatures and ``get_config()`` keys differ),
    for calling ``_init_cluster_axis()`` there to set ``cluster_axis`` /
    ``_cluster_axis_arg``, and for calling ``_setup_cluster_axes()`` from
    ``build()``.

    **Mixed-precision contract (the single home for this rationale).** Every
    prototype-bearing weight in this package is created with ``autocast=False``
    and an explicit ``dtype=self.dtype``, and every ``call()`` casts its inputs
    to ``self.variable_dtype`` on entry and its result back to
    ``self.compute_dtype`` on return. Reason: the forward math here
    (``exp`` / ``log`` / ``softmax`` / division, and triangular solves on the
    low-rank path) is numerically unsafe in float16, so it must run in float32
    even under a ``mixed_float16`` policy. Dropping ``autocast=False`` gives an
    autocast float16 weight against float32 inputs and raises
    ``InvalidArgumentError: Sub half vs float``; dropping either cast makes the
    layer emit the wrong dtype for the active policy. Under the default float32
    policy all of it is a no-op. ``RBFLayer`` is deliberately outside this
    hierarchy (see the module docstring) but follows the same contract, and its
    call sites point back to this paragraph rather than inheriting it.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.input_rank: Optional[int] = None
        self.feature_dims: Optional[int] = None
        self.non_feature_dims: Optional[List[int]] = None
        self.original_shape: Optional[List[int]] = None

    def _init_cluster_axis(self, cluster_axis: Any) -> None:
        """Store the constructor's ``cluster_axis`` in both of its required forms.

        Called from each concrete layer's ``__init__``. Sets the two attributes
        the mixin reads: ``self.cluster_axis`` (a fresh mutable list, rewritten
        in place to positive-and-sorted form by ``_setup_cluster_axes()`` at
        ``build()``) and ``self._cluster_axis_arg`` (an independent copy of the
        ORIGINAL value, never mutated).

        :param cluster_axis: A single axis index, or an iterable of them.
            Negative values are preserved as passed.
        :type cluster_axis: Union[int, List[int]]
        :return: ``None``. The two attributes above are set as a side effect.
        :rtype: None
        """
        self.cluster_axis = (
            [cluster_axis] if isinstance(cluster_axis, int) else list(cluster_axis)
        )
        # DECISION plan_2026-06-14_8c7365d0/D-005: serialize the ORIGINAL (pre-build)
        # cluster_axis, not the build()-mutated positive form. build() rewrites negative
        # axes to positive against input_rank (_setup_cluster_axes), so serializing
        # self.cluster_axis would bake in a rank-specific value -> cross-rank reload picks
        # the wrong logical axis. Stash the constructor value here and emit it in
        # get_config. The list() copy is load-bearing: the two attributes must not alias,
        # or _setup_cluster_axes' in-place sort would mutate the serialized source. The
        # matching D-005 anchors on the get_config() side are a DIFFERENT site and stay.
        self._cluster_axis_arg = list(self.cluster_axis)

    # DECISION plan-2026-07-20T141712-e03557c8/D-007: this property is a pure NAMING seam and
    # nothing more. GMMLayer calls its prototype count `n_components`, KMeansLayer calls
    # it `n_clusters`; that is the ONLY way the two copies of compute_output_shape and
    # _reshape_output ever differed (4 code lines). Do NOT "simplify" this by renaming
    # either public attribute to a shared name -- both appear in get_config() keys and in
    # MIXTURE_REGISTRY params, so a rename is a breaking public-API and serialization
    # change, and it would force test edits. Do NOT add further abstract members here to
    # absorb other differences either. See decisions.md D-007.
    @property
    @abstractmethod
    def _n_prototypes(self) -> int:
        """Number of prototypes (mixture components / centroids) this layer defines.

        Read by the mixin's shape logic to size the collapsed cluster axis.
        Concrete subclasses return their own public count attribute.

        :return: Prototype count.
        :rtype: int
        """
        ...

    @abstractmethod
    def call(self, inputs: keras.KerasTensor, **kwargs: Any) -> keras.KerasTensor:
        """Forward pass. Implemented by concrete subclasses.

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :return: Layer output.
        :rtype: keras.KerasTensor
        """
        ...

# ---------------------------------------------------------------------
