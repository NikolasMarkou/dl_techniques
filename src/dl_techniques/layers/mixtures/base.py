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
    mixin with ``keras.layers.Layer``, declares ``call`` abstract, and initializes
    the four build-derived placeholders the mixin reads.

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
from typing import Optional, List, Any, Tuple

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

        :raises ValueError: If cluster axes are invalid.
        """
        # DECISION plan_2026-06-14_7384c2e3/D-003: re-derive from the ORIGINAL constructor
        # value (_cluster_axis_arg), not in-place on self.cluster_axis. This makes build()
        # idempotent -- a second build() re-normalizes from the stable source instead of
        # double-shifting an already-positive axis (which would corrupt cluster_axis).
        # Convert negative axes to positive
        self.cluster_axis = [
            axis if axis >= 0 else self.input_rank + axis
            for axis in self._cluster_axis_arg
        ]

        # Validate axes
        if not all(0 <= axis < self.input_rank for axis in self.cluster_axis):
            raise ValueError(
                f"Invalid cluster_axis: {self.cluster_axis} for input rank {self.input_rank}"
            )

        # Sort axes for consistent processing
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

            # Handle multiple clustering axes
            if len(axes) > 1:
                # Replace clustered dimensions with the prototype count
                # Remove extra axes in reverse order to preserve indices
                for axis in reversed(axes[1:]):
                    output_shape.pop(axis)
                output_shape[axes[0]] = self._n_prototypes
            else:
                output_shape[axes[0]] = self._n_prototypes

            return tuple(output_shape)

        # For mixture mode, output shape matches input
        return tuple(input_shape)

    def _reshape_for_clustering(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Reshape input tensor for clustering operations.

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :return: Reshaped tensor with shape ``(batch * non_feature_dims, feature_dims)``.
        :rtype: keras.KerasTensor
        """
        # Optimize for common case of single axis at end
        if len(self.cluster_axis) == 1 and self.cluster_axis[0] == self.input_rank - 1:
            return keras.ops.reshape(inputs, [-1, self.feature_dims])

        # General case requires transpose
        perm = self.non_feature_dims + self.cluster_axis
        transposed = keras.ops.transpose(inputs, perm)
        return keras.ops.reshape(transposed, [-1, self.feature_dims])

    def _reshape_output(self, output: keras.KerasTensor) -> keras.KerasTensor:
        """Reshape clustering output to match desired output shape.

        :param output: Output tensor from clustering.
        :type output: keras.KerasTensor
        :return: Reshaped output tensor.
        :rtype: keras.KerasTensor
        """
        if self.output_mode == 'assignments':
            output_shape = list(self.original_shape)

            # Handle multiple clustering axes
            if len(self.cluster_axis) > 1:
                # Remove extra axes in reverse order to preserve indices
                for axis in reversed(self.cluster_axis[1:]):
                    output_shape.pop(axis)
                output_shape[self.cluster_axis[0]] = self._n_prototypes
            else:
                output_shape[self.cluster_axis[0]] = self._n_prototypes

            # Set batch dimension to -1 for dynamic reshaping
            output_shape[0] = -1

        else:  # output_mode == 'mixture'
            output_shape = list(self.original_shape)
            output_shape[0] = -1

        return keras.ops.reshape(output, output_shape)

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

    Subclasses are responsible for setting ``output_mode``, ``cluster_axis`` and
    ``_cluster_axis_arg`` in their own ``__init__`` (their constructor signatures
    and ``get_config()`` keys differ), and for calling ``_setup_cluster_axes()``
    from ``build()``.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Initialize shared attribute placeholders - all derived in build()
        self.input_rank: Optional[int] = None
        self.feature_dims: Optional[int] = None
        self.non_feature_dims: Optional[List[int]] = None
        self.original_shape: Optional[List[int]] = None

    # DECISION plan-2026-07-20-e03557c8/D-007: this property is a pure NAMING seam and
    # nothing more. GMMLayer calls its prototype count `n_components`, KMeansLayer calls
    # it `n_clusters`; that is the ONLY way the two copies of compute_output_shape and
    # _reshape_output ever differed (4 code lines). Do NOT "simplify" this by renaming
    # either public attribute to a shared name -- both appear in get_config() keys and in
    # MIXTURE_REGISTRY params, so a rename is a breaking public-API and serialization
    # change, and it would force test edits. Do NOT add further abstract members here to
    # absorb other differences either; this is an authorized one-off amendment to the
    # plan's abstraction budget, not a precedent. See decisions.md D-007.
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
