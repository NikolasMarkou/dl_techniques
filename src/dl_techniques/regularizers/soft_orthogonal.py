"""Soft orthogonality and orthonormality constraints for kernel weights.

Provides :class:`SoftOrthogonalConstraintRegularizer`, which penalizes only the
off-diagonal entries of the kernel's Gram matrix,
:class:`SoftOrthonormalConstraintRegularizer`, which penalizes the full
deviation from the identity, and :func:`_kernel_gram`, the shared Gram
construction they both use.

Background
----------
Orthogonal weight matrices preserve gradient magnitudes through
backpropagation, which mitigates vanishing and exploding gradients and
covariate shift. "Can We Gain More from Orthogonality Regularizations in
Training Deep CNNs?" (2018) shows that soft versions of the constraint improve
model performance.

For a weight matrix W reshaped to 2D as (fan_in, units):

1. Orthogonality: the off-diagonal entries of the Gram matrix vanish, so the
   compared directions are mutually uncorrelated. Magnitudes are untouched.
2. Orthonormality: the Gram matrix equals the identity, so the compared
   directions are mutually uncorrelated and unit norm. Equivalently, every
   nonzero singular value of W equals one, so W is a partial isometry.

Enforcing either exactly during optimization is hard, so both are applied as
soft penalties added to the loss.

Which Gram matrix
-----------------
A (fan_in, units) matrix cannot have `units` mutually orthonormal columns when
units > fan_in: the Gram ``W^T W`` is rank deficient and ``||W^T W - I||_F^2``
has an irreducible floor of ``units - fan_in``. Penalizing it anyway spends
gradient on an unreachable target and, under matrix scaling, hides the floor
inside the reported loss value.

The Gram is therefore built over whichever axis is smaller::

    units <= fan_in :  G = W^T W,  shape (units, units)
                       "the output channels are mutually orthonormal"

    units >  fan_in :  G = W W^T,  shape (fan_in, fan_in)
                       "the input directions are mutually orthonormal"

Both say the same thing about W, that every nonzero singular value is one, and
both are reachable. The orientation is read off the static kernel shape and
logged once per regularizer instance, because the semantics of the off-diagonal
variant do change with it: for an expansion layer you are decorrelating input
directions, not output channels. At that shape no other decorrelation
statement is available.

The smaller Gram is also the cheaper one: the cost is
``O(fan_in * units * min(fan_in, units))`` rather than ``O(fan_in * units^2)``.

Size normalization
------------------
`use_matrix_scaling` divides the entire regularization value, orthogonality
term plus L1 plus L2, by ``sqrt(rank)``, where rank is the side length of the
Gram matrix actually used.

``sqrt(rank)`` is the divisor that makes the achieved Gram deviation
independent of layer width. Write the deviation entries as ``eps``, the column
norm as ``nu``, and the fan-in as ``f``. The penalty gradient reaching one
kernel coordinate is::

    |grad R|_ij ~ (4 * lambda / D) * sqrt(rank) * eps * nu / sqrt(f)

with the ``sqrt(rank)`` coming from the sum over Gram entries. Balancing
against a per-coordinate task gradient ``g`` gives::

    eps = g * D * sqrt(f) / (4 * lambda * sqrt(rank) * nu)

so ``D = 1`` gives ``eps ~ rank^-0.5``, ``D = rank^2`` gives ``eps ~ rank^1.5``,
and ``D = sqrt(rank)`` gives ``eps ~ rank^0``. Do not use ``rank^2``: it
over-corrects by ``rank^1.5`` and leaves wide layers effectively
unregularized.

A residual ``sqrt(fan_in)`` dependence survives and is left in place: for a
normalized layer the per-coordinate gradient ``g`` carries a compensating
``1/sqrt(fan_in)``, so removing it here would double-count.

The divisor is applied uniformly rather than to the orthogonality term alone,
so toggling the flag is a pure global gain and never changes the balance
between the orthogonality, L1 and L2 terms. For L1 or L2 at an absolute
per-coordinate strength independent of width, multiply their coefficients by
``sqrt(rank)`` yourself, or set ``use_matrix_scaling=False`` and pre-divide
``lambda_coefficient`` by ``sqrt(rank)``.

Formulations
------------
1. :class:`SoftOrthogonalConstraintRegularizer`
   - Builds the smaller Gram matrix
   - Masks the diagonal with ``(1.0 - eye)``
   - Penalizes ``||G - diag(G)||_F^2``
   - Leaves weight magnitudes alone, provided l1_coefficient and
     l2_coefficient stay at their default of 0.0

2. :class:`SoftOrthonormalConstraintRegularizer`
   - Builds the smaller Gram matrix
   - Penalizes ``||G - I||_F^2``, so both orthogonality and unit magnitude

Hybrid terms
------------
Both regularizers support optional L1/L2 terms, off by default. An L2 term
enabled here is a coupled penalty: it passes through the optimizer's
preconditioner, unlike the decoupled decay of AdamW, and it competes with the
orthogonality term for control of the weight norm. Prefer the optimizer's
``weight_decay`` unless you want the coupled behaviour.

`EPSILON` floors the size-normalization divisor for the degenerate ``rank == 0``
case only; it is not a division-by-zero guard in any realistic shape.

Usage guidelines
----------------
- For very deep networks, use SoftOrthogonalConstraintRegularizer
- For convolutional layers, SoftOrthonormalConstraintRegularizer often works
  better
- Both classes share the same size normalization and the same
  use_matrix_scaling default, so their lambda_coefficient values are directly
  comparable
- With use_matrix_scaling=True, lambda_coefficient in [1e-4, 1e-2] is a
  reasonable starting range and does not need retuning across layer widths
- A value tuned against a ``rank^2`` divisor does not transfer; divide it by
  ``rank^1.5`` as a starting point
- Leave l1_coefficient and l2_coefficient at 0.0 unless you have a specific
  reason; use the optimizer's weight_decay for magnitude control
"""

import math
import keras
from typing import Dict, Any, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# dl_techniques.utils.tensors.gram_matrix is not used here: it fixes the Gram
# orientation to the output-channel axis, which is the unreachable target when
# units > fan_in. The orientation is chosen explicitly below instead. See
# "Which Gram matrix" in the module docstring.

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

EPSILON: float = 1e-12

# Off by default. An L2 term here is a coupled penalty and competes directly
# with the orthogonality term for control of the weight norm; at a value of
# 1e-4 it dominated it by roughly three orders of magnitude for a 512-unit
# layer. Use the optimizer's weight_decay instead unless the coupled behaviour
# is what you want.
DEFAULT_SOFTORTHOGONAL_L1: float = 0.0
DEFAULT_SOFTORTHOGONAL_L2: float = 0.0

DEFAULT_SOFTORTHOGONAL_LAMBDA: float = 1e-3
DEFAULT_SOFTORTHOGONAL_STDDEV: float = 0.02

# Shared by both subclasses, so their lambda_coefficient values mean the same
# effective strength.
DEFAULT_USE_MATRIX_SCALING: bool = True

# String constants
STR_FRO: str = "fro"
STR_L1_COEFFICIENT: str = "l1_coefficient"
STR_L2_COEFFICIENT: str = "l2_coefficient"
STR_LAMBDA_COEFFICIENT: str = "lambda_coefficient"
STR_USE_MATRIX_SCALING: str = "use_matrix_scaling"


# ---------------------------------------------------------------------
# Gram construction
# ---------------------------------------------------------------------


def _kernel_gram(
        x: Union[keras.KerasTensor, Any]
) -> Tuple[Union[keras.KerasTensor, Any], int, bool]:
    """Build the Gram matrix over whichever kernel axis is smaller.

    The kernel is flattened to 2D as (fan_in, units), following the Keras
    convention that the output-channel axis is last for both Dense (d, units)
    and Conv (kh, kw, cin, cout) kernels.

    When units <= fan_in the Gram is taken over output channels, ``W^T W``, and
    ``W^T W = I`` asks for orthonormal output channels. When units > fan_in
    that target is rank deficient and unreachable, so the Gram is taken over
    input directions instead, ``W W^T``, and ``W W^T = I`` asks for orthonormal
    input directions. Both express the same condition on W, that every nonzero
    singular value equals one, and only one of them is reachable at any given
    shape.

    :param x: Weight tensor of any rank >= 2, output channels last.
    :type x: tensor
    :return: ``(gram, rank, over_output_channels)``, where ``gram`` is a square
        matrix of side ``rank``, ``rank`` is ``min(fan_in, units)``, and
        ``over_output_channels`` is ``True`` when the Gram is ``W^T W`` and
        ``False`` when it is ``W W^T``.
    :rtype: tuple
    :raises ValueError: If the kernel shape has rank below 2 or is not fully
        static.
    """
    kernel_shape = tuple(x.shape)

    if len(kernel_shape) < 2:
        raise ValueError(
            f"Orthogonality regularizers need a kernel of rank >= 2, "
            f"got shape {kernel_shape}"
        )
    if any(dim is None for dim in kernel_shape):
        raise ValueError(
            f"Orthogonality regularizers need a fully static kernel shape, "
            f"got {kernel_shape}"
        )

    units = int(kernel_shape[-1])
    fan_in = int(math.prod(kernel_shape[:-1]))

    # Rows of w2d are input directions, columns are output channels.
    w2d = keras.ops.reshape(x, (fan_in, units))

    if units <= fan_in:
        # Reachable: `units` orthonormal columns in R^fan_in.
        gram = keras.ops.matmul(keras.ops.transpose(w2d), w2d)
        return gram, units, True

    # units > fan_in. W^T W is rank deficient, so constrain the other side.
    gram = keras.ops.matmul(w2d, keras.ops.transpose(w2d))
    return gram, fan_in, False


# ---------------------------------------------------------------------


class _SoftOrthogonalBaseRegularizer(keras.regularizers.Regularizer):
    """Shared machinery for the two soft orthogonality regularizers.

    Owns the Gram construction, the optional L1/L2 terms, the size
    normalization and serialization. Subclasses implement
    :meth:`_compute_deviation` to define the orthogonality target.

    **Penalty pipeline:**

    .. code-block:: text

        weights (any rank >= 2, output channels last)
             |
             v
        ┌──────────────────────────────────────┐
        │ _kernel_gram                         │
        │  reshape to (fan_in, units), then    │
        │  the smaller of W^T W and W W^T      │
        └───────────────┬──────────────────────┘
                        │ gram [rank, rank], rank = min(fan_in, units)
                        v
        ┌──────────────────────────────────────┐
        │ _compute_deviation(gram, eye)        │  ('lambda' > 0 only)
        │  subclass defines the target         │
        └───────────────┬──────────────────────┘
                        v
              lambda * sum(deviation^2)
                        │
                        +<--- l1_coefficient * L1(w)   (optional)
                        │
                        +<--- l2_coefficient * L2(w)   (optional)
                        │
                        v
        ┌──────────────────────────────────────┐
        │ / sqrt(rank)   ('use_matrix_scaling') │
        │ applied to the whole sum, so the      │
        │ balance between terms is unchanged    │
        └───────────────┬──────────────────────┘
                        v
                     scalar

    **Gram orientation:**

    .. code-block:: text

        shape                 gram      rank     reads as
        -------------------   -------   ------   ------------------------
        units <= fan_in       W^T W     units    output channels are
                                                 mutually orthonormal
        units >  fan_in       W W^T     fan_in   input directions are
                                                 mutually orthonormal

    :param lambda_coefficient: Weight for the orthogonality Frobenius norm
        term. Must be non-negative.
    :type lambda_coefficient: float
    :param l1_coefficient: Weight for L1 regularization. This is a coupled
        penalty and competes with the orthogonality term for control of the
        weight norm. Must be non-negative.
    :type l1_coefficient: float
    :param l2_coefficient: Weight for L2 regularization. Same caveat as L1;
        prefer the optimizer's decoupled ``weight_decay``. Must be
        non-negative.
    :type l2_coefficient: float
    :param use_matrix_scaling: Divide the entire regularization value by
        ``sqrt(rank)``, the side length of the Gram matrix actually used. See
        the module docstring for why that is the width-invariant divisor.
    :type use_matrix_scaling: bool
    :param kwargs: Must be empty. ``keras.regularizers.Regularizer`` defines no
        ``__init__``, so any keyword forwarded here reaches ``object.__init__``
        and raises ``TypeError``.

    :raises ValueError: If any coefficient is negative.
    :raises TypeError: If any keyword argument is supplied.
    """

    def __init__(
            self,
            lambda_coefficient: float = DEFAULT_SOFTORTHOGONAL_LAMBDA,
            l1_coefficient: float = DEFAULT_SOFTORTHOGONAL_L1,
            l2_coefficient: float = DEFAULT_SOFTORTHOGONAL_L2,
            use_matrix_scaling: bool = DEFAULT_USE_MATRIX_SCALING,
            **kwargs: Any
    ) -> None:
        """Validate the coefficients and build the optional L1/L2 sub-terms.

        :param lambda_coefficient: Non-negative orthogonality weight.
        :type lambda_coefficient: float
        :param l1_coefficient: Non-negative L1 weight.
        :type l1_coefficient: float
        :param l2_coefficient: Non-negative L2 weight.
        :type l2_coefficient: float
        :param use_matrix_scaling: Whether to divide by ``sqrt(rank)``.
        :type use_matrix_scaling: bool
        :param kwargs: Must be empty; see the class docstring.
        :raises ValueError: If any coefficient is negative.
        :raises TypeError: If any keyword argument is supplied.
        """
        super().__init__(**kwargs)

        if lambda_coefficient < 0.0:
            raise ValueError(f"lambda_coefficient must be non-negative, got {lambda_coefficient}")
        if l1_coefficient < 0.0:
            raise ValueError(f"l1_coefficient must be non-negative, got {l1_coefficient}")
        if l2_coefficient < 0.0:
            raise ValueError(f"l2_coefficient must be non-negative, got {l2_coefficient}")

        self._lambda_coefficient = lambda_coefficient
        self._l1_coefficient = l1_coefficient
        self._l2_coefficient = l2_coefficient
        self._use_matrix_scaling = use_matrix_scaling

        # Cache which terms are active so __call__ skips the inactive ones.
        self._use_lambda = self._lambda_coefficient > 0.0
        self._use_l1 = self._l1_coefficient > 0.0
        self._use_l2 = self._l2_coefficient > 0.0

        # Build the L1/L2 sub-regularizers once rather than per call.
        self._l1: Optional[keras.regularizers.L1] = None
        self._l2: Optional[keras.regularizers.L2] = None

        if self._use_l1:
            self._l1 = keras.regularizers.L1(l1=self._l1_coefficient)
        if self._use_l2:
            self._l2 = keras.regularizers.L2(l2=self._l2_coefficient)

        # The Gram orientation is a property of the kernel shape, which is not
        # known until the first call. Log it once so an expansion layer
        # switching to the input-direction Gram is visible in the logs.
        self._logged_orientation = False

        logger.debug(
            f"Initialized {self.__class__.__name__} with "
            f"lambda={lambda_coefficient}, l1={l1_coefficient}, "
            f"l2={l2_coefficient}, scaling={use_matrix_scaling}"
        )

    def _compute_deviation(
            self,
            gram: Union[keras.KerasTensor, Any],
            eye: Union[keras.KerasTensor, Any],
    ) -> Union[keras.KerasTensor, Any]:
        """Return the matrix whose squared Frobenius norm is penalized.

        :param gram: Square Gram matrix, already oriented to the reachable
            axis.
        :type gram: tensor
        :param eye: Identity matrix of matching size and dtype.
        :type eye: tensor
        :return: The deviation matrix.
        :rtype: tensor
        :raises NotImplementedError: Always; subclasses define the target.
        """
        raise NotImplementedError

    def _log_orientation_once(self, rank: int, over_output_channels: bool) -> None:
        """Log the chosen Gram orientation the first time it is known.

        :param rank: Side length of the Gram matrix.
        :type rank: int
        :param over_output_channels: ``True`` for ``W^T W``, ``False`` for
            ``W W^T``.
        :type over_output_channels: bool
        :return: Nothing.
        :rtype: None
        """
        if self._logged_orientation:
            return
        self._logged_orientation = True

        if over_output_channels:
            logger.debug(
                f"{self.__class__.__name__}: Gram over output channels, "
                f"W^T W with rank {rank}"
            )
        else:
            logger.info(
                f"{self.__class__.__name__}: units exceed fan_in, so "
                f"W^T W = I is rank deficient and unreachable. Using the input "
                f"direction Gram W W^T with rank {rank} instead. This target is "
                f"equivalent as a statement about the singular values of W, but "
                f"for the off-diagonal variant it decorrelates INPUT directions "
                f"rather than output channels."
            )

    def __call__(self, x: Union[keras.KerasTensor, Any], **kwargs) -> Union[keras.KerasTensor, Any]:
        """Compute the regularization loss for a weight tensor.

        :param x: Weight tensor to regularize.
        :type x: tensor
        :param kwargs: Additional keyword arguments Keras may pass, such as
            ``dtype``. Unused.
        :return: The scalar regularization loss.
        :rtype: tensor
        :raises ValueError: If the kernel shape has rank below 2 or is not
            fully static.
        """
        result = keras.ops.cast(0.0, dtype=x.dtype)

        # rank is needed for the size normalization even when the orthogonality
        # term is disabled, so that toggling lambda_coefficient to zero does not
        # silently change how L1 and L2 are scaled.
        gram, rank, over_output_channels = _kernel_gram(x)
        self._log_orientation_once(rank, over_output_channels)

        if self._use_lambda:
            eye = keras.ops.eye(rank, dtype=gram.dtype)
            deviation = self._compute_deviation(gram, eye)
            frob_norm_sq = keras.ops.sum(keras.ops.square(deviation))
            result = keras.ops.add(result, keras.ops.multiply(self._lambda_coefficient, frob_norm_sq))

        if self._use_l1 and self._l1 is not None:
            result = keras.ops.add(result, self._l1(x))

        if self._use_l2 and self._l2 is not None:
            result = keras.ops.add(result, self._l2(x))

        # Size normalization, applied to the whole value rather than to the
        # orthogonality term alone, which keeps the relative weighting of the
        # three terms invariant to the flag. sqrt(rank) is the divisor that
        # makes the achieved Gram deviation width independent; see the module
        # docstring. The maximum() only guards rank == 0, which no real kernel
        # produces.
        if self._use_matrix_scaling:
            scaling_factor = keras.ops.maximum(
                keras.ops.cast(math.sqrt(float(rank)), dtype=x.dtype),
                keras.ops.cast(EPSILON, dtype=x.dtype),
            )
            result = keras.ops.divide(result, scaling_factor)

        return result

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments for serialization.

        :return: A dict holding the three coefficients and the scaling flag.
        :rtype: dict
        """
        return {
            STR_L1_COEFFICIENT: self._l1_coefficient,
            STR_L2_COEFFICIENT: self._l2_coefficient,
            STR_LAMBDA_COEFFICIENT: self._lambda_coefficient,
            STR_USE_MATRIX_SCALING: self._use_matrix_scaling,
        }


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.regularizers.soft_orthogonal")
class SoftOrthogonalConstraintRegularizer(_SoftOrthogonalBaseRegularizer):
    """Penalize only the off-diagonal entries of the kernel's Gram matrix.

    Drives the compared directions toward mutual decorrelation and leaves
    weight magnitudes alone, provided ``l1_coefficient`` and
    ``l2_coefficient`` stay at 0.0.

    **Deviation matrix:**

    .. code-block:: text

        gram [rank, rank]        eye [rank, rank]
              |                        |
              |            1.0 - eye <-+
              |                 |
              +--------*--------+          elementwise
                       |
                       v
             off-diagonal entries only, diagonal zeroed
                       |
                       v
                 sum of squares

    :param lambda_coefficient: Weight for the off-diagonal Frobenius norm term.
    :type lambda_coefficient: float
    :param l1_coefficient: Weight for L1 regularization.
    :type l1_coefficient: float
    :param l2_coefficient: Weight for L2 regularization. Enabling this makes
        the regularizer constrain magnitudes, which is what separates the
        orthogonal variant from the orthonormal one.
    :type l2_coefficient: float
    :param use_matrix_scaling: Divide the whole regularization value by
        ``sqrt(rank)``.
    :type use_matrix_scaling: bool
    :param kwargs: Must be empty; see :class:`_SoftOrthogonalBaseRegularizer`.

    :raises ValueError: If any coefficient is negative.
    :raises TypeError: If any keyword argument is supplied.

    Note:
        The Gram matrix is built over whichever kernel axis is smaller. For an
        expansion layer (units > fan_in) the output channels cannot be mutually
        decorrelated at all, since there are more of them than the rank
        permits, so the off-diagonal penalty is applied to the input-direction
        Gram instead. That is a different statement about the layer, and it is
        logged at INFO the first time the regularizer is called.

    Example:
        >>> regularizer = SoftOrthogonalConstraintRegularizer(lambda_coefficient=1e-3)
        >>> dense_layer = keras.layers.Dense(64, kernel_regularizer=regularizer)
    """

    def __init__(
            self,
            lambda_coefficient: float = DEFAULT_SOFTORTHOGONAL_LAMBDA,
            l1_coefficient: float = DEFAULT_SOFTORTHOGONAL_L1,
            l2_coefficient: float = DEFAULT_SOFTORTHOGONAL_L2,
            use_matrix_scaling: bool = DEFAULT_USE_MATRIX_SCALING,
            **kwargs: Any
    ) -> None:
        """Forward every argument to the base regularizer.

        :param lambda_coefficient: Non-negative off-diagonal weight.
        :type lambda_coefficient: float
        :param l1_coefficient: Non-negative L1 weight.
        :type l1_coefficient: float
        :param l2_coefficient: Non-negative L2 weight.
        :type l2_coefficient: float
        :param use_matrix_scaling: Whether to divide by ``sqrt(rank)``.
        :type use_matrix_scaling: bool
        :param kwargs: Must be empty.
        :raises ValueError: If any coefficient is negative.
        :raises TypeError: If any keyword argument is supplied.
        """
        super().__init__(
            lambda_coefficient=lambda_coefficient,
            l1_coefficient=l1_coefficient,
            l2_coefficient=l2_coefficient,
            use_matrix_scaling=use_matrix_scaling,
            **kwargs,
        )

    def _compute_deviation(self, gram, eye):
        """Mask the diagonal, leaving only the cross-correlation entries.

        :param gram: Square Gram matrix.
        :type gram: tensor
        :param eye: Identity matrix of matching size and dtype.
        :type eye: tensor
        :return: The Gram matrix with its diagonal zeroed.
        :rtype: tensor
        """
        off_diagonal_mask = keras.ops.subtract(keras.ops.cast(1.0, dtype=gram.dtype), eye)
        return keras.ops.multiply(gram, off_diagonal_mask)


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.regularizers.soft_orthogonal")
class SoftOrthonormalConstraintRegularizer(_SoftOrthogonalBaseRegularizer):
    """Penalize the full deviation of the kernel's Gram matrix from the identity.

    Minimizing ``||G - I||_F^2`` drives every nonzero singular value of the
    kernel toward one, so the layer becomes a partial isometry.

    **Deviation matrix:**

    .. code-block:: text

        gram [rank, rank]        eye [rank, rank]
              |                        |
              +-----------  -  --------+
                       |
                       v
             G - I, diagonal included
                       |
                       v
                 sum of squares

    :param lambda_coefficient: Weight for the Frobenius norm term.
    :type lambda_coefficient: float
    :param l1_coefficient: Weight for L1 regularization.
    :type l1_coefficient: float
    :param l2_coefficient: Weight for L2 regularization. Leave it at 0.0
        unless you want a coupled magnitude penalty: for a wide layer a value
        of 1e-4 dominates the orthonormality term it is attached to.
    :type l2_coefficient: float
    :param use_matrix_scaling: Divide the whole regularization value by
        ``sqrt(rank)``.
    :type use_matrix_scaling: bool
    :param kwargs: Must be empty; see :class:`_SoftOrthogonalBaseRegularizer`.

    :raises ValueError: If any coefficient is negative.
    :raises TypeError: If any keyword argument is supplied.

    Note:
        The Gram matrix is built over whichever kernel axis is smaller, so the
        target is reachable at every shape. For units <= fan_in this asks for
        orthonormal output channels; for units > fan_in it asks for orthonormal
        input directions. Both amount to "all nonzero singular values equal
        one".

    Example:
        >>> regularizer = SoftOrthonormalConstraintRegularizer(lambda_coefficient=1e-3)
        >>> conv_layer = keras.layers.Conv2D(32, 3, kernel_regularizer=regularizer)
    """

    def __init__(
            self,
            lambda_coefficient: float = DEFAULT_SOFTORTHOGONAL_LAMBDA,
            l1_coefficient: float = DEFAULT_SOFTORTHOGONAL_L1,
            l2_coefficient: float = DEFAULT_SOFTORTHOGONAL_L2,
            use_matrix_scaling: bool = DEFAULT_USE_MATRIX_SCALING,
            **kwargs: Any
    ) -> None:
        """Forward every argument to the base regularizer.

        :param lambda_coefficient: Non-negative Frobenius norm weight.
        :type lambda_coefficient: float
        :param l1_coefficient: Non-negative L1 weight.
        :type l1_coefficient: float
        :param l2_coefficient: Non-negative L2 weight.
        :type l2_coefficient: float
        :param use_matrix_scaling: Whether to divide by ``sqrt(rank)``.
        :type use_matrix_scaling: bool
        :param kwargs: Must be empty.
        :raises ValueError: If any coefficient is negative.
        :raises TypeError: If any keyword argument is supplied.
        """
        super().__init__(
            lambda_coefficient=lambda_coefficient,
            l1_coefficient=l1_coefficient,
            l2_coefficient=l2_coefficient,
            use_matrix_scaling=use_matrix_scaling,
            **kwargs,
        )

    def _compute_deviation(self, gram, eye):
        """Return ``G - I``.

        :param gram: Square Gram matrix.
        :type gram: tensor
        :param eye: Identity matrix of matching size and dtype.
        :type eye: tensor
        :return: The deviation from the identity.
        :rtype: tensor
        """
        return keras.ops.subtract(gram, eye)


# ---------------------------------------------------------------------
