"""Soft orthogonality and orthonormality constraints for kernel weights.

This file holds `SoftOrthogonalConstraintRegularizer`, which penalizes only the
off-diagonal entries of a kernel's Gram matrix, and
`SoftOrthonormalConstraintRegularizer`, which penalizes the full deviation from
the identity, over a shared base class and the `_kernel_gram` helper. Both
build the Gram over whichever kernel axis is smaller, `W^T W` when
`units <= fan_in` and `W W^T` otherwise, because `W^T W = I` is rank deficient
for an expansion layer and `||W^T W - I||_F^2` then has an irreducible floor of
`units - fan_in`. Either orientation states the same condition on W, that every
nonzero singular value equals one, and the penalty is

    lambda * ||G - target||_F^2,  G the smaller Gram

Callers should know that `use_matrix_scaling` divides the whole value,
orthogonality plus L1 plus L2, by `sqrt(rank)`, which holds the achieved Gram
deviation independent of layer width; that the kernel shape must be rank 2 or
more and fully static; and that the L1 and L2 terms are off by default and are
coupled penalties, unlike an optimizer's decoupled weight decay.

References:
    - Bansal et al., 2018. Can We Gain More from Orthogonality
      Regularizations in Training Deep Networks?
      (https://arxiv.org/abs/1810.09102)
"""

import math
import keras
from typing import Dict, Any, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# dl_techniques.utils.tensors.gram_matrix fixes the Gram to the output-channel
# axis, the unreachable target when units > fan_in, so _kernel_gram is used.

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

EPSILON: float = 1e-12

# A coupled penalty competing with the orthogonality term for the weight norm:
# at 1e-4 it dominated by three orders of magnitude on a 512-unit layer.
DEFAULT_SOFTORTHOGONAL_L1: float = 0.0
DEFAULT_SOFTORTHOGONAL_L2: float = 0.0

DEFAULT_SOFTORTHOGONAL_LAMBDA: float = 1e-3
DEFAULT_SOFTORTHOGONAL_STDDEV: float = 0.02

# Shared by both subclasses, so their lambda_coefficient values mean the same
# effective strength.
DEFAULT_USE_MATRIX_SCALING: bool = True

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

    The smaller Gram is also the cheaper one, at
    ``O(fan_in * units * min(fan_in, units))`` rather than
    ``O(fan_in * units^2)``.

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

    Penalty pipeline:

    .. code-block:: text

        weights, rank >= 2, output channels last
              │
              ▼
        ┌──────────────────────────────────────┐
        │ _kernel_gram                         │
        │ reshape to (fan_in, units), then     │
        │ the smaller of W^T W and W W^T       │
        └──────────────────────────────────────┘
              │ gram [rank, rank], rank = min(fan_in, units)
              ▼
        ┌──────────────────────────────────────┐
        │ _compute_deviation(gram, eye)        │  (lambda > 0 only)
        │ the subclass defines the target      │
        └──────────────────────────────────────┘
              │
              ▼
        lambda * sum(deviation^2)
              │
              ├◄── l1_coefficient * L1(w)      (optional)
              ├◄── l2_coefficient * L2(w)      (optional)
              ▼
        ┌──────────────────────────────────────┐
        │ divide by sqrt(rank)                 │  (use_matrix_scaling)
        └──────────────────────────────────────┘
              │
              ▼
        scalar loss

    Gram orientation:

    .. code-block:: text

        shape              gram     rank     reads as
        ----------------   ------   ------   ----------------------
        units <= fan_in    W^T W    units    output channels are
                                             mutually orthonormal
        units >  fan_in    W W^T    fan_in   input directions are
                                             mutually orthonormal

    ``_kernel_gram`` runs on every call, including when
    ``lambda_coefficient`` is 0, so the shape checks apply and ``rank`` is
    available for the size normalization whatever the coefficients are.

    Note:
        The orientation depends on the kernel shape, which is unknown until the
        first call, so it is logged then and only then. One instance shared
        across layers of different shapes logs the orientation of the first
        kernel it sees.

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
        ``sqrt(rank)``, the side length of the Gram matrix actually used. The
        divisor covers all three terms, so toggling the flag is a pure global
        gain and never changes their relative weighting. For an L1 or L2 term
        at a width-independent absolute strength, multiply its coefficient by
        ``sqrt(rank)``, or pass ``False`` here and pre-divide
        ``lambda_coefficient``.
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
        """Validate the coefficients and build the optional L1/L2 sub-terms."""
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

        # Cached so __call__ skips the inactive terms.
        self._use_lambda = self._lambda_coefficient > 0.0
        self._use_l1 = self._l1_coefficient > 0.0
        self._use_l2 = self._l2_coefficient > 0.0

        # Built once here rather than on every call.
        self._l1: Optional[keras.regularizers.L1] = None
        self._l2: Optional[keras.regularizers.L2] = None

        if self._use_l1:
            self._l1 = keras.regularizers.L1(l1=self._l1_coefficient)
        if self._use_l2:
            self._l2 = keras.regularizers.L2(l2=self._l2_coefficient)

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

        The output-channel case logs at debug and the switched case at info,
        since only the second changes what the off-diagonal variant means.

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
        # term is off, so that setting lambda_coefficient to zero does not
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

        # Dividing the whole value keeps the relative weighting of the three
        # terms invariant to the flag. The maximum() only guards rank == 0.
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
    ``l2_coefficient`` stay at 0.0. Often used for very deep networks.

    Deviation matrix:

    .. code-block:: text

        gram [rank, rank]          eye [rank, rank]
              │                          │
              │                          ▼
              │                     1.0 - eye
              │                          │
              └────────────┬─────────────┘
                           ▼
                  elementwise product
                           │
                           ▼
              gram with a zeroed diagonal
                           │
                           ▼
                    sum of squares

    :param lambda_coefficient: Weight for the off-diagonal Frobenius norm term.
        With ``use_matrix_scaling=True``, values in ``[1e-4, 1e-2]`` are a
        reasonable starting range and need no retuning across layer widths.
    :type lambda_coefficient: float
    :param l1_coefficient: Weight for L1 regularization.
    :type l1_coefficient: float
    :param l2_coefficient: Weight for L2 regularization. Enabling it adds a
        coupled magnitude penalty, which is not the same target as the
        orthonormal variant's unit-norm directions.
    :type l2_coefficient: float
    :param use_matrix_scaling: Divide the whole regularization value by
        ``sqrt(rank)``.
    :type use_matrix_scaling: bool
    :param kwargs: Must be empty; see :class:`_SoftOrthogonalBaseRegularizer`.

    :raises ValueError: If any coefficient is negative.
    :raises TypeError: If any keyword argument is supplied.

    Note:
        The Gram matrix is built over whichever kernel axis is smaller. For an
        expansion layer (units > fan_in) the output channels cannot all be
        mutually decorrelated, since there are more of them than the rank
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
        """Forward every argument to the base regularizer."""
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
    kernel toward one, so the layer becomes a partial isometry. Often used for
    convolutional layers.

    Deviation matrix:

    .. code-block:: text

        gram [rank, rank]          eye [rank, rank]
              │                          │
              └────────────┬─────────────┘
                           ▼
                        subtract
                           │
                           ▼
              G - I, diagonal included
                           │
                           ▼
                    sum of squares

    :param lambda_coefficient: Weight for the Frobenius norm term. With
        ``use_matrix_scaling=True``, values in ``[1e-4, 1e-2]`` are a
        reasonable starting range and need no retuning across layer widths.
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
        """Forward every argument to the base regularizer."""
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
