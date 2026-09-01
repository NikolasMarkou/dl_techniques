"""
Theory and Implementation of Soft Orthogonality and Orthonormality Constraints
------------------------------------------------------------------------------

Background
----------
Neural networks often suffer from issues like vanishing/exploding gradients and covariate shift.
Orthogonal weight matrices have been shown to mitigate these issues by preserving gradient
magnitudes during backpropagation. The paper "Can We Gain More from Orthogonality Regularizations
in Training Deep CNNs?" (2018) demonstrates that soft orthogonality constraints can significantly
improve model performance.

Mathematical Foundation
----------------------
For a weight matrix W reshaped to 2D as (fan_in, units):

1. Orthogonality: the off-diagonal entries of the Gram matrix vanish, so the
   compared directions are mutually uncorrelated. Magnitudes are untouched.
2. Orthonormality: the Gram matrix equals the identity, so the compared
   directions are mutually uncorrelated AND unit norm. Equivalently, every
   nonzero singular value of W equals one, i.e. W is a partial isometry.

Enforcing these properties exactly is challenging during optimization. Instead, we use "soft"
constraints implemented as regularization terms in the loss function.

Which Gram matrix
-----------------
A (fan_in, units) matrix cannot have `units` mutually orthonormal columns when
units > fan_in: the Gram W^T W is rank deficient and ||W^T W - I||_F^2 has an
irreducible floor of (units - fan_in). Penalizing it anyway spends gradient on
an unreachable target and, under matrix scaling, hides the floor inside the
reported loss value.

This implementation therefore builds the Gram over whichever axis is smaller:

    units <= fan_in :  G = W^T W,  shape (units, units)
                       "the output channels are mutually orthonormal"

    units >  fan_in :  G = W W^T,  shape (fan_in, fan_in)
                       "the input directions are mutually orthonormal"

Both say the same thing about W - every nonzero singular value is one - and
both are reachable. The orientation is decided from the static kernel shape and
logged once per regularizer instance, because the semantics of the OFF-DIAGONAL
variant do change with it: for an expansion layer you are decorrelating input
directions, not output channels, which is the only decorrelation statement that
is available at that shape.

The smaller Gram is also the cheaper one: cost is O(fan_in * units *
min(fan_in, units)) rather than O(fan_in * units^2).

Regularization Formulations
---------------------------
1. Soft Orthogonal Constraint:
   - Penalizes the off-diagonal entries of G
   - Regularization term: ||G - diag(G)||_F^2
   - Does NOT constrain the magnitudes of the weights, provided
     l1_coefficient and l2_coefficient are left at their default of 0.0

2. Soft Orthonormal Constraint:
   - Penalizes the full deviation from the identity
   - Regularization term: ||G - I||_F^2
   - Encourages both orthogonality AND unit magnitude

Size normalization
------------------
`use_matrix_scaling` divides the ENTIRE regularization value, orthogonality
term plus L1 plus L2, by sqrt(rank), where rank is the side length of the Gram
matrix actually used.

Two properties follow, and both are deliberate.

First, sqrt(rank) is the divisor that makes the achieved Gram deviation
independent of layer width. Write the deviation entries as `eps`, the column
norm as `nu`, and the fan-in as `f`. The penalty gradient reaching one kernel
coordinate is

    |grad R|_ij ~ (4 * lambda / D) * sqrt(rank) * eps * nu / sqrt(f)

the sqrt(rank) coming from the sum over Gram entries. Balancing against a
per-coordinate task gradient `g` gives

    eps = g * D * sqrt(f) / (4 * lambda * sqrt(rank) * nu)

so D = 1 gives eps ~ rank^-0.5, D = rank^2 gives eps ~ rank^1.5, and
D = sqrt(rank) gives eps ~ rank^0. Earlier releases used rank^2, which
over-corrected by rank^1.5 and left wide layers effectively unregularized.

A residual sqrt(fan_in) dependence survives and is left in place: for a
normalized layer the per-coordinate gradient `g` carries a compensating
1/sqrt(fan_in), so removing it here would double-count.

Second, the divisor is applied uniformly rather than to the orthogonality term
alone. Toggling the flag is then a pure global gain: it never changes the
balance between the orthogonality, L1, and L2 terms. Earlier releases scaled
only the orthogonality term, so flipping the flag silently rescaled that
balance by rank^2. If you want L1 or L2 at an absolute per-coordinate strength
independent of width, multiply their coefficients by sqrt(rank) yourself, or
set use_matrix_scaling=False and pre-divide lambda_coefficient by sqrt(rank).

Migration: a given lambda_coefficient is rank^1.5 stronger than in the previous
release for the orthonormal regularizer, and rank^-0.5 weaker for the
orthogonal one, which previously defaulted to no scaling at all. Previously
tuned values do not transfer.

Implementation Details
---------------------
1. SoftOrthogonalConstraintRegularizer:
   - Builds the smaller Gram matrix
   - Masks the diagonal using (1.0 - eye)
   - Computes the squared Frobenius norm of the masked matrix

2. SoftOrthonormalConstraintRegularizer:
   - Builds the smaller Gram matrix
   - Computes the squared Frobenius norm of (G - I)

Additional Features
------------------
1. Hybrid regularization:
   - Both regularizers support optional L1/L2 terms, OFF by default. An L2 term
     enabled here is a coupled penalty: it passes through the optimizer's
     preconditioner, unlike the decoupled decay of AdamW, and it competes with
     the orthogonality term for control of the weight norm. Prefer the
     optimizer's weight_decay unless you specifically want the coupled
     behaviour.
   - Lambda coefficient controls strength of the orthogonality constraint

2. Numerical stability:
   - Epsilon floors the size-normalization divisor for the degenerate rank == 0
     case only; it is not a division-by-zero guard in any realistic shape

3. Performance:
   - Caches flags for enabled regularization terms (_use_lambda, _use_l1, _use_l2)
   - Creates L1/L2 regularizers only once during initialization
   - Note that the dominant cost is the Gram matmul itself,
     O(fan_in * units * min(fan_in, units)) per step, which for a wide layer
     with a small batch can exceed the cost of the layer's own forward pass

Practical Benefits
-----------------
1. Improved gradient flow during training
2. Better conditioning of the optimization landscape
3. Enhanced generalization performance
4. Reduced sensitivity to initialization
5. Faster convergence in deep networks

Usage Guidelines
---------------
- For very deep networks, use SoftOrthogonalConstraintRegularizer
- For convolutional layers, often SoftOrthonormalConstraintRegularizer works better
- Both classes now share the same size normalization and the same
  use_matrix_scaling default, so their lambda_coefficient values are directly
  comparable. They were not comparable before: they differed by rank^2.
- With use_matrix_scaling=True, lambda_coefficient in [1e-4, 1e-2] is a
  reasonable starting range and does not need retuning across layer widths
- Leave l1_coefficient and l2_coefficient at 0.0 unless you have a specific
  reason; use the optimizer's weight_decay for magnitude control
"""

import math
import keras
from keras import ops
from typing import Dict, Any, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# NOTE: dl_techniques.utils.tensors.gram_matrix is deliberately NOT used here.
# It fixes the Gram orientation to the output-channel axis, which is the
# unreachable target when units > fan_in. The orientation is chosen explicitly
# below instead. See "Which Gram matrix" in the module docstring.

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

EPSILON: float = 1e-12

# Off by default. An L2 term here is a COUPLED penalty and competes directly
# with the orthogonality term for control of the weight norm; at the previous
# default of 1e-4 it dominated it by roughly three orders of magnitude for a
# 512-unit layer. Use the optimizer's weight_decay instead unless the coupled
# behaviour is what you want.
DEFAULT_SOFTORTHOGONAL_L1: float = 0.0
DEFAULT_SOFTORTHOGONAL_L2: float = 0.0

DEFAULT_SOFTORTHOGONAL_LAMBDA: float = 1e-3
DEFAULT_SOFTORTHOGONAL_STDDEV: float = 0.02

# Both subclasses share this default. They previously disagreed, which made
# their lambda_coefficient values differ in effective strength by rank^2.
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

    When units <= fan_in the Gram is taken over output channels, W^T W, and
    W^T W = I asks for orthonormal output channels. When units > fan_in that
    target is rank deficient and unreachable, so the Gram is taken over input
    directions instead, W W^T, and W W^T = I asks for orthonormal input
    directions. Both express the same condition on W - every nonzero singular
    value equals one - and only one of them is reachable at any given shape.

    Parameters
    ----------
    x : tensor
        Weight tensor of any rank >= 2, output channels last.

    Returns
    -------
    tuple of (gram, rank, over_output_channels)
        gram: square Gram matrix of side `rank`
        rank: min(fan_in, units)
        over_output_channels: True if the Gram is W^T W, False if it is W W^T

    Raises
    ------
    ValueError
        If the kernel shape is not fully static, or has rank < 2.
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

    # (fan_in, units): rows are input directions, columns are output channels
    w2d = ops.reshape(x, (fan_in, units))

    if units <= fan_in:
        # Reachable: `units` orthonormal columns in R^fan_in
        gram = ops.matmul(ops.transpose(w2d), w2d)
        return gram, units, True

    # units > fan_in. W^T W is rank deficient, so constrain the other side.
    gram = ops.matmul(w2d, ops.transpose(w2d))
    return gram, fan_in, False


# ---------------------------------------------------------------------


class _SoftOrthogonalBaseRegularizer(keras.regularizers.Regularizer):
    """Base class for soft orthogonality constraint regularizers.

    Provides shared initialization, optional L1/L2 terms, size normalization,
    and serialization logic. Subclasses implement _compute_deviation to define
    the specific orthogonality target.

    Parameters
    ----------
    lambda_coefficient : float
        Weight for the orthogonality Frobenius norm term
    l1_coefficient : float
        Weight for L1 regularization. Defaults to 0.0. This is a COUPLED
        penalty and competes with the orthogonality term for control of the
        weight norm.
    l2_coefficient : float
        Weight for L2 regularization. Defaults to 0.0. Same caveat as L1;
        prefer the optimizer's decoupled weight_decay.
    use_matrix_scaling : bool
        Divide the entire regularization value by sqrt(rank), where rank is the
        side length of the Gram matrix actually used. sqrt(rank) is the divisor
        that makes the achieved Gram deviation independent of layer width; see
        the module docstring for the derivation. Applying it to every term
        rather than to the orthogonality term alone keeps the balance between
        terms invariant to this flag.
    """

    def __init__(
            self,
            lambda_coefficient: float = DEFAULT_SOFTORTHOGONAL_LAMBDA,
            l1_coefficient: float = DEFAULT_SOFTORTHOGONAL_L1,
            l2_coefficient: float = DEFAULT_SOFTORTHOGONAL_L2,
            use_matrix_scaling: bool = DEFAULT_USE_MATRIX_SCALING,
            **kwargs: Any
    ) -> None:
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

        # Cache flags for performance optimization
        self._use_lambda = self._lambda_coefficient > 0.0
        self._use_l1 = self._l1_coefficient > 0.0
        self._use_l2 = self._l2_coefficient > 0.0

        # Initialize L1/L2 regularizers once for efficiency
        self._l1: Optional[keras.regularizers.L1] = None
        self._l2: Optional[keras.regularizers.L2] = None

        if self._use_l1:
            self._l1 = keras.regularizers.L1(l1=self._l1_coefficient)
        if self._use_l2:
            self._l2 = keras.regularizers.L2(l2=self._l2_coefficient)

        # The Gram orientation is a property of the kernel shape, which is not
        # known until the first call. Log it once so that an expansion layer
        # quietly switching to the input-direction Gram is visible in the logs
        # rather than only in the maths.
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
        """Compute the deviation matrix from the orthogonality target.

        Parameters
        ----------
        gram : tensor
            Square Gram matrix, already oriented to the reachable axis
        eye : tensor
            Identity matrix of matching size and dtype

        Returns
        -------
        tensor
            Matrix whose squared Frobenius norm will be penalized
        """
        raise NotImplementedError

    def _log_orientation_once(self, rank: int, over_output_channels: bool) -> None:
        """Log the chosen Gram orientation the first time it is known."""
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
        """Compute regularization loss for given weights.

        Parameters
        ----------
        x : Union[keras.KerasTensor, Any]
            Weight tensor to regularize
        **kwargs : Any
            Additional keyword arguments (e.g., dtype) that Keras may pass

        Returns
        -------
        Union[keras.KerasTensor, Any]
            Scalar regularization loss value
        """
        result = ops.cast(0.0, dtype=x.dtype)

        # rank is needed for the size normalization even when the orthogonality
        # term is disabled, so that toggling lambda_coefficient to zero does not
        # silently change how L1 and L2 are scaled.
        gram, rank, over_output_channels = _kernel_gram(x)
        self._log_orientation_once(rank, over_output_channels)

        # Add Frobenius norm term if enabled
        if self._use_lambda:
            eye = ops.eye(rank, dtype=gram.dtype)
            deviation = self._compute_deviation(gram, eye)
            frob_norm_sq = ops.sum(ops.square(deviation))
            result = ops.add(result, ops.multiply(self._lambda_coefficient, frob_norm_sq))

        # Add L1 regularization if enabled
        if self._use_l1 and self._l1 is not None:
            result = ops.add(result, self._l1(x))

        # Add L2 regularization if enabled
        if self._use_l2 and self._l2 is not None:
            result = ops.add(result, self._l2(x))

        # Size normalization, applied to the WHOLE value rather than to the
        # orthogonality term alone. This keeps the relative weighting of the
        # three terms invariant to the flag. sqrt(rank) is the divisor that
        # makes the achieved Gram deviation width independent; see the module
        # docstring. The maximum() only guards rank == 0, which no real kernel
        # produces.
        if self._use_matrix_scaling:
            scaling_factor = ops.maximum(
                ops.cast(math.sqrt(float(rank)), dtype=x.dtype),
                ops.cast(EPSILON, dtype=x.dtype),
            )
            result = ops.divide(result, scaling_factor)

        return result

    def get_config(self) -> Dict[str, Any]:
        """Get regularizer configuration for serialization.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing configuration parameters
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
    """Implements soft orthogonality constraint regularization.

    Penalizes deviations from orthogonality by minimizing the squared Frobenius
    norm of the off-diagonal entries of the Gram matrix. Weight magnitudes are
    left alone, provided l1_coefficient and l2_coefficient stay at 0.0.

    Parameters
    ----------
    lambda_coefficient : float, optional
        Weight for the off-diagonal Frobenius norm term, by default 1e-3
    l1_coefficient : float, optional
        Weight for L1 regularization, by default 0.0
    l2_coefficient : float, optional
        Weight for L2 regularization, by default 0.0. Enabling this makes the
        regularizer constrain magnitudes, which is the opposite of what the
        orthogonal (as distinct from orthonormal) variant is for.
    use_matrix_scaling : bool, optional
        Divide the whole regularization value by sqrt(rank), by default True.
        This default changed: it was False, which made this class differ from
        SoftOrthonormalConstraintRegularizer by rank^2 at equal
        lambda_coefficient.
    **kwargs : Any
        Additional arguments passed to parent regularizer

    Notes
    -----
    The Gram matrix is built over whichever kernel axis is smaller. For an
    expansion layer (units > fan_in) the output channels cannot be mutually
    decorrelated at all, since there are more of them than the rank permits, so
    the off-diagonal penalty is applied to the input-direction Gram instead.
    That is a different statement about the layer, and it is logged at INFO the
    first time the regularizer is called.

    Examples
    --------
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
        super().__init__(
            lambda_coefficient=lambda_coefficient,
            l1_coefficient=l1_coefficient,
            l2_coefficient=l2_coefficient,
            use_matrix_scaling=use_matrix_scaling,
            **kwargs,
        )

    def _compute_deviation(self, gram, eye):
        """Mask the diagonal, leaving only the cross-correlation entries."""
        off_diagonal_mask = ops.subtract(ops.cast(1.0, dtype=gram.dtype), eye)
        return ops.multiply(gram, off_diagonal_mask)


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.regularizers.soft_orthogonal")
class SoftOrthonormalConstraintRegularizer(_SoftOrthogonalBaseRegularizer):
    """Implements soft orthonormality constraint regularization.

    Penalizes deviations from orthonormality by minimizing the squared
    Frobenius norm of (G - I), which drives every nonzero singular value of the
    kernel toward one.

    Parameters
    ----------
    lambda_coefficient : float, optional
        Weight for the Frobenius norm term, by default 1e-3
    l1_coefficient : float, optional
        Weight for L1 regularization, by default 0.0
    l2_coefficient : float, optional
        Weight for L2 regularization, by default 0.0. This default changed: it
        was 1e-4, so an L2 penalty was active in every instance that did not
        explicitly disable it, and for a wide layer it dominated the
        orthonormality term it was attached to.
    use_matrix_scaling : bool, optional
        Divide the whole regularization value by sqrt(rank), by default True.
        The divisor changed: it was rank^2, which over-corrected by rank^1.5.
    **kwargs : Any
        Additional arguments passed to parent regularizer

    Notes
    -----
    The Gram matrix is built over whichever kernel axis is smaller, so the
    target is reachable at every shape. For units <= fan_in this asks for
    orthonormal output channels; for units > fan_in it asks for orthonormal
    input directions. Both amount to "all nonzero singular values equal one".

    A given lambda_coefficient is rank^1.5 stronger than in the previous
    release. Divide previously tuned values by rank^1.5 as a starting point.

    Examples
    --------
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
        super().__init__(
            lambda_coefficient=lambda_coefficient,
            l1_coefficient=l1_coefficient,
            l2_coefficient=l2_coefficient,
            use_matrix_scaling=use_matrix_scaling,
            **kwargs,
        )

    def _compute_deviation(self, gram, eye):
        """Compute the deviation from identity: G - I."""
        return ops.subtract(gram, eye)


# ---------------------------------------------------------------------
