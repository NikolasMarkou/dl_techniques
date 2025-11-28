"""
Poincaré Ball Model Geometry Utilities.

This module provides stateless utility functions for Riemannian geometry operations
in the Poincaré Ball model of hyperbolic space. All operations are designed for
numerical stability following the principles outlined in Arevalo et al.

Mathematical Background:
    The Poincaré ball model (D^n_c, g) represents hyperbolic space as points
    within the open unit ball with Riemannian metric scaled by curvature c.

    Key operations:
    - Exponential map: Maps tangent vectors (Euclidean) to manifold (Hyperbolic)
    - Logarithmic map: Maps manifold points to tangent space
    - Möbius addition: Hyperbolic equivalent of vector addition
    - Projection: Ensures points remain within valid ball boundary

References:
    - Eq 17: Möbius Addition
    - Eq 19: Exponential and Logarithmic maps at origin
    - Appendix B: Numerical stability considerations
"""

import keras
from keras import ops
import tensorflow as tf
from typing import Union

# ---------------------------------------------------------------------

class PoincareMath:
    """
    Stateless utility class for Poincaré Ball hyperbolic geometry operations.

    This class provides low-level primitives for working with hyperbolic embeddings,
    including exponential/logarithmic maps between Euclidean tangent space and the
    hyperbolic manifold, Möbius addition for geometric operations, and projection
    to ensure numerical stability.

    All methods are designed to be numerically stable through careful epsilon
    handling and boundary management.

    Attributes:
        eps: Small constant to prevent division by zero and NaN gradients.
        boundary_eps: Safety margin from ball boundary to prevent numerical issues.

    Example:
        ```python
        math_util = PoincareMath(eps=1e-5)

        # Map Euclidean vector to hyperbolic space
        euclidean_vector = ops.random.normal((32, 64))
        hyperbolic_point = math_util.exp_map_0(euclidean_vector, c=1.0)

        # Add bias in hyperbolic space
        bias = ops.random.normal((64,))
        bias_hyp = math_util.exp_map_0(bias, c=1.0)
        shifted = math_util.mobius_add(hyperbolic_point, bias_hyp, c=1.0)

        # Map back to Euclidean space
        tangent = math_util.log_map_0(shifted, c=1.0)
        ```
    """

    def __init__(self, eps: float = 1e-5) -> None:
        """
        Initialize Poincaré Ball math utilities.

        Args:
            eps: Epsilon value for numerical stability in division and square roots.
                Prevents NaN gradients when norms approach zero. Defaults to 1e-5.
        """
        self.eps = eps
        self.boundary_eps = 1e-4

    def safe_norm(
            self,
            x: keras.KerasTensor,
            axis: int = -1,
            keepdims: bool = False
    ) -> keras.KerasTensor:
        """
        Compute Euclidean norm with gradient-safe handling of zero vectors.

        The standard norm computation ||x|| = sqrt(sum(x^2)) produces NaN gradients
        when x=0 because d/dx sqrt(x)|_{x=0} is undefined. This method ensures
        the minimum norm is eps to maintain valid gradients everywhere.

        Args:
            x: Input tensor of shape (..., D).
            axis: Axis along which to compute the norm. Defaults to -1 (last axis).
            keepdims: Whether to keep the reduced dimension. Defaults to False.

        Returns:
            Tensor of norms with shape (..., 1) if keepdims=True, else (...).
            All values are guaranteed >= eps.

        Mathematical Operation:
            ||x|| = max(sqrt(sum(x_i^2)), eps)
        """
        norm = ops.norm(x, axis=axis, keepdims=keepdims)
        return ops.maximum(norm, self.eps)

    def project(
            self,
            x: keras.KerasTensor,
            c: Union[float, keras.KerasTensor]
    ) -> keras.KerasTensor:
        """
        Project points onto the Poincaré ball to ensure valid hyperbolic coordinates.

        The Poincaré ball with curvature c has radius 1/sqrt(c). Points must satisfy
        ||x|| < 1/sqrt(c) to be valid. This operation clamps points that exceed the
        boundary (minus a safety margin) back onto a valid radius.

        Critical for numerical stability before logarithmic map operations, which
        become singular at the boundary.

        Args:
            x: Hyperbolic points with shape (..., D).
            c: Curvature parameter (c > 0). Can be scalar or tensor.

        Returns:
            Projected points with shape (..., D), guaranteed to satisfy
            ||x|| <= (1/sqrt(c)) - boundary_eps.

        Mathematical Operation:
            For each point:
            - If ||x|| < max_radius: return x unchanged
            - If ||x|| >= max_radius: return x * (max_radius / ||x||)

            where max_radius = (1/sqrt(c)) - boundary_eps
        """
        norm_x = self.safe_norm(x, axis=-1, keepdims=True)
        max_r = (1.0 / ops.sqrt(c)) - self.boundary_eps

        cond = norm_x >= max_r
        projected = x * (max_r / norm_x)
        return tf.where(cond, projected, x)

    def exp_map_0(
            self,
            v: keras.KerasTensor,
            c: Union[float, keras.KerasTensor]
    ) -> keras.KerasTensor:
        """
        Exponential map at the origin: Tangent space (Euclidean) → Manifold (Hyperbolic).

        Maps a vector from the flat tangent space at the origin to a point on the
        curved hyperbolic manifold. This is the fundamental operation for embedding
        Euclidean representations into hyperbolic space.

        The exponential map preserves the direction of v but distorts magnitude:
        points far from origin in tangent space get compressed near the ball boundary.

        Args:
            v: Tangent vectors (Euclidean) with shape (..., D).
            c: Curvature parameter (c > 0). Can be scalar or tensor.

        Returns:
            Hyperbolic points with shape (..., D), satisfying ||x|| < 1/sqrt(c).

        Mathematical Operation (Eq 19):
            x = tanh(sqrt(c) * ||v||) * (v / (sqrt(c) * ||v||))
              = (tanh(sqrt(c) * ||v||) / (sqrt(c) * ||v||)) * v

        Special case:
            As ||v|| → 0, we use the limit tanh(x)/x → 1, returning v unchanged.

        Note:
            This operation is differentiable and preserves the origin: exp_0(0) = 0.
        """
        sqrt_c = ops.sqrt(c)
        norm_v = self.safe_norm(v, axis=-1, keepdims=True)

        scale = ops.tanh(sqrt_c * norm_v) / (sqrt_c * norm_v)

        return tf.where(norm_v < self.eps, v, v * scale)

    def log_map_0(
            self,
            y: keras.KerasTensor,
            c: Union[float, keras.KerasTensor]
    ) -> keras.KerasTensor:
        """
        Logarithmic map at the origin: Manifold (Hyperbolic) → Tangent space (Euclidean).

        Inverse of the exponential map. Maps a point from the curved hyperbolic
        manifold to a vector in the flat tangent space at the origin. This enables
        efficient aggregation operations in Euclidean space.

        The logarithmic map expands distances: points near the boundary get mapped
        to large magnitude vectors in the tangent space.

        Args:
            y: Hyperbolic points with shape (..., D).
            c: Curvature parameter (c > 0). Can be scalar or tensor.

        Returns:
            Tangent vectors (Euclidean) with shape (..., D).

        Mathematical Operation (Eq 19):
            v = atanh(sqrt(c) * ||y||) * (y / (sqrt(c) * ||y||))
              = (atanh(sqrt(c) * ||y||) / (sqrt(c) * ||y||)) * y

        Special case:
            As ||y|| → 0, we use the limit atanh(x)/x → 1, returning y unchanged.

        Note:
            Input points are clamped to stay within valid ball radius to prevent
            atanh(1) = infinity. This operation inverts exp_map_0: log_0(exp_0(v)) = v.
        """
        sqrt_c = ops.sqrt(c)
        norm_y = self.safe_norm(y, axis=-1, keepdims=True)

        max_r = (1.0 / sqrt_c) - self.boundary_eps
        norm_y = ops.minimum(norm_y, max_r)

        scale = ops.arctanh(sqrt_c * norm_y) / (sqrt_c * norm_y)
        return tf.where(norm_y < self.eps, y, y * scale)

    def mobius_add(
            self,
            x: keras.KerasTensor,
            y: keras.KerasTensor,
            c: Union[float, keras.KerasTensor]
    ) -> keras.KerasTensor:
        """
        Möbius addition: Hyperbolic equivalent of vector addition in Euclidean space.

        In hyperbolic geometry, standard vector addition x + y doesn't preserve the
        metric structure. Möbius addition is the geometrically correct operation that
        accounts for the curvature of space.

        Used in sHGCN specifically for adding bias vectors to feature matrices in
        hyperbolic space, enabling the model to learn hierarchical transformations.

        Args:
            x: First hyperbolic points with shape (..., D).
            y: Second hyperbolic points with shape (..., D) or broadcastable shape
                like (1, D) for bias addition.
            c: Curvature parameter (c > 0). Can be scalar or tensor.

        Returns:
            Möbius sum (x ⊕_c y) with shape (..., D).

        Mathematical Operation (Eq 17):
            x ⊕_c y = [(1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y] /
                      [1 + 2c<x,y> + c²||x||²||y||²]

        where:
            <x,y> = sum(x_i * y_i) is the Euclidean inner product
            ||x||² = sum(x_i²) is the squared Euclidean norm

        Properties:
            - Commutative: x ⊕ y = y ⊕ x
            - Associative: (x ⊕ y) ⊕ z = x ⊕ (y ⊕ z)
            - Identity: x ⊕ 0 = x
            - Preserves the ball: if x,y in D^n_c, then x ⊕ y in D^n_c
        """
        xy = ops.sum(x * y, axis=-1, keepdims=True)
        x2 = ops.sum(ops.square(x), axis=-1, keepdims=True)
        y2 = ops.sum(ops.square(y), axis=-1, keepdims=True)

        den = 1.0 + 2.0 * c * xy + (c ** 2) * x2 * y2
        den = ops.maximum(den, self.eps)

        coeff1 = 1.0 + 2.0 * c * xy + c * y2
        term1 = coeff1 * x

        coeff2 = 1.0 - c * x2
        term2 = coeff2 * y

        return (term1 + term2) / den

# ---------------------------------------------------------------------
