"""
SIGReg — Sketch Isotropic Gaussian Regularizer.

Implements the regularizer from LeWM (Sobal et al., 2024):

.. math::

    \\text{SIGReg}(Z) = \\mathrm{mean}_j \\sum_k w_k \\big[
       (\\overline{\\cos(t_k (Z A)_j)} - \\phi(t_k))^2
       + (\\overline{\\sin(t_k (Z A)_j)})^2
    \\big] \\cdot N

where :math:`Z \\in \\mathbb{R}^{N \\times D}` is a batch of feature vectors,
:math:`A \\in \\mathbb{R}^{D \\times P}` is a freshly sampled, column-normalized
Gaussian projection matrix, :math:`t_k \\in [0, 3]` are integration knots,
:math:`\\phi(t) = \\exp(-t^2/2)` is the standard-Gaussian characteristic
function (real part), and :math:`w_k` are trapezoidal-rule weights (already
multiplied by the window :math:`\\phi`). The regularizer pushes the empirical
distribution of random 1-D projections of `Z` toward a standard Gaussian
characteristic function — a sliced isotropic-Gaussian fit — in the spirit of
sliced-Wasserstein regularization but using characteristic-function residuals.

**Input convention.** Input is shaped `(..., N, D)` — typically `(T, B, D)`.
The characteristic-function estimate is averaged over the **last-but-one axis
`N`** (the sample axis); for a `(T, B, D)` input that is the batch axis `B`.
One statistic is computed per leading index (per timestep when the input is
`(T, B, D)`), and the scalar returned is the mean over all of them. The
reference PyTorch implementation reduces the same axis.

**Why a Layer and not a Regularizer.** Upstream SIGReg depends on the forward
activations (not weights), samples a fresh random projection each call, and
has buffers (t, phi, weights). Modeling it as `keras.layers.Layer` is the
cleanest fit: it gets a standard `build` / `call` / `get_config` lifecycle,
plays nicely with `model.add_loss(...)`, and the buffers are tracked as
non-trainable weights so they are (1) saved with the model and (2) placed on
the right device.

References:
    - Upstream PyTorch: `/tmp/lewm_source/module.py:SIGReg`.
"""

from typing import Any, Dict, Optional, Tuple

import keras
import numpy as np
from keras import ops

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SIGRegLayer(keras.layers.Layer):
    """Sketch Isotropic Gaussian Regularizer as a Keras Layer.

    **Intent**: Push the empirical distribution of random 1-D projections of
    an activation tensor toward a standard Gaussian, by penalizing the
    squared residual between the empirical characteristic function (real and
    imaginary parts evaluated on a fixed knot grid) and the target
    :math:`\\phi(t) = \\exp(-t^2/2)`. Used as a side-loss via
    ``model.add_loss(...)``; the layer itself returns a scalar 0-D tensor.

    **Architecture**:

    ```
    Input(..., N, D)
           |
           v
    +--------------------+   (sampled per call)
    | A ~ N(0, I)        |   shape (D, num_proj), columns L2-normalized
    +--------------------+
           |
           v
    proj @ A  --->  x  shape (..., N, num_proj)
           |
           v
    outer with t (knots,)  --->  x_t shape (..., N, num_proj, knots)
           |
           v
    cos_mean, sin_mean    averaged over the N (sample) axis
           |
           v
    err = (cos_mean - phi)^2 + sin_mean^2     shape (..., num_proj, knots)
           |
           v
    statistic = err @ weights_  ---> (..., num_proj)
           |  (* N)
           v
    return  ops.mean(statistic)   shape ()
    ```

    The integration grid ``t``, the target window ``phi`` and the
    pre-windowed trapezoidal weights ``weights_`` are stored as non-trainable
    weights so that they are serialized with the layer and placed on the
    correct device.

    Args:
        knots: Integer, number of trapezoidal-rule integration knots on
            ``[0, 3]``. Must be ``>= 2``. Defaults to ``17`` (upstream
            default).
        num_proj: Integer, number of random slicing directions sampled per
            forward pass. Must be ``>= 1``. Defaults to ``1024`` (upstream
            default).
        seed: Optional integer seed for the random projection (for
            reproducible tests). When ``None`` (default) the projection is
            re-sampled each call using ``keras.random.normal`` without a
            fixed seed — matching upstream which uses ``torch.randn`` with
            the global generator.
        **kwargs: Passthrough to ``keras.layers.Layer``.

    Input shape:
        N-D tensor with shape ``(..., N, D)`` where ``D`` (last dimension)
        must be statically known and ``N`` is the sample axis being averaged
        over. Rank must be ``>= 2``. Typical use: ``(T, B, D)``.

    Output shape:
        Scalar (0-D tensor).

    Example:
        ```python
        x = keras.Input(shape=(8, 16))           # (B, T=8, D=16)
        sig = SIGRegLayer(knots=17, num_proj=1024, seed=0)
        loss = sig(x)                            # scalar
        model = keras.Model(x, loss)
        ```

    Raises:
        ValueError: If ``knots < 2`` or ``num_proj < 1`` at construction
            time, or if at ``build()`` the input rank is ``< 2`` or the last
            dimension is ``None``.
    """

    def __init__(
        self,
        knots: int = 17,
        num_proj: int = 1024,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if knots < 2:
            raise ValueError(f"knots must be >= 2, got {knots}")
        if num_proj < 1:
            raise ValueError(f"num_proj must be >= 1, got {num_proj}")

        # Store ALL configuration — needed for get_config() and re-creation
        # of sub-state in build(). No weight creation here (Golden Rule).
        self.knots = knots
        self.num_proj = num_proj
        self.seed = seed
        self._seed_gen = (
            keras.random.SeedGenerator(seed) if seed is not None else None
        )

        # Buffer attributes created in build(); declared here for clarity.
        self.t = None
        self.phi = None
        self.weights_ = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Validate input shape and create the non-trainable buffer weights.

        Computes the integration grid, the Gaussian window evaluated on the
        grid, and the pre-windowed trapezoidal weights once. These are
        shape-independent of the input but registered here (per the Keras 3
        Golden Rule that all ``add_weight`` calls live in ``build``).
        """
        if len(input_shape) < 2:
            raise ValueError(
                f"SIGRegLayer expects input with rank >= 2, got shape {input_shape}"
            )
        if input_shape[-1] is None:
            raise ValueError(
                "SIGRegLayer requires a known last dimension (D). "
                f"Got input_shape={input_shape}."
            )

        # Pre-compute the integration grid, window, and trapezoidal weights
        # once — these are static across all calls. Computed in numpy so the
        # values are baked into Constant initializers (identical bit-for-bit
        # to the pre-refactor in-__init__ values).
        t_np = np.linspace(0.0, 3.0, self.knots, dtype="float32")
        dt = 3.0 / (self.knots - 1)
        weights_np = np.full((self.knots,), 2.0 * dt, dtype="float32")
        weights_np[0] = dt
        weights_np[-1] = dt
        window_np = np.exp(-0.5 * t_np * t_np).astype("float32")
        final_weights_np = (weights_np * window_np).astype("float32")

        # Store as non-trainable weights so they serialize with the layer.
        self.t = self.add_weight(
            name="t",
            shape=(self.knots,),
            dtype="float32",
            initializer=keras.initializers.Constant(t_np.tolist()),
            trainable=False,
        )
        self.phi = self.add_weight(
            name="phi",
            shape=(self.knots,),
            dtype="float32",
            initializer=keras.initializers.Constant(window_np.tolist()),
            trainable=False,
        )
        self.weights_ = self.add_weight(
            name="weights",
            shape=(self.knots,),
            dtype="float32",
            initializer=keras.initializers.Constant(final_weights_np.tolist()),
            trainable=False,
        )

        super().build(input_shape)

    def call(
        self, proj: keras.KerasTensor, training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Compute the SIGReg statistic.

        Args:
            proj: Tensor of shape ``(..., N, D)`` — typically ``(T, B, D)``
                as in upstream. Averaging happens over the last-but-one axis
                ``N``.
            training: Unused; SIGReg runs identically in train/eval.

        Returns:
            Scalar tensor (0-D).
        """
        D = proj.shape[-1]
        if D is None:
            raise ValueError("SIGRegLayer requires a known last dimension.")

        # Sample random projection matrix A: (D, num_proj).
        if self._seed_gen is not None:
            A = keras.random.normal(
                (D, self.num_proj), dtype=proj.dtype, seed=self._seed_gen
            )
        else:
            A = keras.random.normal((D, self.num_proj), dtype=proj.dtype)

        # Normalize each column of A to unit L2 norm.
        col_norm = ops.sqrt(ops.sum(ops.square(A), axis=0, keepdims=True) + 1e-12)
        A = A / col_norm  # (D, num_proj)

        # proj @ A: (..., N, num_proj)
        x = ops.matmul(proj, A)

        # Outer product with t: (..., N, num_proj, knots)
        x_t = ops.expand_dims(x, axis=-1) * ops.reshape(self.t, (1,) * x.ndim + (-1,))

        # Average cos and sin along axis -3 (the "N" axis).
        cos_mean = ops.mean(ops.cos(x_t), axis=-3)  # (..., num_proj, knots)
        sin_mean = ops.mean(ops.sin(x_t), axis=-3)

        # Residual vs target characteristic function phi(t) = exp(-t^2/2).
        # phi shape (knots,) broadcasts over num_proj.
        err = ops.square(cos_mean - self.phi) + ops.square(sin_mean)
        # Weighted sum over knots: err @ weights_ → (..., num_proj)
        statistic = ops.matmul(err, self.weights_)

        return ops.mean(statistic)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple:
        """Output is a scalar."""
        return ()

    def get_config(self) -> Dict[str, Any]:
        """Return all ``__init__`` parameters for serialization."""
        config = super().get_config()
        config.update(
            {
                "knots": self.knots,
                "num_proj": self.num_proj,
                "seed": self.seed,
            }
        )
        return config

# ---------------------------------------------------------------------
