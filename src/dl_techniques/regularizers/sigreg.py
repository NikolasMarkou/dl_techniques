"""SIGReg, the Sketch Isotropic Gaussian Regularizer.

Provides :class:`SIGRegLayer`, which pushes the empirical distribution of
random 1-D projections of an activation tensor toward a standard Gaussian. It
is the regularizer from LeWM (Sobal et al., 2024).

The statistic is

.. math::

    \\text{SIGReg}(Z) = \\mathrm{mean}_j \\sum_k w_k \\big[
       (\\overline{\\cos(t_k (Z A)_j)} - \\phi(t_k))^2
       + (\\overline{\\sin(t_k (Z A)_j)})^2
    \\big] \\cdot N

The trailing :math:`\\cdot N` factor is applied only when the layer is
constructed with ``normalize_by_n=True``; the default, ``False``, omits it
(see :class:`SIGRegLayer`'s constructor for the naming caveat relative to
the upstream reference).

where :math:`Z \\in \\mathbb{R}^{N \\times D}` is a batch of feature vectors,
:math:`A \\in \\mathbb{R}^{D \\times P}` is a freshly sampled, column-normalized
Gaussian projection matrix, :math:`t_k \\in [0, 3]` are integration knots,
:math:`\\phi(t) = \\exp(-t^2/2)` is the real part of the standard-Gaussian
characteristic function, and :math:`w_k` are trapezoidal-rule weights already
multiplied by the window :math:`\\phi`. This is a sliced isotropic-Gaussian
fit, in the spirit of sliced-Wasserstein regularization but using
characteristic-function residuals.

Input convention
----------------
Input is shaped ``(..., N, D)``, typically ``(T, B, D)``. The
characteristic-function estimate is averaged over the last-but-one axis ``N``,
the sample axis; for a ``(T, B, D)`` input that is the batch axis ``B``. One
statistic is computed per leading index, so per timestep for a ``(T, B, D)``
input, and the returned scalar is the mean over all of them. The reference
PyTorch implementation reduces the same axis.

Why a Layer and not a Regularizer
---------------------------------
SIGReg depends on the forward activations rather than on weights, samples a
fresh random projection each call, and holds buffers (``t``, ``phi``,
``weights_``). ``keras.layers.Layer`` gives it a standard ``build`` / ``call``
/ ``get_config`` lifecycle, works with ``model.add_loss(...)``, and tracks the
buffers as non-trainable weights so they are saved with the model and placed on
the right device.

References:
    - Upstream PyTorch: the ``SIGReg`` module of LeWM (Sobal et al., 2024).
"""

from typing import Any, Dict, Optional, Tuple

import keras
import numpy as np
from keras import ops
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.regularizers.sigreg")
class SIGRegLayer(keras.layers.Layer):
    """Penalize how far random 1-D projections of an input are from Gaussian.

    Evaluates the empirical characteristic function of the projections on a
    fixed knot grid and penalizes its squared residual, real and imaginary
    parts, against :math:`\\phi(t) = \\exp(-t^2/2)`. Used as a side loss via
    ``model.add_loss(...)``; the layer returns a scalar.

    **Architecture:**

    .. code-block:: text

        Input(..., N, D)
               |
               v
        ┌────────────────────┐   (sampled per call)
        │ A ~ N(0, I)        │   shape (D, num_proj)
        │ columns L2-normed  │
        └─────────┬──────────┘
                  v
        proj @ A  ------->  x   (..., N, num_proj)
                  |
                  v
        outer with t (knots,)  --->  x_t  (..., N, num_proj, knots)
                  |
                  v
        cos_mean, sin_mean    averaged over the N (sample) axis
                  |
                  v
        err = (cos_mean - phi)^2 + sin_mean^2    (..., num_proj, knots)
                  |
                  v
        statistic = err @ weights_   ---> (..., num_proj)
                  |
                  v
        return  ops.mean(statistic)   shape ()

    **Buffers, all created in build():**

    .. code-block:: text

        name       shape      contents
        --------   --------   -------------------------------------------
        t          (knots,)   linspace(0, 3, knots)
        phi        (knots,)   exp(-t^2/2), the target and the window
        weights    (knots,)   trapezoidal weights * phi
                              dt at the ends, 2*dt in between, dt = 3/(knots-1)

    They are stored as non-trainable weights so they serialize with the layer
    and land on the correct device.

    :param knots: Number of trapezoidal-rule integration knots on ``[0, 3]``.
        Must be at least 2. The default matches upstream.
    :type knots: int
    :param num_proj: Number of random slicing directions sampled per forward
        pass. Must be at least 1. The default matches upstream.
    :type num_proj: int
    :param seed: Seed for the random projection, for reproducible tests.
        ``None`` re-samples each call without a fixed seed, matching upstream's
        use of ``torch.randn`` with the global generator.
    :type seed: int or None
    :param normalize_by_n: If ``True``, multiply the statistic by ``N``, the
        sample-axis size (``proj.shape[-2]``), before the final mean over
        ``num_proj``. Defaults to ``False``, which preserves this layer's
        original, unscaled behavior (both existing consumers, ``lewm`` and
        ``video_jepa``, rely on this default and do not pass this argument).
        **Naming is intentionally inverted from the upstream PyTorch
        reference**: the reference's own ``normalize_by_n`` flag, when
        ``True``, *skips* the ``* N`` multiplication (``if
        self.normalize_by_n: statistic = err @ self.weights`` — no
        multiply), and applies it only when ``normalize_by_n=False``. Here,
        ``normalize_by_n=True`` means "apply the ``* N`` scaling" — the
        natural reading of the name — because this layer's pre-existing
        default (no multiplication) already matched the reference's
        ``normalize_by_n=True`` state before this parameter existed, and
        that default must not silently change. Set ``True`` to reproduce the
        reference's *shipped config* magnitude (``normalize_by_n: false`` in
        its ``conf/config.yaml``).
    :type normalize_by_n: bool
    :param kwargs: Passed through to ``keras.layers.Layer``.

    :ivar knots: The knot count.
    :vartype knots: int
    :ivar num_proj: The projection count.
    :vartype num_proj: int
    :ivar seed: The seed as passed by the caller.
    :vartype seed: int or None
    :ivar normalize_by_n: Whether the statistic is scaled by the sample-axis
        size ``N``. See the constructor parameter above for the naming
        caveat relative to the upstream reference.
    :vartype normalize_by_n: bool
    :ivar t: The integration grid buffer, created in :meth:`build`.
    :ivar phi: The target window buffer, created in :meth:`build`.
    :ivar weights_: The pre-windowed trapezoidal weights, created in
        :meth:`build`.

    Input shape:
        N-D tensor with shape ``(..., N, D)`` where ``D``, the last dimension,
        must be statically known and ``N`` is the sample axis being averaged
        over. Rank must be at least 2. Typical use: ``(T, B, D)``.

    Output shape:
        Scalar (0-D tensor).

    :raises ValueError: If ``knots < 2`` or ``num_proj < 1`` at construction,
        or if at :meth:`build` the input rank is below 2 or the last dimension
        is ``None``.

    Example:
        ```python
        x = keras.Input(shape=(8, 16))           # (B, T=8, D=16)
        sig = SIGRegLayer(knots=17, num_proj=1024, seed=0)
        loss = sig(x)                            # scalar
        model = keras.Model(x, loss)
        ```
    """

    def __init__(
        self,
        knots: int = 17,
        num_proj: int = 1024,
        seed: Optional[int] = None,
        normalize_by_n: bool = False,
        **kwargs: Any,
    ) -> None:
        """Validate the grid settings and store them.

        :param knots: Number of integration knots, at least 2.
        :type knots: int
        :param num_proj: Number of random projections, at least 1.
        :type num_proj: int
        :param seed: Optional seed for the projection draw.
        :type seed: int or None
        :param normalize_by_n: If ``True``, scale the statistic by the
            sample-axis size ``N`` before the final mean. See the class
            docstring for the naming caveat relative to the upstream
            reference. Defaults to ``False``, preserving prior behavior.
        :type normalize_by_n: bool
        :param kwargs: Passed through to ``keras.layers.Layer``.
        :raises ValueError: If ``knots < 2`` or ``num_proj < 1``.
        """
        super().__init__(**kwargs)
        if knots < 2:
            raise ValueError(f"knots must be >= 2, got {knots}")
        if num_proj < 1:
            raise ValueError(f"num_proj must be >= 1, got {num_proj}")

        # Store all configuration here: get_config() needs it and build()
        # recreates the buffers from it. No weight creation in __init__.
        self.knots = knots
        self.num_proj = num_proj
        self.seed = seed
        self.normalize_by_n = normalize_by_n
        self._seed_gen = (
            keras.random.SeedGenerator(seed) if seed is not None else None
        )

        # Buffers are created in build(); declared here for clarity.
        self.t = None
        self.phi = None
        self.weights_ = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Validate the input shape and create the non-trainable buffers.

        Computes the integration grid, the Gaussian window on that grid, and
        the pre-windowed trapezoidal weights once. They do not depend on the
        input shape but are registered here, since every ``add_weight`` call
        belongs in ``build``.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple
        :return: Nothing.
        :rtype: None
        :raises ValueError: If the input rank is below 2 or the last dimension
            is ``None``.
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

        # Computed in numpy so the values are baked into Constant initializers.
        t_np = np.linspace(0.0, 3.0, self.knots, dtype="float32")
        dt = 3.0 / (self.knots - 1)
        weights_np = np.full((self.knots,), 2.0 * dt, dtype="float32")
        weights_np[0] = dt
        weights_np[-1] = dt
        window_np = np.exp(-0.5 * t_np * t_np).astype("float32")
        final_weights_np = (weights_np * window_np).astype("float32")

        # Non-trainable weights, so they serialize with the layer.
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

        :param proj: Tensor of shape ``(..., N, D)``, typically ``(T, B, D)``.
            Averaging happens over the last-but-one axis ``N``.
        :type proj: tensor
        :param training: Unused; SIGReg runs identically in train and eval.
        :type training: bool or None
        :return: A scalar tensor.
        :rtype: tensor
        :raises ValueError: If the last dimension is not statically known.
        """
        D = proj.shape[-1]
        if D is None:
            raise ValueError("SIGRegLayer requires a known last dimension.")

        # Random projection matrix A: (D, num_proj).
        if self._seed_gen is not None:
            A = keras.random.normal(
                (D, self.num_proj), dtype=proj.dtype, seed=self._seed_gen
            )
        else:
            A = keras.random.normal((D, self.num_proj), dtype=proj.dtype)

        # Normalize each column of A to unit L2 norm.
        col_norm = ops.sqrt(ops.sum(ops.square(A), axis=0, keepdims=True) + 1e-12)
        # A is now (D, num_proj) with unit columns.
        A = A / col_norm

        # Shape after this: (..., N, num_proj).
        x = ops.matmul(proj, A)

        # Outer product with t gives (..., N, num_proj, knots).
        x_t = ops.expand_dims(x, axis=-1) * ops.reshape(self.t, (1,) * x.ndim + (-1,))

        # Average cos and sin along axis -3, the N axis.
        # Shape after this: (..., num_proj, knots).
        cos_mean = ops.mean(ops.cos(x_t), axis=-3)
        sin_mean = ops.mean(ops.sin(x_t), axis=-3)

        # Residual against the target characteristic function phi(t).
        # phi has shape (knots,) and broadcasts over num_proj.
        err = ops.square(cos_mean - self.phi) + ops.square(sin_mean)
        # Weighted sum over knots gives (..., num_proj).
        statistic = ops.matmul(err, self.weights_)

        if self.normalize_by_n:
            # Reintroduces the * N scaling removed in commit 13e1ac626 for
            # lewm's own unrelated reason, gated behind this flag so lewm's
            # and video_jepa's unmodified construction (normalize_by_n
            # defaults to False) stays byte-identical. See D-003.
            n = ops.cast(ops.shape(proj)[-2], proj.dtype)
            statistic = statistic * n

        return ops.mean(statistic)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple:
        """Report the scalar output shape.

        :param input_shape: Shape of the input tensor. Unused.
        :type input_shape: tuple
        :return: The empty tuple.
        :rtype: tuple
        """
        return ()

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments for serialization.

        :return: The base layer config plus ``knots``, ``num_proj``,
            ``seed`` and ``normalize_by_n``.
        :rtype: dict
        """
        config = super().get_config()
        config.update(
            {
                "knots": self.knots,
                "num_proj": self.num_proj,
                "seed": self.seed,
                "normalize_by_n": self.normalize_by_n,
            }
        )
        return config

# ---------------------------------------------------------------------
