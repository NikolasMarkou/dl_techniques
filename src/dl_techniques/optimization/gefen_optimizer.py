"""
Gefen-lite: shared-second-moment AdamW optimizer (block-shared ``vmean``).

This module provides a Keras 3 / TensorFlow 2.18 implementation of a memory-lean
AdamW variant inspired by the Gefen optimizer (arxiv 2606.13894). The full Gefen
algorithm compresses optimizer state in three ways: (1) a *block-shared* second
moment (one ``vmean`` scalar per block of ``period`` contiguous flattened gradient
elements instead of one per element), (2) a uint8 codebook-quantized first moment
(momentum) with a 256-entry codebook learned once from gradient histograms on the
first step, and (3) a *period* (block size) selected per-parameter from gradient
statistics on the first step. Together these yield roughly 8x reduction in
optimizer-state memory with claimed parity to AdamW.

This implementation is **Gefen-lite (shared-v)**: it keeps only the dominant,
fully graph-compatible memory saving — the block-shared second moment — and
deliberately diverges from the full algorithm as follows:

- **No uint8 momentum quantization.** Momentum (``m``) is stored full-shape and
  full-precision (standard Adam first moment). The learned-codebook quantization
  requires a global cross-variable pass to fit the codebook, which is incompatible
  with the per-variable ``update_step`` contract of the Keras optimizer API.
- **No learned codebook.** Same reason: a 256-entry codebook fitted from a global
  gradient histogram on step 1 cannot be expressed inside a per-variable,
  ``jit_compile``-safe ``update_step``.
- **Shape-based (not gradient-statistic) period selection.** ``period`` is chosen
  deterministically in ``build()`` from each variable's *shape* (the largest
  divisor of ``numel`` that is ``<= max_block_size``, falling back to ``1``).
  Gradient-statistic period selection violates the Keras "all state shapes known
  before the first gradient step" contract and would force eager execution on the
  first step, breaking ``model.fit`` tracing and ``jit_compile=True``.

These divergences are intentional and were chosen for full compatibility with
graph-mode execution, ``jit_compile=True``, and the standard ``model.fit`` training
loop. Because ``period`` is derived from shape, it is recomputed identically on
load and is NOT serialized.

Update Rule:
-----------
Given gradient ``g`` (flattened to length ``N``) at step ``t`` for a variable with
``K = N / period`` blocks, learning rate ``lr``:

    m       = beta_1 * m + (1 - beta_1) * g                  # full-shape momentum
    g_block = reshape(g, [K, period])
    bmsq    = mean(g_block^2, axis=1)                        # [K] block mean-square
    vmean   = beta_2 * vmean + (1 - beta_2) * bmsq           # [K] shared 2nd moment
    bc1     = 1 - beta_1^t
    bc2     = 1 - beta_2^t
    h_block = sqrt(vmean / bc2) + eps                        # [K]
    denom   = broadcast(h_block) reshaped to var.shape       # one h per block
    p      *= (1 - lr * weight_decay)                        # decoupled (AdamW) WD
    p      -= lr * (m / bc1) / denom

Edge cases:
-----------
- Scalar / tiny variables (``N < min_block_size`` or whose largest valid divisor
  ``< min_block_size``): ``period = 1``, ``K = N`` — the update degenerates to
  standard per-element AdamW for that variable. Acceptable, documented.
- Prime ``N`` larger than ``min_block_size``: only divisors are ``1`` and ``N``; if
  ``N > max_block_size`` then ``period = 1``, else ``period = N`` (single block).
- Mixed precision: ``vmean`` and all block math are float32 regardless of variable
  dtype; the denominator is cast back to ``variable.dtype`` for the final assign.

References:
-----------
.. code-block:: bibtex

    @misc{gefen2026,
      title  = {Gefen: Memory-Efficient Optimization via Block-Shared Second
                Moments and Quantized Momentum},
      year   = {2026},
      eprint = {2606.13894},
      archivePrefix = {arXiv}
    }
"""

import math
import keras
from keras import ops
from typing import Any, Dict, List, Union

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class Gefen(keras.optimizers.Optimizer):
    """Gefen-lite (shared-v) optimizer: AdamW with a block-shared second moment.

    A drop-in AdamW replacement that stores one second-moment estimate per *block*
    of ``period`` contiguous (flattened) gradient elements instead of one per
    element, while keeping a full-shape, full-precision first moment (momentum).
    ``period`` is chosen deterministically from each variable's shape in
    ``build()`` (the largest divisor of ``numel`` that is ``<= max_block_size``,
    else ``1``), so the ``reshape(g, (K, period))`` is graph-static and
    ``jit_compile``-safe.

    Weight decay is applied in the decoupled (AdamW) style — multiplying the
    parameter by ``(1 - lr * weight_decay)`` before the gradient step — and is
    managed manually rather than delegated to the Keras base class.

    This is **Gefen-lite**: it diverges from the full Gefen algorithm
    (arxiv 2606.13894) by omitting uint8 momentum quantization, the learned
    codebook, and gradient-statistic period selection. See the module docstring
    for the full rationale.

    Args:
        learning_rate: Base learning rate. Accepts a float or a Keras
            ``LearningRateSchedule``. Defaults to ``1e-3``.
        beta_1: Exponential decay rate for the first moment (momentum). Must be
            in ``[0, 1)``. Defaults to ``0.9``.
        beta_2: Exponential decay rate for the block-shared second moment. Must be
            in ``[0, 1)``. Defaults to ``0.999``.
        epsilon: Small constant added to the denominator for numerical stability.
            Must be ``>= 0``. Defaults to ``1e-8``.
        weight_decay: Decoupled (AdamW) weight decay coefficient, applied
            multiplicatively before the gradient step. Must be ``>= 0``. Defaults
            to ``0.0`` (no decay).
        max_block_size: Upper bound on the block size (``period``). The chosen
            ``period`` is the largest divisor of ``numel`` not exceeding this.
            Must be ``>= 1``. Defaults to ``1024``.
        min_block_size: Lower bound below which block sharing is abandoned: if the
            largest valid divisor is ``< min_block_size``, ``period`` falls back to
            ``1`` (per-element AdamW). Must be ``>= 1``. Defaults to ``8``.
        name: Optimizer name. Defaults to ``"gefen"``.
        **kwargs: Additional keyword arguments forwarded to the base optimizer
            (e.g. ``clipnorm``, ``global_clipnorm``).

    Raises:
        ValueError: If any hyperparameter is out of its valid range.
    """

    def __init__(
            self,
            learning_rate: Union[
                float, keras.optimizers.schedules.LearningRateSchedule
            ] = 1e-3,
            beta_1: float = 0.9,
            beta_2: float = 0.999,
            epsilon: float = 1e-8,
            weight_decay: float = 0.0,
            max_block_size: int = 1024,
            min_block_size: int = 8,
            name: str = "gefen",
            **kwargs: Any,
    ) -> None:
        # --- Range validation -------------------------------------------------
        if isinstance(learning_rate, (int, float)) and learning_rate < 0.0:
            raise ValueError(
                f"learning_rate must be >= 0, got {learning_rate}."
            )
        if not (0.0 <= beta_1 < 1.0):
            raise ValueError(f"beta_1 must be in [0, 1), got {beta_1}.")
        if not (0.0 <= beta_2 < 1.0):
            raise ValueError(f"beta_2 must be in [0, 1), got {beta_2}.")
        if epsilon < 0.0:
            raise ValueError(f"epsilon must be >= 0, got {epsilon}.")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}.")
        if max_block_size < 1:
            raise ValueError(
                f"max_block_size must be >= 1, got {max_block_size}."
            )
        if min_block_size < 1:
            raise ValueError(
                f"min_block_size must be >= 1, got {min_block_size}."
            )

        # Pass weight_decay=0.0 to base class; we manage it manually so that the
        # decoupled decay step happens at exactly the right point in the update
        # rule (before the gradient step).
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=0.0,
            name=name,
            **kwargs,
        )

        self._beta_1 = float(beta_1)
        self._beta_2 = float(beta_2)
        self._epsilon = float(epsilon)
        self._gefen_weight_decay = float(weight_decay)
        self._max_block_size = int(max_block_size)
        self._min_block_size = int(min_block_size)

        logger.info(
            "Gefen-lite optimizer constructed "
            f"(lr={learning_rate}, beta_1={self._beta_1}, "
            f"beta_2={self._beta_2}, weight_decay={self._gefen_weight_decay}, "
            f"max_block_size={self._max_block_size}, "
            f"min_block_size={self._min_block_size})."
        )

    def _compute_period(self, n: int) -> int:
        """Compute the block size (``period``) for a variable of ``n`` elements.

        Returns the largest divisor ``d`` of ``n`` with ``d <= max_block_size``.
        If that largest valid divisor is ``< min_block_size``, returns ``1``
        (which always divides ``n``), degenerating to per-element AdamW.

        Args:
            n: Number of elements in the variable (``math.prod(var.shape)``).

        Returns:
            The chosen period as a pure Python ``int`` (never a tensor).
        """
        if n <= 0:
            return 1

        cap = min(self._max_block_size, n)
        # Scan divisors from largest-allowed downward; first hit is the largest.
        for d in range(cap, 0, -1):
            if n % d == 0:
                if d < self._min_block_size:
                    return 1
                return d
        # Unreachable (d == 1 always divides), but be safe.
        return 1

    def build(self, var_list: List[keras.Variable]) -> None:
        """Initialise per-variable optimizer state.

        Allocates a full-shape ``momentum`` slot and a float32 ``vmean`` slot of
        shape ``(K,)`` per variable, and records the per-variable ``period`` and
        block count ``K`` as Python ints (keyed by variable index) so that the
        block reshape is graph-static.

        Args:
            var_list: List of trainable variables.
        """
        if self.built:
            return

        super().build(var_list)

        self._momentum: List[keras.Variable] = []
        self._vmean: List[keras.Variable] = []
        self._period: Dict[int, int] = {}
        self._blocks: Dict[int, int] = {}

        for var in var_list:
            idx = self._get_variable_index(var)
            n = int(math.prod(var.shape))
            p = self._compute_period(n)
            # By construction period divides numel; if this ever fires the
            # divisor scan in _compute_period is wrong — halt, do not patch.
            assert n % p == 0, (
                f"period {p} does not divide numel {n} for variable {var.name}"
            )
            k = n // p
            self._period[idx] = p
            self._blocks[idx] = k

            self._momentum.append(
                self.add_variable_from_reference(var, name="momentum")
            )
            self._vmean.append(
                self.add_variable(shape=(k,), dtype="float32", name="vmean")
            )

    def update_step(
            self,
            gradient: keras.KerasTensor,
            variable: keras.Variable,
            learning_rate: keras.KerasTensor,
    ) -> None:
        """Apply a single Gefen-lite update to a variable.

        Performs an AdamW update whose denominator uses a second-moment estimate
        shared across each block of ``period`` contiguous flattened elements.
        Block math is done in float32 and the denominator cast back to
        ``variable.dtype`` for mixed-precision safety.

        Args:
            gradient: Gradient tensor for ``variable``.
            variable: Variable to update in place.
            learning_rate: Scalar learning rate tensor (already evaluated from any
                schedule by the base class).
        """
        dtype = variable.dtype

        lr = ops.cast(learning_rate, dtype)
        beta_1 = ops.cast(self._beta_1, dtype)
        beta_2 = ops.cast(self._beta_2, dtype)
        eps = ops.cast(self._epsilon, dtype)
        wd = ops.cast(self._gefen_weight_decay, dtype)

        idx = self._get_variable_index(variable)
        m = self._momentum[idx]
        vmean = self._vmean[idx]
        p = self._period[idx]
        k = self._blocks[idx]

        # self.iterations has already been incremented by the base class before
        # update_step is called, so it equals 1 on the very first step.
        local_step = ops.cast(self.iterations + 1, dtype)
        one = ops.cast(1.0, dtype)
        bc1 = one - ops.power(beta_1, local_step)
        bc2 = one - ops.power(beta_2, local_step)

        # --- First moment (momentum): full-shape EMA --------------------------
        m_new = beta_1 * m + (one - beta_1) * gradient
        m.assign(m_new)

        # --- Block-shared second moment (float32) -----------------------------
        beta2_32 = ops.cast(self._beta_2, "float32")
        one_32 = ops.cast(1.0, "float32")
        g32 = ops.cast(ops.reshape(gradient, (-1,)), "float32")
        gb = ops.reshape(g32, (k, p))
        block_msq = ops.mean(ops.square(gb), axis=1)  # [K]
        new_v = beta2_32 * vmean + (one_32 - beta2_32) * block_msq
        vmean.assign(new_v)

        # --- Denominator: one h per block, broadcast to variable shape --------
        bc2_32 = ops.cast(bc2, "float32")
        eps_32 = ops.cast(self._epsilon, "float32")
        h_block = ops.sqrt(new_v / bc2_32) + eps_32  # [K]
        denom32 = ops.reshape(
            ops.broadcast_to(ops.reshape(h_block, (k, 1)), (k, p)),
            variable.shape,
        )
        denom = ops.cast(denom32, dtype)

        m_hat = m_new / bc1

        # --- Decoupled (AdamW) weight decay BEFORE the gradient step ----------
        # Applied unconditionally (no-op at weight_decay == 0) for graph
        # simplicity, matching the VSGD pattern.
        variable.assign(variable * (one - lr * wd))

        # --- Parameter update -------------------------------------------------
        variable.assign_sub(lr * m_hat / denom)

    def get_config(self) -> Dict[str, Any]:
        """Return the full optimizer configuration for serialization.

        Note: ``period`` / ``K`` are derived deterministically from each
        variable's shape in ``build()`` and are intentionally NOT serialized;
        only the hyperparameters that determine them (``max_block_size``,
        ``min_block_size``) are emitted, and ``period`` is recomputed identically
        on load.

        Returns:
            JSON-serializable dictionary of all constructor arguments.
        """
        config = super().get_config()
        config.update({
            "beta_1": self._beta_1,
            "beta_2": self._beta_2,
            "epsilon": self._epsilon,
            # Use our stored value, not the Keras base (we passed 0.0 to super).
            "weight_decay": self._gefen_weight_decay,
            "max_block_size": self._max_block_size,
            "min_block_size": self._min_block_size,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Gefen":
        """Reconstruct a Gefen instance from a serialized config.

        Args:
            config: Configuration dictionary produced by ``get_config()``.

        Returns:
            New Gefen optimizer with deserialized hyperparameters.
        """
        return cls(**config)

# ---------------------------------------------------------------------
