"""
VSGD: Variational Stochastic Gradient Descent Optimizer.

This module provides a Keras 3 implementation of the VSGD optimizer from
Chen et al. (2024). VSGD treats each parameter's gradient update as a
probabilistic model and applies Stochastic Variational Inference (SVI) to
derive an adaptive closed-form update rule.

Per-variable running statistics (``mug``, ``bg``, ``bhg``) are maintained
across steps and used to compute an effective, curvature-aware step size.
The update is conceptually similar to Adam in that it normalises the gradient
signal by a running second-moment estimate, but the SVI derivation gives a
principled interpretation of the statistics and avoids the need for explicit
bias-correction factors.

Update Rule:
-----------
Given gradient ``ghat`` at step ``t`` with learning rate ``lr``:

    sg   = bg  / pa2   (prior on g-variance)
    shg  = bhg / pa2   (prior on ghat-variance)

    mug_new = (ghat * sg + mug_prev * shg) / (sg + shg)
    sigg    = sg * shg / (sg + shg)
    mug_sq  = sigg + mug_new**2

    bg2   = pbg2  + mug_sq - 2*mug_new*mug_prev + mug_prev**2
    bhg2  = pbhg2 + mug_sq - 2*ghat*mug_new     + ghat**2

    rho1  = t ** (-tau1)
    rho2  = t ** (-tau2)

    bg_new  = bg  * (1-rho1) + bg2  * rho1
    bhg_new = bhg * (1-rho2) + bhg2 * rho2

    param -= lr / (sqrt(mug_sq) + eps) * mug_new

References:
-----------
.. code-block:: bibtex

    @misc{chen2024vsgd,
      author  = {Chen, X. and others},
      title   = {VSGD: Variational Stochastic Gradient Descent via Bayesian
                 Online Natural Gradient},
      year    = {2024}
    }
"""

import keras
from keras import ops
from typing import Any, Dict, List, Union

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class VSGD(keras.optimizers.Optimizer):
    """VSGD (Variational Stochastic Gradient Descent) optimizer.

    Implements the SVI-derived adaptive optimizer from Chen et al. (2024).
    Per-variable running statistics ``mug``, ``bg``, and ``bhg`` are
    maintained across steps to compute a principled adaptive step size.

    Weight decay is applied in the decoupled (AdamW) style — multiplying the
    parameter by ``(1 - lr * weight_decay)`` before the gradient step — and is
    managed manually rather than delegated to the Keras base class.

    :param ghattg: Gradient hat target (prior mean for ghat).  Controls the
        scale of the prior on the ghat variance. Defaults to 30.0.
    :type ghattg: float
    :param ps: Prior scale. Small positive constant that anchors the priors.
        Defaults to 1e-8.
    :type ps: float
    :param tau1: EMA exponent for the ``bg`` running statistic.  Smaller
        values give faster decay (more weight on recent observations).
        Defaults to 0.81.
    :type tau1: float
    :param tau2: EMA exponent for the ``bhg`` running statistic. Defaults
        to 0.90.
    :type tau2: float
    :param learning_rate: Base learning rate. Accepts a float or a Keras
        ``LearningRateSchedule``. Defaults to 0.1.
    :type learning_rate: Union[float, keras.optimizers.schedules.LearningRateSchedule]
    :param weight_decay: Decoupled weight decay coefficient. Applied
        multiplicatively before the gradient step. Defaults to 0.0
        (no decay).
    :type weight_decay: float
    :param eps: Small constant added to the denominator for numerical
        stability. Defaults to 1e-8.
    :type eps: float
    :param name: Optimizer name. Defaults to ``"VSGD"``.
    :type name: str
    """

    def __init__(
            self,
            ghattg: float = 30.0,
            ps: float = 1e-8,
            tau1: float = 0.81,
            tau2: float = 0.90,
            learning_rate: Union[
                float, keras.optimizers.schedules.LearningRateSchedule
            ] = 0.1,
            weight_decay: float = 0.0,
            eps: float = 1e-8,
            name: str = "VSGD",
            **kwargs: Any,
    ) -> None:
        # Pass weight_decay=0.0 to base class; we manage it manually so that
        # the decay step happens at exactly the right point in the update rule
        # (before the statistics assignment and gradient step).
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=0.0,
            name=name,
            **kwargs,
        )

        self._ghattg = float(ghattg)
        self._ps = float(ps)
        self._tau1 = float(tau1)
        self._tau2 = float(tau2)
        self._weight_decay = float(weight_decay)
        self._eps = float(eps)

        # Derived scalar constants (Python floats; cast to variable.dtype
        # inside update_step for graph-safe mixed-precision support).
        self._pa2 = 2.0 * ps + 1.0 + 1e-4
        self._pbg2 = 2.0 * ps
        self._pbhg2 = 2.0 * ghattg * ps

    def build(self, var_list: List[keras.Variable]) -> None:
        """Initialise per-variable optimizer state.

        Allocates three zero-filled state tensors for each variable:
        ``mug`` (running mean estimate), ``bg`` (running g-variance),
        and ``bhg`` (running ghat-variance).

        :param var_list: List of trainable variables.
        :type var_list: List[keras.Variable]
        """
        if self.built:
            return

        super().build(var_list)

        self._mug: List[keras.Variable] = []
        self._bg: List[keras.Variable] = []
        self._bhg: List[keras.Variable] = []

        for var in var_list:
            self._mug.append(
                self.add_variable_from_reference(var, name="mug")
            )
            self._bg.append(
                self.add_variable_from_reference(var, name="bg")
            )
            self._bhg.append(
                self.add_variable_from_reference(var, name="bhg")
            )

    def update_step(
            self,
            gradient: keras.KerasTensor,
            variable: keras.Variable,
            learning_rate: keras.KerasTensor,
    ) -> None:
        """Apply a single VSGD update to a variable.

        Implements the full SVI-derived update rule.  All intermediate
        tensors are cast to ``variable.dtype`` to support mixed-precision
        workflows.  The ``is_first`` branch uses ``ops.where`` (not a Python
        ``if``) for graph-safe execution under ``@tf.function`` / XLA.

        :param gradient: Gradient tensor for ``variable``.
        :type gradient: keras.KerasTensor
        :param variable: Variable to update in place.
        :type variable: keras.Variable
        :param learning_rate: Scalar learning rate tensor (already evaluated
            from any schedule by the base class).
        :type learning_rate: keras.KerasTensor
        """
        dtype = variable.dtype

        lr = ops.cast(learning_rate, dtype)
        wd = ops.cast(self._weight_decay, dtype)
        eps = ops.cast(self._eps, dtype)
        pa2 = ops.cast(self._pa2, dtype)
        pbg2 = ops.cast(self._pbg2, dtype)
        pbhg2 = ops.cast(self._pbhg2, dtype)
        tau1 = ops.cast(self._tau1, dtype)
        tau2 = ops.cast(self._tau2, dtype)

        idx = self._get_variable_index(variable)
        mug = self._mug[idx]
        bg = self._bg[idx]
        bhg = self._bhg[idx]

        # self.iterations has already been incremented by the base class
        # before update_step is called, so it equals 1 on the very first step.
        step = ops.cast(self.iterations, dtype)
        ghat = gradient
        is_first = step <= ops.cast(1.0, dtype)

        # Prior-based initialisation on step 1; running estimates thereafter.
        sg = ops.where(is_first, pbg2 / (pa2 - ops.cast(1.0, dtype)), bg / pa2)
        shg = ops.where(is_first, pbhg2 / (pa2 - ops.cast(1.0, dtype)), bhg / pa2)

        mug_prev = ops.copy(mug)
        mug_new = (ghat * sg + mug_prev * shg) / (sg + shg)
        sigg = sg * shg / (sg + shg)
        mug_sq = sigg + mug_new ** 2

        bg2 = pbg2 + mug_sq - 2.0 * mug_new * mug_prev + mug_prev ** 2
        bhg2 = pbhg2 + mug_sq - 2.0 * ghat * mug_new + ghat ** 2

        rho1 = ops.power(step, -tau1)
        rho2 = ops.power(step, -tau2)

        bg_new = bg * (ops.cast(1.0, dtype) - rho1) + bg2 * rho1
        bhg_new = bhg * (ops.cast(1.0, dtype) - rho2) + bhg2 * rho2

        # AdamW-style decoupled weight decay applied BEFORE the gradient step.
        variable.assign(variable * (ops.cast(1.0, dtype) - lr * wd))

        mug.assign(mug_new)
        bg.assign(bg_new)
        bhg.assign(bhg_new)

        variable.assign_sub(lr / (ops.sqrt(mug_sq) + eps) * mug_new)

    def get_config(self) -> Dict[str, Any]:
        """Return the full optimizer configuration for serialization.

        :return: JSON-serializable dictionary of all constructor arguments.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "ghattg": self._ghattg,
            "ps": self._ps,
            "tau1": self._tau1,
            "tau2": self._tau2,
            # Use our stored value, not the Keras base (we passed 0.0 to super)
            "weight_decay": self._weight_decay,
            "eps": self._eps,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VSGD":
        """Reconstruct a VSGD instance from a serialized config.

        :param config: Configuration dictionary produced by ``get_config()``.
        :type config: Dict[str, Any]
        :return: New VSGD optimizer with deserialized hyperparameters.
        :rtype: VSGD
        """
        return cls(**config)

# ---------------------------------------------------------------------
