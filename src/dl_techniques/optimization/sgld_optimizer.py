"""
SGLD: Stochastic Gradient Langevin Dynamics Optimizer.

This module provides a Keras 3 implementation of Stochastic Gradient Langevin
Dynamics (SGLD), introduced by Welling & Teh (2011). SGLD bridges stochastic
optimization and Bayesian sampling by augmenting a standard SGD step with
isotropic Gaussian noise whose magnitude is calibrated to the learning rate.

Update Rule:
-----------
For a parameter ``w`` with mini-batch gradient ``g = ∇L(w)``, learning rate
``lr`` and noise scale ``s`` (defaults to 1.0, the canonical SGLD setting):

    w_{t+1} = w_t  -  lr · g  +  s · sqrt(2 · lr) · ε,    ε ~ N(0, I)

The deterministic drift term ``-lr · g`` is identical to vanilla SGD. The
stochastic diffusion term ``s · sqrt(2·lr) · ε`` is what turns the iterate
sequence into an (approximate) sample from the Bayesian posterior
``p(w | data) ∝ exp(-L(w))`` as ``lr → 0``.

Why SGLD?
---------
1.  **Escape shallow local minima**: the injected Gaussian noise gives the
    optimizer enough energy to climb out of narrow attractors that trap pure
    SGD, while the gradient drift still pulls toward low-loss regions. This
    makes SGLD a robust choice for highly non-convex landscapes (e.g. Ackley
    or rugged loss surfaces in deep nets).
2.  **Bayesian posterior sampling**: as ``lr → 0`` and step count ``→ ∞``,
    SGLD produces samples from the (mini-batch-approximated) posterior over
    weights. This enables uncertainty quantification, ensembling without
    re-training, and MCMC-style Bayesian deep learning.
3.  **Exploration/exploitation trade-off**: a larger ``lr`` (or ``noise_scale``)
    favours exploration of the landscape; decaying the LR over training
    "anneals" the system into a stable mode (exploitation), exactly like
    simulated annealing.

Practical Notes:
---------------
*   The noise coefficient ``sqrt(2 · lr)`` is the canonical Langevin scaling
    derived from discretising the continuous-time Langevin SDE with a
    Euler-Maruyama integrator at temperature ``T = 1``. A user-controllable
    ``noise_scale`` parameter exposes the temperature (``s² = T``) for
    empirical tuning.
*   Some reference implementations use ``sqrt(lr)`` (without the factor of
    two). This deviates from the canonical formula in the literature; this
    implementation follows the standard derivation.
*   SGLD is intentionally **stateless** per-variable (no momentum, no
    second-moment buffers). This keeps the algorithm aligned with the
    underlying SDE; variants like pSGLD (Preconditioned SGLD) introduce
    state and should be implemented as separate subclasses.
*   For reproducibility, supply an integer ``seed``. The optimizer manages
    its own ``keras.random.SeedGenerator`` so noise is graph-safe and
    backend-agnostic.

References:
-----------
.. code-block:: bibtex

    @inproceedings{welling2011bayesian,
      author    = {Max Welling and Yee Whye Teh},
      title     = {Bayesian Learning via Stochastic Gradient Langevin Dynamics},
      booktitle = {Proceedings of the 28th International Conference on Machine
                   Learning (ICML)},
      year      = {2011}
    }

    @article{brosse2018sgld,
      author    = {Nicolas Brosse and Eric Moulines and Alain Durmus},
      title     = {The promises and pitfalls of Stochastic Gradient Langevin
                   Dynamics},
      journal   = {NeurIPS},
      year      = {2018}
    }
"""

import keras
from keras import ops
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SGLD(keras.optimizers.Optimizer):
    """Stochastic Gradient Langevin Dynamics (SGLD) optimizer.

    Implements the canonical Langevin update

        ``w <- w - lr * grad + noise_scale * sqrt(2 * lr) * N(0, I)``

    which combines standard SGD drift with an isotropic Gaussian diffusion
    term scaled by ``sqrt(2 * lr)``. With ``noise_scale = 1.0`` (default)
    the resulting iterates approximately sample from the Bayesian posterior
    over the parameters as the learning rate is annealed to zero.

    SGLD inherits weight decay, gradient clipping (``clipnorm``,
    ``global_clipnorm``, ``clipvalue``) and learning rate scheduling from
    the Keras ``Optimizer`` base class.

    Args:
        learning_rate: A float, a ``keras.optimizers.schedules.LearningRateSchedule``
            instance, or a callable returning the current learning rate.
            Controls both the drift magnitude and (through ``sqrt(2 * lr)``)
            the noise magnitude. Defaults to ``1e-2``.
        noise_scale: Non-negative float multiplier on the Langevin noise.
            ``1.0`` is the canonical, temperature-1 setting. Set ``> 1.0``
            to encourage exploration, ``< 1.0`` to dampen noise (purely
            SGD when ``0.0``). Defaults to ``1.0``.
        weight_decay: Optional decoupled weight decay coefficient forwarded
            to the base optimizer. Defaults to ``None`` (no weight decay).
        seed: Optional integer seed for the internal
            ``keras.random.SeedGenerator``. Supply for reproducible noise
            streams. Defaults to ``None`` (non-deterministic).
        name: Optimizer name. Defaults to ``"SGLD"``.
        **kwargs: Additional keyword arguments forwarded to
            ``keras.optimizers.Optimizer`` (e.g. ``clipnorm``,
            ``global_clipnorm``, ``clipvalue``, ``ema_momentum``).

    Raises:
        ValueError: If ``noise_scale`` is negative.

    Example:
        >>> import keras
        >>> from dl_techniques.optimization.sgld_optimizer import SGLD
        >>>
        >>> model = keras.Sequential([
        ...     keras.layers.Dense(64, activation="relu"),
        ...     keras.layers.Dense(10),
        ... ])
        >>> model.compile(
        ...     optimizer=SGLD(learning_rate=1e-3, noise_scale=1.0, seed=42),
        ...     loss="sparse_categorical_crossentropy",
        ... )
    """

    def __init__(
            self,
            learning_rate: Union[
                float, keras.optimizers.schedules.LearningRateSchedule
            ] = 1e-2,
            noise_scale: float = 1.0,
            weight_decay: Optional[float] = None,
            seed: Optional[int] = None,
            name: str = "SGLD",
            **kwargs: Any,
    ) -> None:
        if noise_scale < 0.0:
            raise ValueError(
                f"noise_scale must be non-negative, got {noise_scale}"
            )

        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            name=name,
            **kwargs,
        )

        self.noise_scale = float(noise_scale)
        self.seed = seed

        # The SeedGenerator is created lazily in build() so it tracks the
        # optimizer's lifecycle and is reconstructed cleanly after restore.
        self._seed_generator: Optional[keras.random.SeedGenerator] = None

        logger.info(
            f"Built SGLD optimizer: learning_rate={learning_rate}, "
            f"noise_scale={self.noise_scale}, weight_decay={weight_decay}, "
            f"seed={self.seed}"
        )

    def build(self, var_list: List[keras.Variable]) -> None:
        """Initialize optimizer state.

        SGLD is stateless per-variable (no momentum, no second moments),
        so this only delegates to the base class and constructs the
        ``SeedGenerator`` used by ``update_step`` for noise sampling.

        Args:
            var_list: List of trainable variables.
        """
        if self.built:
            return

        super().build(var_list)
        self._seed_generator = keras.random.SeedGenerator(seed=self.seed)

    def update_step(
            self,
            gradient: keras.KerasTensor,
            variable: keras.Variable,
            learning_rate: keras.KerasTensor,
    ) -> None:
        """Apply a single SGLD update to a variable.

        Performs ``w <- w - lr * g + noise_scale * sqrt(2 * lr) * N(0, I)``
        in the variable's native dtype.

        Args:
            gradient: Gradient tensor for ``variable``.
            variable: Variable to update in place.
            learning_rate: Scalar learning rate tensor for this step
                (already evaluated from the schedule by the base class).
        """
        dtype = variable.dtype

        # Cast scalars into the variable's dtype to keep mixed-precision
        # workflows numerically consistent and avoid implicit upcasts.
        lr = ops.cast(learning_rate, dtype)
        noise_scale = ops.cast(self.noise_scale, dtype)

        # Langevin noise standard deviation: sqrt(2 * lr).
        # When lr == 0 this is exactly 0, so noise injection vanishes
        # (lr=0 is a useful diagnostic for ablating the diffusion term).
        noise_std = ops.sqrt(ops.cast(2.0, dtype) * lr)

        # Backend-agnostic, graph-safe Gaussian sample.
        noise = keras.random.normal(
            shape=ops.shape(variable),
            mean=0.0,
            stddev=1.0,
            dtype=dtype,
            seed=self._seed_generator,
        )

        # Update: deterministic drift + Langevin diffusion.
        update = lr * gradient - noise_scale * noise_std * noise
        variable.assign_sub(update)

    def get_config(self) -> Dict[str, Any]:
        """Return the full optimizer configuration for serialization.

        Returns:
            A JSON-serializable dictionary containing every constructor
            argument needed to faithfully reconstruct the optimizer
            (the learning-rate schedule is handled by the base class).
        """
        config = super().get_config()
        config.update({
            "noise_scale": self.noise_scale,
            "seed": self.seed,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SGLD":
        """Reconstruct an ``SGLD`` instance from a serialized config.

        Args:
            config: Configuration dictionary, typically produced by
                ``get_config()``.

        Returns:
            A new ``SGLD`` optimizer with the deserialized hyperparameters.
        """
        return cls(**config)

# ---------------------------------------------------------------------
