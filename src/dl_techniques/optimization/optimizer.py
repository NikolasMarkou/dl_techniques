"""
Optimizer Builder Module for Deep Learning Techniques.

This module provides `optimizer_builder`, which creates optimizers from a
configuration dictionary. Learning rate schedules are built separately by
`schedule.schedule_builder` (exported as `learning_rate_schedule_builder`) and
passed in.

Supported optimizers:
- Adam: Adaptive moment estimation optimizer
- AdamW: Adam with decoupled weight decay
- SGD: Stochastic gradient descent with optional (Nesterov) momentum
- RMSprop: Adaptive learning rate with momentum
- Adadelta: Adaptive learning rate method

Each optimizer supports gradient clipping options:
- By value (clipvalue): Clip each gradient to a specific range
- By local norm (clipnorm): Clip each gradient independently by its norm
- By global norm (global_clipnorm): Clip all gradients by their combined norm

All optimizers forward `weight_decay` when it is set, and decay-capable ones
additionally support an `exclude_from_weight_decay` list of name patterns
(e.g. ["bias", "gamma", "beta"]) applied after construction.
"""

import keras
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

from keras.api.optimizers import Optimizer
from keras.api.optimizers.schedules import LearningRateSchedule

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .sgld_optimizer import SGLD
from .vsgd_optimizer import VSGD
from .gefen_optimizer import Gefen
from .constants import *

# ---------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------


class OptimizerType(str, Enum):
    """Enumeration of available optimizer types."""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADADELTA = "adadelta"
    SGLD = "sgld"
    VSGD = "vsgd"
    GEFEN = "gefen"


# NOTE: this module previously re-declared every DEFAULT_* constant locally,
# shadowing the identical definitions pulled in by `from .constants import *`
# above (verified: all 39 names matched by value). The local block has been
# removed -- `constants.py` is the single source of truth for optimizer
# defaults.


# ---------------------------------------------------------------------
# Main Functions
# ---------------------------------------------------------------------


# NOTE: this module previously defined its own `learning_rate_schedule_builder`.
# It was an unused, behaviourally-divergent duplicate of `schedule.schedule_builder`
# (it returned a BARE schedule at warmup_steps=0 where schedule_builder returns a
# WarmupSchedule wrapper; the two are numerically identical otherwise). Nothing
# imported it except the test suite -- every consumer gets schedule.py's version
# via `from dl_techniques.optimization import learning_rate_schedule_builder`.
# It has been deleted; build schedules with `schedule.schedule_builder`.


# ---------------------------------------------------------------------

def optimizer_builder(
        config: Dict[str, Any],
        lr_schedule: Union[float, LearningRateSchedule]
) -> Optimizer:
    """Build and configure a Keras optimizer from configuration dictionary.

    Creates an optimizer instance based on the specified type and configuration,
    with support for gradient clipping and custom hyperparameters. Falls back
    to default values from constants when parameters are not specified.

    Args:
        config: Configuration dictionary containing optimizer settings.
            Required keys:
                - type: Optimizer type ('adam', 'adamw', 'sgd', 'rmsprop',
                       'adadelta', 'sgld', 'vsgd', 'gefen')
            Optional keys:
                - Optimizer-specific hyperparameters (beta_1, beta_2, rho,
                  momentum, nesterov, etc.)
                - gradient_clipping_by_value: Clip gradients by absolute value
                - gradient_clipping_by_norm_local: Clip gradients by local norm
                - gradient_clipping_by_norm: Clip gradients by global norm
                - weight_decay: Decoupled weight decay. Forwarded to every
                  optimizer type; when omitted, the Keras default applies
                  (None for all types except AdamW, which defaults to 0.004).
                - exclude_from_weight_decay: List of variable-name patterns to
                  exclude from weight decay (matched with ``re.search``). The
                  conventional recipe is ``["bias", "gamma", "beta"]``. Ignored
                  (with a warning) on optimizers that do not support it.
        lr_schedule: Learning rate as float or LearningRateSchedule instance.

    Returns:
        Configured Keras optimizer instance.

    Raises:
        ValueError: If config is not a dictionary or optimizer type is unknown.

    Example:
        >>> config = {
        ...     "type": "adam",
        ...     "beta_1": 0.9,
        ...     "beta_2": 0.999,
        ...     "epsilon": 1e-7,
        ...     "gradient_clipping_by_norm": 1.0
        ... }
        >>> optimizer = optimizer_builder(config, 0.001)
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")

    # Extract and validate optimizer type
    optimizer_type = config.get("type")
    if not optimizer_type:
        raise ValueError("optimizer type must be specified in config")

    optimizer_type = optimizer_type.strip().lower()

    # Extract gradient clipping configuration
    gradient_clipvalue = config.get("gradient_clipping_by_value")
    gradient_clipnorm = config.get("gradient_clipping_by_norm_local")
    gradient_global_clipnorm = config.get("gradient_clipping_by_norm")

    logger.info(f"Building optimizer: [{optimizer_type}] with lr_schedule type: [{type(lr_schedule).__name__}]")

    # Build base optimizer parameters common to all optimizers
    base_params = {
        "learning_rate": lr_schedule,
        "clipvalue": gradient_clipvalue,
        "clipnorm": gradient_clipnorm,
        "global_clipnorm": gradient_global_clipnorm
    }

    # Build the appropriate optimizer
    if optimizer_type == OptimizerType.RMSPROP:
        optimizer = _build_rmsprop_optimizer(config, base_params)
    elif optimizer_type == OptimizerType.ADAM:
        optimizer = _build_adam_optimizer(config, base_params)
    elif optimizer_type == OptimizerType.ADAMW:
        optimizer = _build_adamw_optimizer(config, base_params)
    elif optimizer_type == OptimizerType.SGD:
        optimizer = _build_sgd_optimizer(config, base_params)
    elif optimizer_type == OptimizerType.ADADELTA:
        optimizer = _build_adadelta_optimizer(config, base_params)
    elif optimizer_type == OptimizerType.SGLD:
        optimizer = _build_sgld_optimizer(config, base_params)
    elif optimizer_type == OptimizerType.VSGD:
        optimizer = _build_vsgd_optimizer(config, base_params)
    elif optimizer_type == OptimizerType.GEFEN:
        optimizer = _build_gefen_optimizer(config, base_params)
    else:
        raise ValueError(
            f"Unknown optimizer_type: [{optimizer_type}]. "
            f"Supported types: {[t.value for t in OptimizerType]}"
        )

    _apply_weight_decay_exclusions(optimizer, config.get("exclude_from_weight_decay"))

    logger.info(f"Successfully built {optimizer.__class__.__name__} optimizer")
    return optimizer


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------


def _apply_weight_decay_exclusions(
        optimizer: Optimizer,
        var_names: Optional[Sequence[str]]
) -> None:
    """Exclude variables whose names match ``var_names`` from weight decay.

    Keras matches each pattern with ``re.search`` against the variable name, so
    the conventional no-decay recipe is ``["bias", "gamma", "beta"]`` — biases
    plus the two LayerNorm/BatchNorm parameters, which Keras spells ``gamma``
    and ``beta``.

    This is a no-op when ``var_names`` is empty or ``None``. The call is guarded
    because ``exclude_from_weight_decay`` only exists on decay-capable
    optimizers and rejects being called after the optimizer has been built;
    neither condition should abort optimizer construction.

    Args:
        optimizer: The freshly built optimizer to configure.
        var_names: Name patterns to exclude from weight decay. ``None`` or an
            empty sequence means no exclusions.
    """
    if not var_names:
        return

    if not hasattr(optimizer, "exclude_from_weight_decay"):
        logger.warning(
            f"{optimizer.__class__.__name__} has no exclude_from_weight_decay(); "
            f"ignoring exclude_from_weight_decay={list(var_names)}"
        )
        return

    try:
        optimizer.exclude_from_weight_decay(var_names=list(var_names))
        logger.info(f"Excluded from weight decay: {list(var_names)}")
    except (ValueError, AttributeError) as e:
        logger.warning(
            f"Could not apply exclude_from_weight_decay={list(var_names)} to "
            f"{optimizer.__class__.__name__}: {e}"
        )


def _build_rmsprop_optimizer(
        config: Dict[str, Any],
        base_params: Dict[str, Any]
) -> keras.optimizers.RMSprop:
    """Build RMSprop optimizer with configuration parameters.

    Args:
        config: Configuration dictionary with RMSprop-specific parameters.
        base_params: Base parameters common to all optimizers.

    Returns:
        Configured RMSprop optimizer instance.
    """
    optimizer_params = {
        "name": "RMSprop",
        "rho": config.get("rho", DEFAULT_RMSPROP_RHO),
        "momentum": config.get("momentum", DEFAULT_RMSPROP_MOMENTUM),
        "epsilon": config.get("epsilon", DEFAULT_RMSPROP_EPSILON),
        "centered": config.get("centered", DEFAULT_RMSPROP_CENTERED),
        **base_params
    }

    # Keras accepts weight_decay on every optimizer, but only forward it when
    # the caller sets it -- otherwise Keras' own default (None) applies. Without
    # this, a config carrying weight_decay would have it SILENTLY dropped.
    if config.get("weight_decay") is not None:
        optimizer_params["weight_decay"] = config["weight_decay"]

    return keras.optimizers.RMSprop(**optimizer_params)


def _build_adam_optimizer(
        config: Dict[str, Any],
        base_params: Dict[str, Any]
) -> keras.optimizers.Adam:
    """Build Adam optimizer with configuration parameters.

    Args:
        config: Configuration dictionary with Adam-specific parameters.
        base_params: Base parameters common to all optimizers.

    Returns:
        Configured Adam optimizer instance.
    """
    optimizer_params = {
        "name": "Adam",
        "beta_1": config.get("beta_1", DEFAULT_ADAM_BETA_1),
        "beta_2": config.get("beta_2", DEFAULT_ADAM_BETA_2),
        "epsilon": config.get("epsilon", DEFAULT_ADAM_EPSILON),
        "amsgrad": config.get("amsgrad", DEFAULT_ADAM_AMSGRAD),
        **base_params
    }

    # Keras accepts weight_decay on every optimizer, but only forward it when
    # the caller sets it -- otherwise Keras' own default (None) applies. Without
    # this, a config carrying weight_decay would have it SILENTLY dropped.
    if config.get("weight_decay") is not None:
        optimizer_params["weight_decay"] = config["weight_decay"]

    return keras.optimizers.Adam(**optimizer_params)


def _build_adamw_optimizer(
        config: Dict[str, Any],
        base_params: Dict[str, Any]
) -> keras.optimizers.AdamW:
    """Build AdamW optimizer with configuration parameters.

    Args:
        config: Configuration dictionary with AdamW-specific parameters.
        base_params: Base parameters common to all optimizers.

    Returns:
        Configured AdamW optimizer instance.
    """
    optimizer_params = {
        "name": "AdamW",
        "weight_decay": config.get("weight_decay", 0.004),
        "beta_1": config.get("beta_1", DEFAULT_ADAMW_BETA_1),
        "beta_2": config.get("beta_2", DEFAULT_ADAMW_BETA_2),
        "epsilon": config.get("epsilon", DEFAULT_ADAMW_EPSILON),
        "amsgrad": config.get("amsgrad", DEFAULT_ADAMW_AMSGRAD),
        **base_params
    }

    return keras.optimizers.AdamW(**optimizer_params)


def _build_sgd_optimizer(
        config: Dict[str, Any],
        base_params: Dict[str, Any]
) -> keras.optimizers.SGD:
    """Build SGD optimizer with configuration parameters.

    Plain stochastic gradient descent with optional (Nesterov) momentum. The
    defaults mirror ``keras.optimizers.SGD`` so that
    ``optimizer_builder({"type": "sgd"}, lr)`` is equivalent to
    ``keras.optimizers.SGD(learning_rate=lr)``.

    ``weight_decay`` is passed through only when the caller sets it; when
    omitted, Keras' own default (``None``, i.e. no decay) applies.

    Args:
        config: Configuration dictionary with SGD-specific parameters.
        base_params: Base parameters common to all optimizers.

    Returns:
        Configured SGD optimizer instance.
    """
    optimizer_params = {
        "name": "SGD",
        "momentum": config.get("momentum", DEFAULT_SGD_MOMENTUM),
        "nesterov": config.get("nesterov", DEFAULT_SGD_NESTEROV),
        **base_params
    }

    if config.get("weight_decay") is not None:
        optimizer_params["weight_decay"] = config["weight_decay"]

    return keras.optimizers.SGD(**optimizer_params)


def _build_adadelta_optimizer(
        config: Dict[str, Any],
        base_params: Dict[str, Any]
) -> keras.optimizers.Adadelta:
    """Build Adadelta optimizer with configuration parameters.

    Args:
        config: Configuration dictionary with Adadelta-specific parameters.
        base_params: Base parameters common to all optimizers.

    Returns:
        Configured Adadelta optimizer instance.
    """
    optimizer_params = {
        "name": "Adadelta",
        "rho": config.get("rho", DEFAULT_ADADELTA_RHO),
        "epsilon": config.get("epsilon", DEFAULT_ADADELTA_EPSILON),
        **base_params
    }

    # Keras accepts weight_decay on every optimizer, but only forward it when
    # the caller sets it -- otherwise Keras' own default (None) applies. Without
    # this, a config carrying weight_decay would have it SILENTLY dropped.
    if config.get("weight_decay") is not None:
        optimizer_params["weight_decay"] = config["weight_decay"]

    return keras.optimizers.Adadelta(**optimizer_params)


def _build_sgld_optimizer(
        config: Dict[str, Any],
        base_params: Dict[str, Any]
) -> SGLD:
    """Build SGLD optimizer with configuration parameters.

    SGLD (Stochastic Gradient Langevin Dynamics) is the SGD update augmented
    with isotropic Gaussian noise of standard deviation `sqrt(2 * lr)`, scaled
    by `noise_scale`. With `noise_scale=1.0` (default) the iterates approximate
    samples from the Bayesian posterior over the parameters as the learning
    rate is annealed.

    Args:
        config: Configuration dictionary with SGLD-specific parameters.
            Optional keys:
                - noise_scale: Multiplier on Langevin noise (default 1.0).
                - seed: Integer seed for reproducible noise (default None).
                - weight_decay: Decoupled weight decay coefficient.
        base_params: Base parameters common to all optimizers (learning_rate,
            clipvalue, clipnorm, global_clipnorm).

    Returns:
        Configured SGLD optimizer instance.
    """
    optimizer_params = {
        "name": "SGLD",
        "noise_scale": config.get("noise_scale", DEFAULT_SGLD_NOISE_SCALE),
        "seed": config.get("seed", DEFAULT_SGLD_SEED),
        "weight_decay": config.get("weight_decay"),
        **base_params,
    }

    return SGLD(**optimizer_params)


def _build_vsgd_optimizer(
        config: Dict[str, Any],
        base_params: Dict[str, Any]
) -> VSGD:
    """Build VSGD optimizer with configuration parameters.

    VSGD (Variational Stochastic Gradient Descent) models gradient updates as
    a probabilistic model and uses Stochastic Variational Inference to derive
    an adaptive closed-form update rule with per-variable running statistics.

    Args:
        config: Configuration dictionary with VSGD-specific parameters.
            Optional keys:
                - ghattg: Gradient hat target (default 30.0).
                - ps: Prior scale (default 1e-8).
                - tau1: EMA exponent for bg (default 0.81).
                - tau2: EMA exponent for bhg (default 0.90).
                - weight_decay: Decoupled weight decay coefficient (default 0.0).
                - eps: Numerical stability floor (default 1e-8).
        base_params: Base parameters common to all optimizers (learning_rate,
            clipvalue, clipnorm, global_clipnorm).

    Returns:
        Configured VSGD optimizer instance.
    """
    optimizer_params = {
        "name": "VSGD",
        "ghattg": config.get("ghattg", DEFAULT_VSGD_GHATTG),
        "ps": config.get("ps", DEFAULT_VSGD_PS),
        "tau1": config.get("tau1", DEFAULT_VSGD_TAU1),
        "tau2": config.get("tau2", DEFAULT_VSGD_TAU2),
        "weight_decay": config.get("weight_decay", DEFAULT_VSGD_WEIGHT_DECAY),
        "eps": config.get("eps", DEFAULT_VSGD_EPS),
        **base_params,
    }

    return VSGD(**optimizer_params)


def _build_gefen_optimizer(
        config: Dict[str, Any],
        base_params: Dict[str, Any]
) -> Gefen:
    """Build Gefen-lite optimizer with configuration parameters.

    Gefen-lite (shared-v) is an AdamW-style optimizer with a block-shared
    second moment (one `vmean` per block of `period` elements) and
    full-precision momentum. The block `period` is chosen deterministically
    from each variable's shape, keeping the update graph-static and
    `jit_compile`-safe. Drop-in for AdamW.

    Args:
        config: Configuration dictionary with Gefen-specific parameters.
            Optional keys:
                - beta_1: First-moment EMA decay (default 0.9).
                - beta_2: Second-moment EMA decay (default 0.999).
                - epsilon: Numerical stability floor (default 1e-8).
                - weight_decay: Decoupled weight decay coefficient (default 0.0).
                - max_block_size: Largest allowed block period (default 1024).
                - min_block_size: Smallest block period before falling back to
                    per-element AdamW (default 8).
        base_params: Base parameters common to all optimizers (learning_rate,
            clipvalue, clipnorm, global_clipnorm).

    Returns:
        Configured Gefen optimizer instance.
    """
    optimizer_params = {
        "name": "gefen",
        "beta_1": config.get("beta_1", DEFAULT_GEFEN_BETA_1),
        "beta_2": config.get("beta_2", DEFAULT_GEFEN_BETA_2),
        "epsilon": config.get("epsilon", DEFAULT_GEFEN_EPSILON),
        "weight_decay": config.get("weight_decay", DEFAULT_GEFEN_WEIGHT_DECAY),
        "max_block_size": config.get("max_block_size", DEFAULT_GEFEN_MAX_BLOCK_SIZE),
        "min_block_size": config.get("min_block_size", DEFAULT_GEFEN_MIN_BLOCK_SIZE),
        **base_params,
    }

    return Gefen(**optimizer_params)