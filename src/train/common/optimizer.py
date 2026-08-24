"""The trainer-config -> optimizer adapter shared by every image/graph ET-family trainer.

One function, :func:`build_optimizer`. It is deliberately an ADAPTER and not optimizer
math: it reads a trainer ``TrainingConfig``-shaped object and delegates every actual
construction decision to :mod:`dl_techniques.optimization`
(``learning_rate_schedule_builder`` / ``optimizer_builder``). Schedule and optimizer
CONSTRUCTION logic belongs there; only the config-field-to-builder-dict mapping,
which is a ``src/train/`` concern, lives here.

It was originally written in ``train.energy_transformer.common`` and is consumed by four
trainer packages (energy_transformer, beit, graph_energy_transformer, dino) across nine
call sites, so it was promoted here; ``train.energy_transformer.common`` re-exports it and
that path still resolves to the SAME OBJECT.
"""

from typing import Any, Dict

import keras

from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
)

__all__ = ["build_optimizer"]


# ---------------------------------------------------------------------
# optimization
# ---------------------------------------------------------------------

# DECISION plan-2026-08-12T123743-e798a9e1/D-007
# This module is an ADAPTER over `dl_techniques.optimization`, and that is the whole
# reason it is allowed to exist under `src/train/`.
# WHAT NOT TO DO: do NOT grow schedule or optimizer CONSTRUCTION logic here (a new
# decay curve, a new optimizer type, a warmup formula). That belongs in
# `dl_techniques/optimization/` -- `learning_rate_schedule_builder` and
# `optimizer_builder` are the single home for it, and a second implementation under
# `src/train/` is exactly the duplication this plan is removing. What may live here is
# only the mapping from trainer `TrainingConfig` FIELD NAMES onto those builders' dicts.
# WHAT NOT TO DO (2): do NOT copy this body into a trainer package "so it is
# self-contained" -- `tests/test_train/test_beit/test_common.py` and
# `tests/test_train/test_dino/test_train_dino.py` both assert object IDENTITY through
# the re-export chain, which a copy fails by construction.
# See decisions.md D-007.
def build_optimizer(config: Any, steps_per_epoch: int) -> keras.optimizers.Optimizer:
    """Build the LR schedule + optimizer from a trainer ``TrainingConfig``.

    Reads ``lr_schedule_type``, ``learning_rate``, ``epochs``, ``warmup_epochs``,
    ``optimizer_type``, ``weight_decay`` and ``gradient_clipping`` off ``config``.

    Double-weight-decay guard (H10 / LESSONS L72): when the optimizer is AdamW, the decay goes
    through ``optimizer_builder`` ONLY. No model layer in this feature sets a
    ``kernel_regularizer=L2(...)`` -- setting both inflates the loss with an L2 penalty AND
    decays the parameters again on the update.

    Args:
        config: The trainer's ``TrainingConfig``.
        steps_per_epoch: Optimizer steps per epoch (drives decay/warmup horizons).

    Returns:
        A configured ``keras.optimizers.Optimizer``.
    """
    lr_schedule = learning_rate_schedule_builder({
        "type": config.lr_schedule_type,
        "learning_rate": config.learning_rate,
        "decay_steps": steps_per_epoch * config.epochs,
        "warmup_steps": steps_per_epoch * config.warmup_epochs,
        "alpha": 0.01,
    })

    opt_config: Dict[str, Any] = {
        "type": config.optimizer_type,
        "gradient_clipping_by_norm": config.gradient_clipping,
    }
    if config.optimizer_type.lower() == "adamw":
        opt_config["weight_decay"] = config.weight_decay

    logger.info(
        f"optimizer={config.optimizer_type}, lr={config.learning_rate}, "
        f"schedule={config.lr_schedule_type}, warmup_steps={steps_per_epoch * config.warmup_epochs}, "
        f"clip_by_norm={config.gradient_clipping}, weight_decay={config.weight_decay}"
    )
    return optimizer_builder(opt_config, lr_schedule)

