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

from typing import Any, Dict, Optional

import keras

from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
)

__all__ = ["build_optimizer", "sgd_weight_decay_regularizer"]


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

    refuse_a_dropped_weight_decay(config.optimizer_type, config.weight_decay, trainer="train.common.build_optimizer")
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


def refuse_a_dropped_weight_decay(optimizer_type: str, weight_decay: Optional[float], *, trainer: str) -> None:
    """Raise when ``weight_decay`` is a positive number the named optimizer would be handed no part of.

    Interface contract: ``optimizer_type`` is the trainer's ``--optimizer`` value, ``weight_decay`` its ``--weight-decay``
    (``None``, zero and negative mean no decay), ``trainer`` names the caller for the message. Only AdamW takes the decay
    (decoupled, inside the optimizer); an SGD, Adam or RMSprop run needs the decay as an L2 penalty on the model's kernels,
    built by :func:`sgd_weight_decay_regularizer` where the model factory takes a ``kernel_regularizer``. A trainer that does
    neither used to train with NO decay and say nothing. Failure mode: ``ValueError`` naming the flag to change.
    """
    # DECISION plan-2026-11-03T221936-98b866fb/D-063
    # REFUSE, do not apply and do not ignore (D-043's rule). Applying Keras' decoupled decay to SGD would be a third meaning of
    # "weight decay" next to the paper's coupled one; ignoring it is the defect this function exists for. Do NOT turn this into a
    # warning, and do NOT make a trainer's default decay silently 0 under sgd: the user typed the optimizer, the flag says what to do.
    if weight_decay is None or weight_decay <= 0 or str(optimizer_type).lower() == "adamw":
        return
    raise ValueError(
        f"{trainer}: weight_decay={weight_decay!r} would be dropped without a word, because optimizer {optimizer_type!r} "
        "is handed no decay (only adamw applies it, inside the optimizer; an sgd decay is an L2 penalty on the model's kernels "
        "that this trainer does not build). Pass --weight-decay 0 for no decay, or --optimizer adamw."
    )


def sgd_weight_decay_regularizer(weight_decay: Optional[float]) -> Optional[keras.regularizers.Regularizer]:
    """Return ``L2(weight_decay / 2)``, the Keras penalty that IS the SGD weight decay ``weight_decay``; ``None`` for none.

    Interface contract: ``weight_decay`` is the ``wd`` of He et al. 2015 (section 3.4) and of ``torch.optim.SGD(weight_decay=wd)``,
    whose update adds ``wd * w`` to the gradient; ``None``, zero and a negative number mean no decay and give ``None``. The result
    goes to a model's ``kernel_regularizer`` and is then the ONLY decay: the optimizer gets none (an optimizer with decoupled decay,
    AdamW, must not be paired with this at all: the weights would decay twice). The same rule serves Adam without the W, whose
    coupled decay adds ``wd * w`` to the gradient as well. Callers: ``train.api.recipes._weight_decay_regularizer``,
    ``train.resnet.train_resnet``, ``train.vit.train_vit``, ``train.rbf_protonet.train_rbf_protonet``;
    ``tests/test_train/test_the_sgd_weight_decay_has_one_rule.py`` finds, by AST, any other module under ``src/train`` or
    ``src/applications`` that builds an ``L2`` from a decay. Failure mode: none (a plain number in, a regularizer or ``None`` out).
    """
    # DECISION plan-2026-11-03T221936-98b866fb/D-042 (the rule, step 10e) and D-048 (one function for it, step 10f)
    # The penalty is `L2(weight_decay / 2)`, NOT `L2(weight_decay)`. Keras `L2(l)` adds `l * sum(w^2)` to the loss, whose
    # gradient is `2 * l * w`; the SGD weight decay adds `wd * w`. `L2(wd)` decayed twice as hard as the number says (measured in
    # D-042: a real SGD step shrinks a weight 2.0000 times as far). `train_vit` and `train_rbf_protonet` kept `L2(config.weight_decay)`
    # under `--optimizer sgd` until D-048. Do NOT build an `L2` from a decay inline in a trainer: call this function.
    if weight_decay is None or weight_decay <= 0:
        return None
    return keras.regularizers.L2(weight_decay / 2)

