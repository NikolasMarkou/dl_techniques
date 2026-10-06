"""Trainer for TopoLM, a topographic causal language model.

The package holds the pipeline in :mod:`~train.topolm.common` and the CLI in
:mod:`~train.topolm.pretrain`::

    MPLBACKEND=Agg python -m train.topolm.pretrain --variant tiny \\
        --vocab-size 50257 --epochs 2 --eval-every-steps 50

Two things about this trainer are worth knowing before reading its code, because
both would otherwise look like mistakes:

**The validation cadence is the virtual epoch length.** Pretraining is roughly
one pass over a stream, and Keras only validates at epoch ends, so a run that
validates once per pass cannot early-stop on the rule the paper used.
``fit`` therefore receives ``steps_per_epoch=eval_every_steps`` and the run's
step count becomes ``eval_every_steps * epochs`` -- not one pass over the data.
``--train-steps`` pins the step count instead, leaving the cadence unchanged;
both numbers are logged before the first step.

**The training head must be told to aggregate the backbone's losses.**
``create_topolm_model`` passes ``aggregate_backbone_losses=True``, and it is
load-bearing: the taps reach the objective only through ``backbone.losses``. On
the same weights at ``alpha = 2.5`` a reported training loss of 15.31 aggregated
became 5.36 not -- computed every step and discarded, with a healthy curve. The
run's ``spatial/unaccounted`` field is what makes that visible if it ever
recurs.

The ``--paired`` flag trains the topographic model and an ``alpha = 0`` control
with identical seeds, data order and every other hyperparameter. The control's
taps are still created and still built, so the arms differ in their objective
alone rather than in their weights.

References:
    - Rathi, Mehrer, AlKhamissi, Binhuraib, Blauch & Schrimpf, 2025. TopoLM.
      ICLR 2025. (https://arxiv.org/abs/2410.11516)
"""

from .common import (
    PAPER_CONDITIONS,
    SMOKE_STIMULI,
    TapCapture,
    TopoLMTrainingConfig,
    build_backbone,
    build_callbacks,
    compile_model,
    create_topolm_model,
    evaluate_topography,
    extract_tap_activations,
    resolve_cadence,
    resolve_tap_sites,
    run_topographic_evaluation,
    train_paired,
    train_topolm,
)

__all__ = [
    "PAPER_CONDITIONS",
    "SMOKE_STIMULI",
    "TapCapture",
    "TopoLMTrainingConfig",
    "build_backbone",
    "build_callbacks",
    "compile_model",
    "create_topolm_model",
    "evaluate_topography",
    "extract_tap_activations",
    "resolve_cadence",
    "resolve_tap_sites",
    "run_topographic_evaluation",
    "train_paired",
    "train_topolm",
]