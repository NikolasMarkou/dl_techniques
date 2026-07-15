"""Shared plumbing for the graph Energy Transformer trainers.

Deliberately TINY. The two graph trainers (``train_anomaly.py``, ``train_classification.py``)
each own their ``TrainingConfig`` / ``parse_arguments`` / ``config_from_args`` so a dropped
argparse flag is a LOCAL, greppable, testable defect rather than an inherited one — the same
discipline as the image ``train.energy_transformer.common``.

The ONE genuinely shared thing is :func:`build_optimizer`, which reads the same set of
optimization fields off any trainer ``TrainingConfig`` (``lr_schedule_type``,
``learning_rate``, ``epochs``, ``warmup_epochs``, ``optimizer_type``, ``weight_decay``,
``gradient_clipping``). It is dataset- and model-agnostic, so rather than re-implement it we
IMPORT and re-export the image trainer's implementation (DRY — one optimizer block, one
double-weight-decay guard, one place to fix).

**No ``EnergyTraceCallback`` here.** The image trainers log the ET block's energy-descent
trace out of graph via a probe backbone rebuilt with ``return_energy=True``. The graph
backbone (:class:`~dl_techniques.models.graph_energy_transformer.GraphEnergyTransformerBackbone`)
has NO ``return_energy`` flag and never surfaces the energy trace — its blocks are built with
``return_energy=False`` unconditionally — so there is no energy trace to probe and no fp16
energy-trace hazard to guard against. The callback is intentionally NOT duplicated here.
"""

# The optimizer block is model/dataset-agnostic — reuse it verbatim, do not re-implement.
from train.energy_transformer.common import build_optimizer

__all__ = ["build_optimizer"]
