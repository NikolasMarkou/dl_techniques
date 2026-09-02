r"""Train the class-conditional latent ``DiT`` through stock ``compile()``/``fit()``.

Usage::

    # wiring proof, no GPU, ~seconds
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" \
        python -m train.dit.train_dit --smoke

    # a real (still synthetic-latent) run
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 \
        python -m train.dit.train_dit --variant DiT-S/2 --input-size 32 \
            --num-classes 10 --epochs 100

    # real, pre-encoded latents (see ``synthetic_data.py`` for the contract)
    ... --train-npz /data/dit/train-00000.npz --val-npz /data/dit/val-00000.npz

The pipeline
------------

.. code-block:: text

    ┌──────────────────────────────────────────────────────────────────┐
    │ records:  latent [N, H, W, C] float32   +   label [N] int32      │
    │           synthetic_records(...)   or   load_records_npz(--*-npz)│
    └────────────────────────────┬─────────────────────────────────────┘
                                 │  build_dit_dataset  (tf.data, infinite)
                                 ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │ element:  (x_t [B, H, W, C], t [B], y [B])                       │
    │           y_true [B, H, W, 2C+1] = noise ⊕ x_start ⊕ t-plane     │
    └────────────────────────────┬─────────────────────────────────────┘
                                 │  model.fit(...)   ← NO custom train_step
                                 ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │ DiT(x, t, y)  ──▶  y_pred [B, H, W, 2C]                          │
    │                    [0:C] eps_pred │ [C:2C] variance logits       │
    └────────────────────────────┬─────────────────────────────────────┘
                                 │  DDPMHybridLoss  (re-derives x_t)
                                 ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │ loss [B]  =  mean_flat((noise - eps_pred)**2)  ⊕  frozen-out VB   │
    └────────────────────────────┬─────────────────────────────────────┘
                                 │  callbacks
                                 ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │ create_callbacks(monitor='val_loss', TerminateOnNaN)             │
    │        + WeightEMACallback(decay=0.9999)                         │
    │ artifacts -> <repo>/results/<experiment_name>/                   │
    └──────────────────────────────────────────────────────────────────┘

NO CUSTOM ``train_step``
------------------------
A ``keras.losses.Loss`` sees only ``(y_true, y_pred, sample_weight)``, and the
DDPM objective additionally needs ``x_start`` and the per-sample timestep ``t``.
Both ride inside ``y_true``, packed by the data pipeline (D-002), so nothing
here overrides ``train_step`` and
``tests/test_train/test_dit/test_the_trainer_uses_stock_fit.py`` asserts
``type(model).train_step is keras.Model.train_step``.

The model is BUILT before ``fit()``
-----------------------------------
``model.build([x_shape, t_shape, y_shape])`` runs before ``compile()``. Keras 3
fires ``on_train_begin`` BEFORE the first batch, so a lazily-built model has an
EMPTY ``trainable_weights`` list there (measured: 0 weights at
``on_train_begin``, 2 after ``fit``) and a callback that snapshots it would
average nothing, silently. ``WeightEMACallback`` defends itself by deferring,
but building here is what matches upstream and keeps the EMA covering the very
first update.

Sampling is NOT wired into this trainer
---------------------------------------
There is no ``--sample`` flag: no VAE exists anywhere in this repository, so a
sample is a latent nothing can decode.
:class:`~dl_techniques.models.vision_language.dit.GaussianDiffusion` already
provides ``p_sample_loop`` / ``ddim_sample_loop``, and ``WeightEMACallback``
exposes ``applied_to(model)``, so sampling from the EMA weights of a finished
run is a few lines in a notebook rather than an untested branch here.

Nothing runs at import time -- no GPU is touched and no tensor is allocated
until ``main()`` has parsed ``argv``, so ``--help`` is free.

References:
    Peebles, W., & Xie, S. (2022). Scalable Diffusion Models with Transformers.
    arXiv:2212.09748. https://arxiv.org/abs/2212.09748
    Reference implementation: ``chuanyangjin/fast-DiT``, transcribed under
    ``reference/`` (``train_and_sample_excerpts.py`` for the optimizer, the EMA
    decay and the latent scale factor).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set, Tuple

import numpy as np

from dl_techniques.losses.ddpm_hybrid_loss import DDPMHybridLoss
from dl_techniques.models.vision_language.dit import (
    DIT_VARIANTS,
    DiT,
    DiffusionConfig,
    create_dit,
    normalize_variant_name,
)
from dl_techniques.optimization import (
    create_learning_rate_schedule,
    optimizer_builder,
)
from dl_techniques.utils.ddpm_schedule import VALID_BETA_SCHEDULES
from dl_techniques.utils.logger import logger
from train.common import (
    config_values_from_args,
    create_callbacks,
    prepare_run_dir,
    resolved_run_dir,
    save_training_history_json,
    set_seeds,
    setup_gpu,
)
from train.dit.ema_callback import DEFAULT_EMA_DECAY, WeightEMACallback
from train.dit.synthetic_data import (
    build_dit_dataset,
    load_records_npz,
    synthetic_records,
    validate_records,
)

__all__ = [
    "CLI_TO_CONFIG",
    "NON_CONFIG_DESTS",
    "SMOKE_PRESET",
    "TrainingConfig",
    "build_dit_optimizer",
    "build_parser",
    "config_from_argv",
    "create_datasets",
    "create_model",
    "load_or_draw_records",
    "main",
    "parse_arguments",
    "train_dit",
]


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Every knob of one DiT training run.

    Interface contract: a plain dataclass, validated in ``__post_init__``
    (raising :class:`ValueError` naming the offending field), and consumed by
    :func:`train_dit`. Every field is reachable from the CLI through
    :data:`CLI_TO_CONFIG` -- whose completeness in BOTH directions is pinned by
    ``tests/test_train/test_dit/test_cli_contract.py`` -- and every field is
    READ on the run path. A field nothing reads is a knob that silently does
    nothing, which this repository has shipped before.

    ``learn_sigma`` is deliberately NOT a field: :class:`DDPMHybridLoss` is
    upstream's ``LEARNED_RANGE`` objective and requires the ``2 * in_channels``
    output, so the only value that trains is ``True``. A flag whose other
    setting raises is not a knob.

    :param input_size: Side of the square latent grid ``H == W``. Upstream's
        Stable-Diffusion latents are ``32`` (a 256x256 image at VAE stride 8).
    :param in_channels: Latent channel count ``C``; ``4`` for the SD VAE.
    :param num_classes: Number of real class labels. The label table carries one
        extra NULL row at index ``num_classes``.
    :param class_dropout_rate: Probability of replacing a label with the null
        row during training, which is what makes classifier-free guidance
        available at sampling time. Upstream trains at ``0.1``.
    :param mlp_ratio: Block FFN hidden width as a multiple of ``hidden_size``.
    :param num_timesteps: Length of the diffusion chain ``T``. NOTE: the
        ``linear`` schedule is only defined on ``{1} u [20, inf)`` (D-010), so a
        short chain needs ``--schedule-name squaredcos_cap_v2``.
    :param schedule_name: Beta schedule; one of :data:`VALID_BETA_SCHEDULES`.
    :param num_train_samples: Synthetic training records to draw.
    :param num_val_samples: Synthetic validation records to draw.
    :param train_npz: Path to a real pre-encoded ``.npz`` shard. Overrides the
        synthetic training records. See ``synthetic_data.py`` for the contract.
    :param val_npz: Same, for validation.
    :param class_signal: Multiplier on the synthetic generator's per-class mean
        field. ``0.0`` removes the class signal, leaving pure noise -- an
        ablation, not a training setting.
    :param noise_std: Standard deviation of the synthetic per-sample noise.
    :param variant: One of the twelve keys of ``DIT_VARIANTS``; supplies
        ``depth``, ``hidden_size``, ``patch_size`` and ``num_heads``.
    :param dropout_rate: Block dropout rate. Upstream trains at ``0.0``.
    :param learning_rate: Learning rate. Upstream: constant ``1e-4``.
    :param optimizer_type: ``adamw`` / ``adam`` / ``sgd`` / ... Upstream: AdamW.
    :param lr_schedule_type: ``constant`` / ``cosine`` / ``exponential``.
        Upstream uses no schedule at all, hence the ``constant`` default.
    :param warmup_epochs: Warmup horizon in epochs (``cosine`` only).
    :param weight_decay: AdamW decoupled weight decay. Upstream: ``0``.
    :param gradient_clipping: Global-norm gradient clip. ``0.0`` disables it.
    :param batch_size: Records per step.
    :param epochs: Training epochs.
    :param steps_per_epoch: Steps per epoch. The training dataset is INFINITE
        (every element is redrawn with a fresh ``t`` and fresh noise), so an
        epoch is defined here and not by the record count.
    :param validation_steps: Validation steps per epoch. The validation dataset
        is finite and re-seeded identically each epoch, so ``val_loss`` is
        comparable across epochs.
    :param early_stopping_patience: Epochs without ``val_loss`` improvement.
    :param ema_decay: Decay of :class:`WeightEMACallback`. Upstream: ``0.9999``.
    :param seed: Seed for every RNG source.
    :param output_dir: Results root. A RELATIVE path resolves against the REPO
        ROOT via ``resolved_run_dir``, never the working directory.
    :param experiment_name: Run directory name under ``output_dir``.
    :param smoke: Shrink the run to a wiring proof (see :data:`SMOKE_PRESET`).
    """

    # -- latent geometry / diffusion ------------------------------------
    input_size: int = 32
    in_channels: int = 4
    num_classes: int = 1000
    class_dropout_rate: float = 0.1
    mlp_ratio: float = 4.0
    num_timesteps: int = 1000
    schedule_name: str = "linear"

    # -- data -----------------------------------------------------------
    num_train_samples: int = 1024
    num_val_samples: int = 128
    train_npz: Optional[str] = None
    val_npz: Optional[str] = None
    class_signal: float = 1.0
    noise_std: float = 1.0

    # -- model ----------------------------------------------------------
    variant: str = "DiT-S/2"
    dropout_rate: float = 0.0

    # -- optimization ---------------------------------------------------
    learning_rate: float = 1e-4
    optimizer_type: str = "adamw"
    lr_schedule_type: str = "constant"
    warmup_epochs: int = 0
    weight_decay: float = 0.0
    gradient_clipping: float = 0.0
    batch_size: int = 32
    epochs: int = 20
    steps_per_epoch: int = 32
    validation_steps: int = 4
    early_stopping_patience: int = 10
    ema_decay: float = DEFAULT_EMA_DECAY
    seed: int = 42

    # -- output ---------------------------------------------------------
    output_dir: str = "results"
    experiment_name: str = "dit"
    smoke: bool = False

    def __post_init__(self) -> None:
        try:
            self.variant = normalize_variant_name(self.variant)
        except ValueError as error:
            raise ValueError(f"variant: {error}") from error

        for name in (
            "num_train_samples", "num_val_samples", "batch_size", "epochs",
            "steps_per_epoch", "validation_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive, got {getattr(self, name)}"
                )
        for name in ("warmup_epochs", "early_stopping_patience"):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} must be non-negative, got {getattr(self, name)}"
                )
        for name in ("noise_std", "gradient_clipping", "weight_decay"):
            if getattr(self, name) < 0.0:
                raise ValueError(
                    f"{name} must be non-negative, got {getattr(self, name)}"
                )
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if not 0.0 <= self.ema_decay <= 1.0:
            raise ValueError(
                f"ema_decay must lie in [0, 1], got {self.ema_decay}"
            )
        if self.batch_size > self.num_train_samples:
            raise ValueError(
                f"batch_size ({self.batch_size}) exceeds num_train_samples "
                f"({self.num_train_samples})"
            )
        if self.batch_size > self.num_val_samples:
            raise ValueError(
                f"batch_size ({self.batch_size}) exceeds num_val_samples "
                f"({self.num_val_samples})"
            )

        # The latent geometry, the chain length and the schedule name are
        # validated by `DiffusionConfig` itself -- including the pairing of
        # `num_timesteps` with `schedule_name`, which is NOT expressible as a
        # threshold (D-010). Delegating keeps one owner for that rule.
        try:
            config = self.diffusion_config
        except ValueError as error:
            raise ValueError(f"latent/diffusion geometry: {error}") from error

        variant_row = DIT_VARIANTS[self.variant]
        try:
            config.validate_patch_size(int(variant_row["patch_size"]))
        except ValueError as error:
            raise ValueError(
                f"variant {self.variant!r} has patch_size "
                f"{variant_row['patch_size']} which does not tile an "
                f"input_size of {self.input_size}: {error}"
            ) from error

    @property
    def diffusion_config(self) -> DiffusionConfig:
        """The run's latent geometry and diffusion chain.

        Built fresh on every access (a frozen value object, cheap to construct)
        so no caller can mutate the run's geometry through a shared instance.

        :return: The validated geometry, shared by the model, the data pipeline
            and the loss.
        :rtype: DiffusionConfig
        :raises ValueError: If any geometry field is invalid.
        """
        return DiffusionConfig(
            input_size=self.input_size,
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            class_dropout_rate=self.class_dropout_rate,
            learn_sigma=True,
            mlp_ratio=self.mlp_ratio,
            num_timesteps=self.num_timesteps,
            schedule_name=self.schedule_name,
        )


#: argparse ``dest`` -> :class:`TrainingConfig` field. THE wiring, in one place.
#: A flag without a row here, or a field without a flag, fails
#: ``tests/test_train/test_dit/test_cli_contract.py`` by name. This repository
#: has shipped CLI flags that were parsed and then never forwarded; the map
#: exists so that failure mode is a red test rather than a silent no-op.
CLI_TO_CONFIG: Dict[str, str] = {
    "input_size": "input_size",
    "in_channels": "in_channels",
    "num_classes": "num_classes",
    "class_dropout_rate": "class_dropout_rate",
    "mlp_ratio": "mlp_ratio",
    "num_timesteps": "num_timesteps",
    "schedule_name": "schedule_name",
    "num_train_samples": "num_train_samples",
    "num_val_samples": "num_val_samples",
    "train_npz": "train_npz",
    "val_npz": "val_npz",
    "class_signal": "class_signal",
    "noise_std": "noise_std",
    "variant": "variant",
    "dropout_rate": "dropout_rate",
    "learning_rate": "learning_rate",
    "optimizer_type": "optimizer_type",
    "lr_schedule_type": "lr_schedule_type",
    "warmup_epochs": "warmup_epochs",
    "weight_decay": "weight_decay",
    "gradient_clipping": "gradient_clipping",
    "batch_size": "batch_size",
    "epochs": "epochs",
    "steps_per_epoch": "steps_per_epoch",
    "validation_steps": "validation_steps",
    "early_stopping_patience": "early_stopping_patience",
    "ema_decay": "ema_decay",
    "seed": "seed",
    "output_dir": "output_dir",
    "experiment_name": "experiment_name",
    "smoke": "smoke",
}

#: argparse dests that deliberately do NOT reach the config: they act on the
#: process, not on the run's parameters.
NON_CONFIG_DESTS: Set[str] = {"help", "gpu"}

# DECISION plan-2026-09-02T170923-1285ed83/D-005
# `variant`, `input_size`, `in_channels`, `num_classes` and `num_timesteps` ARE
# in this preset, which BREAKS the SAM-family rule that a smoke preset may
# change only HOW MUCH is measured, never WHAT. Do NOT remove them "for
# consistency": the smallest published variant (DiT-S/2) at upstream's 32x32
# latent grid is a 33M-parameter model over 256 tokens, and a CPU wiring proof
# on it is a timeout, not a proof. The preset therefore measures a
# DIFFERENT-SIZED model than the default, so a capacity-dependent defect
# (memory, attention length, a variant-table row) can pass smoke and fail the
# real run -- that cost is accepted because the alternative is a smoke path
# nobody runs. `num_timesteps` stays >= 20 because the `linear` schedule is
# undefined between 2 and 19 (D-010); shortening it further requires
# `--schedule-name squaredcos_cap_v2`. If you need a capacity smoke, type
# `--variant` explicitly: provenance makes your value win over this preset.
# See decisions.md D-005.
#: The ``--smoke`` preset, applied only to fields the caller did not type.
SMOKE_PRESET: Dict[str, Any] = {
    "variant": "DiT-S/2",
    "input_size": 8,
    "in_channels": 4,
    "num_classes": 4,
    "num_timesteps": 50,
    "num_train_samples": 64,
    "num_val_samples": 32,
    "batch_size": 8,
    "epochs": 3,
    "steps_per_epoch": 8,
    "validation_steps": 2,
    "warmup_epochs": 0,
    "early_stopping_patience": 3,
}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    :return: The parser. Every option is a long (``--``) spelling plus
        argparse's own ``-h``: ``train.common.args.explicitly_set_flags``
        REFUSES a parser carrying any other short option, because it cannot see
        the attached (``-b8``) or grouped (``-vb 8``) forms.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train the class-conditional latent Diffusion Transformer (DiT) on "
            "upstream's MSE + LEARNED_RANGE objective through stock "
            "compile()/fit(), with x_start and t packed into y_true and no "
            "custom train_step."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    defaults = TrainingConfig()

    geometry = parser.add_argument_group("latent geometry / diffusion")
    geometry.add_argument("--input-size", type=int, default=defaults.input_size,
                          help="Side of the square latent grid (H == W).")
    geometry.add_argument("--in-channels", type=int,
                          default=defaults.in_channels,
                          help="Latent channel count C (4 for the SD VAE).")
    geometry.add_argument("--num-classes", type=int,
                          default=defaults.num_classes,
                          help="Real class labels; the null row is index "
                               "num_classes.")
    geometry.add_argument("--class-dropout-rate", type=float,
                          default=defaults.class_dropout_rate,
                          help="Label-dropout probability; what makes "
                               "classifier-free guidance available.")
    geometry.add_argument("--mlp-ratio", type=float, default=defaults.mlp_ratio)
    geometry.add_argument("--num-timesteps", type=int,
                          default=defaults.num_timesteps,
                          help="Diffusion chain length T. The 'linear' "
                               "schedule is undefined between 2 and 19.")
    geometry.add_argument("--schedule-name", type=str,
                          default=defaults.schedule_name,
                          choices=list(VALID_BETA_SCHEDULES))

    data = parser.add_argument_group("data")
    data.add_argument("--num-train-samples", type=int,
                      default=defaults.num_train_samples,
                      help="Synthetic training records (ignored with "
                           "--train-npz).")
    data.add_argument("--num-val-samples", type=int,
                      default=defaults.num_val_samples)
    data.add_argument("--train-npz", type=str, default=defaults.train_npz,
                      help="Pre-encoded .npz shard; see synthetic_data.py for "
                           "the input contract.")
    data.add_argument("--val-npz", type=str, default=defaults.val_npz)
    data.add_argument("--class-signal", type=float,
                      default=defaults.class_signal,
                      help="Multiplier on the synthetic per-class mean field. "
                           "0.0 removes the class signal (an ablation).")
    data.add_argument("--noise-std", type=float, default=defaults.noise_std)

    model = parser.add_argument_group("model")
    model.add_argument("--variant", type=str, default=defaults.variant,
                       choices=sorted(DIT_VARIANTS),
                       help="Capacity and patch size; the latent geometry "
                            "comes from --input-size / --in-channels.")
    model.add_argument("--dropout-rate", type=float,
                       default=defaults.dropout_rate)

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--learning-rate", type=float,
                       default=defaults.learning_rate,
                       help="Upstream trains at a constant 1e-4.")
    optim.add_argument("--optimizer-type", type=str,
                       default=defaults.optimizer_type)
    optim.add_argument("--lr-schedule-type", type=str,
                       default=defaults.lr_schedule_type,
                       choices=["constant", "cosine", "exponential"],
                       help="Upstream uses no schedule at all.")
    optim.add_argument("--warmup-epochs", type=int,
                       default=defaults.warmup_epochs,
                       help="Warmup horizon in epochs ('cosine' only).")
    optim.add_argument("--weight-decay", type=float,
                       default=defaults.weight_decay,
                       help="AdamW decoupled weight decay. Upstream: 0.")
    optim.add_argument("--gradient-clipping", type=float,
                       default=defaults.gradient_clipping,
                       help="Global-norm clip; 0.0 disables it.")
    optim.add_argument("--batch-size", type=int, default=defaults.batch_size)
    optim.add_argument("--epochs", type=int, default=defaults.epochs)
    optim.add_argument("--steps-per-epoch", type=int,
                       default=defaults.steps_per_epoch,
                       help="The training dataset is infinite (every element "
                            "is redrawn), so an epoch is defined here.")
    optim.add_argument("--validation-steps", type=int,
                       default=defaults.validation_steps)
    optim.add_argument("--early-stopping-patience", type=int,
                       default=defaults.early_stopping_patience)
    optim.add_argument("--ema-decay", type=float, default=defaults.ema_decay,
                       help="WeightEMACallback decay. Upstream: 0.9999.")
    optim.add_argument("--seed", type=int, default=defaults.seed)

    output = parser.add_argument_group("output")
    output.add_argument("--output-dir", type=str, default=defaults.output_dir,
                        help="Relative paths resolve against the REPO ROOT, "
                             "never the current directory, so a run from "
                             "anywhere writes to <repo>/results/.")
    output.add_argument("--experiment-name", type=str,
                        default=defaults.experiment_name)
    output.add_argument("--gpu", type=int, default=None,
                        help="GPU index for setup_gpu (process-level; not a "
                             "config field).")
    output.add_argument("--smoke", action=argparse.BooleanOptionalAction,
                        default=defaults.smoke,
                        help="Shrink the run to a wiring proof. Any flag you "
                             "type explicitly wins over the preset.")
    return parser


def parse_arguments(
    argv: Optional[Sequence[str]] = None,
) -> Tuple[argparse.Namespace, TrainingConfig]:
    """Run the FULL ``argv -> parse -> config`` path.

    Interface contract: the single entry point :func:`main` uses AND the single
    entry point the CLI guard drives, so there is no path a test can pass while
    the trainer fails.

    The ``--smoke`` preset is applied HERE, gated on PROVENANCE
    (``train.common.args.config_values_from_args`` ->
    ``explicitly_set_flags``, a raw token scan). It must not move into
    ``__post_init__``: by then the argv tokens are gone, so ``--smoke --epochs
    3`` with ``3`` the preset's own value is indistinguishable from a bare
    ``--smoke``, and a value the caller really typed is silently overridden.

    :param argv: Tokens without the program name. ``None`` reads
        ``sys.argv[1:]``.
    :type argv: Optional[Sequence[str]]
    :return: ``(namespace, config)``. The namespace is returned only so
        :func:`main` can read the process-level dests in
        :data:`NON_CONFIG_DESTS`.
    :rtype: Tuple[argparse.Namespace, TrainingConfig]
    """
    parser = build_parser()
    args, values = config_values_from_args(
        parser, argv, CLI_TO_CONFIG, SMOKE_PRESET
    )
    return args, TrainingConfig(**values)


def config_from_argv(argv: Optional[Sequence[str]] = None) -> TrainingConfig:
    """The config half of :func:`parse_arguments`.

    :param argv: Tokens without the program name.
    :type argv: Optional[Sequence[str]]
    :return: A validated :class:`TrainingConfig`.
    :rtype: TrainingConfig
    """
    return parse_arguments(argv)[1]


# ---------------------------------------------------------------------
# RUN PIECES
# ---------------------------------------------------------------------


def create_model(config: TrainingConfig) -> DiT:
    """Construct the model and BUILD it.

    Building here, rather than letting ``fit()`` build lazily on the first
    batch, is load-bearing: Keras 3 fires ``on_train_begin`` BEFORE the first
    batch, so a lazily-built model exposes an EMPTY ``trainable_weights`` list
    to every callback that runs there (measured: 0 weights at
    ``on_train_begin``, 2 after ``fit``).

    :param config: The run's config.
    :type config: TrainingConfig
    :return: A BUILT :class:`DiT`.
    :rtype: DiT
    """
    geometry = config.diffusion_config
    model = create_dit(
        config.variant,
        input_size=geometry.input_size,
        in_channels=geometry.in_channels,
        num_classes=geometry.num_classes,
        class_dropout_rate=geometry.class_dropout_rate,
        learn_sigma=geometry.learn_sigma,
        mlp_ratio=geometry.mlp_ratio,
        dropout_rate=config.dropout_rate,
        label_seed=config.seed,
    )
    model.build([
        (None, geometry.input_size, geometry.input_size, geometry.in_channels),
        (None,),
        (None,),
    ])
    logger.info(
        "Built %s: %d weight tensors, %d parameters",
        config.variant,
        len(model.weights),
        sum(int(np.prod(w.shape)) for w in model.weights),
    )
    return model


def build_dit_optimizer(
    config: TrainingConfig, steps_per_epoch: int
) -> Any:
    """Build the LR schedule and the optimizer.

    Interface contract: pure; returns a fresh ``keras.optimizers.Optimizer``
    configured from ``learning_rate``, ``lr_schedule_type``, ``epochs``,
    ``warmup_epochs``, ``optimizer_type``, ``weight_decay`` and
    ``gradient_clipping``. The defaults reproduce upstream's
    ``AdamW(lr=1e-4, weight_decay=0)`` with NO schedule
    (``reference/train_and_sample_excerpts.py:18``).

    :param config: The run's config.
    :type config: TrainingConfig
    :param steps_per_epoch: Optimizer steps per epoch (the decay/warmup
        horizon).
    :type steps_per_epoch: int
    :return: The configured optimizer.
    :rtype: keras.optimizers.Optimizer
    """
    # DECISION plan-2026-09-02T170923-1285ed83/D-025
    # `train.common.build_optimizer` -- the shared adapter every sibling trainer
    # calls, including `bit_diffusion` -- is deliberately NOT used here. Do NOT
    # "restore consistency" by switching to it: it routes every run through
    # `learning_rate_schedule_builder`, whose accepted types are exactly
    # cosine_decay / exponential_decay / cosine_decay_restarts, so it CANNOT
    # express a constant learning rate and RAISES on `type='constant'`. Upstream
    # DiT trains at a constant 1e-4 with no schedule at all
    # (`reference/train_and_sample_excerpts.py:18`), and silently substituting a
    # cosine decay changes the published recipe with no shape, dtype or
    # finiteness symptom. Both halves still come from
    # `dl_techniques.optimization`; only the epoch-facing schedule adapter
    # differs. See decisions.md D-025.
    lr_schedule = create_learning_rate_schedule(
        initial_lr=config.learning_rate,
        schedule_type=config.lr_schedule_type,
        total_epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_steps=config.warmup_epochs * steps_per_epoch,
    )
    optimizer_config: Dict[str, Any] = {"type": config.optimizer_type}
    if config.gradient_clipping > 0.0:
        # `optimizer_builder` RENAMES the clipping keys: a literal "clipnorm"
        # is silently ignored. `gradient_clipping_by_norm` is the global one.
        optimizer_config["gradient_clipping_by_norm"] = config.gradient_clipping
    if config.optimizer_type.strip().lower() == "adamw":
        # AdamW's Keras default is 0.004, not 0 -- omitting the key would
        # silently decay the weights against upstream's recipe.
        optimizer_config["weight_decay"] = config.weight_decay
    return optimizer_builder(optimizer_config, lr_schedule)


def load_or_draw_records(
    config: TrainingConfig, split: str
) -> Dict[str, np.ndarray]:
    """Records for one split: the ``.npz`` shard if given, else synthetic.

    :param config: The run's config.
    :type config: TrainingConfig
    :param split: ``"train"`` or ``"val"``.
    :type split: str
    :return: A contract-valid record batch.
    :rtype: Dict[str, np.ndarray]
    :raises ValueError: If ``split`` is not one of the two names, or if a
        supplied shard violates the input contract.
    """
    if split == "train":
        path, count, seed_offset = (
            config.train_npz, config.num_train_samples, 0
        )
    elif split == "val":
        path, count, seed_offset = config.val_npz, config.num_val_samples, 1
    else:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    geometry = config.diffusion_config
    if path:
        records = load_records_npz(path)
        validate_records(records, geometry)
        logger.info(
            "dit: %s split read from %s (%d records)",
            split, path, records["latent"].shape[0],
        )
        return records
    return synthetic_records(
        count,
        geometry,
        seed=config.seed + seed_offset,
        class_signal=config.class_signal,
        noise_std=config.noise_std,
    )


def create_datasets(config: TrainingConfig) -> Tuple[Any, Any]:
    """Build the train and validation ``tf.data`` pipelines.

    The training dataset is INFINITE (hence ``steps_per_epoch`` at the
    ``fit()`` call) and the validation one is FINITE and identically re-seeded
    each epoch, so ``val_loss`` compares like with like across epochs.

    :param config: The run's config.
    :type config: TrainingConfig
    :return: ``(train_dataset, val_dataset)``.
    :rtype: Tuple[Any, Any]
    """
    geometry = config.diffusion_config
    train = build_dit_dataset(
        load_or_draw_records(config, "train"),
        geometry,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=True,
    )
    val = build_dit_dataset(
        load_or_draw_records(config, "val"),
        geometry,
        batch_size=config.batch_size,
        seed=config.seed + 1,
        shuffle=False,
        steps=config.validation_steps,
    )
    return train, val


def train_dit(config: TrainingConfig) -> Tuple[DiT, Any, Path]:
    """Run the training and return ``(model, history, run_dir)``.

    :param config: The run's config.
    :type config: TrainingConfig
    :return: The trained model, the ``fit()`` history, and the run directory.
    :rtype: Tuple[DiT, Any, Path]
    """
    run_dir = prepare_run_dir(config, output_dir=resolved_run_dir(config))
    logger.info("dit run '%s' -> %s", config.experiment_name, run_dir)

    model = create_model(config)
    train_dataset, val_dataset = create_datasets(config)

    # STOCK compile/fit. `DDPMHybridLoss` reads `x_start` and `t` out of the
    # packed `y_true` (D-002) and re-derives `x_t` from its own schedule, so
    # nothing here overrides `train_step`, on purpose.
    model.compile(
        optimizer=build_dit_optimizer(config, config.steps_per_epoch),
        loss=DDPMHybridLoss(
            schedule_name=config.schedule_name,
            num_timesteps=config.num_timesteps,
            in_channels=config.in_channels,
        ),
    )

    callbacks, _ = create_callbacks(
        model_name="dit",
        run_dir=str(run_dir),
        monitor="val_loss",
        monitor_mode="min",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
        include_analyzer=False,
        include_terminate_on_nan=True,
    )
    ema = WeightEMACallback(decay=config.ema_decay)
    callbacks.append(ema)

    start = time.time()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        callbacks=callbacks,
        verbose=1,
    )
    logger.info("Training finished in %.2f s", time.time() - start)
    logger.info(
        "EMA: %d updates over %d shadowed tensors at decay %.6f",
        ema.updates, len(ema.shadow_values()), ema.decay,
    )

    for key in ("loss", "val_loss"):
        values = [float(v) for v in history.history.get(key, [])]
        if values:
            logger.info(
                "%s %.6f (epoch 1) -> %.6f (epoch %d), delta %+.6f",
                key, values[0], values[-1], len(values),
                values[-1] - values[0],
            )

    save_training_history_json(history, run_dir)

    try:
        model.save(run_dir / "final_model.keras")
        logger.info("Final model saved to %s", run_dir / "final_model.keras")
    except Exception as error:  # pragma: no cover - reporting path
        logger.error("Failed to save the final model: %s", error)

    return model, history, run_dir


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse the CLI, set the process up, and train.

    ``parse_arguments`` is the FIRST statement: nothing expensive runs before
    argparse can print ``--help`` and exit.

    :param argv: Tokens without the program name. ``None`` reads
        ``sys.argv[1:]``.
    :type argv: Optional[Sequence[str]]
    """
    args, config = parse_arguments(argv)
    setup_gpu(gpu_id=args.gpu)
    set_seeds(config.seed)
    logger.info(
        "Config: %s",
        {item.name: getattr(config, item.name) for item in fields(config)},
    )
    train_dit(config)
    logger.info("dit training completed.")


if __name__ == "__main__":
    main()
