"""Train ``DiTXA`` on the bidirectional bridge objective through stock ``fit()``.

Usage::

    # wiring proof, no GPU, ~seconds
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" \\
        python -m train.bit_diffusion.train_bit_diffusion --smoke

    # a real (still synthetic-data) run
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 \\
        python -m train.bit_diffusion.train_bit_diffusion --variant S --epochs 50

    # D-002 ablations
    ... --direction forward           # forward-only (text -> image)
    ... --direction reverse           # reverse-only (image -> text)
    ... --text-as-noise               # replace the text endpoint with noise
    ... --image-as-noise              # replace the image endpoint with noise
    ... --sde-type flow_matching      # the rectified-flow baseline

    # classifier-free guidance needs a TRAINED unconditional branch (D-031)
    ... --unconditional-percent 0.3   # upstream's value, and the default here
    ... --unconditional-percent 0.0   # disable it; forward_with_cfg then asks
                                      #   the model for a regime it never saw

    # real, pre-encoded data (see synthetic_data.py for the contract)
    ... --train-npz /data/bib/train-00000.npz --val-npz /data/bib/val-00000.npz

NO CUSTOM ``train_step``
-----------------------
``t`` and the direction-specific weighting ``w(t)`` reach the loss as
``sample_weight``, the third ``tf.data`` tuple element; ``direction`` and
``cond_mask`` reach the model as ordinary inputs. The mechanism, the measurement
behind it and the rank-3 weight are documented in this package's ``__init__.py``
and in ``synthetic_data.prepare_training_batch``. Nothing here subclasses
``train_step``, and ``tests/test_train/test_bit_diffusion/`` asserts that
``type(model).train_step is keras.Model.train_step``.

Geometry is derived, not configured twice
-----------------------------------------
``--bridge-preset`` is the single source of the bridge geometry: the model's
``input_size`` and ``in_channels`` are read off it, so a preset and a variant can
never disagree about the tensor shape. ``--variant`` then supplies only the
capacity knobs (hidden size, depth, heads).

Nothing runs at import time -- no GPU is touched and no tensor is allocated until
``main()`` has parsed ``argv``, so ``--help`` is free.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set, Tuple

from dl_techniques.losses.flow_matching_velocity_loss import (
    FlowMatchingVelocityLoss,
)
from dl_techniques.models.vision_language.bit_diffusion.config import (
    BRIDGE_PRESETS,
    BridgeConfig,
    get_bridge_config,
)
from dl_techniques.models.vision_language.bit_diffusion.model import (
    DiTXA,
    create_ditxa,
)
from dl_techniques.models.vision_language.bit_diffusion.sde import (
    SDE_TYPES,
    BridgeSDE,
    create_bridge_sde,
)
from dl_techniques.utils.logger import logger
from train.common import (
    build_optimizer,
    config_values_from_args,
    create_callbacks,
    prepare_run_dir,
    resolved_run_dir,
    set_seeds,
    setup_gpu,
)
from train.bit_diffusion.synthetic_data import (
    DEFAULT_UNCONDITIONAL_PERCENT,
    DIRECTION_MODES,
    TIME_SAMPLERS,
    build_bridge_dataset,
    load_records_npz,
    synthetic_records,
    validate_records,
)

__all__ = [
    "CLI_TO_CONFIG",
    "SMOKE_PRESET",
    "TrainingConfig",
    "build_parser",
    "build_sde",
    "config_from_argv",
    "create_model",
    "load_or_draw_records",
    "main",
    "parse_arguments",
    "train_bit_diffusion",
]


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Every knob of one bit-diffusion run.

    Interface contract: a plain dataclass, validated in ``__post_init__``, and
    consumed by :func:`train_bit_diffusion`. Every field is reachable from the
    CLI (see :data:`CLI_TO_CONFIG`, whose completeness is pinned by
    ``tests/test_train/test_bit_diffusion/test_cli_contract.py``) and every field
    is READ somewhere on the run path -- a field nothing reads is a knob that
    silently does nothing, which this repo has shipped before.

    :param bridge_preset: Key of ``BRIDGE_PRESETS``; the single source of the
        bridge geometry, from which the model's ``input_size`` and
        ``in_channels`` are derived.
    :param num_train_samples: Synthetic training records to draw.
    :param num_val_samples: Synthetic validation records to draw.
    :param train_npz: Path to a real pre-encoded ``.npz`` shard. Overrides the
        synthetic training records.
    :param val_npz: Same, for validation.
    :param synthetic_latent_std: Standard deviation of the synthetic latent.
    :param min_tokens: Minimum non-padding token rows per synthetic sample.
    :param sde_type: One of ``SDE_TYPES``.
    :param sde_alpha: ``alpha`` of the periodic / cosine-decay volatility.
    :param sde_k: ``k`` (frequency) of the periodic volatility. Ignored by the
        cosine-decay variant, which fixes ``k = 0.5``.
    :param sde_eps: ``eps`` floor of the periodic / cosine-decay volatility.
    :param sde_uniform_k: ``K``, the constant volatility of the uniform variant.
    :param sde_drift: ``A``, the Ornstein-Uhlenbeck drift of the uniform variant.
    :param time_sampler: One of ``TIME_SAMPLERS``.
    :param direction: One of ``DIRECTION_MODES`` (D-002 ablation).
    :param unconditional_percent: Per-sample probability that a training example
        is made UNCONDITIONAL (``cond_mask = 0``). Upstream's
        ``--unconditional-percent``; ``0.3`` on every reference launch script.
        This is what makes ``DiTXA.forward_with_cfg``'s unconditional branch a
        regime the model has actually seen. NOT the same knob as
        ``class_dropout_rate``, which drops only the prompt-kind LABEL.
    :param text_as_noise: Replace the text endpoint with noise (D-002 ablation).
    :param image_as_noise: Replace the image endpoint with noise (D-002).
    :param variant: Key of ``DiTXA.MODEL_VARIANTS`` -- capacity only.
    :param forward_cond_scale: Forward-direction raw-pixel conditioning scale.
    :param class_dropout_rate: CFG label-dropout probability.
    :param drop_path_rate: Terminal stochastic-depth rate.
    :param dropout_rate: Block dropout rate.
    :param learning_rate: Peak learning rate.
    :param optimizer_type: ``adam`` / ``adamw`` / ``sgd`` / ...
    :param lr_schedule_type: ``cosine_decay`` / ``exponential_decay`` /
        ``cosine_decay_restarts``.
    :param warmup_epochs: Warmup horizon, in epochs, for the schedule.
    :param weight_decay: AdamW decoupled weight decay.
    :param gradient_clipping: Global-norm gradient clip.
    :param batch_size: Records per step.
    :param epochs: Training epochs.
    :param steps_per_epoch: Steps per epoch. The dataset is infinite (every
        element is redrawn), so an epoch is defined here, not by the record count.
    :param validation_steps: Validation steps per epoch.
    :param early_stopping_patience: Epochs without ``val_loss`` improvement.
    :param seed: Seed for every RNG source.
    :param output_dir: Results root. A RELATIVE path resolves against the REPO
        ROOT via ``resolved_run_dir``, never the working directory.
    :param experiment_name: Run directory name under ``output_dir``.
    :param smoke: Shrink the run to a wiring proof.
    """

    # -- data ----------------------------------------------------------
    bridge_preset: str = "sd"
    num_train_samples: int = 1024
    num_val_samples: int = 128
    train_npz: Optional[str] = None
    val_npz: Optional[str] = None
    synthetic_latent_std: float = 1.0
    min_tokens: int = 1

    # -- bridge process -------------------------------------------------
    sde_type: str = "periodic"
    sde_alpha: float = 0.95
    sde_k: float = 1.0
    sde_eps: float = 0.05
    sde_uniform_k: float = 1.0
    sde_drift: float = 0.0
    time_sampler: str = "logit_normal"
    direction: str = "both"
    unconditional_percent: float = DEFAULT_UNCONDITIONAL_PERCENT
    text_as_noise: bool = False
    image_as_noise: bool = False

    # -- model ----------------------------------------------------------
    variant: str = "S"
    forward_cond_scale: float = 1.0
    class_dropout_rate: float = 0.1
    drop_path_rate: float = 0.0
    dropout_rate: float = 0.0

    # -- optimization ---------------------------------------------------
    learning_rate: float = 1e-4
    optimizer_type: str = "adamw"
    lr_schedule_type: str = "cosine_decay"
    warmup_epochs: int = 1
    weight_decay: float = 0.0
    gradient_clipping: float = 1.0
    batch_size: int = 32
    epochs: int = 20
    steps_per_epoch: int = 32
    validation_steps: int = 4
    early_stopping_patience: int = 10
    seed: int = 42

    # -- output ---------------------------------------------------------
    output_dir: str = "results"
    experiment_name: str = "bit_diffusion"
    smoke: bool = False

    def __post_init__(self) -> None:
        if self.bridge_preset not in BRIDGE_PRESETS:
            raise ValueError(
                f"Unknown bridge preset '{self.bridge_preset}'. Available: "
                f"{sorted(BRIDGE_PRESETS)}"
            )
        if self.variant not in DiTXA.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{self.variant}'. Available: "
                f"{sorted(DiTXA.MODEL_VARIANTS)}"
            )
        if self.sde_type not in SDE_TYPES:
            raise ValueError(
                f"Unknown SDE type '{self.sde_type}'. Available: {sorted(SDE_TYPES)}"
            )
        if self.direction not in DIRECTION_MODES:
            raise ValueError(
                f"Unknown direction '{self.direction}'. Available: "
                f"{list(DIRECTION_MODES)}"
            )
        if self.time_sampler not in TIME_SAMPLERS:
            raise ValueError(
                f"Unknown time sampler '{self.time_sampler}'. Available: "
                f"{list(TIME_SAMPLERS)}"
            )
        if not 0.0 <= self.unconditional_percent <= 1.0:
            raise ValueError(
                "unconditional_percent must be a probability in [0, 1], got "
                f"{self.unconditional_percent!r}"
            )
        for name in ("batch_size", "epochs", "steps_per_epoch",
                     "validation_steps", "num_train_samples",
                     "num_val_samples"):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive, got {getattr(self, name)}"
                )
        if self.warmup_epochs < 0:
            raise ValueError(
                f"warmup_epochs must be non-negative, got {self.warmup_epochs}"
            )

    @property
    def bridge_config(self) -> BridgeConfig:
        """The run's bridge geometry, with the two ablation flags applied.

        A COPY of the preset, never the shared module-level object: the
        ``*_as_noise`` flags would otherwise mutate ``BRIDGE_PRESETS`` for the
        whole process, and a second run in the same interpreter (a test session)
        would inherit them.

        :return: The validated per-run geometry.
        :rtype: BridgeConfig
        """
        preset = get_bridge_config(self.bridge_preset)
        return BridgeConfig(
            token_seq_len=preset.token_seq_len,
            token_emb_dim=preset.token_emb_dim,
            bridge_shape=preset.bridge_shape,
            patch_size=preset.patch_size,
            text_as_noise=self.text_as_noise,
            image_as_noise=self.image_as_noise,
            latent_scale=preset.latent_scale,
            latent_shift=preset.latent_shift,
        ).validate()


#: argparse ``dest`` -> :class:`TrainingConfig` field. THE wiring, in one place.
#: A flag without a row here, or a field without a flag, fails
#: ``test_cli_contract.py`` by name.
CLI_TO_CONFIG: Dict[str, str] = {
    "bridge_preset": "bridge_preset",
    "num_train_samples": "num_train_samples",
    "num_val_samples": "num_val_samples",
    "train_npz": "train_npz",
    "val_npz": "val_npz",
    "synthetic_latent_std": "synthetic_latent_std",
    "min_tokens": "min_tokens",
    "sde_type": "sde_type",
    "sde_alpha": "sde_alpha",
    "sde_k": "sde_k",
    "sde_eps": "sde_eps",
    "sde_uniform_k": "sde_uniform_k",
    "sde_drift": "sde_drift",
    "time_sampler": "time_sampler",
    "direction": "direction",
    "unconditional_percent": "unconditional_percent",
    "text_as_noise": "text_as_noise",
    "image_as_noise": "image_as_noise",
    "variant": "variant",
    "forward_cond_scale": "forward_cond_scale",
    "class_dropout_rate": "class_dropout_rate",
    "drop_path_rate": "drop_path_rate",
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
    "seed": "seed",
    "output_dir": "output_dir",
    "experiment_name": "experiment_name",
    "smoke": "smoke",
}

#: argparse dests that deliberately do NOT reach the config: they act on the
#: process, not on the run's parameters.
NON_CONFIG_DESTS: Set[str] = {"help", "gpu"}

#: The ``--smoke`` preset, applied only to fields the caller did not type.
#:
#: DELIBERATE DEVIATION from the SAM-family rule that a smoke preset may change
#: only HOW MUCH is measured, never WHAT. ``variant`` and ``bridge_preset`` ARE
#: in here, because the default ``S`` variant at the ``sd`` geometry is a 50.6M-
#: parameter model and a CPU wiring proof on it is a timeout, not a proof. The
#: ``tiny`` pair is the smallest configuration that exercises every code path
#: (both directions, both embedders, the packing, the weighting). If you need a
#: capacity smoke, type ``--variant`` explicitly -- provenance makes your value
#: win over this preset.
# DECISION plan-2026-09-02T094601-77d4a04e/D-023
# `variant`/`bridge_preset` ARE in this preset, breaking the SAM-family rule that
# a smoke preset changes only HOW MUCH. Do NOT remove them "for consistency": the
# default `S`/`sd` pair is a 50.6M-parameter model and a CPU wiring proof on it is
# a timeout, not a proof. See decisions.md D-023.
SMOKE_PRESET: Dict[str, Any] = {
    "variant": "tiny",
    "bridge_preset": "tiny",
    "num_train_samples": 32,
    "num_val_samples": 8,
    "batch_size": 4,
    "epochs": 2,
    "steps_per_epoch": 4,
    "validation_steps": 2,
    "warmup_epochs": 0,
    "early_stopping_patience": 2,
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
            "Train the BiT/BiB bidirectional text<->image diffusion bridge "
            "(DiTXA) through stock compile()/fit(), with w(t) riding in as "
            "sample_weight and no custom train_step."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    defaults = TrainingConfig()

    data = parser.add_argument_group("data")
    data.add_argument("--bridge-preset", type=str, default=defaults.bridge_preset,
                      choices=sorted(BRIDGE_PRESETS),
                      help="Bridge geometry. Also fixes the model's input_size "
                           "and in_channels.")
    data.add_argument("--num-train-samples", type=int,
                      default=defaults.num_train_samples,
                      help="Synthetic training records (ignored with --train-npz).")
    data.add_argument("--num-val-samples", type=int,
                      default=defaults.num_val_samples)
    data.add_argument("--train-npz", type=str, default=defaults.train_npz,
                      help="Pre-encoded .npz shard; see synthetic_data.py for "
                           "the input contract.")
    data.add_argument("--val-npz", type=str, default=defaults.val_npz)
    data.add_argument("--synthetic-latent-std", type=float,
                      default=defaults.synthetic_latent_std)
    data.add_argument("--min-tokens", type=int, default=defaults.min_tokens,
                      help="Minimum non-padding token rows per synthetic sample.")

    process = parser.add_argument_group("bridge process")
    process.add_argument("--sde-type", type=str, default=defaults.sde_type,
                         choices=sorted(SDE_TYPES))
    process.add_argument("--sde-alpha", type=float, default=defaults.sde_alpha)
    process.add_argument("--sde-k", type=float, default=defaults.sde_k,
                         help="Periodic volatility frequency. The cosine-decay "
                              "variant fixes k=0.5 and ignores this.")
    process.add_argument("--sde-eps", type=float, default=defaults.sde_eps)
    process.add_argument("--sde-uniform-k", type=float,
                         default=defaults.sde_uniform_k,
                         help="Constant volatility K of the uniform variant.")
    process.add_argument("--sde-drift", type=float, default=defaults.sde_drift,
                         help="Ornstein-Uhlenbeck drift A of the uniform variant.")
    process.add_argument("--time-sampler", type=str,
                         default=defaults.time_sampler, choices=list(TIME_SAMPLERS))
    process.add_argument("--direction", type=str, default=defaults.direction,
                         choices=list(DIRECTION_MODES),
                         help="Ablation: train both directions, or only one.")
    process.add_argument("--unconditional-percent", type=float,
                         default=defaults.unconditional_percent,
                         help="Per-sample probability of zeroing cond_mask "
                              "during training, so classifier-free guidance "
                              "has a trained unconditional branch. Upstream "
                              "runs 0.3; 0.0 disables it.")
    process.add_argument("--text-as-noise",
                         action=argparse.BooleanOptionalAction,
                         default=defaults.text_as_noise)
    process.add_argument("--image-as-noise",
                         action=argparse.BooleanOptionalAction,
                         default=defaults.image_as_noise)

    model = parser.add_argument_group("model")
    model.add_argument("--variant", type=str, default=defaults.variant,
                       choices=sorted(DiTXA.MODEL_VARIANTS),
                       help="Capacity only; geometry comes from --bridge-preset.")
    model.add_argument("--forward-cond-scale", type=float,
                       default=defaults.forward_cond_scale)
    model.add_argument("--class-dropout-rate", type=float,
                       default=defaults.class_dropout_rate)
    model.add_argument("--drop-path-rate", type=float,
                       default=defaults.drop_path_rate)
    model.add_argument("--dropout-rate", type=float, default=defaults.dropout_rate)

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--learning-rate", type=float,
                       default=defaults.learning_rate)
    optim.add_argument("--optimizer-type", type=str,
                       default=defaults.optimizer_type)
    optim.add_argument("--lr-schedule-type", type=str,
                       default=defaults.lr_schedule_type)
    optim.add_argument("--warmup-epochs", type=int, default=defaults.warmup_epochs)
    optim.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    optim.add_argument("--gradient-clipping", type=float,
                       default=defaults.gradient_clipping)
    optim.add_argument("--batch-size", type=int, default=defaults.batch_size)
    optim.add_argument("--epochs", type=int, default=defaults.epochs)
    optim.add_argument("--steps-per-epoch", type=int,
                       default=defaults.steps_per_epoch,
                       help="The dataset is infinite (every element is redrawn), "
                            "so an epoch is defined here.")
    optim.add_argument("--validation-steps", type=int,
                       default=defaults.validation_steps)
    optim.add_argument("--early-stopping-patience", type=int,
                       default=defaults.early_stopping_patience)
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

    Interface contract: the single entry point ``main()`` uses AND the single
    entry point the wiring guard drives, so there is no path a test can pass
    while the trainer fails.

    The ``--smoke`` preset is applied HERE, in the builder, gated on PROVENANCE
    (``train.common.args.config_values_from_args`` -> ``explicitly_set_flags``,
    a raw token scan). It must not move into ``__post_init__``: by then the argv
    tokens are gone, so ``--smoke --epochs 2`` with ``2`` the preset's own value
    is indistinguishable from a bare ``--smoke``, and a value the caller really
    typed is silently overridden.

    :param argv: Tokens without the program name. ``None`` reads ``sys.argv[1:]``.
    :type argv: Optional[Sequence[str]]
    :return: ``(namespace, config)``. The namespace is returned only so
        ``main()`` can read the process-level dests in :data:`NON_CONFIG_DESTS`.
    :rtype: Tuple[argparse.Namespace, TrainingConfig]
    """
    parser = build_parser()
    args, values = config_values_from_args(
        parser, argv, CLI_TO_CONFIG, SMOKE_PRESET
    )
    return args, TrainingConfig(**values)


def config_from_argv(
    argv: Optional[Sequence[str]] = None,
) -> TrainingConfig:
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


def build_sde(config: TrainingConfig) -> BridgeSDE:
    """Construct the base process named by ``config.sde_type``.

    The four variants take DIFFERENT constructor arguments -- passing them all
    to every variant raises -- so the mapping is explicit rather than a blanket
    ``**kwargs`` forward.

    :param config: The run's config.
    :type config: TrainingConfig
    :return: The base process.
    :rtype: BridgeSDE
    """
    if config.sde_type == "uniform":
        kwargs: Dict[str, Any] = {
            "A": config.sde_drift,
            "K": config.sde_uniform_k,
        }
    elif config.sde_type == "periodic":
        kwargs = {
            "alpha": config.sde_alpha,
            "k": config.sde_k,
            "eps": config.sde_eps,
        }
    elif config.sde_type == "cosine_decay":
        # `k` is fixed at 0.5 by the class (the half-period shift IS the
        # variant), so `--sde-k` is deliberately not forwarded here.
        kwargs = {"alpha": config.sde_alpha, "eps": config.sde_eps}
    else:
        kwargs = {}
    return create_bridge_sde(config.sde_type, **kwargs)


def create_model(config: TrainingConfig) -> DiTXA:
    """Build the model, with its geometry DERIVED from the bridge preset.

    :param config: The run's config.
    :type config: TrainingConfig
    :return: An unbuilt :class:`DiTXA`.
    :rtype: DiTXA
    :raises ValueError: If the bridge preset is not square, which the shared
        square ``pos_embed`` grid requires.
    """
    bridge = config.bridge_config
    if bridge.height != bridge.width:
        raise ValueError(
            f"bridge preset '{config.bridge_preset}' is "
            f"{bridge.height}x{bridge.width}; DiTXA's 2D sin-cos position grid "
            "is square by construction."
        )
    return create_ditxa(
        config.variant,
        input_size=bridge.height,
        in_channels=bridge.channels,
        patch_size=bridge.patch_size,
        forward_cond_scale=config.forward_cond_scale,
        class_dropout_rate=config.class_dropout_rate,
        drop_path_rate=config.drop_path_rate,
        dropout_rate=config.dropout_rate,
        label_seed=config.seed,
    )


def load_or_draw_records(
    config: TrainingConfig, split: str
) -> Dict[str, Any]:
    """Records for one split: the ``.npz`` shard if given, else synthetic.

    :param config: The run's config.
    :type config: TrainingConfig
    :param split: ``"train"`` or ``"val"``.
    :type split: str
    :return: A contract-valid record batch.
    :rtype: Dict[str, Any]
    :raises ValueError: If ``split`` is not one of the two names.
    """
    if split == "train":
        path, count, seed_offset = (
            config.train_npz, config.num_train_samples, 0
        )
    elif split == "val":
        path, count, seed_offset = (
            config.val_npz, config.num_val_samples, 1
        )
    else:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    bridge = config.bridge_config
    if path:
        records = load_records_npz(path)
        validate_records(records, bridge)
        logger.info(
            "bit_diffusion: %s split read from %s (%d records)",
            split, path, records["latent"].shape[0],
        )
        return records
    return synthetic_records(
        count,
        bridge,
        seed=config.seed + seed_offset,
        latent_std=config.synthetic_latent_std,
        min_tokens=config.min_tokens,
    )


def create_datasets(
    config: TrainingConfig, sde: BridgeSDE
) -> Tuple[Any, Any]:
    """Build the train and validation ``tf.data`` pipelines.

    The validation dataset is FINITE (``steps=validation_steps``) and the
    training one infinite, which is why ``fit()`` is given ``steps_per_epoch``.

    :param config: The run's config.
    :type config: TrainingConfig
    :param sde: The base process.
    :type sde: BridgeSDE
    :return: ``(train_dataset, val_dataset)``.
    :rtype: Tuple[Any, Any]
    """
    bridge = config.bridge_config
    common = dict(
        config=bridge,
        sde=sde,
        batch_size=config.batch_size,
        direction_mode=config.direction,
        time_sampler=config.time_sampler,
        unconditional_percent=config.unconditional_percent,
    )
    train = build_bridge_dataset(
        load_or_draw_records(config, "train"),
        seed=config.seed,
        shuffle=True,
        **common,
    )
    val = build_bridge_dataset(
        load_or_draw_records(config, "val"),
        seed=config.seed + 1,
        shuffle=False,
        steps=config.validation_steps,
        **common,
    )
    return train, val


def train_bit_diffusion(config: TrainingConfig) -> Tuple[DiTXA, Any, Path]:
    """Run the training and return ``(model, history, run_dir)``.

    :param config: The run's config.
    :type config: TrainingConfig
    :return: The trained model, the ``fit()`` history, and the run directory.
    :rtype: Tuple[DiTXA, Any, Path]
    """
    run_dir = prepare_run_dir(config, output_dir=resolved_run_dir(config))
    logger.info(
        "bit_diffusion run '%s' -> %s", config.experiment_name, run_dir
    )

    sde = build_sde(config)
    model = create_model(config)
    train_dataset, val_dataset = create_datasets(config, sde)

    # STOCK compile/fit. `FlowMatchingVelocityLoss` reduces the channel axis,
    # giving a (B, H, W) value tensor; the pipeline's (B, H, W) sample_weight
    # multiplies it elementwise and `sum_over_batch_size` then reproduces
    # upstream's `mean((pred - target)**2 * w(t))` exactly. There is no
    # train_step override anywhere in this file, on purpose.
    model.compile(
        optimizer=build_optimizer(config, config.steps_per_epoch),
        loss=FlowMatchingVelocityLoss(),
    )

    callbacks, _ = create_callbacks(
        model_name="bit_diffusion",
        run_dir=str(run_dir),
        monitor="val_loss",
        monitor_mode="min",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
        include_analyzer=False,
        include_terminate_on_nan=True,
    )

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

    losses = [float(value) for value in history.history.get("loss", [])]
    if losses:
        logger.info(
            "loss %.6f (epoch 1) -> %.6f (epoch %d), delta %+.6f",
            losses[0], losses[-1], len(losses), losses[-1] - losses[0],
        )

    try:
        payload = {
            "history": {
                key: [float(v) for v in values]
                for key, values in history.history.items()
            },
        }
        with open(run_dir / "training_history.json", "w") as handle:
            json.dump(payload, handle, indent=2)
    except Exception as error:  # pragma: no cover - reporting path
        logger.warning("Failed to write training history: %s", error)

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

    :param argv: Tokens without the program name. ``None`` reads ``sys.argv[1:]``.
    :type argv: Optional[Sequence[str]]
    """
    args, config = parse_arguments(argv)
    setup_gpu(gpu_id=args.gpu)
    set_seeds(config.seed)
    logger.info(
        "Config: %s",
        {item.name: getattr(config, item.name) for item in fields(config)},
    )
    train_bit_diffusion(config)
    logger.info("bit_diffusion training completed.")


if __name__ == "__main__":
    main()
