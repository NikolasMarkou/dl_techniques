"""Command-line entry point for TopoLM pre-training.

.. code-block:: bash

    # A topographic run at the paper's shape.
    MPLBACKEND=Agg python -m train.topolm.pretrain \\
        --variant paper --alpha 2.5 --train-steps 100000

    # The non-topographic control the paper compares against.
    MPLBACKEND=Agg python -m train.topolm.pretrain --variant paper --alpha 0

    # Both arms, identical seeds, back to back.
    MPLBACKEND=Agg python -m train.topolm.pretrain --variant tiny --paired

    # The Fig. 12 ablation: one shared unit layout instead of one per tap.
    MPLBACKEND=Agg python -m train.topolm.pretrain --no-permute

    # A 3D unit grid (7x7x7, radius 2 from the variant row, raw arm only).
    MPLBACKEND=Agg python -m train.topolm.pretrain --variant tiny3d

    # Watch the map form: a frame at step 0, every 200 steps and at the end.
    MPLBACKEND=Agg python -m train.topolm.pretrain --variant tiny3d \\
        --viz-every-steps 200

    # A single-tap ablation, and the paper's stopping rule relaxed.
    MPLBACKEND=Agg python -m train.topolm.pretrain \\
        --tap-sites attention --early-stop-patience 10

``MPLBACKEND=Agg`` is required on a headless machine: the run draws training
curves at the end and an unset backend reaches for X11 and takes the process
down with it.

Every flag maps onto exactly one field of
:class:`~train.topolm.common.TopoLMTrainingConfig`, so ``--help`` and
``config.json`` describe the same run.

The module is named ``pretrain`` rather than ``train_topolm`` because the
package also re-exports a FUNCTION of that name: a module and a function of the
same name in one package makes ``from train.topolm import train_topolm`` return
whichever one the import machinery reaches first, and every other trainer here
(``gpt2``, ``wave_field``) already uses ``pretrain.py``.

References:
    - Rathi, Mehrer, AlKhamissi, Binhuraib, Blauch & Schrimpf, 2025. TopoLM.
      ICLR 2025. (https://arxiv.org/abs/2410.11516)
"""

import argparse
from typing import Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from train.common import setup_gpu
from dl_techniques.models.language.topolm import MODEL_VARIANTS, grid_shape_of
from dl_techniques.utils.logger import logger

from .common import (
    DEFAULT_READOUT_FWHM,
    TopoLMTrainingConfig,
    _grid_label,
    check_viz_every_steps,
    resolve_spatial_radius,
    train_paired,
    train_topolm,
)

# ---------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Every default here mirrors the dataclass default, so an unflagged run is the
    paper's recipe rather than the trainer's idea of one. Two flags are the
    exception on purpose: ``--radius`` and ``--readout-fwhm`` default to ``None``
    ("not typed") and :func:`_config_from_args` resolves them per variant, because
    a typed value and an untyped one must be distinguishable (a 3D variant has its
    own radius and no readout) and the resolved integer is what ``config.json``
    records.
    """
    parser = argparse.ArgumentParser(
        description="TopoLM topographic causal language-model pre-training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Hardware
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")

    # Model
    parser.add_argument(
        "--variant",
        type=str,
        default="small",
        choices=sorted(MODEL_VARIANTS),
        help=(
            "Model variant. 'paper' is the published shape (784 units on a 28x28 "
            "grid, 12 blocks); 'small' and 'tiny' are repo-authored scales. "
            "'paper3d' (10x10x10, radius 4) and 'tiny3d' (7x7x7, radius 2) "
            "are 3D layouts: the grid's rank comes from the variant, there is "
            "no grid flag, and they are scored on the raw arm only."
        ),
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=100277,
        help=(
            "Embedding table rows. Must hold every id the tokenizer emits, "
            "special ids included. The trainer's tokenizer is Tiktoken 'gpt2' "
            "(50257 base ids plus special ids 50257..50260), so at least 50261 "
            "rows; the default 100277 satisfies that. 50257 rows fail on the "
            "first batch with an out-of-range Gather in word_embeddings "
            "(measured on the earlier pipeline, not re-run on the packed path); "
            "the start-up guard refuses anything below 50261."
        ),
    )
    parser.add_argument(
        "--num-layers", type=int, default=None, help="Override the block count"
    )
    parser.add_argument(
        "--num-heads", type=int, default=None, help="Override the head count"
    )
    parser.add_argument(
        "--tie-word-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tie the LM head to the token embedding table",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.0,
        help="Dropout. The paper uses none.",
    )
    parser.add_argument(
        "--attention-dropout-rate", type=float, default=0.0
    )

    # Topographic objective
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.5,
        help=(
            "Weight on every tap's spatial loss. 0 trains the "
            "non-topographic control, with an identical weight set."
        ),
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=None,
        help=(
            "Neighbourhood radius; a radius-r patch needs (2r+1)**grid_dim "
            "units, so every axis of the grid must be at least 2r+1 across. "
            "Unset takes the variant's own radius (tiny3d 2, paper3d 4) and "
            "5 for the 2D variants, the paper's."
        ),
    )
    parser.add_argument(
        "--neighborhoods",
        type=int,
        default=5,
        help="Neighbourhoods sampled per tap per step, and averaged",
    )
    parser.add_argument(
        "--distance",
        type=str,
        default="linf",
        choices=["linf", "l1", "l2"],
        help=(
            "Metric behind the inverse-distance prior. 'linf' is the paper's, "
            "and it matches the contiguity the post-hoc cluster growing and "
            "Moran's I use."
        ),
    )
    parser.add_argument(
        "--permute",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Give each tap its own random unit permutation. Disabling this is "
            "the paper's Fig. 12 ablation: with one shared layout the residual "
            "stream satisfies the loss by copying a single map forward."
        ),
    )
    parser.add_argument(
        "--tap-sites",
        type=str,
        default="both",
        choices=["both", "attention", "mlp"],
        help="Which branch outputs to tap",
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help=(
            "Number of VIRTUAL epochs. Combined with --eval-every-steps this "
            "fixes the optimizer-step count unless --train-steps is given."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help=(
            "Packed chunk length, AND the position table's size -- "
            "build_backbone forwards it to max_seq_len because the two must "
            "agree. The paper trains at 1024."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=6e-4,
        help="Peak learning rate; the paper's value",
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.05, help="Warmup fraction"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.1, help="AdamW weight decay"
    )

    # Loss
    parser.add_argument(
        "--loss-type", type=str, default="ce", choices=["ce", "focal"]
    )
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)

    # Cadence and stopping
    parser.add_argument(
        "--eval-every-steps",
        type=int,
        default=2000,
        help=(
            "Validation cadence in optimizer steps, and the virtual epoch "
            "length. Every epoch-level callback fires on this cadence."
        ),
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=32,
        help="Validation batches per evaluation",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=None,
        help=(
            "Pin the optimizer-step count. The evaluation cadence is "
            "unchanged; the number of virtual epochs is derived from it."
        ),
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        help=(
            "Consecutive validation-loss INCREASES tolerated. The paper's 3, "
            "and deliberately not the same rule as Keras' EarlyStopping."
        ),
    )
    parser.add_argument(
        "--spatial-log-every",
        type=int,
        default=100,
        help="Batches between spatial-loss log entries",
    )

    # Data source
    parser.add_argument(
        "--dataset-source", type=str, default="huggingface",
        choices=["tfds", "huggingface"],
    )
    parser.add_argument("--dataset-name", type=str, default="imdb_reviews")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default="/media/arxwn/data0_4tb/datasets/wikipedia",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--min-article-length", type=int, default=0)
    parser.add_argument("--shuffle-shards", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Checkpointing
    parser.add_argument("--checkpoint-every-steps", type=int, default=25000)
    parser.add_argument("--analyze-every-steps", type=int, default=50000)
    parser.add_argument("--max-checkpoints", type=int, default=3)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Override the chunk-aware step estimate (data volume only)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a .keras checkpoint to resume from",
    )

    # Topographic evaluation
    parser.add_argument(
        "--readout-fwhm",
        type=float,
        default=None,
        help=(
            "Full width at half maximum of the simulated fMRI readout, in "
            "grid units. Unset is 2.0 on a 2D variant and no readout on a 3D "
            "one (the Gaussian readout is 2-D, so a value typed for a 3D "
            "variant is refused). Pass 0 to score the raw activations only."
        ),
    )
    parser.add_argument("--readout-unit-spacing", type=float, default=1.0)
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=10,
        help="Smallest cluster the post-hoc sweep keeps",
    )
    parser.add_argument(
        "--permutation-p-value",
        action="store_true",
        help=(
            "Also score Moran's I against spatial randomness by permutation. "
            "The paper's third Moran statistic, and the expensive one: "
            "--num-permutations shuffles per tap per arm, so 9999 x 24 x 2 is "
            "about half a million recomputations. Off by default; the standard "
            "and islands values are reported either way."
        ),
    )
    parser.add_argument(
        "--num-permutations",
        type=int,
        default=9999,
        help=(
            "Shuffles behind the permutation p-value, which also sets its floor "
            "at 1/(num_permutations + 1). Only used with --permutation-p-value."
        ),
    )
    parser.add_argument(
        "--contrast",
        type=str,
        nargs=2,
        default=["a", "b"],
        metavar=("CONDITION_A", "CONDITION_B"),
        help=(
            "The two stimulus conditions the t-map contrasts. With the default "
            "smoke stimuli these are the 'a' and 'b' keys."
        ),
    )
    parser.add_argument(
        "--no-topography",
        dest="run_topography",
        action="store_false",
        default=True,
        help=(
            "Skip the post-hoc topographic evaluation after training. It does "
            "not switch off --viz-every-steps, which is a separate output."
        ),
    )
    parser.add_argument(
        "--viz-every-steps",
        type=int,
        default=0,
        help=(
            "Draw the topography WHILE training: at step 0, every N optimizer "
            "steps and at the end, into <run_dir>/topography_live/ (open "
            "latest_t_maps_raw.png, tail metrics.jsonl). 0 is off. N counts "
            "optimizer steps and is independent of --eval-every-steps, the "
            "virtual epoch length. Each draw scores the built-in smoke "
            "stimuli on the raw arm in eval mode and does not change the run. "
            "Works for 2D and 3D variants and for --paired."
        ),
    )

    # Paired control
    parser.add_argument(
        "--paired",
        action="store_true",
        help=(
            "Train the topographic model and then an alpha=0 control, with "
            "identical seeds, data order and every other hyperparameter."
        ),
    )

    # Output
    parser.add_argument(
        "--save-dir", type=str, default="results/topolm_pretrain"
    )

    return parser


def _resolve_readout_fwhm(args: argparse.Namespace) -> Optional[float]:
    """The readout FWHM the run gets: typed value, else the variant's rule.

    Unset gives :data:`~train.topolm.common.DEFAULT_READOUT_FWHM` on a 2D variant
    (the config and ``config.json`` as they were before 3D support) and ``None``
    on a 3D one; ``0`` gives ``None``. The Gaussian readout is a ``Conv2D``, so a
    nonzero value TYPED for a 3D variant is refused here, by flag and variant,
    rather than silently dropped.

    :param args: Parsed arguments (``variant`` is one of ``MODEL_VARIANTS``).
    :type args: argparse.Namespace
    :return: The FWHM, or ``None`` for the raw arm only.
    :rtype: Optional[float]
    :raises ValueError: If a nonzero ``--readout-fwhm`` is typed for a 3D variant.
    """
    row = MODEL_VARIANTS[args.variant]
    grid_dim = int(row.get("grid_dim", 2))
    value = args.readout_fwhm
    if grid_dim == 2:
        return DEFAULT_READOUT_FWHM if value is None else (value or None)
    if value:
        grid = _grid_label(grid_shape_of(int(row["embed_dim"]), grid_dim))
        raise ValueError(
            f"--readout-fwhm {value:g} was typed for variant {args.variant!r}, "
            f"which lays its {int(row['embed_dim'])} units out on a {grid} grid "
            f"(grid_dim={grid_dim}); the Gaussian readout is 2-D, so a 3D "
            f"variant is scored on its raw arm only. Drop --readout-fwhm (or pass "
            f"0), or choose a 2D variant."
        )
    return None


def _config_from_args(args: argparse.Namespace) -> TopoLMTrainingConfig:
    """Map parsed CLI arguments onto one :class:`TopoLMTrainingConfig`.

    Every argument appears here exactly once, so a new flag cannot be added
    without deciding where in the dataclass it lands. ``--radius`` and
    ``--readout-fwhm`` are RESOLVED here (see :func:`_build_parser`), so the
    returned config, and the ``config.json`` written from it, hold the values the
    run uses.

    :raises ValueError: If a nonzero ``--readout-fwhm`` is typed for a 3D variant,
        or ``--viz-every-steps`` is negative.
    """
    config = TopoLMTrainingConfig(
        # Model
        model_variant=args.variant,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        tie_word_embeddings=args.tie_word_embeddings,
        dropout_rate=args.dropout_rate,
        attention_dropout_rate=args.attention_dropout_rate,
        # Topographic objective
        spatial_alpha=args.alpha,
        spatial_radius=args.radius,
        spatial_neighborhoods=args.neighborhoods,
        spatial_distance=args.distance,
        spatial_permute=args.permute,
        tap_sites=args.tap_sites,
        # Training
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        # Loss
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        # Cadence and stopping
        eval_every_steps=args.eval_every_steps,
        eval_batches=args.eval_batches,
        train_steps=args.train_steps,
        early_stop_patience=args.early_stop_patience,
        spatial_log_every=args.spatial_log_every,
        # Data source
        dataset_source=args.dataset_source,
        dataset_name=args.dataset_name,
        max_samples=args.max_samples,
        hf_cache_dir=args.hf_cache_dir,
        max_train_samples=args.max_train_samples,
        val_fraction=args.val_fraction,
        min_article_length=args.min_article_length,
        shuffle_shards=args.shuffle_shards,
        seed=args.seed,
        # Checkpointing
        checkpoint_every_steps=args.checkpoint_every_steps,
        analyze_every_steps=args.analyze_every_steps,
        max_checkpoints=args.max_checkpoints,
        steps_per_epoch=args.steps_per_epoch,
        resume_from=args.resume,
        # Topographic evaluation
        readout_fwhm=_resolve_readout_fwhm(args),
        readout_unit_spacing=args.readout_unit_spacing,
        min_cluster_size=args.min_cluster_size,
        permutation_p_value=args.permutation_p_value,
        num_permutations=args.num_permutations,
        contrast_conditions=(args.contrast[0], args.contrast[1]),
        # Live topography
        viz_every_steps=check_viz_every_steps(
            args.viz_every_steps, "--viz-every-steps"
        ),
        # Output
        save_dir=args.save_dir,
    )
    config.spatial_radius = resolve_spatial_radius(config)
    return config


def main() -> None:
    """Parse arguments, configure the GPU, and train.

    :raises SystemExit: If ``--paired`` is combined with ``--alpha 0``, which has
        no topographic arm to pair against. Raised through the trainer's own
        check rather than duplicated here.
    """
    args = _build_parser().parse_args()
    setup_gpu(gpu_id=args.gpu)

    config = _config_from_args(args)
    logger.info(
        f"Config: variant={config.model_variant}, "
        f"alpha={config.spatial_alpha}, radius={config.spatial_radius}, "
        f"distance={config.spatial_distance}, permute={config.spatial_permute}, "
        f"sites={config.tap_sites}, epochs={config.num_epochs}, "
        f"eval_every_steps={config.eval_every_steps}, "
        f"train_steps={config.train_steps}, batch={config.batch_size}, "
        f"lr={config.learning_rate}, source={config.dataset_source}"
    )

    if args.paired:
        results = train_paired(
            config, run_topography=args.run_topography
        )
        for name, result in results.items():
            logger.info(
                f"{name}: wrote {result['results_dir']} "
                f"({result['epochs']} x {result['steps_per_epoch']} steps)"
            )
        return

    result = train_topolm(config, run_topography=args.run_topography)
    logger.info(f"Wrote {result['results_dir']}")


if __name__ == "__main__":
    main()