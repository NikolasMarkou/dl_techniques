r"""Bespoke, non-``fit()`` trainer for ``MiniVec2VecAligner``.

Usage::

    # what the CLI offers -- allocates nothing, touches no GPU/data
    .venv/bin/python -m train.mini_vec2vec.train_mini_vec2vec --help

    # a tiny synthetic smoke run
    MPLBACKEND=Agg .venv/bin/python -m train.mini_vec2vec.train_mini_vec2vec \
        --embedding-dim 16 --n-samples 200 --n-eval 50 --n-clusters 4 \
        --approx-clusters 4 --approx-runs 2 --approx-neighbors 3 \
        --refine1-iterations 3 --refine1-sample-size 50 --refine1-neighbors 3 \
        --refine2-clusters 4 --output-dir results

Why this trainer is not ``fit()``-shaped
-----------------------------------------
``MiniVec2VecAligner`` (``dl_techniques.models.language.mini_vec2vec.model``)
learns a single square linear map between two embedding spaces by
clustering, quadratic assignment and orthogonal Procrustes -- five numpy/
scikit-learn/scipy stages, none of them a gradient step. Its own module
docstring says so explicitly: "``W`` is a trainable weight but is only ever
written by ``assign`` from numpy; the entry point is ``align``, not
``fit``." Wrapping it in a Keras ``compile()``/``fit()`` loop would
misrepresent the model's real training procedure, which is exactly what
this plan's Pre-Mortem #3 warns against fabricating.

Falsification check against ``example_alignment.py`` (Plan Assumption A4)
---------------------------------------------------------------------------
``dl_techniques/models/language/mini_vec2vec/example_alignment.py`` was read
in full before writing this file. It is a genuine, self-contained, runnable
reference, not dead code or a notebook dump:

- ``generate_synthetic_data`` builds two aligned embedding spaces in-process
  (a gaussian-mixture base cloud plus a random orthogonal ground-truth
  transform) -- no external file or network dependency.
- ``run_alignment_example`` calls the model's own public factory
  (``create_mini_vec2vec_aligner`` is used indirectly via
  ``MiniVec2VecAligner(embedding_dim=...)``) and the model's own public
  entry point (``aligner.align(...)``), then evaluates and serializes the
  result.
- ``if __name__ == "__main__":`` actually runs it end to end, including a
  "transform new embeddings" demo.

This trainer is therefore BASED ON that reference rather than inventing a
new procedure: ``generate_synthetic_data``, ``align_frame``,
``compute_top1_accuracy``, ``compute_mean_cosine_similarity`` and
``compute_transformation_error`` are imported and reused verbatim (DRY --
they are not re-implemented here), and ``run_alignment`` below mirrors
``run_alignment_example``'s five steps (generate data -> build aligner ->
``align`` -> evaluate in the fitted frame -> persist artifacts).

Why there is no ``--gpu`` flag
-------------------------------
Every stage of ``align()`` (``KMeans``, ``quadratic_assignment``,
``NearestNeighbors``, the SVD in ``_procrustes``) runs on CPU through numpy/
scikit-learn/scipy; ``model.py``'s own docstring states this outright ("All
heavy work runs on CPU through scikit-learn and scipy"). The only Keras/GPU-
eligible op anywhere in the model is ``call()``'s single ``X @ W`` matmul,
which is negligible next to the CPU-bound clustering/QAP stages -- pinning a
GPU via ``setup_gpu``/``CUDA_VISIBLE_DEVICES`` would configure a device this
trainer never meaningfully uses. Per this plan's own constraint ("``--gpu``
process-level only if GPU is used at all"), it is omitted rather than added
for house-shape consistency alone.

Why there is no ``common.py``
-------------------------------
See ``train/mini_vec2vec/__init__.py``'s module docstring.

``main()`` PARSES FIRST.
    ``args, config = parse_arguments(argv)`` is the first statement, so
    ``--help`` prints a ``usage:`` line and exits without generating data,
    seeding RNGs, or constructing the aligner. This HARD invariant
    (``src/train/CLAUDE.md``) applies to every entry point regardless of the
    training loop's shape, algorithmic or ``fit()``-based alike.

THE FILE IS ``train_mini_vec2vec.py``, NEVER ``train.py``.
    A module named ``train.py`` inside a package on ``sys.path`` shadows the
    ``train`` package itself and breaks ``from train.common import ...``.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import keras
import numpy as np

from dl_techniques.models.language.mini_vec2vec import (
    MiniVec2VecAligner,
    create_mini_vec2vec_aligner,
)
from dl_techniques.models.language.mini_vec2vec.example_alignment import (
    align_frame,
    compute_mean_cosine_similarity,
    compute_top1_accuracy,
    compute_transformation_error,
    generate_synthetic_data,
)
from dl_techniques.utils.logger import logger
from train.common import default_experiment_name, save_config_json, set_seeds

__all__ = [
    "MiniVec2VecTrainingConfig",
    "PROGRAM_NAME",
    "build_aligner",
    "build_parser",
    "build_synthetic_embedding_spaces",
    "config_from_args",
    "main",
    "parse_arguments",
    "run_alignment",
]

PROGRAM_NAME: str = "train_mini_vec2vec.py"
"""``prog`` for the parser, so ``--help`` names the script rather than
``__main__`` when the module is run with ``python -m``."""


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass
class MiniVec2VecTrainingConfig:
    """Config for a mini-vec2vec alignment run.

    Defaults mirror ``example_alignment.run_alignment_example``'s own
    defaults (the reference this trainer is based on), not a tiny smoke
    configuration -- pass smaller ``--embedding-dim``/``--n-samples``/
    iteration-count flags for a fast smoke run, the same convention the
    CLM trainers in this plan use for their ``--variant``/model-size flags.

    Args:
        embedding_dim: Dimensionality of both embedding spaces and of the
            square transform ``W``.
        n_samples: Number of synthetic samples used to FIT the alignment.
        n_eval: Number of held-out synthetic samples used to EVALUATE it.
        n_clusters: Number of gaussian modes in the synthetic base cloud.
            Kept equal to ``approx_clusters`` is recommended (see
            ``generate_synthetic_data``'s own docstring on why a mismatch
            defeats recovery).
        cluster_noise: Standard deviation around each synthetic gaussian
            mode.
        approx_clusters: Clusters used by the approximate-matching stage
            (``align``'s Algorithm 2).
        approx_runs: Ensemble runs for approximate matching.
        approx_neighbors: Neighbors averaged into each pseudo-pair target.
        refine1_iterations: Iterations of matching-based refinement
            (Algorithm 3).
        refine1_sample_size: Samples drawn per Refine-1 iteration.
        refine1_neighbors: Neighbors used for matching in Refine-1.
        refine2_clusters: Clusters used by clustering-based refinement
            (Algorithm 4); should exceed ``approx_clusters``.
        smoothing_alpha: Exponential smoothing factor blended into ``W`` at
            each refinement step, in ``(0, 1]``.
        seed: Seed for every RNG source (``set_seeds``) and for synthetic
            data generation.
        output_dir: Root directory under which a timestamped run directory
            is created and artifacts are written.
    """

    embedding_dim: int = 128
    n_samples: int = 25000
    n_eval: int = 5000
    n_clusters: int = 20
    cluster_noise: float = 0.3
    approx_clusters: int = 20
    approx_runs: int = 30
    approx_neighbors: int = 10
    refine1_iterations: int = 50
    refine1_sample_size: int = 5000
    refine1_neighbors: int = 10
    refine2_clusters: int = 200
    smoothing_alpha: float = 0.5
    seed: int = 42
    output_dir: str = "results"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the parser.

    Interface contract: pure. Constructs and returns a parser; parses
    nothing, reads no environment, touches no filesystem, allocates no
    device, and generates no data.

    Returns:
        The parser.
    """
    defaults = MiniVec2VecTrainingConfig()
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description=(
            "Run the mini-vec2vec unsupervised embedding-space alignment "
            "algorithm (clustering + quadratic assignment + orthogonal "
            "Procrustes, no gradient descent) on a synthetic pair of "
            "embedding spaces, based on the model package's own "
            "example_alignment.py reference."
        ),
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=defaults.embedding_dim,
        help="Dimensionality of both embedding spaces / size of W.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=defaults.n_samples,
        help="Number of synthetic samples used to fit the alignment.",
    )
    parser.add_argument(
        "--n-eval", type=int, default=defaults.n_eval,
        help="Number of held-out synthetic samples used to evaluate it.",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=defaults.n_clusters,
        help="Gaussian modes in the synthetic base cloud.",
    )
    parser.add_argument(
        "--cluster-noise", type=float, default=defaults.cluster_noise,
        help="Std dev around each synthetic gaussian mode.",
    )
    parser.add_argument(
        "--approx-clusters", type=int, default=defaults.approx_clusters,
        help="Clusters for the approximate-matching stage (Algorithm 2).",
    )
    parser.add_argument(
        "--approx-runs", type=int, default=defaults.approx_runs,
        help="Ensemble runs for approximate matching.",
    )
    parser.add_argument(
        "--approx-neighbors", type=int, default=defaults.approx_neighbors,
        help="Neighbors averaged into each pseudo-pair target.",
    )
    parser.add_argument(
        "--refine1-iterations", type=int, default=defaults.refine1_iterations,
        help="Iterations of matching-based refinement (Algorithm 3).",
    )
    parser.add_argument(
        "--refine1-sample-size", type=int, default=defaults.refine1_sample_size,
        help="Samples drawn per Refine-1 iteration.",
    )
    parser.add_argument(
        "--refine1-neighbors", type=int, default=defaults.refine1_neighbors,
        help="Neighbors used for matching in Refine-1.",
    )
    parser.add_argument(
        "--refine2-clusters", type=int, default=defaults.refine2_clusters,
        help="Clusters for clustering-based refinement (Algorithm 4).",
    )
    parser.add_argument(
        "--smoothing-alpha", type=float, default=defaults.smoothing_alpha,
        help="Exponential smoothing factor blended into W, in (0, 1].",
    )
    parser.add_argument(
        "--seed", type=int, default=defaults.seed,
        help="Seed for every RNG source and for synthetic data generation.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=defaults.output_dir,
        help="Root directory under which a timestamped run directory is created.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> MiniVec2VecTrainingConfig:
    """Build the config from parsed args.

    Args:
        args: The parsed namespace from :func:`build_parser`.

    Returns:
        The config.
    """
    return MiniVec2VecTrainingConfig(
        embedding_dim=args.embedding_dim,
        n_samples=args.n_samples,
        n_eval=args.n_eval,
        n_clusters=args.n_clusters,
        cluster_noise=args.cluster_noise,
        approx_clusters=args.approx_clusters,
        approx_runs=args.approx_runs,
        approx_neighbors=args.approx_neighbors,
        refine1_iterations=args.refine1_iterations,
        refine1_sample_size=args.refine1_sample_size,
        refine1_neighbors=args.refine1_neighbors,
        refine2_clusters=args.refine2_clusters,
        smoothing_alpha=args.smoothing_alpha,
        seed=args.seed,
        output_dir=args.output_dir,
    )


def parse_arguments(
    argv: Optional[Sequence[str]] = None,
) -> Tuple[argparse.Namespace, MiniVec2VecTrainingConfig]:
    """Run the full ``argv -> parse -> config`` path.

    Args:
        argv: Tokens without the program name. ``None`` reads
            ``sys.argv[1:]``.

    Returns:
        ``(namespace, config)``.

    Raises:
        SystemExit: As argparse does, on ``--help`` or a bad flag.
    """
    args = build_parser().parse_args(argv)
    return args, config_from_args(args)


# ---------------------------------------------------------------------
# Data / model construction
# ---------------------------------------------------------------------


def build_synthetic_embedding_spaces(
    config: MiniVec2VecTrainingConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate the synthetic aligned embedding spaces via the model
    package's own reference generator.

    Reuses ``example_alignment.generate_synthetic_data`` rather than
    re-implementing gaussian-mixture + random-orthogonal-transform
    generation (DRY).

    Args:
        config: The run config.

    Returns:
        ``(XA_align, XB_align, XA_eval, XB_eval, ground_truth_W)``, exactly
        ``generate_synthetic_data``'s own return shape.
    """
    return generate_synthetic_data(
        n_samples=config.n_samples,
        n_eval=config.n_eval,
        embed_dim=config.embedding_dim,
        n_clusters=config.n_clusters,
        cluster_noise=config.cluster_noise,
        seed=config.seed,
    )


def build_aligner(config: MiniVec2VecTrainingConfig) -> MiniVec2VecAligner:
    """Build an (unbuilt) ``MiniVec2VecAligner`` via the model's own factory.

    Args:
        config: The run config.

    Returns:
        A fresh ``MiniVec2VecAligner``. ``align()`` builds it if needed.
    """
    return create_mini_vec2vec_aligner(embedding_dim=config.embedding_dim)


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------


def run_alignment(
    config: MiniVec2VecTrainingConfig,
) -> Tuple[MiniVec2VecAligner, Dict[str, float], str]:
    """Run the full mini-vec2vec alignment pipeline and persist artifacts.

    Mirrors ``example_alignment.run_alignment_example``'s five steps:
    generate data, build the aligner, run ``align()``, evaluate in the
    fitted frame (``align_frame``, matching the model's own frame
    contract -- see ``example_alignment.py``'s module docstring on why the
    means must be reproduced rather than reapplied on raw embeddings), then
    persist ``config.json``, ``alignment_metrics.json`` and the saved model.

    Args:
        config: The run config.

    Returns:
        ``(aligner, metrics, results_dir)``.
    """
    set_seeds(config.seed)

    XA_align, XB_align, XA_eval, XB_eval, ground_truth_W = (
        build_synthetic_embedding_spaces(config)
    )
    mean_A = XA_align.mean(axis=0, keepdims=True)
    mean_B = XB_align.mean(axis=0, keepdims=True)

    aligner = build_aligner(config)

    logger.info(
        "mini_vec2vec: aligning embedding_dim=%d over n_samples=%d "
        "(approx_clusters=%d, refine1_iterations=%d, refine2_clusters=%d)",
        config.embedding_dim, config.n_samples,
        config.approx_clusters, config.refine1_iterations,
        config.refine2_clusters,
    )
    aligner.align(
        XA=XA_align,
        XB=XB_align,
        approx_clusters=config.approx_clusters,
        approx_runs=config.approx_runs,
        approx_neighbors=config.approx_neighbors,
        refine1_iterations=config.refine1_iterations,
        refine1_sample_size=config.refine1_sample_size,
        refine1_neighbors=config.refine1_neighbors,
        refine2_clusters=config.refine2_clusters,
        smoothing_alpha=config.smoothing_alpha,
    )

    # Evaluate IN THE FITTED FRAME -- both sides go through `align_frame`,
    # same reasoning as `example_alignment.evaluate_alignment`.
    XA_proc = align_frame(XA_eval, mean_A)
    XB_proc = align_frame(XB_eval, mean_B)
    aligned_A = keras.ops.convert_to_numpy(aligner(XA_proc))

    metrics: Dict[str, float] = {
        "top1_accuracy": float(compute_top1_accuracy(aligned_A, XB_proc)),
        "mean_cosine_sim": float(
            compute_mean_cosine_similarity(aligned_A, XB_proc)
        ),
        "transformation_error": float(
            compute_transformation_error(
                keras.ops.convert_to_numpy(aligner.W), ground_truth_W
            )
        ),
    }
    logger.info("mini_vec2vec alignment metrics: %s", metrics)

    results_dir = os.path.join(
        config.output_dir, default_experiment_name("mini_vec2vec")
    )
    os.makedirs(results_dir, exist_ok=True)
    save_config_json(config, results_dir, "config.json")
    with open(os.path.join(results_dir, "alignment_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    aligner.save(os.path.join(results_dir, "aligner.keras"))

    logger.info("mini_vec2vec: artifacts written to %s", results_dir)
    return aligner, metrics, results_dir


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse the CLI and run the alignment.

    Statement order is the contract: parse first, so ``--help`` costs
    nothing; then :func:`run_alignment`, which seeds, generates data,
    builds the aligner, runs ``align()`` and persists artifacts.

    Args:
        argv: Tokens without the program name. ``None`` reads
            ``sys.argv[1:]``.
    """
    args, config = parse_arguments(argv)

    logger.info(
        "mini_vec2vec: embedding_dim=%d, n_samples=%d, n_eval=%d",
        config.embedding_dim, config.n_samples, config.n_eval,
    )
    _, metrics, results_dir = run_alignment(config)
    logger.info(
        "mini_vec2vec alignment complete; top1_accuracy=%.4f, artifacts in %s",
        metrics["top1_accuracy"], results_dir,
    )


if __name__ == "__main__":
    main()
