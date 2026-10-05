"""Topographic VAE training: the paper's ablations, its two evaluation metrics, and
its capsule-traversal figures.

What this trainer reproduces
----------------------------
The paper's evaluation has two halves and this script runs both.

**Likelihood** — Tables 1 and 2 report ``log p(x)`` in nats, estimated by
importance sampling with 10 draws (:meth:`TopographicVAE.log_likelihood`). That is
the ``test.log_likelihood`` summary key here.

**Equivariance** — and this is where the paper's most careful result lives.
``equivariance_error`` alone is NOT the equivariance measurement: it is low for an
*invariant* representation too, which is exactly what the stationary-coherence
baseline learns. ``capcorr`` is the measurement. Both are computed, and
``results_summary.json`` records which is which so the two can never be read
interchangeably.

Baselines are flags, not separate scripts, because they differ from the headline
model in exactly one mechanism:

===============================  ===========================================
``--temporal-coherence``         what it is
===============================  ===========================================
``shifting`` (default)           the Topographic VAE (Eq. 9)
``stationary``                   BubbleVAE (Eq. 8) — the invariance baseline
``none``                         topographic, no temporal coherence
``--no-variance-variables``      the plain VAE: no ``u``, so no topography
===============================  ===========================================

Running the ablation is therefore four runs with different flags, and the summary
keys line up across them.

The figures are the paper's qualitative claims made falsifiable. A capsule
traversal decodes a whole sequence from ONE encoded activation by rolling it
within each capsule; if the model learned equivariance rather than invariance, the
traversed frames track the input sequence. ``visualizations/`` writes those as a
three-row grid (input / direct reconstruction / capsule traversal) — the paper's
Figure 1 and Figures 4, 7-21 in one panel. The topological-map figure is Figure 3,
drawn only for ``--topography torus_2d`` where it is defined.

Run it with::

    MPLBACKEND=Agg python -m train.topographic_vae.train_topographic_vae \\
        --dataset mnist --transform rotation --epochs 100

References:
    - Keller & Welling, 2022. Topographic VAEs learn Equivariant Capsules.
      NeurIPS 2021. (https://arxiv.org/abs/2109.01394)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.vision.topographic_vae import (
    TopographicVAE,
    create_topographic_vae,
)
from dl_techniques.losses.topographic_vae_loss import TopographicVAELoss
from dl_techniques.metrics.topographic import (
    capcorr_correlation,
    capcorr_per_capsule,
    equivariance_error,
)
from dl_techniques.datasets.vision.transform_sequences import (
    DEFAULT_DSPRITES_CACHE,
    DSPRITES_TRANSFORMS,
    MNIST_TRANSFORMS,
    create_transform_sequence_dataset,
)
from dl_techniques.layers.generative.topographic_product import (
    TEMPORAL_COHERENCE_TYPES,
    TOPOGRAPHY_TYPES,
)
from train.common import (
    create_callbacks,
    create_learning_rate_schedule,
    resolve_monitor_mode,
    save_training_history_json,
    setup_gpu,
)
from train.common.run_artifacts import attach_run_log
from train.common.run_io import default_experiment_name, prepare_run_dir
from train.common.run_summary import run_data_free_analysis
from train.common.seed import set_seeds

# ---------------------------------------------------------------------

#: Transformations each dataset offers. `--transform` is validated against the
#: union so `--help` lists them, and against the SELECTED dataset at run time so a
#: rotation sequence is not requested from dSprites.
ALL_TRANSFORMS = tuple(sorted(set(MNIST_TRANSFORMS) | set(DSPRITES_TRANSFORMS)))

#: The datasets this trainer can build sequences for. The frame size, the
#: sequence length and the available transforms all follow from this choice.
DATASETS = ("mnist", "dsprites")

#: Learning-rate schedules. ``"constant"`` is the paper's setting and is the
#: default; the other two route through ``train.common.create_learning_rate_schedule``
#: and switch ``ReduceLROnPlateau`` off, because an external schedule and a
#: plateau reducer both own the learning rate.
LR_SCHEDULE_TYPES = ("constant", "cosine", "exponential")

#: Global dtype policies this trainer will set. Listed here rather than read from
#: ``keras.mixed_precision`` so ``--help`` and the config validation agree.
MIXED_PRECISION_TYPES = ("float32", "mixed_float16")

#: Per-frame image shape of each dataset, fixed by its generator. This is the
#: DATALOGUE rather than the model: the decoder's output width must equal it or
#: the reconstruction is not the input's shape. Asserted against the variant table
#: in ``__post_init__`` so the two cannot drift apart.
DATASET_FRAME_SHAPES: Dict[str, Tuple[int, int, int]] = {
    "mnist": (28, 28, 3),
    "dsprites": (64, 64, 1),
}

#: Coherence-window presets as a fraction of the sequence length. Each row is an
#: ``L`` the paper actually reports (Section A.6), written as a fraction of ``S``
#: because the paper states it that way ("``L = S/3``"). On MNIST's ``S = 18``
#: these resolve to L = 0, 3 (``1/6``), 6 (``1/3``) and 9 (``1/2``); on dSprites'
#: ``S = 15`` to 0, 3 (``1/5``, the paper's ``L = 3/15``) and 5 (``1/3``). The
#: paper's lowest non-zero settings (``L = 2/36 S`` and ``L = 1/12 S``) are not
#: offered as presets because they round to 1 or 2 on both datasets and
#: ``--coherence-window`` covers them explicitly.
L_PRESETS: Dict[str, float] = {
    "none": 0.0,
    "sixth": 1.0 / 6.0,
    "third": 1.0 / 3.0,
    "half": 1.0 / 2.0,
}


# ---------------------------------------------------------------------
# config
# ---------------------------------------------------------------------


@dataclass
class TopographicVAEConfig:
    """Every knob of one Topographic VAE run.

    The defaults reproduce the paper's MNIST setting of Section 6.3 as closely as
    a run on one GPU allows: 18 capsules of 18 dimensions, ``L`` near ``S/3``,
    ``K = 3``, SGD at ``1e-4`` with momentum 0.9, batch size 8.

    Note the optimizer defaults to SGD + momentum, NOT Adam: that is what
    Section A.1 specifies, and it is also why ``--learning-rate`` defaults to
    ``1e-4`` rather than the ``1e-3`` that would suit Adam.
    """

    # --- data ---
    dataset: str = "mnist"
    transform: str = "rotation"
    sequence_length: Optional[int] = None
    num_train_sequences: int = 4096
    num_val_sequences: int = 512
    num_test_sequences: int = 512
    dsprites_cache: str = DEFAULT_DSPRITES_CACHE

    # --- model ---
    variant: str = "mnist"
    num_capsules: int = 18
    capsule_dim: int = 18
    l_preset: str = "third"
    coherence_window: Optional[int] = None
    neighborhood_size: int = 3
    temporal_coherence: str = "shifting"
    topography: str = "capsule_1d"
    grid_shape: Optional[Tuple[int, int]] = None
    use_variance_variables: bool = True
    degrees_of_freedom: float = 1.0
    prior_mean: float = 30.0
    encoder_hidden_dims: Optional[List[int]] = None
    decoder_hidden_dims: Optional[List[int]] = None

    # --- objective ---
    kl_loss_weight: float = 1.0
    reconstruction_sum_reduction: bool = False

    # --- optimisation ---
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    momentum: float = 0.9
    lr_schedule: str = "constant"
    patience: int = 20

    # --- evaluation ---
    likelihood_samples: int = 10
    likelihood_batch_size: int = 64
    num_traversals: int = 12

    # --- plumbing ---
    seed: int = 0
    gpu: Optional[int] = None
    output_dir: str = "results"
    experiment_name: Optional[str] = None
    mixed_precision: str = "float32"
    run_analysis: bool = True
    save_visualizations: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate every field the run would otherwise fail on later.

        Deliberately NOT resolving ``L`` or the experiment name here: both depend
        on the sequence length or the clock, and a config re-read from
        ``config.json`` must not silently produce a different run than the one
        that wrote it. Only IMPOSSIBLE values are rejected -- ones no default
        could have produced -- so that a config round trip stays exact.

        This exists because every one of these was previously discovered at
        ``epochs = 0`` or ``batch_size = 0`` INSIDE ``train()``, i.e. after the
        dataset had been generated and a run directory had been created.
        """
        if self.dataset not in DATASETS:
            raise ValueError(
                f"dataset must be one of {sorted(DATASETS)}, "
                f"got {self.dataset!r}"
            )
        allowed = MNIST_TRANSFORMS if self.dataset == "mnist" else DSPRITES_TRANSFORMS
        if self.transform not in allowed:
            raise ValueError(
                f"transform must be one of the {self.dataset!r} transforms "
                f"{list(allowed)}, got {self.transform!r}"
            )
        if self.variant not in TopographicVAE.MODEL_VARIANTS:
            raise ValueError(
                f"variant must be one of "
                f"{sorted(TopographicVAE.MODEL_VARIANTS)}, got {self.variant!r}"
            )
        # The dataset fixes the frame size and the variant fixes the model's, so
        # the two must agree. Without this, `--dataset dsprites` with the default
        # `--variant mnist` builds a 28x28 decoder over 64x64 frames.
        variant_shape = tuple(
            TopographicVAE.MODEL_VARIANTS[self.variant]["input_shape"]
        )
        dataset_shape = DATASET_FRAME_SHAPES[self.dataset]
        if variant_shape != dataset_shape:
            raise ValueError(
                f"variant {self.variant!r} expects frames {variant_shape} but "
                f"dataset {self.dataset!r} produces {dataset_shape}. Pass "
                f"--variant matching the dataset."
            )
        if self.l_preset not in L_PRESETS:
            raise ValueError(
                f"l_preset must be one of {sorted(L_PRESETS)}, "
                f"got {self.l_preset!r}"
            )
        if self.temporal_coherence not in TEMPORAL_COHERENCE_TYPES:
            raise ValueError(
                f"temporal_coherence must be one of "
                f"{sorted(TEMPORAL_COHERENCE_TYPES)}, "
                f"got {self.temporal_coherence!r}"
            )
        if self.topography not in TOPOGRAPHY_TYPES:
            raise ValueError(
                f"topography must be one of {sorted(TOPOGRAPHY_TYPES)}, "
                f"got {self.topography!r}"
            )
        if self.lr_schedule not in LR_SCHEDULE_TYPES:
            raise ValueError(
                f"lr_schedule must be one of {sorted(LR_SCHEDULE_TYPES)}, "
                f"got {self.lr_schedule!r}"
            )
        if self.mixed_precision not in MIXED_PRECISION_TYPES:
            raise ValueError(
                f"mixed_precision must be one of "
                f"{sorted(MIXED_PRECISION_TYPES)}, got {self.mixed_precision!r}"
            )

        for name in (
            "num_train_sequences",
            "num_val_sequences",
            "num_test_sequences",
            "sequence_length",
            "num_capsules",
            "capsule_dim",
            "neighborhood_size",
            "epochs",
            "batch_size",
            "likelihood_batch_size",
            "likelihood_samples",
            "num_traversals",
            "patience",
        ):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(
                    f"{name} must be positive, got {value!r}"
                )

        for name in ("learning_rate", "degrees_of_freedom"):
            value = float(getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value!r}")

        if float(self.momentum) < 0.0:
            raise ValueError(
                f"momentum must be non-negative, got {self.momentum!r}"
            )
        if float(self.prior_mean) < 0.0:
            raise ValueError(
                f"prior_mean must be non-negative, got {self.prior_mean!r}"
            )
        if float(self.kl_loss_weight) < 0.0:
            raise ValueError(
                f"kl_loss_weight must be non-negative, "
                f"got {self.kl_loss_weight!r}"
            )

        if self.coherence_window is not None and int(self.coherence_window) < 0:
            raise ValueError(
                f"coherence_window must be non-negative, "
                f"got {self.coherence_window!r}"
            )
        if self.grid_shape is not None and len(self.grid_shape) != 2:
            raise ValueError(
                f"grid_shape must be a (height, width) pair, "
                f"got {self.grid_shape!r}"
            )
        if self.topography == "torus_2d" and self.grid_shape is None:
            raise ValueError(
                "topography 'torus_2d' requires grid_shape=(height, width): the "
                "capsule_1d layout gets its neighbourhood from the capsule "
                "dimensions and needs no lattice"
            )
        # The model refuses this combination too, but only after the dataset has
        # been generated -- and the refusal is far from obvious, because a
        # dropped coherence window is a DIFFERENT MODEL, not a smaller one.
        if not self.use_variance_variables and self.coherence_window:
            raise ValueError(
                "a non-zero coherence_window correlates the energy built from u, "
                "which use_variance_variables=False does not have. Either set "
                "use_variance_variables=True or leave --coherence-window unset so "
                "--l-preset none resolves L=0."
            )
        # The mirror image: an explicit `L` of 0 with the u variables ON is
        # legal and meaningful (the model accepts it), so it is NOT rejected
        # here. Asserted below so the asymmetry is deliberate rather than an
        # oversight.

    def resolved_coherence_window(self, sequence_length: int) -> int:
        """Resolve ``--l-preset`` (or an explicit ``L``) to a concrete integer.

        The presets are fractions of the sequence length, so they must be resolved
        AFTER the sequence length is known: ``--l-preset third`` on an 18-frame
        sequence is ``L = 6``, which is the setting the paper reports its best
        equivariance at. Resolving them in ``__post_init__`` would freeze the value
        against a default sequence length that ``--sequence-length`` can then move.

        :param sequence_length: Frames per sequence ``S``.
        :type sequence_length: int
        :return: ``L``.
        :rtype: int
        :raises ValueError: On an unknown preset name.
        """
        if self.coherence_window is not None:
            return int(self.coherence_window)
        if self.l_preset not in L_PRESETS:
            raise ValueError(
                f"l_preset must be one of {sorted(L_PRESETS)}, got {self.l_preset!r}"
            )
        fraction = L_PRESETS[self.l_preset]
        if fraction == 0:
            return 0
        return max(1, int(round(sequence_length * float(fraction))))

    def resolved_experiment_name(self) -> str:
        """The run directory name, or an explicit override.

        :return: ``config.experiment_name`` when set, else a timestamped name.
        :rtype: str
        """
        if self.experiment_name:
            return self.experiment_name
        return default_experiment_name(
            "topographic_vae",
            self.dataset,
            self.transform,
            self.temporal_coherence,
        )


# ---------------------------------------------------------------------
# data
# ---------------------------------------------------------------------


def load_transform_sequences(
    config: TopographicVAEConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build train / validation / test sequence splits for one run.

    Each split is generated from a DIFFERENT seed, so the three sets draw
    different base images and different start poses. Reusing one generator across
    splits would put the same sequence in train and test and inflate every number
    in the summary.

    :param config: The run configuration.
    :type config: TopographicVAEConfig
    :return: ``(x_train, x_val, x_test, factors_test)``. The first three are
        ``(N, S, H, W, C)`` float32 on ``[0, 1]``; ``factors_test`` is
        ``(N, S)``, the ground-truth transformation parameter, which only the
        CapCorr metric consumes.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    :raises ValueError: On a transform the chosen dataset does not offer, or a
        non-positive split size.
    """
    allowed = (
        MNIST_TRANSFORMS if config.dataset == "mnist" else DSPRITES_TRANSFORMS
    )
    if config.transform not in allowed:
        raise ValueError(
            f"dataset '{config.dataset}' offers transforms {list(allowed)}, "
            f"got {config.transform!r}"
        )
    for name in (
        "num_train_sequences",
        "num_val_sequences",
        "num_test_sequences",
    ):
        if getattr(config, name) <= 0:
            raise ValueError(
                f"{name} must be positive, got {getattr(config, name)}"
            )

    def _build(count: int, seed: int) -> np.ndarray:
        sequences, _ = create_transform_sequence_dataset(
            config.dataset,
            transform=config.transform,
            sequence_length=config.sequence_length,
            num_sequences=count,
            seed=seed,
            cache_root=config.dsprites_cache,
        )
        return sequences

    x_train = _build(config.num_train_sequences, config.seed)
    x_val = _build(config.num_val_sequences, config.seed + 10_000)
    x_test, factors_test = create_transform_sequence_dataset(
        config.dataset,
        transform=config.transform,
        sequence_length=config.sequence_length,
        num_sequences=config.num_test_sequences,
        seed=config.seed + 20_000,
        cache_root=config.dsprites_cache,
    )

    logger.info(
        f"Sequences: train {x_train.shape}, val {x_val.shape}, "
        f"test {x_test.shape} (dataset={config.dataset}, "
        f"transform={config.transform})"
    )
    return x_train, x_val, x_test, factors_test


# ---------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------


def evaluate_model(
    model: TopographicVAE,
    x_test: np.ndarray,
    factors_test: np.ndarray,
    config: TopographicVAEConfig,
) -> Dict[str, Any]:
    """Compute the paper's Table 1 and Table 2 columns on the test split.

    Four numbers, and the distinction between them is the point of the whole
    evaluation:

    - ``log_likelihood`` — IWAE estimate in nats, summed over pixels and sequence.
    - ``equivariance_error`` — Eq. 13. A **smoothness** measure; low for an
      invariant representation too. Never read it as equivariance alone.
    - ``capcorr`` — Eq. 15/16. The equivariance measurement, ``1.0`` for a
      perfectly equivariant representation.
    - ``capcorr_per_capsule`` — the same metric per capsule, which makes the
      "all capsules roll together" assumption behind the pooled number observable
      rather than assumed.

    The latent ``t`` is encoded from the test sequences under a fixed seed, and
    so is the importance-weighted likelihood. The model's samplers are
    stochastic in BOTH places, so both draws are seeded: an unseeded ``t`` makes
    ``capcorr`` and ``equivariance_error`` functions of the run's RNG state rather
    than of the model, and two baseline rows would not be comparable.

    :param model: A trained model.
    :type model: TopographicVAE
    :param x_test: Test sequences.
    :type x_test: np.ndarray
    :param factors_test: Ground-truth factors, ``(N, S)``.
    :type factors_test: np.ndarray
    :param config: The run configuration, for ``likelihood_samples``.
    :type config: TopographicVAEConfig
    :return: The summary block.
    :rtype: Dict[str, Any]
    """
    # The forward that produces `t` MUST be seeded too, not just the likelihood.
    # The model's samplers are stochastic, so an unseeded `t` makes `capcorr` and
    # `E_eq` differ between two evaluations of the SAME weights -- which makes
    # every number in the summary a function of the run's RNG state rather than of
    # the model, and makes two baseline rows incomparable. MEASURED before the fix:
    # capcorr -0.5635 then -0.8935 for one model on one test split.
    #
    # `seed` also reaches the samplers, so setting the global seed immediately
    # before the forward is what makes the pair reproducible.
    set_seeds(int(config.seed))
    # The samplers are seeded EXPLICITLY, not via `set_seeds`: on this build
    # `keras.utils.set_random_seed` does not reseed `keras.random`, so three
    # `set_seeds(3)` calls still give three different draws (MEASURED), while an
    # explicit `seed=` gives the same draw every time. The model therefore takes a
    # `sampling_seed`, which this sets for the duration of the evaluation and
    # restores afterwards -- training must keep drawing freely, since the ELBO
    # wants an unbiased sample.
    previous_sampling_seed = model.sampling_seed
    model.sampling_seed = int(config.seed)
    try:
        outputs = model(x_test, training=False)
    finally:
        model.sampling_seed = previous_sampling_seed
    latents = np.asarray(
        outputs["t"] if not hasattr(outputs["t"], "numpy")
        else outputs["t"].numpy(),
        dtype=np.float64,
    )
    # (N, S, C*D) -> (N, S, C, D): the capsule split the metrics index.
    capsules = latents.reshape(
        latents.shape[0],
        latents.shape[1],
        model.num_capsules,
        model.capsule_dim,
    )

    likelihoods = []
    for start in range(0, x_test.shape[0], config.likelihood_batch_size):
        batch = x_test[start : start + config.likelihood_batch_size]
        values = model.log_likelihood(
            batch, num_samples=config.likelihood_samples, seed=config.seed
        )
        likelihoods.append(
            np.asarray(
                values.numpy() if hasattr(values, "numpy") else values,
                dtype=np.float64,
            )
        )
    log_likelihood = float(np.concatenate(likelihoods).mean())

    equivariance = float(equivariance_error(capsules))
    capcorr = float(capcorr_correlation(capsules, factors_test))
    per_capsule = capcorr_per_capsule(capsules, factors_test)

    block = {
        "log_likelihood": log_likelihood,
        "log_likelihood_samples": int(config.likelihood_samples),
        "equivariance_error": equivariance,
        "capcorr": capcorr,
        "capcorr_per_capsule": per_capsule.tolist(),
        "capcorr_per_capsule_min": float(np.nanmin(per_capsule))
        if per_capsule.size
        else float("nan"),
        "capcorr_per_capsule_max": float(np.nanmax(per_capsule))
        if per_capsule.size
        else float("nan"),
        "reconstruction_bce": float(
            np.mean(
                np.abs(
                    np.asarray(
                        outputs["reconstruction"].numpy()
                        if hasattr(outputs["reconstruction"], "numpy")
                        else outputs["reconstruction"],
                        dtype=np.float64,
                    )
                    - x_test
                )
            )
        ),
        "num_examples": int(x_test.shape[0]),
        "note": (
            "equivariance_error is a SMOOTHNESS measure and is low for an "
            "invariant representation too; capcorr is the equivariance "
            "measurement (1.0 = perfectly equivariant)"
        ),
    }
    logger.info(
        f"Evaluation: log p(x)={log_likelihood:.2f} nats, "
        f"E_eq={equivariance:.2f}, CapCorr={capcorr:.4f}"
    )
    return block


# ---------------------------------------------------------------------
# visualizations
# ---------------------------------------------------------------------


def plot_capsule_traversals(
    model: TopographicVAE,
    x_test: np.ndarray,
    run_dir: Path,
    num_traversals: int = 12,
) -> Optional[Path]:
    """Draw input / reconstruction / capsule-traversal grids (paper Figures 1, 4).

    Three rows per panel, and the middle row exists to disambiguate a failure: a
    poor traversal caused by a bad reconstruction looks identical to a poor
    traversal caused by a missing equivariance, so the paper plots the direct
    reconstruction between the input and the traversal for exactly this reason.

    The traversal row is generated from the FIRST timestep's ``t`` alone — the
    remaining frames are never encoded, which is the claim being illustrated.

    :param model: A trained model.
    :type model: TopographicVAE
    :param x_test: Test sequences, used as the source panels.
    :type x_test: np.ndarray
    :param run_dir: Directory to write into.
    :type run_dir: Path
    :param num_traversals: How many panels to draw. Defaults to 12.
    :type num_traversals: int
    :return: The written path, or ``None`` if the data cannot be used.
    :rtype: Optional[Path]
    """
    if num_traversals <= 0 or x_test.shape[0] == 0:
        return None

    count = min(num_traversals, x_test.shape[0])
    batch = x_test[:count]
    sequence_length = batch.shape[1]

    direct = np.asarray(
        model(batch, training=False)["reconstruction"].numpy(), dtype=np.float32
    )
    traversal = np.asarray(
        model.traverse_capsules(
            np.asarray(
                model(batch, training=False)["t"].numpy(), dtype=np.float32
            ),
            num_steps=sequence_length,
        ),
        dtype=np.float32,
    )

    rows = (
        ("input sequence", batch),
        ("direct reconstruction", direct),
        ("capsule traversal (from t_0)", traversal),
    )
    figure, axes = plt.subplots(
        len(rows) * count,
        sequence_length,
        figsize=(1.1 * sequence_length, 2.6 * len(rows) * count),
        squeeze=False,
    )
    for panel in range(count):
        for row_index, (title, frames) in enumerate(rows):
            row = row_index * count + panel
            for step in range(sequence_length):
                axis = axes[row][step]
                frame = np.asarray(frames[panel, step], dtype=np.float64)
                axis.imshow(_to_display(frame), cmap="gray", vmin=0.0, vmax=1.0)
                axis.set_xticks([])
                axis.set_yticks([])
                if step == 0:
                    axis.set_ylabel(
                        title if panel == 0 else "",
                        fontsize=5,
                        rotation=0,
                        ha="right",
                        va="center",
                    )
    figure.suptitle(
        "Capsule traversals: the whole sequence decoded from one encoded "
        "activation by rolling within each capsule",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.985))

    visualizations_dir = run_dir / "visualizations"
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    path = visualizations_dir / "capsule_traversals.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    logger.info(f"Wrote {path}")
    return path


def plot_topographic_map(
    model: TopographicVAE,
    x_test: np.ndarray,
    run_dir: Path,
    num_examples: int = 512,
) -> Optional[Path]:
    """Draw a maximum-activating-image map of the latent lattice (paper Figure 3).

    Only defined for a ``torus_2d`` topography — with circular capsules there is
    no 2-D lattice to lay out. Each lattice cell shows the test image that
    maximizes the L2 norm of that latent variable's activation, and a topographic
    map is one where neighbouring cells show similar images.

    :param model: A trained model.
    :type model: TopographicVAE
    :param x_test: Test sequences.
    :type x_test: np.ndarray
    :param run_dir: Directory to write into.
    :type run_dir: Path
    :param num_examples: How many sequences to search. Defaults to 512.
    :type num_examples: int
    :return: The written path, or ``None`` when the topography is not a 2-D grid.
    :rtype: Optional[Path]
    """
    if model.topography != "torus_2d" or model.grid_shape is None:
        return None

    grid_height, grid_width = model.grid_shape
    take = min(num_examples, x_test.shape[0])
    batch = x_test[:take]
    outputs = model(batch, training=False)
    latents = np.asarray(outputs["t"].numpy(), dtype=np.float32)
    energies = (latents**2).sum(axis=1)  # (N, S, C*D)

    cells = []
    for index in range(grid_height * grid_width):
        flat = energies[:, :, index]
        best = np.unravel_index(int(np.argmax(flat)), flat.shape)
        cells.append(batch[best[0], best[1]])

    figure, axes = plt.subplots(
        grid_height, grid_width, figsize=(0.7 * grid_width, 0.7 * grid_height),
        squeeze=False,
    )
    for row in range(grid_height):
        for column in range(grid_width):
            axis = axes[row][column]
            axis.imshow(
                _to_display(cells[row * grid_width + column]),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        "Maximum-activating image per latent on the 2-D torus: neighbouring "
        "cells sharing a feature IS the topographic organization",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))

    visualizations_dir = run_dir / "visualizations"
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    path = visualizations_dir / "topographic_map.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    logger.info(f"Wrote {path}")
    return path


def plot_training_curves(
    history: Dict[str, List[float]], run_dir: Path
) -> Optional[Path]:
    """Plot the training and validation ELBO curves.

    :param history: The ``History.history`` dict.
    :type history: Dict[str, List[float]]
    :param run_dir: Directory to write into.
    :type run_dir: Path
    :return: The written path, or ``None`` when there is nothing to plot.
    :rtype: Optional[Path]
    """
    series = {
        key: values
        for key, values in history.items()
        if key in ("loss", "val_loss") and values
    }
    if len(series) < 1:
        return None

    figure, axis = plt.subplots(figsize=(6, 3.5))
    for key, values in series.items():
        axis.plot(range(1, len(values) + 1), values, label=key)
    axis.set_xlabel("epoch")
    axis.set_ylabel("negative ELBO")
    axis.set_title("Topographic VAE: evidence lower bound")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()

    visualizations_dir = run_dir / "visualizations"
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    path = visualizations_dir / "training_curves.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    logger.info(f"Wrote {path}")
    return path


def _to_display(frame: np.ndarray) -> np.ndarray:
    """Collapse a frame to a 2-D array a colormap can show.

    :param frame: An ``(H, W)``, ``(H, W, 1)`` or ``(H, W, 3)`` frame.
    :type frame: np.ndarray
    :return: A 2-D float array on ``[0, 1]``.
    :rtype: np.ndarray
    """
    array = np.asarray(frame, dtype=np.float64)
    if array.ndim == 2:
        return array
    if array.shape[-1] == 1:
        return array[..., 0]
    # An RGB frame has no single grayscale rendering that preserves hue, and the
    # alternative (three separate panels per frame) triples the figure width.
    # Mean luminance is the honest scalar summary of a coloured MNIST digit.
    return array.mean(axis=-1)


# ---------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------


def build_model(config: TopographicVAEConfig, sequence_length: int) -> TopographicVAE:
    """Create and compile the model for one run.

    :param config: The run configuration.
    :type config: TopographicVAEConfig
    :param sequence_length: Frames per sequence, needed to resolve ``L``.
    :type sequence_length: int
    :return: A compiled model.
    :rtype: TopographicVAE
    :raises ValueError: If the configuration cannot build a model.
    """
    frame_shape = frame_shape_of(config)
    coherence_window = config.resolved_coherence_window(sequence_length)
    # Checked HERE rather than only in __post_init__, because the offending L can
    # come from the PRESET: `__post_init__` sees `coherence_window=None` and the
    # model then refuses the resolved value, i.e. after the dataset has been
    # generated. The message names the flag that fixes it.
    if not config.use_variance_variables and coherence_window > 0:
        raise ValueError(
            f"l_preset={config.l_preset!r} resolves to L={coherence_window} on "
            f"S={sequence_length}, and a non-zero L correlates the energy built "
            f"from u, which use_variance_variables=False does not have. Use "
            f"--l-preset none (or --coherence-window 0) for the plain-VAE "
            f"baseline."
        )
    model_kwargs: Dict[str, Any] = {
        "sequence_length": sequence_length,
        "num_capsules": config.num_capsules,
        "capsule_dim": config.capsule_dim,
        "coherence_window": coherence_window,
        "neighborhood_size": config.neighborhood_size,
        "temporal_coherence": config.temporal_coherence,
        "topography": config.topography,
        "use_variance_variables": config.use_variance_variables,
        "degrees_of_freedom": config.degrees_of_freedom,
        "prior_mean": config.prior_mean,
    }
    if config.grid_shape is not None:
        model_kwargs["grid_shape"] = config.grid_shape
    if config.encoder_hidden_dims is not None:
        model_kwargs["encoder_hidden_dims"] = config.encoder_hidden_dims
    if config.decoder_hidden_dims is not None:
        model_kwargs["decoder_hidden_dims"] = config.decoder_hidden_dims
    model_kwargs.update(config.extra)

    # The frame shape follows from the dataset, so it is read from the variant
    # table rather than from --image-size (which does not exist here). Every other
    # variant key is deliberately NOT splatted in: the CLI flags are the explicit
    # overrides, and letting the table's widths win would make --num-capsules and
    # --coherence-window silent no-ops for the overlapping keys.

    optimizer = keras.optimizers.SGD(
        learning_rate=config.learning_rate, momentum=config.momentum
    )
    loss = TopographicVAELoss(
        kl_loss_weight=config.kl_loss_weight,
        reconstruction_sum_reduction=config.reconstruction_sum_reduction,
        name="topographic_elbo",
    )
    model = TopographicVAE(input_shape=frame_shape, **model_kwargs)
    model.compile(optimizer=optimizer, loss=loss)

    logger.info(
        f"Model: variant={config.variant}, S={sequence_length}, "
        f"capsules={config.num_capsules}x{config.capsule_dim}, "
        f"L={model.coherence_window}, K={config.neighborhood_size}, "
        f"coherence={config.temporal_coherence}, topography={config.topography}, "
        f"u={'on' if config.use_variance_variables else 'off'}"
    )
    return model


def build_learning_rate_schedule(
    config: TopographicVAEConfig, steps_per_epoch: int
) -> Optional[Any]:
    """The LR schedule, or ``None`` for the paper's constant-rate setting.

    The paper trains at a constant ``1e-4`` (Section A.1), so ``"constant"`` is the
    default and this returns ``None`` — which is what ``create_callbacks`` needs to
    leave ``ReduceLROnPlateau`` off. The schedules are produced by the shared
    ``train.common.create_learning_rate_schedule`` rather than locally, so
    ``cosine`` means the same thing here as everywhere else in the tree.

    :param config: The run configuration.
    :type config: TopographicVAEConfig
    :param steps_per_epoch: Batches per epoch, for the step-based schedules.
    :type steps_per_epoch: int
    :return: A Keras schedule, or ``None``.
    :rtype: Optional[Any]
    """
    if config.lr_schedule == "constant":
        return None
    return create_learning_rate_schedule(
        config.learning_rate,
        config.lr_schedule,
        config.epochs,
        steps_per_epoch,
    )


def frame_shape_of(config: TopographicVAEConfig) -> Tuple[int, int, int]:
    """The per-frame image shape for this run, from the DATASET.

    Not from the variant table. The sequence generator produces frames of a size
    fixed by the dataset -- 28x28x3 for MNIST, 64x64x1 for dSprites -- and the
    model's decoder output width must match it or ``reconstruction`` is not the
    input's shape and the ELBO silently reshapes. The variant table happens to
    agree, but only because each dataset has one variant; reading it from the
    variant instead means ``--dataset dsprites`` with the default
    ``--variant mnist`` builds a 28x28 model over 64x64 data. MEASURED before
    this fix: ``frame_shape_of(dsprites) == (28, 28, 3)``.

    The agreement is then ASSERTED in ``__post_init__`` rather than assumed, so a
    third dataset or variant cannot reintroduce the mismatch quietly.

    :param config: The run configuration.
    :type config: TopographicVAEConfig
    :return: ``(height, width, channels)``.
    :rtype: Tuple[int, int, int]
    """
    return DATASET_FRAME_SHAPES[config.dataset]


def train(config: TopographicVAEConfig) -> Dict[str, Any]:
    """Run one full training job and write the run directory.

    :param config: The run configuration.
    :type config: TopographicVAEConfig
    :return: The summary written to ``results_summary.json``.
    :rtype: Dict[str, Any]
    """
    set_seeds(config.seed)
    # Resolved HERE, not in __post_init__: the timestamped default is not
    # deterministic, and a config re-read from `config.json` should not silently
    # produce a second, differently-named directory.
    config.experiment_name = config.resolved_experiment_name()
    run_dir = prepare_run_dir(config)

    # Everything the run narrates goes inside the block, so `run.log` holds the
    # WHOLE story rather than only the lines logged after this point. Placed
    # immediately after the directory exists and before the first log call, which
    # is the ordering every other trainer in the tree uses.
    with attach_run_log(run_dir):
        logger.info(f"Run directory: {run_dir}")
        return _train_in_run_dir(config, run_dir)


def _train_in_run_dir(
    config: TopographicVAEConfig, run_dir: Path
) -> Dict[str, Any]:
    """The body of :func:`train`, with the run directory already prepared.

    Split out so the ``with attach_run_log(run_dir)`` block can wrap the whole
    run: a function whose body is one ``with`` statement over a call cannot be
    read to see what it covers, and a log attached for only part of a run is a log
    that silently omits the part nobody looked at.

    :param config: The run configuration, with ``experiment_name`` resolved.
    :type config: TopographicVAEConfig
    :param run_dir: The prepared run directory.
    :type run_dir: Path
    :return: The summary written to ``results_summary.json``.
    :rtype: Dict[str, Any]
    """
    x_train, x_val, x_test, factors_test = load_transform_sequences(config)
    sequence_length = x_train.shape[1]

    model = build_model(config, sequence_length)
    # Build BEFORE summary. A subclassed model that has never been forwarded
    # reports `Total params: 0` from summary() and count_params() returns 0
    # without raising — the exact defect the research guide records under
    # "`Model.build(shape)` builds a subclassed model". One dummy forward here
    # costs nothing and makes the logged table and the summary's `parameters`
    # key agree with each other.
    model.build(
        (None, sequence_length) + tuple(frame_shape_of(config))
    )
    model.summary(print_fn=logger.info)

    steps_per_epoch = max(
        1, math.ceil(x_train.shape[0] / max(1, config.batch_size))
    )
    lr_schedule = build_learning_rate_schedule(config, steps_per_epoch)
    if lr_schedule is not None:
        logger.info(f"LR schedule: {config.lr_schedule} over {steps_per_epoch} steps/epoch")
        model.compile(
            optimizer=keras.optimizers.SGD(
                learning_rate=lr_schedule, momentum=config.momentum
            ),
            loss=TopographicVAELoss(
                kl_loss_weight=config.kl_loss_weight,
                reconstruction_sum_reduction=config.reconstruction_sum_reduction,
                name="topographic_elbo",
            ),
        )

    callbacks, results_dir = create_callbacks(
        model_name="topographic_vae",
        run_dir=str(run_dir),
        monitor="val_loss",
        monitor_mode=resolve_monitor_mode("val_loss"),
        patience=config.patience,
        # An external schedule owns the LR, so ReduceLROnPlateau must be off;
        # create_callbacks' own flag is exactly this inversion.
        use_lr_schedule=lr_schedule is None,
        include_analyzer=False,
    )
    logger.info(f"Artifacts: {results_dir}")

    history = model.fit(
        x_train,
        x_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_data=(x_val, x_val),
        callbacks=callbacks,
        verbose=2,
    )

    final_model_path = run_dir / "final_model.keras"
    model.save(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")

    evaluation = evaluate_model(model, x_test, factors_test, config)

    # Every figure is a KEY whether or not it was drawn, with ``None`` for the
    # ones that were not. A summary whose `visualizations` block is empty when
    # they were switched off cannot be told apart from one where the plotting
    # failed, and "the figure is absent" reads as a finding rather than as a
    # choice.
    figures: Dict[str, Optional[str]] = {
        "capsule_traversals": None,
        "topographic_map": None,
        "training_curves": None,
    }
    if config.save_visualizations:
        figures["capsule_traversals"] = (
            str(
                plot_capsule_traversals(
                    model, x_test, run_dir, config.num_traversals
                )
                or ""
            )
            or None
        )
        figures["topographic_map"] = (
            str(plot_topographic_map(model, x_test, run_dir) or "") or None
        )
        figures["training_curves"] = (
            str(plot_training_curves(history.history, run_dir) or "") or None
        )

    analysis = run_data_free_analysis(
        model,
        x_test,
        factors_test,
        history,
        "topographic_vae",
        run_dir,
        enabled=config.run_analysis,
    )

    save_training_history_json(history, str(run_dir))

    summary: Dict[str, Any] = {
        "model_name": "topographic_vae",
        "dataset": config.dataset,
        "transform": config.transform,
        "temporal_coherence": config.temporal_coherence,
        "topography": config.topography,
        "use_variance_variables": config.use_variance_variables,
        "baseline": _baseline_name(config),
        "num_capsules": config.num_capsules,
        "capsule_dim": config.capsule_dim,
        "coherence_window": model.coherence_window,
        "sequence_length": sequence_length,
        "neighborhood_size": config.neighborhood_size,
        "degrees_of_freedom": config.degrees_of_freedom,
        "prior_mean": config.prior_mean,
        "latent_dim": model.latent_dim,
        "parameters": int(model.count_params()),
        "epochs_requested": config.epochs,
        "epochs_run": len(history.history.get("loss", [])),
        "best_epoch": int(
            np.argmin(history.history["val_loss"]) + 1
        )
        if history.history.get("val_loss")
        else None,
        "best_val_loss": float(min(history.history["val_loss"]))
        if history.history.get("val_loss")
        else None,
        "final_train_loss": float(history.history["loss"][-1])
        if history.history.get("loss")
        else None,
        "optimizer": {
            "name": "sgd",
            "learning_rate": config.learning_rate,
            "momentum": config.momentum,
        },
        "kl_loss_weight": config.kl_loss_weight,
        "reconstruction_sum_reduction": config.reconstruction_sum_reduction,
        "effective_beta": float(
            TopographicVAELoss(
                kl_loss_weight=config.kl_loss_weight,
                reconstruction_sum_reduction=config.reconstruction_sum_reduction,
            ).effective_beta
        ),
        "train": {
            "sequences": int(x_train.shape[0]),
            "shape": list(x_train.shape[1:]),
        },
        "test_evaluation": evaluation,
        "analysis": analysis,
        "visualizations": figures,
        "data_shapes": {
            "train": list(x_train.shape),
            "validation": list(x_val.shape),
            "test": list(x_test.shape),
        },
    }

    summary_path = run_dir / "results_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, default=float)
    logger.info(f"Wrote {summary_path}")

    logger.info(
        f"Done. log p(x)={evaluation['log_likelihood']:.2f} nats | "
        f"E_eq={evaluation['equivariance_error']:.2f} | "
        f"CapCorr={evaluation['capcorr']:.4f}"
    )
    return summary


def _baseline_name(config: TopographicVAEConfig) -> str:
    """Name the row of the paper's ablation table this run is.

    :param config: The run configuration.
    :type config: TopographicVAEConfig
    :return: A short label.
    :rtype: str
    """
    if not config.use_variance_variables:
        return "plain_vae"
    if config.temporal_coherence == "stationary":
        return "bubble_vae"
    if config.l_preset == "none" or config.coherence_window == 0:
        return "topographic_vae_l0"
    if config.temporal_coherence == "none":
        return "topographic_vae_no_window"
    return "topographic_vae"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def create_argument_parser() -> argparse.ArgumentParser:
    """Build this trainer's argument parser.

    Deliberately NOT ``train.common.create_base_argument_parser``: that helper's
    ``--dataset`` choices are mnist/cifar10/cifar100/imagenet and its
    ``--image-size`` means nothing here, where the frame size follows from the
    dataset and the sequence is the axis that matters.

    :return: The parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Train a Topographic VAE (Keller & Welling, 2022)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--dataset", choices=("mnist", "dsprites"), default="mnist"
    )
    data.add_argument(
        "--transform", choices=ALL_TRANSFORMS, default="rotation",
        help="which single factor the sequences vary; validated per dataset",
    )
    data.add_argument(
        "--sequence-length", type=int, default=None,
        help="frames per sequence (default: 18 for MNIST, 15 for dSprites)",
    )
    data.add_argument("--num-train-sequences", type=int, default=4096)
    data.add_argument("--num-val-sequences", type=int, default=512)
    data.add_argument("--num-test-sequences", type=int, default=512)
    data.add_argument(
        "--dsprites-cache", default=DEFAULT_DSPRITES_CACHE,
        help="directory holding the dSprites .npz (never inside the repository)",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--variant", choices=sorted(TopographicVAE.MODEL_VARIANTS),
                       default="mnist")
    model.add_argument("--num-capsules", type=int, default=18)
    model.add_argument("--capsule-dim", type=int, default=18)
    model.add_argument(
        "--l-preset", choices=sorted(L_PRESETS), default="third",
        help="coherence window as a fraction of the sequence length; the paper "
             "reports its best equivariance near L = S/3",
    )
    model.add_argument(
        "--coherence-window", type=int, default=None,
        help="explicit L; overrides --l-preset",
    )
    model.add_argument("--neighborhood-size", type=int, default=3,
                       help="within-capsule window width K")
    model.add_argument(
        "--temporal-coherence",
        choices=("none", "stationary", "shifting"),
        default="shifting",
        help="shifting is the Topographic VAE, stationary is the BubbleVAE baseline",
    )
    model.add_argument("--topography", choices=("capsule_1d", "torus_2d"),
                       default="capsule_1d")
    model.add_argument(
        "--grid-shape", type=int, nargs=2, default=None,
        metavar=("H", "W"), help="torus_2d lattice shape",
    )
    model.add_argument(
        "--no-variance-variables", action="store_true",
        help="drop u entirely: the plain-VAE baseline, with no topography",
    )
    model.add_argument("--degrees-of-freedom", type=float, default=1.0,
                       help="Student's-t degrees of freedom nu")
    model.add_argument("--prior-mean", type=float, default=30.0)
    model.add_argument("--encoder-hidden-dims", type=int, nargs="+", default=None)
    model.add_argument("--decoder-hidden-dims", type=int, nargs="+", default=None)

    objective = parser.add_argument_group("objective")
    objective.add_argument("--kl-loss-weight", type=float, default=1.0)
    objective.add_argument(
        "--reconstruction-sum-reduction", action="store_true",
        help="sum the reconstruction over pixels, making --kl-loss-weight the "
             "paper's beta; the default mean-over-pixels is a per-pixel average",
    )

    optimisation = parser.add_argument_group("optimisation")
    optimisation.add_argument("--epochs", type=int, default=100)
    optimisation.add_argument("--batch-size", type=int, default=8)
    optimisation.add_argument(
        "--learning-rate", type=float, default=1e-4,
        help="the paper trains with SGD at 1e-4 and momentum 0.9",
    )
    optimisation.add_argument("--momentum", type=float, default=0.9)
    optimisation.add_argument(
        "--lr-schedule", choices=("constant", "cosine", "exponential"),
        default="constant",
        help="the paper trains at a constant rate; anything else swaps in the "
             "shared train.common schedule and turns ReduceLROnPlateau off",
    )
    optimisation.add_argument("--patience", type=int, default=20)

    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument(
        "--likelihood-samples", type=int, default=10,
        help="importance samples for log p(x); the paper uses 10",
    )
    evaluation.add_argument("--likelihood-batch-size", type=int, default=64)
    evaluation.add_argument("--num-traversals", type=int, default=12)
    evaluation.add_argument("--no-model-analysis", action="store_true")

    plumbing = parser.add_argument_group("plumbing")
    plumbing.add_argument("--seed", type=int, default=0)
    plumbing.add_argument("--gpu", type=int, default=None)
    plumbing.add_argument("--output-dir", default="results")
    plumbing.add_argument("--experiment-name", default=None)
    plumbing.add_argument(
        "--mixed-precision", choices=("float32", "mixed_float16"),
        default="float32",
    )
    plumbing.add_argument("--no-visualizations", action="store_true")

    return parser


def config_from_args(args: argparse.Namespace) -> TopographicVAEConfig:
    """Build a config from parsed arguments, resolving every None sentinel.

    :param args: Parsed arguments.
    :type args: argparse.Namespace
    :return: The configuration.
    :rtype: TopographicVAEConfig
    """
    return TopographicVAEConfig(
        dataset=args.dataset,
        transform=args.transform,
        sequence_length=args.sequence_length,
        num_train_sequences=args.num_train_sequences,
        num_val_sequences=args.num_val_sequences,
        num_test_sequences=args.num_test_sequences,
        dsprites_cache=args.dsprites_cache,
        variant=args.variant,
        num_capsules=args.num_capsules,
        capsule_dim=args.capsule_dim,
        l_preset=args.l_preset,
        coherence_window=args.coherence_window,
        neighborhood_size=args.neighborhood_size,
        temporal_coherence=args.temporal_coherence,
        topography=args.topography,
        grid_shape=tuple(args.grid_shape) if args.grid_shape else None,
        use_variance_variables=not args.no_variance_variables,
        degrees_of_freedom=args.degrees_of_freedom,
        prior_mean=args.prior_mean,
        encoder_hidden_dims=args.encoder_hidden_dims,
        decoder_hidden_dims=args.decoder_hidden_dims,
        kl_loss_weight=args.kl_loss_weight,
        reconstruction_sum_reduction=args.reconstruction_sum_reduction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        lr_schedule=args.lr_schedule,
        patience=args.patience,
        likelihood_samples=args.likelihood_samples,
        likelihood_batch_size=args.likelihood_batch_size,
        num_traversals=args.num_traversals,
        run_analysis=not args.no_model_analysis,
        save_visualizations=not args.no_visualizations,
        seed=args.seed,
        gpu=args.gpu,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        mixed_precision=args.mixed_precision,
    )


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """CLI entry point.

    :param argv: Argument list; ``None`` reads ``sys.argv``.
    :type argv: Optional[List[str]]
    :return: The run summary.
    :rtype: Dict[str, Any]
    """
    args = create_argument_parser().parse_args(argv)
    config = config_from_args(args)

    setup_gpu(config.gpu)
    set_seeds(config.seed)

    if config.mixed_precision == "mixed_float16":
        keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Global policy: mixed_float16")
    else:
        keras.mixed_precision.set_global_policy("float32")

    return train(config)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------