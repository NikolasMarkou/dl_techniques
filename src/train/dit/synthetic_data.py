"""The pre-encoded latent input contract, a synthetic generator, and the pipeline.

THE INPUT CONTRACT
==================
:class:`~dl_techniques.models.vision_language.dit.DiT` never sees pixels. It is a
**latent** diffusion transformer: an offline job runs a VAE encoder once over the
corpus and writes its output to disk, and training reads only that output. This
module is where that contract is written down, because **no VAE ships with this
repository** -- there is no trained autoencoder anywhere under
``src/dl_techniques/``, and every ``pretrained=True`` path in this tree raises
:class:`NotImplementedError`. Real latents must therefore be produced externally,
and a contract nobody states is a contract nobody can satisfy.

One RECORD is one (image, class) pair. A batch of ``N`` records is a dict of two
numpy arrays, keyed by :data:`CONTRACT_KEYS`:

=============== ================== ========= =====================================
key             shape              dtype     meaning
=============== ================== ========= =====================================
``latent``      ``(N, H, W, C)``   float32   VAE latent, channels-LAST, ALREADY
                                             multiplied by
                                             :data:`LATENT_SCALE_FACTOR`
``label``       ``(N,)``           int32     class id in ``[0, num_classes)``
=============== ================== ========= =====================================

``(H, W)`` is ``DiffusionConfig.input_size`` squared and ``C`` is
``DiffusionConfig.in_channels``; :func:`validate_records` enforces both against a
concrete config rather than trusting the caller.

Two properties of ``latent`` are part of the contract and are NOT checkable from
shape alone. Both are silent when violated -- the model trains, the loss falls,
and the samples are wrong:

* **Channels-LAST here, channels-FIRST upstream.** Upstream's pre-encoding job
  saves one ``.npy`` per sample of shape ``(1, 4, 32, 32)`` in NCHW
  (``reference/train_and_sample_excerpts.py:5-11``). This port is channels-last
  throughout, so those arrays must be transposed. :func:`latents_nchw_to_nhwc` is
  that transpose and is the only spelling of it in this package::

      latents_nhwc = latents_nchw_to_nhwc(np.load(path))   # (1,4,32,32) -> (1,32,32,4)

  A ``(N, C, H, W)`` array with ``C == H == W`` would pass every shape check
  while being silently transposed, which is why :func:`validate_records` names
  NCHW explicitly in its error when it sees a plausible one.
* **The ``0.18215`` scale is ALREADY APPLIED.** Upstream writes
  ``vae.encode(x).latent_dist.sample() * 0.18215``
  (``reference/train_and_sample_excerpts.py:9``); the constant is Stable
  Diffusion's ``scale_factor``, chosen so the encoder's output has roughly unit
  variance, which is what the DDPM beta schedule assumes. Nothing downstream
  rescales, so an unscaled latent trains a numerically different model with no
  symptom. See :data:`LATENT_SCALE_FACTOR`.

Upstream's image preprocessing, for a producer that wants to reproduce it: center
crop to 256, random horizontal flip, then map to ``[-1, 1]``, then encode.

ON-DISK FORMAT
--------------
One ``.npz`` per shard, holding the two arrays under the contract key names
(:func:`save_records_npz` / :func:`load_records_npz`, reached from the CLI as
``--train-npz`` / ``--val-npz``). ``.npz`` because it is plain ``numpy`` -- no new
dependency, no schema server, no codec -- it keeps dtypes and shapes without a
sidecar, and an encoder job in ANY framework can emit it. Shard so that one file
fits in RAM; the reader loads a whole shard.

THE SYNTHETIC GENERATOR
-----------------------
:func:`synthetic_records` draws **class-correlated** latents: a per-class mean
field plus per-sample Gaussian noise,
``latent_i = class_signal * mu[y_i] + noise_std * eps_i``. The correlation is the
whole point and is not decoration. The smoke run's acceptance criterion is that
``val_loss`` FALLS; against pure noise there is nothing for a class-conditional
model to learn beyond the marginal, so that criterion would be unfalsifiable --
it would pass or fail on optimizer noise. With a per-class mean field the label
carries real information and a falling ``val_loss`` means something. What it does
NOT mean: these are not image latents, and a synthetic number is never a result.

THE PIPELINE
------------
:func:`build_dit_dataset` emits exactly what stock ``fit()`` needs -- 2-tuples
``((x_t, t, y), y_true)`` -- with no custom ``train_step`` anywhere. ``t`` and
``noise`` are drawn here, ``x_t`` is produced by
:meth:`~dl_techniques.models.vision_language.dit.GaussianDiffusion.q_sample`, and
``y_true`` is the packed target :class:`~dl_techniques.losses.DDPMHybridLoss`
reads (D-002):

.. code-block:: text

        record                        draw                    model inputs
    ┌──────────────────┐     ┌────────────────────┐     ┌────────────────────┐
    │ latent [B,H,W,C] │     │ t ~ U{0 .. T-1}    │     │ x_t  [B, H, W, C]  │
    │ label  [B]       │──┬─▶│ noise ~ N(0, I)    │──┬─▶│ t    [B]           │
    └────────┬─────────┘  │  └────────────────────┘  │  │ y    [B]           │
             │            └───────── q_sample ───────┘  └────────────────────┘
             │        x_t = sqrt(a_bar_t) * x_0  ⊕  sqrt(1 - a_bar_t) * noise
             ▼
    ┌──────────────────── y_true [B, H, W, 2C+1] ─────────────────────────┐
    │  [0:C] noise      │  [C:2C] x_start        │  [2C:2C+1] t as plane  │
    └───────────────────┴────────────────────────┴────────────────────────┘

The loss re-derives ``x_t`` from ``(x_start, noise, t)`` instead of receiving it,
so the two sides must agree numerically. That agreement is made STRUCTURAL rather
than coincidental: both construct their tables from the same
``(schedule_name, num_timesteps)`` pair through
:meth:`~dl_techniques.models.vision_language.dit.config.DiffusionConfig.build_schedule`,
and this module calls ``GaussianDiffusion.q_sample`` rather than retyping the
two-term formula. ``tests/test_train/test_dit/test_the_input_contract.py`` pins
the equality at ``atol=1e-6, rtol=0``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

import keras

from dl_techniques.models.vision_language.dit import GaussianDiffusion
from dl_techniques.models.vision_language.dit.config import DiffusionConfig
from dl_techniques.utils.logger import logger

__all__ = [
    "CONTRACT_KEYS",
    "LATENT_SCALE_FACTOR",
    "build_dit_dataset",
    "build_training_diffusion",
    "latents_nchw_to_nhwc",
    "load_records_npz",
    "pack_target",
    "prepare_training_batch",
    "save_records_npz",
    "synthetic_records",
    "validate_records",
]

#: The two arrays one record batch carries. Also the ``.npz`` member names.
CONTRACT_KEYS: Tuple[str, ...] = ("latent", "label")

#: Stable Diffusion's VAE ``scale_factor``, applied by the PRODUCER of the
#: latents and never by this package
#: (``reference/train_and_sample_excerpts.py:9``). Exported so a producer script
#: can import the number instead of typing it a second time.
LATENT_SCALE_FACTOR: float = 0.18215

#: Axis permutation taking upstream's NCHW latents to this port's NHWC.
NCHW_TO_NHWC: Tuple[int, int, int, int] = (0, 2, 3, 1)


# ---------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------


def _rng(seed: Optional[int]) -> np.random.Generator:
    """Build the local NumPy generator for an explicit seed.

    Interface contract: pure. ``None`` means "draw fresh, unreproducibly";
    any integer -- INCLUDING ``0`` -- means "reproduce".

    :param seed: Explicit seed, or ``None`` for a non-deterministic generator.
    :type seed: Optional[int]
    :return: The generator every draw in this module goes through.
    :rtype: np.random.Generator
    """
    # DECISION plan-2026-09-02T170923-1285ed83/D-022
    # `is None`, NOT `if not seed` / `if seed`. A truthiness test makes `seed=0`
    # behave as UNSEEDED, which is a measured defect class in this repo: the
    # caller asks for the most obvious reproducible seed there is and silently
    # gets a fresh entropy draw, with no exception and no shape symptom. Also do
    # NOT reach for `keras.utils.set_random_seed` as a substitute: step 7 of this
    # plan MEASURED that it does not re-seed an already-created global
    # `SeedGenerator` here, so two identically-seeded runs disagree. Every draw
    # in this module goes through an explicitly constructed local generator.
    # Pinned by `test_the_input_contract.py::TestSeedZeroIsHonoured`.
    # See decisions.md D-022.
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


# ---------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------


def latents_nchw_to_nhwc(latents: np.ndarray) -> np.ndarray:
    """Transpose upstream's NCHW latents into this port's NHWC layout.

    Interface contract: pure, shape-only. Takes ``(N, C, H, W)`` and returns
    ``(N, H, W, C)`` as a contiguous ``float32`` array. This is the ONLY
    spelling of the transpose in this package; a second one is how two
    conventions end up in one tree.

    :param latents: Upstream latents, ``(N, C, H, W)`` -- e.g. the
        ``(1, 4, 32, 32)`` arrays ``reference/train_and_sample_excerpts.py:5-11``
        writes, stacked.
    :type latents: np.ndarray
    :return: ``(N, H, W, C)`` float32.
    :rtype: np.ndarray
    :raises ValueError: If ``latents`` is not rank 4.
    """
    array = np.asarray(latents)
    if array.ndim != 4:
        raise ValueError(
            f"latents_nchw_to_nhwc expects a rank-4 (N, C, H, W) array, got "
            f"shape {tuple(array.shape)}"
        )
    return np.ascontiguousarray(
        np.transpose(array, NCHW_TO_NHWC), dtype="float32"
    )


def validate_records(records: Dict[str, np.ndarray], config: DiffusionConfig) -> int:
    """Check a record batch against the input contract and return its size.

    Interface contract: pure. Reads the two :data:`CONTRACT_KEYS` off
    ``records``, raises :class:`ValueError` NAMING the offending key on the first
    violation, and returns ``N`` on success. Shape, dtype and label range only --
    the two semantic properties in this module's docstring (channels-last, scale
    already applied) cannot be read off an array and are the producer's
    responsibility.

    :param records: Batch of records, keyed by :data:`CONTRACT_KEYS`.
    :type records: Dict[str, np.ndarray]
    :param config: The latent geometry the records must match.
    :type config: DiffusionConfig
    :return: ``N``, the number of records.
    :rtype: int
    :raises KeyError: If a contract key is missing.
    :raises ValueError: On a rank/shape mismatch, a non-float ``latent``, a
        non-integer ``label``, a ragged batch, an empty batch, or a label outside
        ``[0, config.num_classes)``.
    """
    missing = [key for key in CONTRACT_KEYS if key not in records]
    if missing:
        raise KeyError(
            f"records is missing contract key(s) {missing}; got {sorted(records)}"
        )

    latent = np.asarray(records["latent"])
    label = np.asarray(records["label"])

    size, channels = config.input_size, config.in_channels
    expected = (size, size, channels)

    if latent.ndim != 4:
        raise ValueError(
            f"'latent' must be rank 4 (N, {size}, {size}, {channels}), got "
            f"shape {tuple(latent.shape)}"
        )
    if tuple(latent.shape[1:]) != expected:
        hint = ""
        if tuple(latent.shape[1:]) == (channels, size, size):
            hint = (
                " -- this looks like upstream's NCHW layout; run it through "
                "latents_nchw_to_nhwc() first"
            )
        raise ValueError(
            f"'latent' must be (N, {size}, {size}, {channels}) channels-LAST, "
            f"got {tuple(latent.shape)}{hint}"
        )
    if not np.issubdtype(latent.dtype, np.floating):
        raise ValueError(
            f"'latent' must be a floating dtype (float32), got {latent.dtype}"
        )

    if label.ndim != 1:
        raise ValueError(f"'label' must be (N,), got shape {tuple(label.shape)}")
    if not np.issubdtype(label.dtype, np.integer):
        raise ValueError(
            f"'label' must be an integer dtype (int32), got {label.dtype}"
        )

    if latent.shape[0] != label.shape[0]:
        raise ValueError(
            "ragged record batch: 'latent' has "
            f"{latent.shape[0]} rows but 'label' has {label.shape[0]}"
        )
    count = int(latent.shape[0])
    if count == 0:
        raise ValueError("record batch is empty ('latent' has 0 rows)")

    if int(label.min()) < 0 or int(label.max()) >= config.num_classes:
        raise ValueError(
            f"'label' must lie in [0, {config.num_classes}), got "
            f"[{int(label.min())}, {int(label.max())}]"
        )
    return count


def synthetic_records(
    num_samples: int,
    config: DiffusionConfig,
    seed: Optional[int] = 0,
    class_signal: float = 1.0,
    noise_std: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Draw ``num_samples`` CLASS-CORRELATED records satisfying the contract.

    Interface contract: pure given ``seed``. Returns a fresh dict of the two
    :data:`CONTRACT_KEYS`, already contract-valid (it is passed through
    :func:`validate_records` before it is returned).

    Each class ``c`` gets one fixed mean field ``mu[c]`` of shape ``(H, W, C)``
    drawn from ``N(0, I)``, and a sample of class ``c`` is
    ``class_signal * mu[c] + noise_std * eps``. Setting ``class_signal = 0.0``
    reduces this to pure noise, which is what a guard uses as its anti-vacuity
    control -- see this module's docstring for why the correlation exists at all
    (without it "the loss decreases" is not a falsifiable criterion).

    Labels are drawn uniformly over ``[0, config.num_classes)``, so a large
    ``num_classes`` with a small ``num_samples`` gives most classes zero
    examples; that is legal and deliberate, not a bug.

    :param num_samples: Number of records to draw.
    :type num_samples: int
    :param config: The latent geometry and class count.
    :type config: DiffusionConfig
    :param seed: Explicit seed. ``0`` IS a seed (see :func:`_rng`); ``None``
        draws fresh.
    :type seed: Optional[int]
    :param class_signal: Multiplier on the per-class mean field. ``0.0`` removes
        the class correlation entirely.
    :type class_signal: float
    :param noise_std: Standard deviation of the per-sample noise.
    :type noise_std: float
    :return: A contract-valid record batch.
    :rtype: Dict[str, np.ndarray]
    :raises ValueError: If ``num_samples`` is not positive, or if
        ``class_signal`` / ``noise_std`` is negative or not finite.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    for name, value in (("class_signal", class_signal), ("noise_std", noise_std)):
        if not np.isfinite(value) or float(value) < 0.0:
            raise ValueError(
                f"{name} must be a finite non-negative float, got {value!r}"
            )

    rng = _rng(seed)
    size, channels = config.input_size, config.in_channels
    field_shape = (size, size, channels)

    labels = rng.integers(0, config.num_classes, size=num_samples).astype("int32")
    class_means = rng.standard_normal((config.num_classes, *field_shape))
    noise = rng.standard_normal((num_samples, *field_shape))

    latent = (
        float(class_signal) * class_means[labels] + float(noise_std) * noise
    ).astype("float32")

    records = {"latent": latent, "label": labels}
    validate_records(records, config)
    logger.debug(
        "dit: drew %d synthetic records (%d distinct classes present, "
        "class_signal=%.3g, noise_std=%.3g)",
        num_samples,
        int(np.unique(labels).size),
        class_signal,
        noise_std,
    )
    return records


def save_records_npz(
    records: Dict[str, np.ndarray], path: Union[str, Path]
) -> Path:
    """Write one record shard to ``path`` as an uncompressed ``.npz``.

    :param records: Batch of records, keyed by :data:`CONTRACT_KEYS`.
    :type records: Dict[str, np.ndarray]
    :param path: Destination file. Parent directories are created.
    :type path: Union[str, Path]
    :return: The written path.
    :rtype: Path
    :raises KeyError: If a contract key is missing.
    """
    missing = [key for key in CONTRACT_KEYS if key not in records]
    if missing:
        raise KeyError(f"records is missing contract key(s) {missing}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **{key: np.asarray(records[key]) for key in CONTRACT_KEYS})
    logger.info("dit: wrote %s", target)
    return target


def load_records_npz(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Read one record shard written by :func:`save_records_npz`.

    Deliberately NOT validated here: the caller owns the
    :class:`~dl_techniques.models.vision_language.dit.config.DiffusionConfig` the
    records must match, so :func:`validate_records` runs at the pipeline
    boundary where that config is in scope. Dtypes are NOT coerced either -- a
    ``float64`` or NCHW shard must be REJECTED by the validator rather than
    silently repaired, because silent repair is how a wrong-layout corpus trains
    a plausible wrong model.

    :param path: A ``.npz`` file carrying the two contract keys.
    :type path: Union[str, Path]
    :return: The record batch, as stored.
    :rtype: Dict[str, np.ndarray]
    :raises KeyError: If the file lacks a contract key.
    """
    with np.load(str(path)) as handle:
        missing = [key for key in CONTRACT_KEYS if key not in handle.files]
        if missing:
            raise KeyError(
                f"{path} is missing contract key(s) {missing}; has {handle.files}"
            )
        return {key: np.asarray(handle[key]) for key in CONTRACT_KEYS}


# ---------------------------------------------------------------------
# Records -> one training element
# ---------------------------------------------------------------------


def build_training_diffusion(config: DiffusionConfig) -> GaussianDiffusion:
    """Build the forward process this pipeline noises with.

    Interface contract: pure. Returns an UNRESPACED
    :class:`~dl_techniques.models.vision_language.dit.GaussianDiffusion` over
    ``config``'s schedule.

    :param config: The diffusion configuration; the loss is constructed from the
        same two fields (``schedule_name``, ``num_timesteps``).
    :type config: DiffusionConfig
    :return: The forward process.
    :rtype: GaussianDiffusion
    """
    # DECISION plan-2026-09-02T170923-1285ed83/D-021
    # Unrespaced, and reached through `GaussianDiffusion.q_sample` rather than by
    # retyping `sqrt(a_bar)*x0 + sqrt(1-a_bar)*eps` here. Both halves matter and
    # neither has a shape symptom. (1) `timestep_respacing` shortens the tables
    # and remaps `t`; the loss owns an UNRESPACED `DDPMSchedule` and would then
    # gather different constants for the same `t`, so the pipeline's `x_t` and
    # the loss's re-derived `x_t` would silently disagree. Respacing is a
    # SAMPLING-time knob only. (2) A local copy of the two-term formula would
    # make the agreement coincidental instead of structural -- one edit to
    # either copy and the objective is trained against a state the model never
    # saw. Pinned at atol=1e-6, rtol=0 by
    # `test_the_input_contract.py::TestTheLossRederivesTheSameXT`.
    # See decisions.md D-021.
    return GaussianDiffusion.from_name(
        schedule_name=config.schedule_name,
        num_timesteps=config.num_timesteps,
        timestep_respacing=None,
    )


def pack_target(
    noise: np.ndarray, x_start: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Pack ``(noise, x_start, t)`` into the target ``DDPMHybridLoss`` reads.

    Interface contract: pure. Returns ``(N, H, W, 2C+1)`` float32 with layout
    ``[0:C] = noise``, ``[C:2C] = x_start``, ``[2C:2C+1] = t`` broadcast over
    ``(H, W)``. THE single producer of that layout in this package.

    :param noise: The epsilon that was added, ``(N, H, W, C)``.
    :type noise: np.ndarray
    :param x_start: The clean latent, ``(N, H, W, C)``.
    :type x_start: np.ndarray
    :param t: Per-sample timestep, ``(N,)``.
    :type t: np.ndarray
    :return: The packed target, ``(N, H, W, 2C+1)`` float32.
    :rtype: np.ndarray
    :raises ValueError: If the three arguments do not agree in shape.
    """
    noise = np.asarray(noise, dtype="float32")
    x_start = np.asarray(x_start, dtype="float32")
    t = np.asarray(t)

    if noise.shape != x_start.shape:
        raise ValueError(
            f"noise {noise.shape} and x_start {x_start.shape} must have the "
            "same shape"
        )
    if noise.ndim != 4:
        raise ValueError(
            f"noise/x_start must be rank 4 (N, H, W, C), got {noise.shape}"
        )
    if t.shape != (noise.shape[0],):
        raise ValueError(
            f"t must be ({noise.shape[0]},), got {tuple(t.shape)}"
        )

    # DECISION plan-2026-09-02T170923-1285ed83/D-002
    # noise FIRST, x_start SECOND, t LAST. This order is the hand-maintained
    # contract with `DDPMHybridLoss._unpack`, and swapping the first two halves
    # produces a target of the IDENTICAL shape and dtype that trains a
    # plausible, wrong model -- there is no shape, dtype or finiteness symptom.
    # Do NOT move `t` onto `sample_weight` instead: Keras MULTIPLIES the
    # per-sample loss by `sample_weight`, so `t` there corrupts the objective.
    # Pinned by `test_the_input_contract.py::TestThePackedLayout`.
    # See decisions.md D-002.
    t_plane = np.broadcast_to(
        t.astype("float32")[:, None, None, None],
        (*noise.shape[:3], 1),
    )
    return np.concatenate([noise, x_start, t_plane], axis=-1).astype("float32")


def prepare_training_batch(
    records: Dict[str, np.ndarray],
    config: DiffusionConfig,
    diffusion: Optional[GaussianDiffusion] = None,
    seed: Optional[int] = 0,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Turn one record batch into the ``((x_t, t, y), y_true)`` training element.

    Interface contract, relied on by ``train_dit.py`` AND by
    ``tests/test_train/test_dit/``: this is THE function that decides the element
    shape, so there is no path a test can pass while the trainer fails. Pure
    given ``seed``; touches no global RNG state and no file.

    ``t`` is drawn uniformly over ``[0, T)`` and ``noise`` from ``N(0, I)``;
    ``x_t`` comes from :meth:`GaussianDiffusion.q_sample` and ``y_true`` from
    :func:`pack_target`. The SAME ``t`` that produced ``x_t`` is the one packed
    into ``y_true`` -- using two different draws leaves every shape intact while
    training the model on a state that does not correspond to its target.

    :param records: A contract-valid record batch (see :data:`CONTRACT_KEYS`).
    :type records: Dict[str, np.ndarray]
    :param config: The latent geometry and chain length.
    :type config: DiffusionConfig
    :param diffusion: The forward process. ``None`` builds one via
        :func:`build_training_diffusion`; pass one in to avoid rebuilding the
        tables per batch.
    :type diffusion: Optional[GaussianDiffusion]
    :param seed: Explicit seed for this batch's ``t``/``noise`` draw. Must vary
        per batch, or every batch trains on identical noise.
    :type seed: Optional[int]
    :return: ``((x_t, t, y), y_true)`` -- ``x_t`` ``(N, H, W, C)`` float32,
        ``t`` ``(N,)`` int32, ``y`` ``(N,)`` int32, ``y_true``
        ``(N, H, W, 2C+1)`` float32.
    :rtype: Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    """
    count = validate_records(records, config)
    process = build_training_diffusion(config) if diffusion is None else diffusion
    rng = _rng(seed)

    x_start = np.asarray(records["latent"], dtype="float32")
    labels = np.asarray(records["label"], dtype="int32")

    t = rng.integers(0, config.num_timesteps, size=count).astype("int32")
    noise = rng.standard_normal(x_start.shape).astype("float32")

    x_t = np.asarray(
        keras.ops.convert_to_numpy(process.q_sample(x_start, t, noise=noise)),
        dtype="float32",
    )
    return (x_t, t, labels), pack_target(noise, x_start, t)


# ---------------------------------------------------------------------
# The tf.data pipeline
# ---------------------------------------------------------------------


def build_dit_dataset(
    records: Dict[str, np.ndarray],
    config: DiffusionConfig,
    batch_size: int,
    seed: Optional[int] = 0,
    shuffle: bool = True,
    steps: Optional[int] = None,
) -> "tf.data.Dataset":
    """Build the ``tf.data`` pipeline stock ``fit()`` consumes.

    Interface contract: returns a dataset of 2-tuples
    ``((x_t, t, y), y_true)`` -- exactly the ``(inputs, target)`` pair
    ``keras.Model.fit`` expects, with ``inputs`` in
    :data:`~dl_techniques.models.vision_language.dit.MODEL_INPUT_NAMES` order and
    no ``sample_weight`` third element (``DDPMHybridLoss`` needs none; ``t``
    rides inside ``y_true``, see D-002). INFINITE unless ``steps`` is given,
    because every element is redrawn with a fresh ``t`` and fresh noise, so an
    epoch is defined by ``steps_per_epoch`` at the ``fit()`` call rather than by
    exhausting the records.

    Built on ``from_generator`` over :func:`prepare_training_batch`,
    deliberately. A graph-mode ``Dataset.map`` cannot draw fresh noise here:
    ``keras.random.*`` is stateless given an integer seed and
    ``keras.utils.set_random_seed`` does not re-seed an already-created global
    ``SeedGenerator`` on this Keras (measured), so a traced map would emit
    byte-identical noise on every step. Here the per-batch seed is advanced by a
    plain NumPy generator in Python, which is auditable. The cost is an eager
    Python step per batch, overlapped by ``prefetch``.

    :param records: A contract-valid record batch.
    :type records: Dict[str, np.ndarray]
    :param config: The latent geometry and chain length.
    :type config: DiffusionConfig
    :param batch_size: Records per element.
    :type batch_size: int
    :param seed: Explicit seed for the record shuffle and the per-batch seeds.
        ``0`` IS a seed; ``None`` draws fresh.
    :type seed: Optional[int]
    :param shuffle: Re-shuffle the record order each pass.
    :type shuffle: bool
    :param steps: Emit exactly this many elements, then stop. ``None`` is
        infinite.
    :type steps: Optional[int]
    :return: The dataset.
    :rtype: tf.data.Dataset
    :raises ValueError: If ``batch_size`` is not positive or exceeds the record
        count, or if ``steps`` is not positive.
    """
    count = validate_records(records, config)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if batch_size > count:
        raise ValueError(
            f"batch_size ({batch_size}) exceeds the record count ({count})"
        )
    if steps is not None and steps <= 0:
        raise ValueError(f"steps must be positive when given, got {steps}")

    size, channels = config.input_size, config.in_channels
    # Built ONCE, outside the generator: the tables are a pure function of the
    # config, and rebuilding them per batch is pure cost.
    process = build_training_diffusion(config)

    signature = (
        (
            tf.TensorSpec((None, size, size, channels), tf.float32),
            tf.TensorSpec((None,), tf.int32),
            tf.TensorSpec((None,), tf.int32),
        ),
        tf.TensorSpec((None, size, size, 2 * channels + 1), tf.float32),
    )

    def generator():
        rng = _rng(seed)
        order = np.arange(count)
        cursor = count  # force a (re)shuffle on the first batch
        emitted = 0
        while steps is None or emitted < steps:
            if cursor + batch_size > count:
                if shuffle:
                    rng.shuffle(order)
                cursor = 0
            take = order[cursor: cursor + batch_size]
            cursor += batch_size
            yield prepare_training_batch(
                {key: np.asarray(records[key])[take] for key in CONTRACT_KEYS},
                config,
                diffusion=process,
                seed=int(rng.integers(1, 2**31 - 1)),
            )
            emitted += 1

    dataset = tf.data.Dataset.from_generator(
        generator, output_signature=signature
    )
    if steps is not None:
        # `from_generator` reports UNKNOWN cardinality and Keras then OVER-RUNS
        # a finite dataset -- `evaluate()` raises StopIteration without this.
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(steps))
    return dataset.prefetch(tf.data.AUTOTUNE)
