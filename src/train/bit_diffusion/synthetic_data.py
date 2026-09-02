"""The pre-encoded input contract, a synthetic generator for it, and the pipeline.

THE INPUT CONTRACT
==================
``DiTXA`` never sees pixels or token ids. It is trained on a bridge tensor built
from **pre-encoded** modalities, exactly as upstream is: an offline job runs the
image encoder and the text encoder once over the corpus and writes their outputs
to disk, and training reads only those outputs. This module is where that
contract is written down, because under D-001 this port ships neither encoder
(they are ``diffusers`` / ``transformers`` dependencies) and a contract nobody
states is a contract nobody can satisfy.

One RECORD is one (image, prompt) pair. A batch of ``N`` records is a dict of
three numpy arrays, keyed by :data:`CONTRACT_KEYS`:

======================= ================== ========= ===================================
key                     shape              dtype     meaning
======================= ================== ========= ===================================
``latent``              ``(N, H, W, C)``   float32   VAE latent, ALREADY scaled/shifted
``text_token_emb``      ``(N, D_flat)``    float32   flat token embeddings, row-major
``prompt_kind_label``   ``(N,)``           int32     ``PROMPT_KIND_TO_LABEL`` value
======================= ================== ========= ===================================

``(H, W, C)`` is ``BridgeConfig.bridge_shape`` and ``D_flat`` is
``BridgeConfig.token_flat_dim``; :func:`validate_records` enforces both against
a concrete preset rather than trusting the caller. Three further properties are
part of the contract and are NOT checkable from shape alone:

* ``latent`` carries the encoder's scaling ALREADY APPLIED
  (``x * latent_scale + latent_shift``). The bridge math does not rescale, so a
  raw un-scaled latent trains a numerically different model with no symptom.
* ``text_token_emb`` reshapes row-major to ``(N, token_seq_len, token_emb_dim)``.
  Each REAL token row has L2 norm exactly ``token_scale = sqrt(token_emb_dim)``
  (i.e. it is unit-norm once divided by ``token_scale``), and each PADDING row is
  exactly zero. That is the property
  :func:`~dl_techniques.models.vision_language.bit_diffusion.token_bridge.norm_based_token_stops`
  reads to recover the sequence length, so it is load-bearing, not cosmetic.
* ``prompt_kind_label`` indexes the prompt-length class the caption was drawn
  from (``original`` / ``short`` / ``medium``), not the image class.

ON-DISK FORMAT
--------------
One ``.npz`` per shard, holding the three arrays under the contract key names
(:func:`save_records_npz` / :func:`load_records_npz`). ``.npz`` because it is
plain ``numpy`` -- no new dependency (D-001), no schema server, no codec -- it
keeps dtypes and shapes without a sidecar, and an encoder job in ANY framework
can emit it. It is deliberately NOT a ``TFRecord``: the payload is a handful of
fixed-shape dense float arrays, so the schema/parse machinery in
``train.common.tfrecord`` would buy nothing, and ``.npz`` stays readable from a
notebook that has no TensorFlow. Shard so that one file fits in RAM; the reader
loads a whole shard.

THE SYNTHETIC GENERATOR
-----------------------
:func:`synthetic_records` produces records with the contract's SHAPE and, on the
text side, its STRUCTURE: unit-norm rows scaled by ``token_scale``, exactly-zero
padding rows, and a per-sample stop position drawn uniformly. Dense Gaussian
noise would have been cheaper and would have made every token row "real",
silently disabling the padding path the token decoder depends on.

Two honest limitations, stated rather than hidden:

* the image side IS dense Gaussian noise scaled to the latent's usual magnitude.
  A structurally honest image latent needs an image encoder, which D-001 excludes;
  there is no cheap stand-in the way there is for unit-norm token rows.
* the two modalities are INDEPENDENT. Nothing ties a sample's text to its image,
  so a run on synthetic data proves the loop wires up and the loss descends --
  never that the model learns the correspondence. Do not report a synthetic
  number as a result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.models.vision_language.bit_diffusion.bridge_process import (
    dsm_weight_forward,
    dsm_weight_reverse,
    flow_matching_interpolant,
    flow_matching_target,
    sample_bridge_x_t,
    sample_timesteps_logit_normal,
    sample_timesteps_uniform,
    score_target_forward,
    score_target_reverse,
)
from dl_techniques.models.vision_language.bit_diffusion.config import (
    PROMPT_NUM_CLASSES,
    BridgeConfig,
)
from dl_techniques.models.vision_language.bit_diffusion.sde import (
    BridgeSDE,
    FlowMatchingODE,
)
from dl_techniques.models.vision_language.bit_diffusion.token_bridge import (
    prepare_bridge_batch,
)
from dl_techniques.utils.logger import logger

__all__ = [
    "CONTRACT_KEYS",
    "DIRECTION_MODES",
    "TIME_SAMPLERS",
    "build_bridge_dataset",
    "load_records_npz",
    "prepare_training_batch",
    "save_records_npz",
    "synthetic_records",
    "validate_records",
]

#: The three arrays one record batch carries. Also the ``.npz`` member names.
CONTRACT_KEYS: Tuple[str, ...] = (
    "latent",
    "text_token_emb",
    "prompt_kind_label",
)

#: How ``direction`` is drawn per sample. ``both`` is the paper's bidirectional
#: training; the other two are D-002's forward-only / reverse-only ablations,
#: which are DATA settings rather than model variants (D-005).
DIRECTION_MODES: Tuple[str, ...] = ("both", "forward", "reverse")

#: Registered time samplers. Both clamp to ``[TIME_EPS, 1 - TIME_EPS]``.
TIME_SAMPLERS: Tuple[str, ...] = ("logit_normal", "uniform")

#: ``direction`` encoding, matching ``DiTXA._is_reverse`` (``> 0.5`` is reverse).
FORWARD: float = 0.0
REVERSE: float = 1.0


# ---------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------


def validate_records(records: Dict[str, np.ndarray], config: BridgeConfig) -> int:
    """Check a record batch against the input contract and return its size.

    Interface contract: pure. Reads the three :data:`CONTRACT_KEYS` off
    ``records`` and raises on the first violation; returns ``N`` on success.
    Shape and dtype only -- the three semantic properties in this module's
    docstring (pre-scaled latents, unit-norm token rows, prompt-kind labels)
    cannot be read off an array and are the producer's responsibility.

    :param records: Batch of records, keyed by :data:`CONTRACT_KEYS`.
    :type records: Dict[str, np.ndarray]
    :param config: Bridge geometry the records must match.
    :type config: BridgeConfig
    :return: ``N``, the number of records.
    :rtype: int
    :raises KeyError: If a contract key is missing.
    :raises ValueError: On a shape mismatch, a ragged batch, or a label out of
        ``[0, PROMPT_NUM_CLASSES)``.
    """
    config.validate()
    missing = [key for key in CONTRACT_KEYS if key not in records]
    if missing:
        raise KeyError(
            f"records is missing contract key(s) {missing}; got {sorted(records)}"
        )

    latent = np.asarray(records["latent"])
    token_emb = np.asarray(records["text_token_emb"])
    labels = np.asarray(records["prompt_kind_label"])

    expected_latent = (config.height, config.width, config.channels)
    if latent.ndim != 4 or tuple(latent.shape[1:]) != expected_latent:
        raise ValueError(
            f"latent must be (N, {expected_latent[0]}, {expected_latent[1]}, "
            f"{expected_latent[2]}), got {tuple(latent.shape)}"
        )
    if token_emb.ndim != 2 or token_emb.shape[1] != config.token_flat_dim:
        raise ValueError(
            f"text_token_emb must be (N, {config.token_flat_dim}), got "
            f"{tuple(token_emb.shape)}"
        )
    if labels.ndim != 1:
        raise ValueError(
            f"prompt_kind_label must be (N,), got {tuple(labels.shape)}"
        )

    sizes = {latent.shape[0], token_emb.shape[0], labels.shape[0]}
    if len(sizes) != 1:
        raise ValueError(
            "ragged record batch: latent/text_token_emb/prompt_kind_label have "
            f"lengths {latent.shape[0]}/{token_emb.shape[0]}/{labels.shape[0]}"
        )
    count = latent.shape[0]
    if count == 0:
        raise ValueError("record batch is empty")

    if labels.size and (labels.min() < 0 or labels.max() >= PROMPT_NUM_CLASSES):
        raise ValueError(
            f"prompt_kind_label must lie in [0, {PROMPT_NUM_CLASSES}), got "
            f"[{labels.min()}, {labels.max()}]"
        )
    return count


def synthetic_records(
    num_samples: int,
    config: BridgeConfig,
    seed: int = 0,
    latent_std: float = 1.0,
    min_tokens: int = 1,
) -> Dict[str, np.ndarray]:
    """Draw ``num_samples`` records satisfying the input contract.

    Interface contract: pure given ``seed``. Returns a fresh dict of the three
    :data:`CONTRACT_KEYS`, already contract-valid (it is passed through
    :func:`validate_records` before it is returned).

    The text side is structurally honest: each sample gets a stop position
    ``s ~ U{min_tokens, ..., token_seq_len}``, rows ``[0, s)`` are unit-norm
    Gaussian directions multiplied by ``config.token_scale`` and rows ``[s, T)``
    are exactly zero. The image side is Gaussian noise -- see this module's
    docstring for why there is no cheap honest stand-in for it.

    :param num_samples: Number of records to draw.
    :type num_samples: int
    :param config: Bridge geometry.
    :type config: BridgeConfig
    :param seed: Seed for the local ``numpy`` generator.
    :type seed: int
    :param latent_std: Standard deviation of the synthetic latent.
    :type latent_std: float
    :param min_tokens: Minimum number of non-padding token rows per sample.
        ``0`` allows an all-padding sample (the ``stops == 0`` branch).
    :type min_tokens: int
    :return: A contract-valid record batch.
    :rtype: Dict[str, np.ndarray]
    :raises ValueError: If ``num_samples`` is not positive or ``min_tokens`` is
        outside ``[0, token_seq_len]``.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if not 0 <= min_tokens <= config.token_seq_len:
        raise ValueError(
            f"min_tokens must lie in [0, {config.token_seq_len}], got {min_tokens}"
        )
    config.validate()

    rng = np.random.default_rng(seed)
    seq_len, emb_dim = config.token_seq_len, config.token_emb_dim

    directions = rng.standard_normal((num_samples, seq_len, emb_dim))
    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    # A zero draw is astronomically unlikely but would produce NaN, and a NaN in
    # the data is indistinguishable downstream from a NaN in the loss.
    norms = np.maximum(norms, 1e-12)
    tokens = (directions / norms) * config.token_scale

    stops = rng.integers(min_tokens, seq_len + 1, size=num_samples)
    keep = np.arange(seq_len)[None, :] < stops[:, None]
    tokens = tokens * keep[..., None]

    records = {
        "latent": (
            rng.standard_normal((num_samples, *config.bridge_shape)) * latent_std
        ).astype("float32"),
        "text_token_emb": tokens.reshape(num_samples, -1).astype("float32"),
        "prompt_kind_label": rng.integers(
            0, PROMPT_NUM_CLASSES, size=num_samples
        ).astype("int32"),
    }
    validate_records(records, config)
    logger.debug(
        "bit_diffusion: drew %d synthetic records (stops %d-%d of %d tokens)",
        num_samples,
        int(stops.min()),
        int(stops.max()),
        seq_len,
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
    logger.info("bit_diffusion: wrote %s", target)
    return target


def load_records_npz(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Read one record shard written by :func:`save_records_npz`.

    Not validated here: the caller owns the :class:`BridgeConfig` the records
    must match, so :func:`validate_records` is called at the pipeline boundary
    where that config is in scope.

    :param path: A ``.npz`` file carrying the three contract keys.
    :type path: Union[str, Path]
    :return: The record batch, dtypes coerced to the contract's.
    :rtype: Dict[str, np.ndarray]
    :raises KeyError: If the file lacks a contract key.
    """
    with np.load(str(path)) as handle:
        missing = [key for key in CONTRACT_KEYS if key not in handle.files]
        if missing:
            raise KeyError(
                f"{path} is missing contract key(s) {missing}; has {handle.files}"
            )
        return {
            "latent": handle["latent"].astype("float32"),
            "text_token_emb": handle["text_token_emb"].astype("float32"),
            "prompt_kind_label": handle["prompt_kind_label"].astype("int32"),
        }


# ---------------------------------------------------------------------
# Records -> one training element
# ---------------------------------------------------------------------


def _to_numpy(value: Any) -> np.ndarray:
    """``keras`` tensor -> ``float32`` numpy array."""
    return np.asarray(keras.ops.convert_to_numpy(value), dtype="float32")


def _draw_directions(
    count: int, mode: str, rng: np.random.Generator
) -> np.ndarray:
    """``(count,)`` float32 direction flags for ``mode``.

    :param count: Batch size.
    :type count: int
    :param mode: One of :data:`DIRECTION_MODES`.
    :type mode: str
    :param rng: Source of randomness for ``both``.
    :type rng: np.random.Generator
    :return: ``0.0`` forward / ``1.0`` reverse, per sample.
    :rtype: np.ndarray
    :raises ValueError: If ``mode`` is unknown.
    """
    if mode == "forward":
        return np.full((count,), FORWARD, dtype="float32")
    if mode == "reverse":
        return np.full((count,), REVERSE, dtype="float32")
    if mode == "both":
        return rng.integers(0, 2, size=count).astype("float32")
    raise ValueError(
        f"Unknown direction mode '{mode}'. Available: {list(DIRECTION_MODES)}"
    )


def _draw_times(
    count: int, sampler: str, seed: int, ) -> Any:
    """``(count,)`` times from the named sampler.

    :param count: Batch size.
    :type count: int
    :param sampler: One of :data:`TIME_SAMPLERS`.
    :type sampler: str
    :param seed: Integer seed; ``keras.random`` is stateless given one (D-019),
        so the caller MUST vary it per batch or every batch draws the same ``t``.
    :type seed: int
    :return: Times in ``[TIME_EPS, 1 - TIME_EPS]``.
    :rtype: Any
    :raises ValueError: If ``sampler`` is unknown.
    """
    if sampler == "uniform":
        return sample_timesteps_uniform(count, seed=seed)
    if sampler == "logit_normal":
        return sample_timesteps_logit_normal(count, seed=seed)
    raise ValueError(
        f"Unknown time sampler '{sampler}'. Available: {list(TIME_SAMPLERS)}"
    )


def prepare_training_batch(
    records: Dict[str, np.ndarray],
    config: BridgeConfig,
    sde: BridgeSDE,
    direction_mode: str = "both",
    time_sampler: str = "logit_normal",
    seed: int = 0,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Turn one record batch into the ``(inputs, target, sample_weight)`` triple.

    Interface contract, relied on by ``train_bit_diffusion.py`` AND by
    ``tests/test_train/test_bit_diffusion/``: this is THE function that decides
    the element shape, so there is no path a test can pass while the trainer
    fails. Pure given ``seed``; touches no global RNG state and no file.

    Everything direction-dependent is resolved HERE, host-side: the model sees
    ``direction`` only as an ordinary input tensor, and the loss sees ``w(t)``
    only as ``sample_weight``. Both targets and both weightings are computed for
    the whole batch and then selected per sample, which is the same one-code-path
    argument D-005 makes for the model.

    THE WEIGHT IS RETURNED AT RANK 3, ``(B, H, W)``, NOT ``(B,)``.
    ``FlowMatchingVelocityLoss`` reduces only the channel axis, so its value
    tensor is ``(B, H, W)``; a ``(B,)`` weight against it RAISES
    ``InvalidArgumentError`` (measured at step 1, and stock
    ``keras.losses.MeanSquaredError`` raises identically -- it is a general Keras
    property, not a quirk of one loss). Broadcasting here, where the reduction
    shape is known, is the whole remedy; ``sum_over_batch_size`` then reproduces
    upstream's ``mean((pred - target)**2 * w)`` exactly.

    :param records: A contract-valid record batch (see :data:`CONTRACT_KEYS`).
    :type records: Dict[str, np.ndarray]
    :param config: Bridge geometry, carrying the ``*_as_noise`` ablation flags.
    :type config: BridgeConfig
    :param sde: The base process. A :class:`FlowMatchingODE` switches the whole
        batch onto the rectified-flow baseline (no ``sigma``/``phi``/``C``, an
        unweighted loss), because that class RAISES on all three closed forms.
    :type sde: BridgeSDE
    :param direction_mode: One of :data:`DIRECTION_MODES`.
    :type direction_mode: str
    :param time_sampler: One of :data:`TIME_SAMPLERS`.
    :type time_sampler: str
    :param seed: Batch seed. Must differ per batch.
    :type seed: int
    :return: ``(inputs, target, sample_weight)`` with ``inputs`` carrying
        ``x_t``/``t``/``y``/``x_cond``/``direction``/``cond_mask``.
    :rtype: Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]
    """
    count = validate_records(records, config)
    rng = np.random.default_rng(seed)
    is_flow_matching = isinstance(sde, FlowMatchingODE)

    x_0_process, x_1_process, y, x_0_cond, x_1_cond = prepare_bridge_batch(
        records, config, seed=int(rng.integers(1, 2**31 - 1))
    )

    t = _draw_times(count, time_sampler, int(rng.integers(1, 2**31 - 1)))

    if is_flow_matching:
        x_t = flow_matching_interpolant(x_0_process, x_1_process, t)
        forward_target = reverse_target = flow_matching_target(
            x_0_process, x_1_process
        )
        forward_weight = reverse_weight = keras.ops.ones_like(t)
    else:
        x_t = sample_bridge_x_t(
            sde,
            x_0_process,
            x_1_process,
            t,
            seed=int(rng.integers(1, 2**31 - 1)),
        )
        forward_target = score_target_forward(sde, x_t, t, x_1_process)
        reverse_target = score_target_reverse(sde, x_t, t, x_0_process)
        forward_weight = dsm_weight_forward(sde, t)
        reverse_weight = dsm_weight_reverse(sde, t)

    direction = _draw_directions(count, direction_mode, rng)
    is_reverse = direction > 0.5
    pick_map = is_reverse[:, None, None, None]
    pick_scalar = is_reverse

    target = np.where(pick_map, _to_numpy(reverse_target), _to_numpy(forward_target))
    # Reverse (image -> text) conditions on the IMAGE end, forward on the TEXT
    # end. The `*_cond` pair is deliberately the REAL endpoints, so a
    # `text_as_noise` / `image_as_noise` run still conditions on genuine data
    # while the process itself runs on noise.
    x_cond = np.where(pick_map, _to_numpy(x_1_cond), _to_numpy(x_0_cond))
    per_sample_weight = np.where(
        pick_scalar, _to_numpy(reverse_weight), _to_numpy(forward_weight)
    )

    # DECISION plan-2026-09-02T094601-77d4a04e/D-021
    # Rank 3, NOT rank 1. Do NOT "simplify" this to the per-sample `(B,)` vector:
    # MEASURED, a `(B,)` weight against this loss's `(B,H,W)` value tensor RAISES
    # `InvalidArgumentError`, and stock `MeanSquaredError` raises identically.
    # See decisions.md D-021.
    weight = np.broadcast_to(
        per_sample_weight[:, None, None],
        (count, config.height, config.width),
    ).astype("float32")

    inputs = {
        "x_t": _to_numpy(x_t),
        "t": _to_numpy(t),
        "y": np.asarray(keras.ops.convert_to_numpy(y), dtype="int32"),
        "x_cond": x_cond.astype("float32"),
        "direction": direction.astype("float32"),
        # All-ones: upstream applies NO conditioning dropout during training
        # (its classifier-free branch comes from the class-label embedder's own
        # `class_dropout_rate`, and `forward_with_cfg` zeroes `cond_mask` at
        # INFERENCE time). The key is emitted anyway so the element shape is the
        # model's full input surface rather than a subset a later change could
        # silently drift from. Do NOT add a training-time cond-dropout knob here
        # without a reference for it.
        "cond_mask": np.ones((count,), dtype="float32"),
    }
    return inputs, target.astype("float32"), weight


# ---------------------------------------------------------------------
# The tf.data pipeline
# ---------------------------------------------------------------------


def build_bridge_dataset(
    records: Dict[str, np.ndarray],
    config: BridgeConfig,
    sde: BridgeSDE,
    batch_size: int,
    direction_mode: str = "both",
    time_sampler: str = "logit_normal",
    seed: int = 0,
    shuffle: bool = True,
    steps: Optional[int] = None,
) -> "tf.data.Dataset":
    """Build the ``tf.data`` pipeline ``fit()`` consumes.

    Interface contract: returns an INFINITE dataset of 3-tuples
    ``({"x_t","t","y","x_cond","direction","cond_mask"}, target, sample_weight)``
    -- infinite because every element is redrawn (fresh ``t``, fresh bridge
    noise, fresh directions), so an epoch is defined by ``steps_per_epoch`` at
    the ``fit()`` call rather than by exhausting the records. Pass ``steps`` to
    get a finite dataset instead (what the guards and ``evaluate()`` want).

    Built on ``from_generator`` over :func:`prepare_training_batch`, deliberately.
    The alternative -- a graph-mode ``Dataset.map`` -- cannot draw fresh noise:
    ``keras.random.*`` is STATELESS given an integer seed (D-019), so a traced
    map would emit byte-identical noise on every step under a suite that is
    finite, reproducible AND seed-sensitive, and a ``SeedGenerator`` variable
    inside a ``tf.data`` map has undefined update ordering. Here the batch seed
    is advanced by a plain ``numpy`` generator in Python, which is auditable.
    The cost is an eager Python step per batch, overlapped by ``prefetch``.

    :param records: A contract-valid record batch.
    :type records: Dict[str, np.ndarray]
    :param config: Bridge geometry.
    :type config: BridgeConfig
    :param sde: The base process.
    :type sde: BridgeSDE
    :param batch_size: Records per element.
    :type batch_size: int
    :param direction_mode: One of :data:`DIRECTION_MODES`.
    :type direction_mode: str
    :param time_sampler: One of :data:`TIME_SAMPLERS`.
    :type time_sampler: str
    :param seed: Seed for the batch-index shuffle and the per-batch seeds.
    :type seed: int
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
    # Fail fast, in the caller's stack frame, rather than inside the generator
    # where TensorFlow re-raises it as an opaque dataset error.
    _draw_directions(1, direction_mode, np.random.default_rng(0))
    _draw_times(1, time_sampler, 1)

    height, width, channels = config.bridge_shape
    signature = (
        {
            "x_t": tf.TensorSpec((None, height, width, channels), tf.float32),
            "t": tf.TensorSpec((None,), tf.float32),
            "y": tf.TensorSpec((None,), tf.int32),
            "x_cond": tf.TensorSpec((None, height, width, channels), tf.float32),
            "direction": tf.TensorSpec((None,), tf.float32),
            "cond_mask": tf.TensorSpec((None,), tf.float32),
        },
        tf.TensorSpec((None, height, width, channels), tf.float32),
        tf.TensorSpec((None, height, width), tf.float32),
    )

    def generator():
        rng = np.random.default_rng(seed)
        order = np.arange(count)
        cursor = count  # force a reshuffle on the first batch
        emitted = 0
        while steps is None or emitted < steps:
            if cursor + batch_size > count:
                if shuffle:
                    rng.shuffle(order)
                cursor = 0
            take = order[cursor : cursor + batch_size]
            cursor += batch_size
            slice_records = {key: records[key][take] for key in CONTRACT_KEYS}
            yield prepare_training_batch(
                slice_records,
                config,
                sde,
                direction_mode=direction_mode,
                time_sampler=time_sampler,
                seed=int(rng.integers(1, 2**31 - 1)),
            )
            emitted += 1

    dataset = tf.data.Dataset.from_generator(
        generator, output_signature=signature
    )
    if steps is not None:
        # DECISION plan-2026-09-02T094601-77d4a04e/D-022
        # `from_generator` reports UNKNOWN cardinality and Keras then OVER-RUNS a
        # finite dataset -- `evaluate()` raised `StopIteration` before this line.
        # Do NOT drop it and pass `steps=` at every call site instead: that puts
        # the same number in two places. See decisions.md D-022.
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(steps))
    return dataset.prefetch(tf.data.AUTOTUNE)
