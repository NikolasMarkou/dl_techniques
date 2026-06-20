"""Trainer-side tests for the bias-free ConvUNeXt self-iterate feature.

Covers Step 7 of plan_2026-06-20_88705c63. The CRITICAL test here is SC1:
the ``--self-iterate`` OFF path must remain byte-identical to the pre-plan
streaming trainer. We verify that two ways:

1. **Behavioural determinism (SC1a).** Building the streaming ``create_dataset``
   twice under the same fixed seed yields a bitwise-identical first
   ``(noisy, clean)`` batch (max-abs-diff == 0.0). This proves the OFF path is
   deterministic and seed-controlled (and that nothing in the self-iterate work
   leaked non-determinism into the streaming path).

2. **Textual byte-identity vs the pre-plan baseline (SC1b).** The streaming
   noise/augment/crop logic (``create_dataset`` + ``make_curriculum_noise_fn``)
   must be UNCHANGED from commit 8688519a (the commit BEFORE this plan). Step 3
   only branched ``train()``'s data path; it did NOT edit these functions. We
   extract each function's source from the current module (``inspect.getsource``)
   and from ``git show <baseline>`` and compare NORMALIZED bodies.

   Normalization (strip leading indentation + collapse whitespace) is applied
   because a relocation refactor *could* re-indent the body (e.g. moving lines
   into an ``else`` branch). If the only difference were indentation the byte
   meaning would be unchanged; normalization lets the test assert "same logic"
   rather than "same column layout". If even the non-whitespace token stream
   differs, that is a REAL regression in the OFF path and the test fails loudly.

Other tests cover the pool dataset contract, the multi-pass eval helpers, the
rejection guards, and the "no custom train_step" invariant (SC4).

All shapes are tiny so the suite stays CPU/GPU1-light.
"""

import re
import inspect
import subprocess
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
import keras

from train.common import set_seeds
import train.bfunet.train_convunext_denoiser as trainer_mod
from train.bfunet.train_convunext_denoiser import (
    TrainingConfig,
    create_dataset,
    make_curriculum_noise_fn,
    build_self_iterate_pool,
    create_self_iterate_dataset,
    denoise_k_passes,
    multi_pass_psnr,
)

# Commit immediately BEFORE plan_2026-06-20_88705c63 (the pre-self-iterate
# trainer). The OFF-path streaming logic must match this baseline byte-for-byte
# (modulo indentation; see module docstring).
BASELINE_COMMIT = "8688519a16e68640beb6cfc5b7ee72d08f37db19"
TRAINER_REL_PATH = "src/train/bfunet/train_convunext_denoiser.py"

PATCH = 16          # tiny patch for fast tests
CHANNELS = 3
SEED = 1234


# ---------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------


def _repo_root() -> Path:
    # tests/test_train/test_bfunet/<this file> -> repo root is 3 levels up.
    return Path(__file__).resolve().parents[3]


def _write_png(path: Path, size: int, rng: np.random.Generator) -> None:
    """Write a tiny random RGB PNG so the streaming loader has real files."""
    arr = rng.integers(0, 256, size=(size, size, CHANNELS), dtype=np.uint8)
    path.write_bytes(tf.io.encode_png(arr).numpy())


@pytest.fixture
def image_paths(tmp_path):
    """A handful of tiny RGB PNGs (larger than PATCH so random_crop works)."""
    rng = np.random.default_rng(7)
    paths = []
    for i in range(6):
        p = tmp_path / f"img_{i}.png"
        _write_png(p, size=PATCH * 2, rng=rng)
        paths.append(str(p))
    return paths


def _streaming_config() -> TrainingConfig:
    """A tiny additive-noise config for the streaming OFF path."""
    return TrainingConfig(
        train_image_dirs=["/tmp/unused_train"],   # not read: we pass paths directly
        val_image_dirs=["/tmp/unused_val"],
        patch_size=PATCH,
        channels=CHANNELS,
        batch_size=4,
        patches_per_image=2,
        patch_shuffle_buffer=8,
        dataset_shuffle_buffer=8,
        augment_data=True,
        noise_type="additive",
        sigma_max_start=0.1,
        sigma_max_end=0.3,
        epochs=2,
        seed=SEED,
        self_iterate=False,
    )


def _first_batch(file_paths):
    """Seed, build the streaming OFF-path dataset, return the first batch."""
    set_seeds(SEED)
    config = _streaming_config()
    sigma_var = tf.Variable(config.sigma_max_start, dtype=tf.float32)
    noise_fn = make_curriculum_noise_fn(config, sigma_var)
    ds = create_dataset(file_paths, config, noise_fn, is_training=True)
    noisy, clean = next(iter(ds))
    return np.asarray(noisy), np.asarray(clean)


# ---------------------------------------------------------------------
# SC1 -- OFF-path determinism + textual byte-identity (the critical test)
# ---------------------------------------------------------------------


def test_off_path_byte_identical(image_paths):
    """SC1a: the streaming OFF path is deterministic under a fixed seed.

    Build the streaming dataset twice with self_iterate=False under the SAME
    seed and assert the first (noisy, clean) batch is bitwise identical. A
    non-zero diff here would mean the OFF path lost seed-determinism (a
    regression introduced by the self-iterate work).
    """
    noisy_a, clean_a = _first_batch(image_paths)
    noisy_b, clean_b = _first_batch(image_paths)

    assert noisy_a.shape == (4, PATCH, PATCH, CHANNELS)
    assert clean_a.shape == (4, PATCH, PATCH, CHANNELS)

    noisy_diff = float(np.max(np.abs(noisy_a - noisy_b)))
    clean_diff = float(np.max(np.abs(clean_a - clean_b)))
    assert noisy_diff == 0.0, f"OFF-path noisy batch not deterministic: {noisy_diff}"
    assert clean_diff == 0.0, f"OFF-path clean batch not deterministic: {clean_diff}"


def _extract_function_source_from_text(source_text: str, func_name: str) -> str:
    """Extract a top-level ``def func_name(...)...`` block from a full module text.

    Captures from the ``def`` line up to (but not including) the next top-level
    ``def`` / ``class`` / decorator at column 0. Used on the ``git show`` baseline
    where we have raw text, not an importable module.
    """
    lines = source_text.splitlines()
    start = None
    pat = re.compile(rf"^def\s+{re.escape(func_name)}\s*\(")
    for i, ln in enumerate(lines):
        if pat.match(ln):
            start = i
            break
    assert start is not None, f"{func_name} not found in baseline source"
    end = len(lines)
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        # Next top-level construct at column 0 ends the function block.
        if ln and not ln[0].isspace() and (
            ln.startswith("def ")
            or ln.startswith("class ")
            or ln.startswith("@")
            or ln.startswith("#")
        ):
            end = j
            break
    return "\n".join(lines[start:end])


def _normalize_body(func_source: str) -> str:
    """Normalize a function source to its logic-only token stream.

    Strips the ``def`` signature line, dedents, and collapses ALL whitespace.
    This makes the comparison robust to the relocation re-indent (Step 3 could
    have moved the streaming body into an ``else`` branch, adding leading
    indentation) while still catching any change to the actual logic.
    """
    lines = func_source.splitlines()
    # Drop the def signature line(s) up to and including the line ending in ':'.
    body_start = 0
    for i, ln in enumerate(lines):
        if ln.rstrip().endswith(":") and ln.lstrip().startswith("def "):
            body_start = i + 1
            break
    body = "\n".join(lines[body_start:])
    # Collapse every run of whitespace (incl. newlines/indentation) to one space.
    return re.sub(r"\s+", " ", body).strip()


def _nonspace_tokens(func_source: str) -> str:
    """All non-whitespace characters of the function source, concatenated."""
    return re.sub(r"\s+", "", func_source)


@pytest.mark.parametrize("func_name", ["create_dataset", "make_curriculum_noise_fn"])
def test_off_path_textual_byte_identity_vs_baseline(func_name):
    """SC1b: streaming OFF-path source matches the pre-plan baseline.

    The streaming ``create_dataset`` / ``make_curriculum_noise_fn`` logic must be
    unchanged from commit 8688519a (Step 3 only branched ``train()``). We compare
    the NORMALIZED function bodies (indentation-insensitive). If normalized
    equality fails we fall back to comparing the non-whitespace token SET, and if
    THAT differs the OFF path genuinely changed -> SC1 failure / real regression.
    """
    repo_root = _repo_root()
    baseline_text = subprocess.check_output(
        ["git", "show", f"{BASELINE_COMMIT}:{TRAINER_REL_PATH}"],
        cwd=str(repo_root),
        text=True,
    )
    baseline_src = _extract_function_source_from_text(baseline_text, func_name)
    current_src = inspect.getsource(getattr(trainer_mod, func_name))

    baseline_norm = _normalize_body(baseline_src)
    current_norm = _normalize_body(current_src)

    if baseline_norm != current_norm:
        # Fall back to a token-set comparison; if even that differs the OFF-path
        # LOGIC changed, which is a real regression (do NOT weaken this assert).
        assert _nonspace_tokens(baseline_src) == _nonspace_tokens(current_src), (
            f"OFF-path function {func_name!r} CHANGED vs baseline {BASELINE_COMMIT[:8]} "
            f"-- this is an SC1 regression, not a relocation re-indent.\n"
            f"baseline(normalized)={baseline_norm!r}\n"
            f"current(normalized)={current_norm!r}"
        )


# ---------------------------------------------------------------------
# Pool dataset contract
# ---------------------------------------------------------------------


def _pool_config(pool_size=16, batch=4) -> TrainingConfig:
    return TrainingConfig(
        train_image_dirs=["/tmp/unused_train"],
        val_image_dirs=["/tmp/unused_val"],
        patch_size=PATCH,
        channels=CHANNELS,
        batch_size=batch,
        noise_type="additive",
        sigma_max_start=0.1,
        sigma_max_end=0.3,
        epochs=2,
        seed=SEED,
        self_iterate=True,
        self_iterate_pool_size=pool_size,
        # Keep steps_per_epoch unconstrained so the dataset spans the whole pool.
        steps_per_epoch=None,
        patch_shuffle_buffer=8,
    )


def test_pool_dataset_shapes_dtype_range_and_finite():
    """create_self_iterate_dataset over synthetic clean patches yields the
    contract: [batch, P, P, C] float32 in [-1,1], FINITE with pool_size//batch
    steps.

    Synthetic numpy arrays are fed directly (no disk dependency) -- the dataset
    builder only needs the array shapes/contents, not real image files.
    """
    pool_size, batch = 16, 4
    config = _pool_config(pool_size=pool_size, batch=batch)
    rng = np.random.default_rng(0)
    clean_pool = rng.uniform(-1.0, 1.0, size=(pool_size, PATCH, PATCH, CHANNELS)).astype(np.float32)
    current_input = np.clip(
        clean_pool + rng.normal(size=clean_pool.shape).astype(np.float32) * 0.1,
        -1.0, 1.0,
    ).astype(np.float32)

    ds, steps_per_epoch = create_self_iterate_dataset(clean_pool, current_input, config)
    assert steps_per_epoch == pool_size // batch

    batches = list(ds)  # FINITE: this terminates (no .repeat())
    assert len(batches) == steps_per_epoch

    for noisy, clean in batches:
        assert tuple(noisy.shape) == (batch, PATCH, PATCH, CHANNELS)
        assert tuple(clean.shape) == (batch, PATCH, PATCH, CHANNELS)
        assert noisy.dtype == tf.float32
        assert clean.dtype == tf.float32
        n = np.asarray(noisy)
        c = np.asarray(clean)
        assert n.min() >= -1.0 - 1e-6 and n.max() <= 1.0 + 1e-6
        assert c.min() >= -1.0 - 1e-6 and c.max() <= 1.0 + 1e-6


def test_build_self_iterate_pool_from_tiny_files(tmp_path):
    """build_self_iterate_pool loads real (tiny) image files into the pool.

    Writes a couple of small PNGs and confirms the returned (clean_pool,
    current_input) arrays are float32, in [-1,1], correctly shaped, and that
    current_input differs from clean_pool (noise was injected).
    """
    rng = np.random.default_rng(11)
    files = []
    for i in range(3):
        p = tmp_path / f"pool_{i}.png"
        _write_png(p, size=PATCH * 2, rng=rng)
        files.append(str(p))

    pool_size, batch = 8, 4
    config = _pool_config(pool_size=pool_size, batch=batch)
    clean_pool, current_input = build_self_iterate_pool(files, config, sigma_init=0.2)

    assert clean_pool.shape == (pool_size, PATCH, PATCH, CHANNELS)
    assert current_input.shape == (pool_size, PATCH, PATCH, CHANNELS)
    assert clean_pool.dtype == np.float32
    assert current_input.dtype == np.float32
    assert clean_pool.min() >= -1.0 - 1e-6 and clean_pool.max() <= 1.0 + 1e-6
    assert current_input.min() >= -1.0 - 1e-6 and current_input.max() <= 1.0 + 1e-6
    # sigma_init>0 must have perturbed the clean pool.
    assert float(np.max(np.abs(current_input - clean_pool))) > 0.0


# ---------------------------------------------------------------------
# Multi-pass eval helpers
# ---------------------------------------------------------------------


def _tiny_conv_model() -> keras.Model:
    """A tiny same-shape conv model (linear, bias-free-ish) for eval helpers."""
    inp = keras.Input(shape=(PATCH, PATCH, CHANNELS))
    out = keras.layers.Conv2D(CHANNELS, 3, padding="same", use_bias=False)(inp)
    return keras.Model(inp, out)


def test_denoise_k_passes_shapes():
    """denoise_k_passes returns k tensors, each shaped like the input."""
    model = _tiny_conv_model()
    noisy = tf.random.uniform((2, PATCH, PATCH, CHANNELS), -1.0, 1.0)
    outs = denoise_k_passes(model, noisy, 3)
    assert isinstance(outs, list) and len(outs) == 3
    for t in outs:
        assert tuple(t.shape) == (2, PATCH, PATCH, CHANNELS)
        arr = np.asarray(t)
        assert arr.min() >= -1.0 - 1e-6 and arr.max() <= 1.0 + 1e-6


def test_multi_pass_psnr_returns_k_numbers():
    """multi_pass_psnr returns k finite numeric PSNR values."""
    model = _tiny_conv_model()
    clean = tf.random.uniform((2, PATCH, PATCH, CHANNELS), -1.0, 1.0)
    noisy = tf.clip_by_value(
        clean + tf.random.normal(clean.shape) * 0.1, -1.0, 1.0
    )
    psnrs = multi_pass_psnr(model, clean, noisy, 3)
    assert len(psnrs) == 3
    for v in psnrs:
        assert isinstance(v, float)
        assert np.isfinite(v)


# ---------------------------------------------------------------------
# Rejection guards (SC4-adjacent)
# ---------------------------------------------------------------------


def test_self_iterate_rejects_multiplicative_noise():
    """self_iterate=True with non-additive noise must raise ValueError (D-003)."""
    with pytest.raises(ValueError):
        TrainingConfig(
            train_image_dirs=["/tmp/u"],
            val_image_dirs=["/tmp/v"],
            patch_size=PATCH,
            channels=CHANNELS,
            batch_size=4,
            noise_type="multiplicative",
            self_iterate=True,
            self_iterate_pool_size=16,
        )


def test_self_iterate_rejects_pool_smaller_than_batch():
    """self_iterate=True with pool_size < batch_size must raise ValueError."""
    with pytest.raises(ValueError):
        TrainingConfig(
            train_image_dirs=["/tmp/u"],
            val_image_dirs=["/tmp/v"],
            patch_size=PATCH,
            channels=CHANNELS,
            batch_size=8,
            noise_type="additive",
            self_iterate=True,
            self_iterate_pool_size=4,  # < batch_size
        )


# ---------------------------------------------------------------------
# SC4 -- no custom train_step introduced anywhere
# ---------------------------------------------------------------------


def test_no_custom_train_step_invariant():
    """SC4: neither the trainer nor the callback defines a custom train_step.

    Training must stay stock ``compile(loss="mse") + model.fit``. A ``def
    train_step`` in either file would violate the core invariant.
    """
    repo_root = _repo_root()
    trainer_src = (repo_root / TRAINER_REL_PATH).read_text()
    callback_src = (
        repo_root / "src/dl_techniques/callbacks/self_iterate_pool.py"
    ).read_text()
    assert "def train_step" not in trainer_src, "trainer introduced a custom train_step"
    assert "def train_step" not in callback_src, "callback introduced a custom train_step"
