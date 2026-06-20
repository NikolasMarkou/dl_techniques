"""Trainer-side tests for the bias-free ConvUNeXt self-iterate feature.

Covers Step 7 of plan_2026-06-20_88705c63. The CRITICAL test here is SC1:
the ``--self-iterate`` OFF path must remain byte-identical to the pre-plan
streaming trainer. We verify that two ways:

1. **Behavioural batch contract (SC1a).** Building the streaming ``create_dataset``
   with ``self_iterate=False`` yields a well-formed first ``(noisy, clean)`` batch
   (right shape/dtype, values in [-1, +1], additive noise actually applied). We do
   NOT assert bitwise cross-build determinism: the streaming noise/crop ops are not
   stateless-seeded, so two builds are not guaranteed bitwise-equal even under the
   same seed -- that was never an OFF-path property and asserting it was flaky.

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
    build_model,
    denoise_k_passes,
    multi_pass_psnr,
    parse_arguments,
    _resolve_depthwise_initializer,
)
from dl_techniques.callbacks.self_iterate_pool import SelfIteratePoolCallback
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint

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


def test_off_path_batch_contract(image_paths):
    """SC1a: the streaming OFF path still produces a well-formed batch.

    Build the streaming dataset with self_iterate=False and assert the first
    (noisy, clean) batch honours the contract: right shape, float32 dtype,
    values in the [-1, +1] domain, and additive noise actually applied
    (noisy != clean). This is the behavioural half of SC1; the *byte-identity*
    half (proving the streaming logic is unchanged vs the pre-plan baseline)
    is ``test_off_path_textual_byte_identity_vs_baseline`` below.

    NOTE: we deliberately do NOT assert bitwise determinism across two builds.
    The original streaming pipeline's noise/crop ops are not stateless-seeded,
    so two independent dataset constructions are not guaranteed bitwise-equal
    even under the same global seed -- that was never a property of the OFF
    path, so asserting it produced a flaky test (~1-in-3 failures).
    """
    noisy, clean = _first_batch(image_paths)

    assert noisy.shape == (4, PATCH, PATCH, CHANNELS)
    assert clean.shape == (4, PATCH, PATCH, CHANNELS)
    assert noisy.dtype == np.float32 and clean.dtype == np.float32
    assert float(np.max(np.abs(clean))) <= 1.0 + 1e-6
    assert float(np.max(np.abs(noisy))) <= 1.0 + 1e-6
    # Additive noise was applied -> noisy differs from clean.
    assert float(np.max(np.abs(noisy - clean))) > 0.0


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
# D-005 (Bug-B) -- self-iterate fit must train >=2 epochs + re-read pool
# ---------------------------------------------------------------------


class _BatchCounter(keras.callbacks.Callback):
    """Records, per epoch, how many train batches model.fit actually ran.

    The Bug-B (D-005) failure was: with a FINITE pool dataset passed to fit
    ALONGSIDE steps_per_epoch, Keras exhausted the dataset in epoch 1 and ran
    ZERO batches in epoch 2+ (OUT_OF_RANGE, loss=0.0). Counting on_train_batch_end
    per epoch makes that regression directly observable: the fixed call shape
    (steps_per_epoch=None) must yield >0 batches in BOTH epochs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.per_epoch_batches: list[int] = []
        self._cur = 0

    def on_epoch_begin(self, epoch, logs=None) -> None:
        self._cur = 0

    def on_train_batch_end(self, batch, logs=None) -> None:
        self._cur += 1

    def on_epoch_end(self, epoch, logs=None) -> None:
        self.per_epoch_batches.append(self._cur)


def test_self_iterate_trains_multiple_epochs_and_pool_reread():
    """D-005 regression: self-iterate fit trains >=2 epochs + re-reads pool.

    This is the direct guard for Bug-B. We reproduce the FIXED trainer call shape
    (finite from_generator pool dataset + steps_per_epoch=None) end-to-end on tiny
    synthetic data, with the REAL SelfIteratePoolCallback mutating the SAME live
    pool arrays the dataset indexes. We assert:

      (a) the batch-counter saw >0 batches in BOTH epoch 1 AND epoch 2 (the
          pre-fix bug gave 0 in epoch 2 -- this is the load-bearing assertion);
      (b) the pool callback regenerated in BOTH epochs (history len == 2);
      (c) the live `current_input` buffer actually changed across the two epoch
          regenerations (the pool was re-read / mutated each epoch, not snapshotted).

    No real data, no disk -- synthetic [-1,1] arrays only; tiny conv model so the
    suite stays GPU1-light and non-flaky.
    """
    set_seeds(SEED)
    pool_size, batch = 16, 4
    config = _pool_config(pool_size=pool_size, batch=batch)  # steps_per_epoch=4

    rng = np.random.default_rng(0)
    clean_pool = rng.uniform(
        -1.0, 1.0, size=(pool_size, PATCH, PATCH, CHANNELS)
    ).astype(np.float32)
    current_input = np.clip(
        clean_pool + rng.normal(size=clean_pool.shape).astype(np.float32) * 0.1,
        -1.0, 1.0,
    ).astype(np.float32)

    ds, steps_per_epoch = create_self_iterate_dataset(
        clean_pool, current_input, config
    )
    assert steps_per_epoch == pool_size // batch  # 4

    model = keras.Sequential(
        [keras.Input(shape=(PATCH, PATCH, CHANNELS)),
         keras.layers.Conv2D(CHANNELS, 1, padding="same")]
    )
    model.compile(optimizer="adam", loss="mse")

    # Real callback over the SAME live arrays the dataset indexes (regen every
    # epoch, mix half). get_sigma is a fixed small sigma for determinism.
    pool_cb = SelfIteratePoolCallback(
        clean_pool=clean_pool,
        current_input=current_input,
        get_sigma=lambda: 0.1,
        regen_freq=1,
        mix_ratio=0.5,
        predict_batch_size=batch,
        seed=SEED,
    )
    counter = _BatchCounter()

    # Snapshot the pool BEFORE training (initial state).
    snap_pre = current_input.copy()

    # FIXED trainer call shape: finite pool dataset + steps_per_epoch=None so Keras
    # consumes the whole finite dataset each epoch via a fresh per-epoch iterator.
    model.fit(
        ds,
        epochs=2,
        steps_per_epoch=None,
        callbacks=[pool_cb, counter],
        verbose=0,
    )

    # (a) BOTH epochs trained on batches. The pre-fix bug gave 0 in epoch 2.
    assert len(counter.per_epoch_batches) == 2, counter.per_epoch_batches
    assert counter.per_epoch_batches[0] > 0, (
        f"epoch-1 ran zero batches: {counter.per_epoch_batches}"
    )
    assert counter.per_epoch_batches[1] > 0, (
        "epoch-2 ran ZERO batches -- Bug-B (D-005) regression: finite pool "
        f"dataset exhausted after epoch 1. per_epoch={counter.per_epoch_batches}"
    )
    # Both epochs consume the same finite pool -> equal batch counts == steps.
    assert counter.per_epoch_batches[0] == steps_per_epoch
    assert counter.per_epoch_batches[1] == steps_per_epoch

    # (b) The callback regenerated the pool in BOTH epochs.
    assert len(pool_cb.history["epoch"]) == 2, pool_cb.history
    # Two distinct regeneration records (sanity: mean_residual recorded twice).
    assert len(pool_cb.history["mean_residual"]) == 2

    # (c) The live pool buffer actually changed across the run (re-read + mutated
    # each epoch, NOT snapshotted into a graph constant). Robust difference check,
    # no bitwise cross-run assumption.
    assert float(np.max(np.abs(current_input - snap_pre))) > 0.0, (
        "pool current_input did not change -- regeneration never mutated the "
        "live buffer (A1/D-004 re-read broke)."
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


# ---------------------------------------------------------------------
# --init-from warm-start (self-iterate fine-tuning entry point)
# ---------------------------------------------------------------------


def _tiny_model_config() -> TrainingConfig:
    """Smallest buildable denoiser config for fast warm-start tests."""
    return TrainingConfig(
        variant="tiny",
        convnext_version="v1",
        use_gabor_stem=False,
        patch_size=PATCH,
        channels=CHANNELS,
        batch_size=2,
        self_iterate_pool_size=4,  # >= batch_size (post_init guard)
    )


def test_init_from_round_trips_checkpoint_weights(tmp_path):
    """build_model + load_weights_from_checkpoint(skip_prefixes=()) copies ALL
    layer weights, so a fresh model warm-started from a saved checkpoint matches
    the source bit-for-bit. This is the mechanism the trainer's --init-from uses.
    """
    cfg = _tiny_model_config()

    set_seeds(1)
    src = build_model(cfg)
    ckpt = tmp_path / "src.keras"
    src.save(ckpt)

    set_seeds(2)  # different init -> weights must start DIFFERENT
    tgt = build_model(cfg)

    src_w = src.get_weights()
    tgt_w = tgt.get_weights()
    assert any(
        not np.allclose(a, b) for a, b in zip(src_w, tgt_w)
    ), "fresh model unexpectedly identical before transfer"

    report = load_weights_from_checkpoint(tgt, ckpt_path=str(ckpt), skip_prefixes=())
    assert len(report.loaded) > 0
    assert len(report.shape_mismatch) == 0

    # After transfer the target equals the source weight-for-weight.
    for a, b in zip(src.get_weights(), tgt.get_weights()):
        np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_init_from_missing_file_raises(tmp_path):
    cfg = _tiny_model_config()
    tgt = build_model(cfg)
    with pytest.raises(FileNotFoundError):
        load_weights_from_checkpoint(
            tgt, ckpt_path=str(tmp_path / "nope.keras"), skip_prefixes=()
        )


def test_init_from_field_default_is_none():
    """Default config does not warm-start (byte-identical to pre-feature)."""
    assert TrainingConfig().init_from is None


# ---------------------------------------------------------------------
# SC6 -- depthwise init/regularizer trainer wiring (config + helper)
# ---------------------------------------------------------------------


class TestDepthwiseTrainerWiring:
    """SC6: --depthwise-initializer / --depthwise-l2 opt-in wiring.

    Pure config + helper logic -- no GPU/training. Covers the
    _resolve_depthwise_initializer alias (D-002), the TrainingConfig field
    round-trip, the byte-identical OFF default, and (cheaply) the argparse->
    config propagation via the importable parse_arguments() entry point.
    """

    def test_resolve_orthonormal_alias(self):
        """'orthonormal' -> keras Orthogonal(gain=1.0) (D-002)."""
        init = _resolve_depthwise_initializer("orthonormal")
        assert isinstance(init, keras.initializers.Orthogonal)
        assert float(init.gain) == 1.0

    def test_resolve_none_is_none(self):
        """None -> None (OFF, byte-identical)."""
        assert _resolve_depthwise_initializer(None) is None

    def test_resolve_other_string_passthrough(self):
        """Any other string passes through unchanged for keras.initializers.get."""
        assert _resolve_depthwise_initializer("he_normal") == "he_normal"

    def test_config_fields_round_trip(self):
        """TrainingConfig stores the two new fields verbatim."""
        cfg = TrainingConfig(
            depthwise_initializer="orthonormal",
            depthwise_l2=1e-4,
        )
        assert cfg.depthwise_initializer == "orthonormal"
        assert cfg.depthwise_l2 == 1e-4

    def test_config_defaults_are_off(self):
        """Default config leaves both fields None (byte-identical OFF path)."""
        cfg = TrainingConfig()
        assert cfg.depthwise_initializer is None
        assert cfg.depthwise_l2 is None

    def test_argparse_maps_into_config(self, monkeypatch):
        """parse_arguments() picks up the two new flags from argv.

        parse_arguments() reads sys.argv via parser.parse_args(); patch argv and
        confirm the parsed args carry the values that the smoke/normal config
        blocks propagate into TrainingConfig.
        """
        argv = [
            "train_convunext_denoiser",
            "--smoke",
            "--depthwise-initializer", "orthonormal",
            "--depthwise-l2", "1e-4",
        ]
        monkeypatch.setattr("sys.argv", argv)
        args = parse_arguments()
        assert args.depthwise_initializer == "orthonormal"
        assert args.depthwise_l2 == 1e-4
