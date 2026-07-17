"""Trainer-side tests for the bias-free ConvUNeXt self-iterate feature.

Covers Step 7 of plan_2026-06-20_88705c63. The CRITICAL test here is SC1:
the ``--self-iterate`` OFF path must remain sound. We verify that two ways:

1. **Behavioural batch contract (SC1a).** Building the streaming ``create_dataset``
   with ``self_iterate=False`` yields a well-formed first ``(noisy, clean)`` batch
   (right shape/dtype, values in the [0, 1] domain, additive noise actually applied).
   We do NOT assert bitwise cross-build determinism: the streaming noise/crop ops are
   not stateless-seeded, so two builds are not guaranteed bitwise-equal even under the
   same seed -- that was never an OFF-path property and asserting it was flaky.

2. **Additive RNG draw ORDER (SC1b).** ``make_curriculum_noise_fn``'s additive branch
   must draw the per-image ``noise_level`` scalar FIRST and the Gaussian field SECOND;
   reordering them silently changes every additive noise sample. This used to be pinned
   by an ``inspect.getsource`` text-diff against a historical commit — that guard was
   retired (D-003) because it asserts on TEXT, not behavior. It is now pinned by a
   seeded, self-contained BEHAVIORAL replay: ``test_additive_branch_rng_draw_order``.

Other tests cover the pool dataset contract, the multi-pass eval helpers, the
rejection guards, and the "no custom train_step" invariant (SC4).

All shapes are tiny so the suite stays CPU/GPU1-light.
"""

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
import keras

from train.common import set_seeds
from train.bfunet.common import DATA_MIN, DATA_MAX
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
    values in the [0, 1] domain, and additive noise actually applied
    (noisy != clean). This is the behavioural half of SC1; the RNG-draw-order
    half is ``test_additive_branch_rng_draw_order`` below.

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
    # Two-sided domain gate. The pre-[0,1] version asserted only max(abs(x)) <= 1.0,
    # which a zero-centered array also satisfies -- it could not tell the two domains
    # apart. This one can.
    assert float(np.min(clean)) >= DATA_MIN - 1e-6
    assert float(np.max(clean)) <= DATA_MAX + 1e-6
    assert float(np.min(noisy)) >= DATA_MIN - 1e-6
    assert float(np.max(noisy)) <= DATA_MAX + 1e-6
    # Additive noise was applied -> noisy differs from clean.
    assert float(np.max(np.abs(noisy - clean))) > 0.0


# DECISION plan_2026-07-12_e56909cd/D-003: the textual byte-identity guard
# (`test_off_path_textual_byte_identity_vs_baseline`, `_undo_scale_rescale`,
# `_extract_function_source_from_text`) is RETIRED and replaced by the BEHAVIORAL test
# below. Two reasons: (a) the user waived checkpoint reproducibility for the [0,1] domain
# migration, retiring the guard's reproducibility rationale; (b) an `inspect.getsource`
# text-diff asserts on TEXT, not on runtime BEHAVIOR — it once blocked a provably
# behavior-neutral change and could only be unblocked by hand-folding literals through a
# `_undo_scale_rescale` fiction, which is exactly the failure mode plans/LESSONS.md warns
# about. The invariant that guard actually protected — the additive branch draws
# `noise_level` FIRST (tf.random.uniform), THEN tf.random.normal, so the additive RNG
# stream is never reordered by a refactor — SURVIVES the migration and is now pinned
# behaviorally by `test_additive_branch_rng_draw_order`. Do NOT reintroduce a source-text
# comparison here. See plans/plan_2026-07-12_e56909cd/decisions.md D-003.
def test_additive_branch_rng_draw_order():
    """The additive branch's RNG draw ORDER is `noise_level` FIRST, then the normal.

    ``make_curriculum_noise_fn``'s additive branch must draw the per-image scalar
    ``noise_level = tf.random.uniform([], sigma_min, sigma_max)`` BEFORE
    ``tf.random.normal(tf.shape(patch))``. Any refactor that hoists/reorders those two
    draws (e.g. folding the additive path into a shared helper that samples the normal
    first) changes the whole additive noise stream. This pins that order BEHAVIORALLY:
    we seed TF, run the real noise fn, then replay the two draws by hand IN THE SAME
    ORDER from the same seed and require a bitwise match. A swapped order consumes the
    RNG differently and yields a detectably different output.
    """
    config = _streaming_config()
    assert config.noise_type == "additive"
    sigma_min = float(config.noise_sigma_min)
    sigma_max = float(config.sigma_max_start)

    patch = tf.constant(
        np.linspace(0.0, 1.0, PATCH * PATCH * CHANNELS, dtype=np.float32).reshape(
            PATCH, PATCH, CHANNELS
        )
    )

    # (1) The real noise fn, from a known global seed.
    sigma_var = tf.Variable(sigma_max, dtype=tf.float32)
    noise_fn = make_curriculum_noise_fn(config, sigma_var)
    tf.random.set_seed(SEED)
    noisy, clean = noise_fn(patch)
    noisy = np.asarray(noisy)

    # (2) Hand-replayed reference: SAME seed, draws in the CORRECT order
    #     (uniform scalar first, then the normal field), then the [0,1] clip.
    tf.random.set_seed(SEED)
    ref_level = tf.random.uniform([], sigma_min, sigma_max)
    ref_normal = tf.random.normal(tf.shape(patch))
    ref = np.asarray(
        tf.clip_by_value(patch + ref_normal * ref_level, DATA_MIN, DATA_MAX)
    )

    np.testing.assert_allclose(np.asarray(clean), np.asarray(patch), rtol=0, atol=0)
    np.testing.assert_allclose(
        noisy,
        ref,
        rtol=0,
        atol=0,
        err_msg=(
            "additive-branch RNG draw order CHANGED: the noise fn no longer draws "
            "noise_level (tf.random.uniform) BEFORE tf.random.normal. This silently "
            "changes every additive noise sample."
        ),
    )

    # (3) The SWAPPED order must be detectably different -- proves this test has teeth
    #     and is not trivially satisfied by any implementation.
    tf.random.set_seed(SEED)
    swp_normal = tf.random.normal(tf.shape(patch))
    swp_level = tf.random.uniform([], sigma_min, sigma_max)
    swapped = np.asarray(
        tf.clip_by_value(patch + swp_normal * swp_level, DATA_MIN, DATA_MAX)
    )
    assert not np.allclose(ref, swapped), (
        "swapped-draw-order reference matched the correct-order reference; this test "
        "cannot detect a reordering and is worthless as written."
    )


def test_clip_noise_false_is_unclipped_and_clip_only_difference():
    """clip_noise=False returns a genuinely unclipped noisy tensor, differing from
    clip_noise=True ONLY by the clip (D-001).

    At a HIGH sigma the additive noise pushes the noisy input well outside [0,1]:
    - clip_noise=False must produce values ABOVE DATA_MAX and BELOW DATA_MIN (proves
      it is genuinely unclipped, not a no-op).
    - clip_noise=True must stay within [DATA_MIN, DATA_MAX].
    - The two share the SAME RNG realization (same global seed, identical draw order),
      so wherever the CLIPPED tensor is STRICTLY inside (DATA_MIN, DATA_MAX) the clip was
      a no-op and the two tensors must be bitwise-equal -> the clip is the ONLY
      difference. This pins invariant 2 (draw order preserved) from the flag's side.
    """
    # High, fixed sigma: noise_sigma_min == sigma_max_start so noise_level is exactly
    # this value (tf.random.uniform([], a, a) == a) -> large, deterministic noise.
    config = TrainingConfig(
        train_image_dirs=["/tmp/unused_train"],
        val_image_dirs=["/tmp/unused_val"],
        patch_size=PATCH,
        channels=CHANNELS,
        batch_size=4,
        noise_type="additive",
        noise_sigma_min=0.5,
        sigma_max_start=0.5,
        sigma_max_end=1.0,  # only used by __post_init__ validation; the noise fn reads
                            # the sigma_var we pass below (= sigma_max_start = 0.5).
        epochs=2,
        seed=SEED,
        self_iterate=False,
    )
    assert config.clip_noise is True  # dataclass default preserves today's behavior

    patch = tf.constant(
        np.linspace(0.0, 1.0, PATCH * PATCH * CHANNELS, dtype=np.float32).reshape(
            PATCH, PATCH, CHANNELS
        )
    )
    sigma_var = tf.Variable(float(config.sigma_max_start), dtype=tf.float32)

    clip_fn = make_curriculum_noise_fn(config, sigma_var, clip_noise=True)
    unclip_fn = make_curriculum_noise_fn(config, sigma_var, clip_noise=False)

    # Same global seed -> identical draw order (uniform level THEN normal field) ->
    # identical noise realization; the ONLY difference is the return-value clip.
    tf.random.set_seed(SEED)
    noisy_clipped, clean_c = clip_fn(patch)
    tf.random.set_seed(SEED)
    noisy_unclipped, clean_u = unclip_fn(patch)

    noisy_clipped = np.asarray(noisy_clipped)
    noisy_unclipped = np.asarray(noisy_unclipped)

    # clean target is untouched by the flag in both cases.
    np.testing.assert_allclose(np.asarray(clean_c), np.asarray(patch), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(clean_u), np.asarray(patch), rtol=0, atol=0)

    # (1) clip_noise=False is genuinely unclipped: exceeds BOTH bounds at high sigma.
    assert float(np.max(noisy_unclipped)) > DATA_MAX, (
        "clip_noise=False did not exceed DATA_MAX -> the clip gate is still firing."
    )
    assert float(np.min(noisy_unclipped)) < DATA_MIN, (
        "clip_noise=False did not go below DATA_MIN -> the clip gate is still firing."
    )

    # (2) clip_noise=True stays within [DATA_MIN, DATA_MAX].
    assert float(np.max(noisy_clipped)) <= DATA_MAX + 1e-6
    assert float(np.min(noisy_clipped)) >= DATA_MIN - 1e-6

    # (3) The clip is the ONLY difference: wherever the CLIPPED tensor is strictly
    #     inside the open interval (DATA_MIN, DATA_MAX), the clip was a no-op there, so
    #     the two tensors must be bitwise-equal at those positions.
    interior = (noisy_clipped > DATA_MIN) & (noisy_clipped < DATA_MAX)
    assert interior.any(), "no strictly-interior pixels -> cannot prove clip-only diff"
    np.testing.assert_allclose(
        noisy_clipped[interior],
        noisy_unclipped[interior],
        rtol=0,
        atol=0,
        err_msg=(
            "clipped and unclipped noisy tensors differ at strictly-interior pixels; "
            "the clip flag moved an RNG draw (invariant 2 violated)."
        ),
    )

    # (4) The flag has teeth: at least some pixels WERE clipped (the tensors differ).
    assert not np.array_equal(noisy_clipped, noisy_unclipped), (
        "clip_noise True vs False produced identical tensors at high sigma; the clip "
        "gate is a no-op and this test cannot detect a regression."
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
    contract: [batch, P, P, C] float32 in [0,1], FINITE with pool_size//batch
    steps.

    Synthetic numpy arrays are fed directly (no disk dependency) -- the dataset
    builder only needs the array shapes/contents, not real image files.
    """
    pool_size, batch = 16, 4
    config = _pool_config(pool_size=pool_size, batch=batch)
    rng = np.random.default_rng(0)
    clean_pool = rng.uniform(
        DATA_MIN, DATA_MAX, size=(pool_size, PATCH, PATCH, CHANNELS)
    ).astype(np.float32)
    current_input = np.clip(
        clean_pool + rng.normal(size=clean_pool.shape).astype(np.float32) * 0.1,
        DATA_MIN, DATA_MAX,
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
        assert n.min() >= DATA_MIN - 1e-6 and n.max() <= DATA_MAX + 1e-6
        assert c.min() >= DATA_MIN - 1e-6 and c.max() <= DATA_MAX + 1e-6


def test_build_self_iterate_pool_from_tiny_files(tmp_path):
    """build_self_iterate_pool loads real (tiny) image files into the pool.

    Writes a couple of small PNGs and confirms the returned (clean_pool,
    current_input) arrays are float32, in [0,1], correctly shaped, and that
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
    assert clean_pool.min() >= DATA_MIN - 1e-6 and clean_pool.max() <= DATA_MAX + 1e-6
    assert (
        current_input.min() >= DATA_MIN - 1e-6
        and current_input.max() <= DATA_MAX + 1e-6
    )
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
    noisy = tf.random.uniform((2, PATCH, PATCH, CHANNELS), DATA_MIN, DATA_MAX)
    outs = denoise_k_passes(model, noisy, 3)
    assert isinstance(outs, list) and len(outs) == 3
    for t in outs:
        assert tuple(t.shape) == (2, PATCH, PATCH, CHANNELS)
        arr = np.asarray(t)
        assert arr.min() >= DATA_MIN - 1e-6 and arr.max() <= DATA_MAX + 1e-6


def test_multi_pass_psnr_returns_k_numbers():
    """multi_pass_psnr returns k finite numeric PSNR values."""
    model = _tiny_conv_model()
    clean = tf.random.uniform((2, PATCH, PATCH, CHANNELS), DATA_MIN, DATA_MAX)
    noisy = tf.clip_by_value(
        clean + tf.random.normal(clean.shape) * 0.1, DATA_MIN, DATA_MAX
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

    No real data, no disk -- synthetic [0,1] arrays only; tiny conv model so the
    suite stays GPU1-light and non-flaky.
    """
    set_seeds(SEED)
    pool_size, batch = 16, 4
    config = _pool_config(pool_size=pool_size, batch=batch)  # steps_per_epoch=4

    rng = np.random.default_rng(0)
    clean_pool = rng.uniform(
        DATA_MIN, DATA_MAX, size=(pool_size, PATCH, PATCH, CHANNELS)
    ).astype(np.float32)
    current_input = np.clip(
        clean_pool + rng.normal(size=clean_pool.shape).astype(np.float32) * 0.1,
        DATA_MIN, DATA_MAX,
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


# ---------------------------------------------------------------------
# --dropout trainer wiring (plan_2026-06-29_cbfbbf42 SC3)
# ---------------------------------------------------------------------


class TestDropoutTrainerWiring:
    """SC3: --dropout opt-in MLP-dropout wiring (mirrors TestDepthwiseTrainerWiring).

    Pure config + argparse + lightweight construction -- no training. Covers the
    byte-identical OFF default (0.0), the TrainingConfig field round-trip, the
    argparse->args propagation via the importable parse_arguments() entry point,
    and a construction-only assertion that dropout_rate reaches the ConvNeXt
    blocks through build_model (NOT a full fit).
    """

    def test_config_default_is_off(self):
        """Default config leaves dropout_rate 0.0 (byte-identical OFF path)."""
        assert TrainingConfig().dropout_rate == 0.0

    def test_config_field_round_trip(self):
        """TrainingConfig stores the new field verbatim."""
        cfg = TrainingConfig(dropout_rate=0.1)
        assert cfg.dropout_rate == 0.1

    def test_argparse_maps_into_config(self, monkeypatch):
        """parse_arguments() picks up --dropout from argv as args.dropout."""
        argv = [
            "train_convunext_denoiser",
            "--smoke",
            "--dropout", "0.1",
        ]
        monkeypatch.setattr("sys.argv", argv)
        args = parse_arguments()
        assert args.dropout == 0.1

    def test_build_model_threads_dropout_into_blocks(self):
        """Construction-only: TrainingConfig(dropout_rate=0.1) -> build_model
        reaches the ConvNeXt blocks (get_config dropout_rate == 0.1). No fit."""
        cfg = TrainingConfig(
            variant="tiny",
            convnext_version="v1",
            use_gabor_stem=False,
            patch_size=PATCH,
            channels=CHANNELS,
            batch_size=2,
            self_iterate_pool_size=4,
            dropout_rate=0.1,
        )
        model = build_model(cfg)
        blocks = [
            l for l in model._flatten_layers()
            if l.__class__.__name__ in ("ConvNextV1Block", "ConvNextV2Block")
        ]
        assert len(blocks) > 0, "no ConvNeXt blocks found in the built model"
        for blk in blocks:
            assert blk.get_config()['dropout_rate'] == 0.1


# ---------------------------------------------------------------------
# SC5 -- symmetry-penalty config fields + fail-closed validation
# (plan-2026-07-17-874b11cc step 3, invariant 4)
# ---------------------------------------------------------------------


class TestSymmetryConfigValidation:
    """SC5: symmetry_weight / symmetry_probes fields + fail-closed __post_init__ guards.

    Pure config construction -- no GPU/training. Covers the byte-identical OFF defaults
    (0.0 / 1), the field round-trip, and the three refused combinations that must raise
    ValueError: negative weight, probes < 1, and symmetry_weight>0 together with
    mixed_precision (invariant 4 -- the second-order fp16/XLA silent-death ban, D-003).
    """

    def test_config_defaults_are_off(self):
        """Default config leaves the penalty OFF (weight 0.0, probes 1)."""
        cfg = TrainingConfig()
        assert cfg.symmetry_weight == 0.0
        assert cfg.symmetry_probes == 1

    def test_config_fields_round_trip(self):
        """TrainingConfig stores the two new fields verbatim (fp32, penalty ON)."""
        cfg = TrainingConfig(symmetry_weight=0.1, symmetry_probes=3)
        assert cfg.symmetry_weight == 0.1
        assert cfg.symmetry_probes == 3

    def test_negative_weight_raises(self):
        """symmetry_weight < 0 must raise ValueError."""
        with pytest.raises(ValueError):
            TrainingConfig(symmetry_weight=-0.1)

    def test_zero_probes_raises(self):
        """symmetry_probes = 0 must raise ValueError."""
        with pytest.raises(ValueError):
            TrainingConfig(symmetry_probes=0)

    def test_penalty_with_mixed_precision_raises(self):
        """symmetry_weight>0 together with mixed_precision must raise ValueError
        (invariant 4: fail-closed on the second-order fp16/XLA path, D-003)."""
        with pytest.raises(ValueError):
            TrainingConfig(symmetry_weight=0.1, mixed_precision=True)

    def test_penalty_fp32_constructs(self):
        """symmetry_weight>0 with mixed_precision OFF (fp32) constructs without raising."""
        cfg = TrainingConfig(symmetry_weight=0.1, symmetry_probes=2, mixed_precision=False)
        assert cfg.symmetry_weight == 0.1
        assert cfg.mixed_precision is False
