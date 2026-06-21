"""Tests for SelfIteratePoolCallback (plan_2026-06-20_88705c63).

Covers: regeneration write-back, mix_ratio semantics, off-cadence no-op,
fail-soft on unattached/broken model, get_config/from_config round-trip (scalars
only), clip range, and seed determinism of the regen-slot selection.
"""

import numpy as np
import pytest
import keras

from dl_techniques.callbacks.self_iterate_pool import SelfIteratePoolCallback


# ----------------------------------------------------------------------------
# Helpers / fixtures
# ----------------------------------------------------------------------------
def _build_scale_model(channels: int, scale: float = 0.5):
    """A deterministic 1x1 conv with all-ones weights and no bias.

    With a ``kernel_initializer`` of a constant ``scale / channels`` the output
    of every spatial location is ``scale * mean_over_channels`` broadcast across
    all output channels — a fully predictable, differentiable forward map we can
    reproduce in numpy.
    """
    init = keras.initializers.Constant(scale / channels)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(None, None, channels)),
            keras.layers.Conv2D(
                channels,
                kernel_size=1,
                padding="same",
                use_bias=False,
                kernel_initializer=init,
            ),
        ]
    )
    return model


def _expected_output(x: np.ndarray, channels: int, scale: float = 0.5) -> np.ndarray:
    """Numpy reproduction of ``_build_scale_model``'s forward pass."""
    chan_sum = x.sum(axis=-1, keepdims=True) * (scale / channels)
    return np.broadcast_to(chan_sum, x.shape).astype(np.float32)


def _make_pool(n=8, h=4, w=4, c=3, seed=0):
    rng = np.random.default_rng(seed)
    clean = rng.uniform(-1.0, 1.0, size=(n, h, w, c)).astype(np.float32)
    # current_input starts as a slightly noised copy in-range.
    current = np.clip(
        clean + rng.normal(size=clean.shape).astype(np.float32) * 0.1, -1.0, 1.0
    )
    return clean, current


# ----------------------------------------------------------------------------
class TestSelfIteratePoolCallback:
    # --- 1. regeneration write-back ---------------------------------------
    def test_regeneration_writes_back(self):
        c = 3
        clean, current = _make_pool(n=8, c=c, seed=1)
        before = current.copy()
        model = _build_scale_model(c)
        denoised = _expected_output(current, c)

        cb = SelfIteratePoolCallback(
            clean_pool=clean,
            current_input=current,
            get_sigma=lambda: 0.1,
            regen_freq=1,
            mix_ratio=0.5,
            seed=7,
        )
        cb.set_model(model)
        cb.on_epoch_end(0)

        assert len(cb.history["epoch"]) == 1
        n_regen = cb.history["n_regen"][0]
        assert n_regen == 4  # round(0.5 * 8)

        # Reproduce the exact slot selection with the same seed.
        rng = np.random.default_rng(7)
        perm = rng.permutation(8)
        regen_idx = perm[:n_regen]

        # Regenerated slots equal the clipped model output.
        np.testing.assert_allclose(
            current[regen_idx],
            np.clip(denoised[regen_idx], -1.0, 1.0),
            atol=1e-5,
        )
        # Every slot changed (regen got model output, fresh got new noise).
        assert not np.array_equal(current, before)

    # --- 2. mix_ratio semantics -------------------------------------------
    def test_mix_ratio_zero_no_regen(self):
        c = 3
        clean, current = _make_pool(n=8, c=c, seed=2)
        model = _build_scale_model(c)
        denoised = _expected_output(current.copy(), c)

        cb = SelfIteratePoolCallback(
            clean_pool=clean, current_input=current, get_sigma=lambda: 0.1,
            regen_freq=1, mix_ratio=0.0, seed=3,
        )
        cb.set_model(model)
        cb.on_epoch_end(0)

        assert cb.history["n_regen"][0] == 0
        # No slot equals the model output (all are fresh noise) — overwhelmingly
        # unlikely to coincide, assert strictly none match exactly.
        clipped = np.clip(denoised, -1.0, 1.0)
        matches = np.all(np.isclose(current, clipped, atol=1e-6), axis=(1, 2, 3))
        assert not matches.any()

    def test_mix_ratio_one_all_regen(self):
        c = 3
        clean, current = _make_pool(n=8, c=c, seed=4)
        model = _build_scale_model(c)
        denoised = _expected_output(current.copy(), c)

        cb = SelfIteratePoolCallback(
            clean_pool=clean, current_input=current, get_sigma=lambda: 0.1,
            regen_freq=1, mix_ratio=1.0, seed=5,
        )
        cb.set_model(model)
        cb.on_epoch_end(0)

        assert cb.history["n_regen"][0] == 8
        # Every slot equals the clipped model output.
        np.testing.assert_allclose(
            current, np.clip(denoised, -1.0, 1.0), atol=1e-5
        )

    # --- 3. off-cadence no-op ---------------------------------------------
    def test_off_cadence_is_noop(self):
        c = 3
        clean, current = _make_pool(n=8, c=c, seed=6)
        before = current.copy()
        model = _build_scale_model(c)

        cb = SelfIteratePoolCallback(
            clean_pool=clean, current_input=current, get_sigma=lambda: 0.1,
            regen_freq=2, mix_ratio=0.5, seed=8,
        )
        cb.set_model(model)

        # epoch index 0 -> (0+1) % 2 == 1 -> skip.
        cb.on_epoch_end(0)
        assert len(cb.history["epoch"]) == 0
        np.testing.assert_array_equal(current, before)

        # epoch index 1 -> (1+1) % 2 == 0 -> regenerate.
        cb.on_epoch_end(1)
        assert len(cb.history["epoch"]) == 1
        assert not np.array_equal(current, before)

    # --- 4. fail-soft -----------------------------------------------------
    def test_no_arrays_attached_is_failsoft(self):
        # No live pool/sigma attached -> warns and returns, no raise, no history.
        cb = SelfIteratePoolCallback(regen_freq=1, mix_ratio=0.5, seed=1)
        cb.set_model(_build_scale_model(3))
        cb.on_epoch_end(0)  # must not raise
        assert len(cb.history["epoch"]) == 0

    def test_broken_model_predict_is_failsoft(self):
        c = 3
        clean, current = _make_pool(n=8, c=c, seed=9)
        before = current.copy()

        class _Boom:
            def predict(self, *a, **k):
                raise RuntimeError("predict exploded")

        cb = SelfIteratePoolCallback(
            clean_pool=clean, current_input=current, get_sigma=lambda: 0.1,
            regen_freq=1, mix_ratio=0.5, seed=2,
        )
        # Attach a model whose predict raises (cb.model is read-only -> set_model).
        cb.set_model(_Boom())

        cb.on_epoch_end(0)  # must not raise (fail-soft)
        # Pool unchanged, no history recorded.
        np.testing.assert_array_equal(current, before)
        assert len(cb.history["epoch"]) == 0

    # --- 5. config round-trip (scalars only) ------------------------------
    def test_get_config_scalars_only(self):
        clean, current = _make_pool()
        cb = SelfIteratePoolCallback(
            clean_pool=clean, current_input=current, get_sigma=lambda: 0.1,
            regen_freq=3, mix_ratio=0.25, predict_batch_size=16, seed=11,
        )
        cfg = cb.get_config()
        assert set(cfg.keys()) == {
            "regen_freq", "mix_ratio", "predict_batch_size", "seed",
            "clip_min", "clip_max",
        }
        # No live arrays / model / callable leaked into the config.
        for forbidden in ("clean_pool", "current_input", "get_sigma", "model"):
            assert forbidden not in cfg
        # Every value is a JSON scalar.
        for v in cfg.values():
            assert isinstance(v, (int, float))

        rebuilt = SelfIteratePoolCallback.from_config(cfg)
        assert rebuilt.regen_freq == 3
        assert rebuilt.mix_ratio == 0.25
        assert rebuilt.predict_batch_size == 16
        assert rebuilt.seed == 11
        # Reconstructed callback has no live arrays -> on_epoch_end is a no-op.
        rebuilt.on_epoch_end(0)
        assert len(rebuilt.history["epoch"]) == 0

    def test_keras_serializable_registered(self):
        cb = SelfIteratePoolCallback(regen_freq=2, mix_ratio=0.5, seed=4)
        cfg = keras.saving.serialize_keras_object(cb)
        restored = keras.saving.deserialize_keras_object(cfg)
        assert isinstance(restored, SelfIteratePoolCallback)
        assert restored.regen_freq == 2
        assert restored.mix_ratio == 0.5

    # --- 6. clip range ----------------------------------------------------
    def test_clip_range_after_regen(self):
        c = 3
        clean, current = _make_pool(n=8, c=c, seed=12)
        # Use a large sigma so unclipped fresh noise would exceed [-1, 1].
        model = _build_scale_model(c, scale=3.0)  # output well outside [-1,1]

        cb = SelfIteratePoolCallback(
            clean_pool=clean, current_input=current, get_sigma=lambda: 2.0,
            regen_freq=1, mix_ratio=0.5, seed=13,
        )
        cb.set_model(model)
        cb.on_epoch_end(0)

        assert current.min() >= -1.0 - 1e-6
        assert current.max() <= 1.0 + 1e-6

    # --- 7. determinism ---------------------------------------------------
    def test_same_seed_same_regen_selection(self):
        c = 3
        clean0, current0 = _make_pool(n=8, c=c, seed=14)
        clean1, current1 = _make_pool(n=8, c=c, seed=14)  # identical inputs
        before = current0.copy()
        model = _build_scale_model(c)

        cb0 = SelfIteratePoolCallback(
            clean_pool=clean0, current_input=current0, get_sigma=lambda: 0.1,
            regen_freq=1, mix_ratio=0.5, seed=99,
        )
        cb1 = SelfIteratePoolCallback(
            clean_pool=clean1, current_input=current1, get_sigma=lambda: 0.1,
            regen_freq=1, mix_ratio=0.5, seed=99,
        )
        cb0.set_model(model)
        cb1.set_model(model)
        cb0.on_epoch_end(0)
        cb1.on_epoch_end(0)

        # Same seed + same inputs -> byte-identical pool after one epoch.
        np.testing.assert_array_equal(current0, current1)

        # And the SET of changed slots is identical (sanity on selection).
        changed0 = np.any(current0 != before, axis=(1, 2, 3))
        changed1 = np.any(current1 != before, axis=(1, 2, 3))
        np.testing.assert_array_equal(changed0, changed1)

    def test_different_seed_different_selection(self):
        c = 3
        clean0, current0 = _make_pool(n=16, c=c, seed=15)
        clean1, current1 = _make_pool(n=16, c=c, seed=15)
        model = _build_scale_model(c)

        cb0 = SelfIteratePoolCallback(
            clean_pool=clean0, current_input=current0, get_sigma=lambda: 0.1,
            regen_freq=1, mix_ratio=0.5, seed=1,
        )
        cb1 = SelfIteratePoolCallback(
            clean_pool=clean1, current_input=current1, get_sigma=lambda: 0.1,
            regen_freq=1, mix_ratio=0.5, seed=2,
        )
        cb0.set_model(model)
        cb1.set_model(model)
        cb0.on_epoch_end(0)
        cb1.on_epoch_end(0)

        # Different seeds -> different regen selection / fresh noise -> pools differ.
        assert not np.array_equal(current0, current1)
