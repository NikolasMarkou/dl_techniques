"""Fast CPU unit tests for the ``DenoiserPrior`` core wrapper.

Uses a TINY in-memory bias-free Keras model (no checkpoint, no GPU) to verify the
wrapper's mechanics: ``residual`` / ``denoise`` symbolic identities, ``ingest`` /
``denorm`` domain math, and the non-overlapping ``tile`` / ``untile`` round-trip.
The real 22 MB checkpoint is exercised separately (step 6 / drift-guard script),
never in this fast suite.
"""

import json
import logging

import keras
import numpy as np
import pytest

from applications.bias_free_denoiser.denoiser_prior import DenoiserPrior


def _tiny_bias_free_model() -> keras.Model:
    """A minimal bias-free (no-bias) conv model on ``(None, None, 3)``.

    Not an identity — a single random-but-fixed 1x1 conv with ``use_bias=False`` so
    it is degree-1 homogeneous, enough to test ``residual = model(y) - y`` exactly.
    """
    inputs = keras.Input(shape=(None, None, 3))
    outputs = keras.layers.Conv2D(
        3, kernel_size=1, use_bias=False,
        kernel_initializer=keras.initializers.GlorotUniform(seed=0),
    )(inputs)
    return keras.Model(inputs, outputs, name="tiny_bias_free")


@pytest.fixture
def prior() -> DenoiserPrior:
    return DenoiserPrior(_tiny_bias_free_model())


class TestDomainHelpers:
    """The exact ingest/denorm numerics ARE the contract for the [0,1] domain (D-004)."""

    def test_ingest_uint8_maps_to_domain(self):
        img = np.array([[[0, 128, 255]]], dtype=np.uint8)  # [1,1,3]
        out = DenoiserPrior.ingest(img)
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, np.array([[[0.0, 128 / 255, 1.0]]]), atol=1e-6)

    def test_ingest_maps_0_255_extremes(self):
        img = np.zeros((1, 4, 4, 3), dtype=np.uint8)
        img[..., 0] = 255
        out = DenoiserPrior.ingest(img)
        assert float(out.min()) == pytest.approx(0.0)
        assert float(out.max()) == pytest.approx(1.0)

    def test_ingest_float_0_1_is_identity(self):
        # Model domain == display domain: a float [0,1] image ingests unchanged.
        img = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        out = DenoiserPrior.ingest(img)
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_ingest_is_idempotent(self):
        # D-004: the "has negatives => already ingested" heuristic is GONE, and is not
        # needed — under [0,1] ingest is idempotent, so a double-ingest is a no-op.
        img = np.random.default_rng(0).integers(0, 256, size=(1, 8, 8, 3)).astype(np.uint8)
        once = DenoiserPrior.ingest(img)
        twice = DenoiserPrior.ingest(once)
        np.testing.assert_allclose(twice, once, atol=1e-7)

    def test_ingest_clips_out_of_domain_float(self):
        # Negatives are OUT of domain now (not a "already ingested" signal): clip them.
        img = np.array([[[-0.4, 0.25, 1.3]]], dtype=np.float32)
        out = DenoiserPrior.ingest(img)
        np.testing.assert_allclose(out, np.array([[[0.0, 0.25, 1.0]]]), atol=1e-6)

    def test_ingest_warns_on_a_legacy_zero_centered_array(self, caplog):
        # A float array with real negative mass is the signature of a LEGACY [-0.5,+0.5]
        # image. Clipping it to [0,1] destroys its entire lower half, so ingest must SAY so
        # rather than corrupt it silently -- the silent failure is the one this migration
        # exists to eliminate. The clip itself is unconditional (no domain branch).
        legacy = np.random.default_rng(0).uniform(-0.5, 0.5, size=(1, 8, 8, 3)).astype(np.float32)
        with caplog.at_level(logging.WARNING):
            out = DenoiserPrior.ingest(legacy)
        assert any("LEGACY" in r.message or "LEGACY" in r.getMessage() for r in caplog.records), (
            "ingest() silently clipped a negative-valued (legacy zero-centered) array"
        )
        assert float(out.min()) >= 0.0

    def test_ingest_does_not_warn_on_float_noise(self, caplog):
        # A hair below zero is numerical overshoot from an upstream op, not a legacy array.
        # Warning on it would make the diagnostic noise, and noisy diagnostics get ignored.
        img = np.array([[[-1e-7, 0.25, 1.0]]], dtype=np.float32)
        with caplog.at_level(logging.WARNING):
            DenoiserPrior.ingest(img)
        assert not any("LEGACY" in r.getMessage() for r in caplog.records)

    def test_denorm_inverts_ingest_for_uint8(self):
        img = np.random.randint(0, 256, size=(1, 8, 8, 3)).astype(np.uint8)
        recovered = DenoiserPrior.denorm(DenoiserPrior.ingest(img))
        np.testing.assert_allclose(recovered, img.astype(np.float32) / 255.0, atol=1e-6)

    def test_denorm_is_identity_in_domain(self):
        x = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        np.testing.assert_allclose(DenoiserPrior.denorm(x), x, atol=1e-6)

    def test_denorm_clips_out_of_domain_iterate(self):
        # denorm's ONLY remaining job: rectify a solver iterate that stepped outside.
        x = np.array([[[-0.3, 0.5, 1.7]]], dtype=np.float32)
        np.testing.assert_allclose(DenoiserPrior.denorm(x),
                                   np.array([[[0.0, 0.5, 1.0]]]), atol=1e-6)


class TestCoreMethods:
    def test_domain_attributes(self, prior):
        assert prior.domain_center == 0.5
        assert prior.domain_halfwidth == 0.5
        assert prior.model is not None

    def test_denoise_shape(self, prior):
        y = np.random.uniform(0.0, 1.0, size=(2, 16, 16, 3)).astype(np.float32)
        d = keras.ops.convert_to_numpy(prior.denoise(y))
        assert d.shape == y.shape

    def test_residual_equals_model_minus_input(self, prior):
        y = np.random.uniform(0.0, 1.0, size=(1, 16, 16, 3)).astype(np.float32)
        f = keras.ops.convert_to_numpy(prior.residual(y))
        d = keras.ops.convert_to_numpy(prior.model(y, training=False))
        np.testing.assert_allclose(f, d - y, atol=1e-6)
        assert f.shape == y.shape


class TestDataRangeGate:
    """`from_pretrained` REFUSES any checkpoint not stamped data_range == "[0,1]" (D-005).

    The gate runs BEFORE any model load, so these tests need only a dummy `.keras` file
    (existence is all `_resolve_paths` checks) — they stay fast and CPU-only.
    """

    @staticmethod
    def _fake_ckpt(tmp_path, config: dict | None):
        (tmp_path / "best_model.keras").write_bytes(b"not-a-real-model")
        if config is not None:
            (tmp_path / "config.json").write_text(json.dumps(config))
        return tmp_path

    def test_absent_data_range_key_is_refused(self, tmp_path):
        # EVERY pre-migration checkpoint looks like this. Absent => legacy => refuse.
        ckpt = self._fake_ckpt(tmp_path, {"convnext_version": "v1", "variant": "base"})
        with pytest.raises(ValueError, match=r"data_range"):
            DenoiserPrior.from_pretrained(str(ckpt))

    def test_legacy_data_range_value_is_refused(self, tmp_path):
        ckpt = self._fake_ckpt(tmp_path, {"data_range": "[-0.5,0.5]"})
        with pytest.raises(ValueError, match=r"REFUSING"):
            DenoiserPrior.from_pretrained(str(ckpt))

    def test_missing_config_json_is_refused(self, tmp_path):
        ckpt = self._fake_ckpt(tmp_path, None)
        with pytest.raises(ValueError, match=r"missing or unreadable"):
            DenoiserPrior.from_pretrained(str(ckpt))

    def test_refusal_message_names_the_checkpoint_and_the_retrain(self, tmp_path):
        ckpt = self._fake_ckpt(tmp_path, {"variant": "base"})
        with pytest.raises(ValueError) as exc:
            DenoiserPrior.from_pretrained(str(ckpt))
        msg = str(exc.value)
        assert str(tmp_path) in msg           # names the offending checkpoint
        assert "[-0.5,+0.5]" in msg           # names the legacy domain
        assert "RETRAIN" in msg.upper()       # states the remedy

    def test_stamped_checkpoint_passes_the_gate(self, tmp_path):
        # The gate itself must ACCEPT a [0,1] stamp — it then falls through to the real
        # loader, which fails on our dummy file. Any error here must NOT be the gate's.
        ckpt = self._fake_ckpt(tmp_path, {"data_range": "[0,1]"})
        with pytest.raises(Exception) as exc:  # noqa: PT011 - keras raises its own type
            DenoiserPrior.from_pretrained(str(ckpt))
        assert "REFUSING" not in str(exc.value)

    def test_direct_construction_bypasses_the_gate(self):
        # The fast unit tests (and any caller holding a model) must stay unaffected.
        assert DenoiserPrior(_tiny_bias_free_model()).model is not None


class TestTiling:
    def test_tile_untile_roundtrip_512(self):
        img = np.random.uniform(0.0, 1.0, size=(1, 512, 512, 3)).astype(np.float32)
        tiles, meta = DenoiserPrior.tile(img, tile_size=256)
        # 512 / 256 = 2 per axis -> 4 non-overlapping blocks.
        assert tiles.shape == (4, 256, 256, 3)
        assert (meta["nh"], meta["nw"]) == (2, 2)
        recon = DenoiserPrior.untile(tiles, meta)
        assert recon.shape == img.shape
        np.testing.assert_array_equal(recon, img)

    def test_tile_untile_roundtrip_with_padding(self):
        # 300 is not a multiple of 256 -> padded to 512, cropped back exactly.
        img = np.random.uniform(0.0, 1.0, size=(2, 300, 256, 3)).astype(np.float32)
        tiles, meta = DenoiserPrior.tile(img, tile_size=256)
        assert tiles.shape == (2 * 2 * 1, 256, 256, 3)
        recon = DenoiserPrior.untile(tiles, meta)
        assert recon.shape == img.shape
        np.testing.assert_array_equal(recon, img)

    # ---- The PAD VALUE itself (D-001). A shape-only round-trip test is blind here:
    # untile() crops the pad away, so a BLACK pad round-trips perfectly while still
    # bleeding a full-contrast artificial edge into the kept region through the
    # denoiser's receptive field. These two tests assert the pad CONTENT.

    def test_tile_padding_injects_no_black_border(self):
        # A pure-WHITE image. Under [0,1], numpy's default constant pad (0.0) is BLACK:
        # the padded tile rows would drop to 0.0, a full-contrast step edge. A neutral
        # pad of ANY kind (reflect, or 0.5) keeps every pixel at 1.0 / >= 0.5.
        img = np.ones((1, 300, 256, 3), dtype=np.float32)
        tiles, _ = DenoiserPrior.tile(img, tile_size=256)
        assert float(tiles.min()) == pytest.approx(1.0), (
            "tile() injected a darker-than-white pad into a pure-white image — on [0,1] "
            "a zero/constant pad is BLACK, not the neutral mid-grey it was on [-0.5,+0.5]"
        )

    def test_tile_padding_is_a_reflection_of_the_content(self):
        # Stronger than "not black": the pad must be a MIRROR of the image, i.e. no
        # artificial edge at all. Reconstruct the padded canvas from the tiles and check
        # the first padded row against the reflected source row.
        img = np.random.uniform(0.0, 1.0, size=(1, 300, 256, 3)).astype(np.float32)
        tiles, meta = DenoiserPrior.tile(img, tile_size=256)
        assert (meta["nh"], meta["nw"]) == (2, 1)
        # nh=2, nw=1 -> tile 0 is rows 0..255, tile 1 is rows 256..511 of the padded canvas.
        canvas = np.concatenate([tiles[0], tiles[1]], axis=0)  # [512, 256, 3]
        np.testing.assert_allclose(canvas[:300], img[0], rtol=0, atol=0)
        # np.pad(mode="reflect"): canvas[299 + k] == img[299 - k].
        for k in (1, 2, 5, 50):
            np.testing.assert_allclose(
                canvas[299 + k], img[0, 299 - k], rtol=0, atol=0,
                err_msg=f"padded row {299 + k} is not the reflection of source row {299 - k}",
            )
        # And no value outside the source range was invented (a constant pad would).
        assert float(canvas.min()) >= float(img.min())
        assert float(canvas.max()) <= float(img.max())

    def test_tile_padding_survives_an_image_smaller_than_one_tile(self):
        # Edge case for mode="reflect": pad width (192) EXCEEDS the dimension (64).
        # NumPy chains reflections rather than raising, so this must just work.
        img = np.random.uniform(0.0, 1.0, size=(1, 64, 100, 3)).astype(np.float32)
        tiles, meta = DenoiserPrior.tile(img, tile_size=256)
        assert tiles.shape == (1, 256, 256, 3)
        assert float(tiles.min()) >= float(img.min())
        assert float(tiles.max()) <= float(img.max())
        np.testing.assert_array_equal(DenoiserPrior.untile(tiles, meta), img)
