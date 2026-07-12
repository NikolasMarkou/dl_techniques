"""Fast CPU unit tests for the ``DenoiserPrior`` core wrapper.

Uses a TINY in-memory bias-free Keras model (no checkpoint, no GPU) to verify the
wrapper's mechanics: ``residual`` / ``denoise`` symbolic identities, ``ingest`` /
``denorm`` domain math, and the non-overlapping ``tile`` / ``untile`` round-trip.
The real 22 MB checkpoint is exercised separately (step 6 / drift-guard script),
never in this fast suite.
"""

import json

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
