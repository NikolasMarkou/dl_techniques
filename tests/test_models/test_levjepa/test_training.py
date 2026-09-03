"""Tests for ``LeVJEPATrainingModel`` (forward shape/finiteness, EMA shadow
wiring, serialization round trip)."""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.levjepa.encoder import LeVJEPAEncoder
from dl_techniques.models.vision.levjepa.training import LeVJEPATrainingModel


def _make_model(**kwargs):
    encoder = LeVJEPAEncoder(
        input_shape=(16, 16, 3),
        num_frames=4,
        patch_size=16,
        tubelet_size=2,
        embed_dim=32,
        depth=1,
        num_heads=2,
    )
    defaults = dict(sigreg_num_proj=16, sigreg_knots=5, sigreg_weight=0.02)
    defaults.update(kwargs)
    return LeVJEPATrainingModel(encoder=encoder, **defaults)


def _make_batch(batch_size=2, num_local=2, num_frames=4, size=16, channels=3):
    return {
        "global_frame": keras.random.normal((batch_size, num_frames, size, size, channels)),
        "local_frames": keras.random.normal(
            (batch_size, num_local, num_frames, size, size, channels)
        ),
    }


class TestLeVJEPATrainingModelForward:
    def test_forward_output_shape(self):
        model = _make_model()
        batch = _make_batch(batch_size=2, num_local=2)
        out = model(batch, training=True)
        assert out.shape == (2, 3, 32)  # 1 global + 2 local views

    def test_forward_output_finite(self):
        model = _make_model()
        batch = _make_batch(batch_size=2, num_local=2)
        out = np.array(model(batch, training=True))
        assert np.all(np.isfinite(out))

    def test_losses_are_finite_and_registered(self):
        model = _make_model()
        batch = _make_batch(batch_size=2, num_local=2)
        model(batch, training=True)
        assert len(model.losses) == 2
        for loss in model.losses:
            assert np.isfinite(np.array(loss))

    # DECISION plan-2026-09-03T113223-2a714a91/D-024
    # The prior version of this test computed `global_emb - embeddings[:, :1, :]`
    # where BOTH operands were the exact same slice of the exact same tensor
    # (`a - a == 0`), a guard that cannot fail regardless of whether call()'s
    # global/local indexing is correct. WHAT NOT TO DO: do not "fix" this by
    # merely renaming the variables or re-deriving the SAME slice a second way
    # from the SAME `embeddings` tensor -- that reproduces the identical
    # tautology under a different name. The real fix independently recomputes
    # the global row via a SEPARATE forward path (encoder -> CLS slice ->
    # projector, called on the global view ALONE) and compares it against
    # `embeddings[:, 0, :]` from the model's own joint forward pass, using
    # `training=False` so BatchNormalization's moving statistics (not batch
    # statistics, which would legitimately differ between a 1-view and a
    # 1+V-view call) make the two paths comparable. See decisions.md D-024.
    def test_global_embedding_matches_independent_recomputation(self):
        """``embeddings[:, 0, :]`` (the global view's row produced by
        ``call()``'s joint global+local forward pass) must equal an
        INDEPENDENTLY recomputed projection of the global CLS token alone --
        a real cross-check on ``call()``'s global/local indexing.

        Prior version of this test computed ``global_emb - embeddings[:, :1, :]``
        where both operands were literally the SAME slice of the SAME tensor
        (``a - a == 0``), an assertion that cannot fail regardless of whether
        ``call()``'s indexing is correct -- see decisions.md D-024.

        Uses ``training=False`` throughout: ``LeVJEPAProjector``'s
        ``BatchNormalization`` then uses its moving statistics, which do not
        depend on how many views are present in a given forward call, so the
        global row computed jointly (alongside the local views) must exactly
        match the global row computed alone from the encoder's own CLS token.
        This recomputation would diverge from ``embeddings[:, 0, :]`` if
        ``call()`` ever mixed up which view is "global" (e.g. wrong slot in
        the ``concatenate`` call, or an off-by-one on the view axis) --
        confirmed by injecting exactly that mutation (swapping the
        ``concatenate`` operand order) and observing this assertion go RED,
        then reverting and confirming GREEN again.
        """
        model = _make_model()
        batch = _make_batch(batch_size=2, num_local=2)
        # One training=True call to build the model and take a real moving-
        # stat update, so `training=False` below exercises actual (non-
        # freshly-initialized) BatchNormalization statistics.
        model(batch, training=True)
        embeddings = np.array(model(batch, training=False))

        global_tokens = model.encoder(batch["global_frame"], training=False)
        global_cls = global_tokens[:, :1, :]  # (B, 1, D), independent path
        expected_global_emb = np.array(model.projector(global_cls, training=False))

        np.testing.assert_allclose(
            embeddings[:, 0, :], expected_global_emb[:, 0, :], atol=1e-5, rtol=0
        )

    def test_requires_video_mode_encoder(self):
        image_encoder = LeVJEPAEncoder(
            input_shape=(16, 16, 3), num_frames=1, patch_size=16,
            embed_dim=32, depth=1, num_heads=2,
        )
        with pytest.raises(ValueError, match="video"):
            LeVJEPATrainingModel(encoder=image_encoder)

    def test_metrics_expose_pred_and_sigreg_trackers(self):
        model = _make_model()
        batch = _make_batch(batch_size=2, num_local=1)
        model(batch, training=True)
        names = {m.name for m in model.metrics}
        assert "pred_loss" in names
        assert "sigreg_loss" in names


class TestLeVJEPATrainingModelEMAShadow:
    def test_update_ema_shadow_moves_shadow_weights(self):
        model = _make_model()
        batch = _make_batch(batch_size=2, num_local=1)
        model(batch, training=True)  # build

        before = [np.array(w) for w in model._ema_shadow_weights]
        # Perturb the live encoder weights so the shadow update has something
        # to move toward.
        for w in model.encoder.weights:
            w.assign(w + 1.0)

        model.update_ema_shadow(decay=0.5)
        after = [np.array(w) for w in model._ema_shadow_weights]

        moved = any(
            not np.allclose(b, a, atol=1e-8) for b, a in zip(before, after)
        )
        assert moved

    def test_ema_shadow_count_matches_encoder_weight_count(self):
        model = _make_model()
        batch = _make_batch(batch_size=2, num_local=1)
        model(batch, training=True)
        assert len(model._ema_shadow_weights) == len(model.encoder.weights)


class TestLeVJEPATrainingModelSerialization:
    def test_get_config_from_config_round_trip(self):
        model = _make_model()
        batch = _make_batch(batch_size=2, num_local=1)
        model(batch, training=True)

        config = model.get_config()
        restored = LeVJEPATrainingModel.from_config(config)

        assert restored.sigreg_weight == model.sigreg_weight
        assert restored.encoder.embed_dim == model.encoder.embed_dim
        assert restored.projector.hidden_dim == model.projector.hidden_dim

    def test_full_model_save_load_round_trip(self, tmp_path):
        model = _make_model()
        batch = _make_batch(batch_size=2, num_local=1)
        y_before = np.array(model(batch, training=False))

        save_path = tmp_path / "levjepa_training.keras"
        model.save(save_path)
        loaded = keras.models.load_model(save_path)
        y_after = np.array(loaded(batch, training=False))

        np.testing.assert_allclose(y_before, y_after, atol=1e-4, rtol=0)
