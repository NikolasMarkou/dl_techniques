"""Video-JEPA-Clifford test suite.

Iter-3 (D-013): telemetry conditioning removed. ``TestTelemetryEmbedder``
deleted. ``TestConfig::test_cond_dim_must_equal_embed_dim`` deleted.
``test_adaln_identity_init`` renamed to ``test_predictor_identity_init``
and reframed against the new ``CausalSelfAttnMLPBlock`` (LayerScale
γ=1e-5 ⇒ near-identity at init).

Hardest-first order:

1. Causality (C1) — ``TestPredictor::test_causality``
2. SIGReg stability (C2) — ``TestPredictor::test_sigreg_finite``
3. Predictor identity-at-init (C3) — ``TestPredictor::test_predictor_identity_init``
4. Serialization round-trip (C4) — on encoder, predictor, top-level VideoJEPA.
5. Shapes (C5) — on encoder, predictor, model.
6. Streaming O(1) (C6) — ``TestVideoJEPA::test_stream_step_timing``.
"""

from __future__ import annotations

import numpy as np
import pytest
import keras

from dl_techniques.models.video_jepa.config import VideoJEPAConfig
from dl_techniques.models.video_jepa.masking import TubeMaskGenerator
from dl_techniques.models.video_jepa.encoder import VideoJEPACliffordEncoder
from dl_techniques.models.video_jepa.predictor import VideoJEPAPredictor
from dl_techniques.models.video_jepa.model import VideoJEPA
from dl_techniques.regularizers.sigreg import SIGRegLayer


class TestConfig:
    """Validate :class:`VideoJEPAConfig` invariants and round-trip."""

    def test_defaults_construct(self) -> None:
        cfg = VideoJEPAConfig()
        assert cfg.img_size == 64
        assert cfg.patch_size == 8
        assert cfg.patches_per_side == 8
        assert cfg.num_patches == 64
        assert cfg.input_image_shape == (64, 64, 3)

    def test_to_from_dict_round_trip(self) -> None:
        cfg = VideoJEPAConfig()
        d = cfg.to_dict()
        # tuples must survive as lists through dict → from_dict and come
        # back as tuples on the reconstructed config.
        assert isinstance(d["encoder_shifts"], list)
        assert isinstance(d["predictor_shifts"], list)
        cfg2 = VideoJEPAConfig.from_dict(d)
        assert cfg2 == cfg
        assert isinstance(cfg2.encoder_shifts, tuple)
        assert isinstance(cfg2.predictor_shifts, tuple)

    def test_custom_fields_survive(self) -> None:
        cfg = VideoJEPAConfig(
            img_size=32, patch_size=8, embed_dim=32,
            num_frames=3, history_size_k=3, predictor_depth=1,
            encoder_clifford_depth=1, sigreg_num_proj=8,
        )
        cfg2 = VideoJEPAConfig.from_dict(cfg.to_dict())
        assert cfg2.img_size == 32
        assert cfg2.patch_size == 8
        assert cfg2.num_frames == 3
        assert cfg2.sigreg_num_proj == 8

    def test_img_size_divisible_by_patch_size(self) -> None:
        with pytest.raises(ValueError, match="divisible by patch_size"):
            VideoJEPAConfig(img_size=65, patch_size=8)

    def test_positive_integer_guards(self) -> None:
        with pytest.raises(ValueError, match="history_size_k"):
            VideoJEPAConfig(history_size_k=0)
        with pytest.raises(ValueError, match="num_frames"):
            VideoJEPAConfig(num_frames=0)
        with pytest.raises(ValueError, match="encoder_clifford_depth"):
            VideoJEPAConfig(encoder_clifford_depth=0)
        with pytest.raises(ValueError, match="predictor_depth"):
            VideoJEPAConfig(predictor_depth=0)

    def test_legacy_telemetry_keys_ignored(self) -> None:
        """``from_dict`` must tolerate legacy ``cond_dim`` / ``telemetry_dim``
        keys in checkpoints written before iter-3 (D-013)."""
        d = VideoJEPAConfig().to_dict()
        d["cond_dim"] = 64
        d["telemetry_dim"] = 7
        cfg = VideoJEPAConfig.from_dict(d)
        assert cfg == VideoJEPAConfig()

    # ------------------------------------------------------------------
    # Iter-2: V-JEPA tube-masked prediction fields (D-008..D-012)
    # ------------------------------------------------------------------
    def test_iter2_defaults(self) -> None:
        """Iter-2 defaults: masking enabled, ratio=0.6, equal λ weights."""
        cfg = VideoJEPAConfig()
        assert cfg.mask_prediction_enabled is True
        assert cfg.mask_ratio == 0.6
        assert cfg.lambda_next_frame == 1.0
        assert cfg.lambda_mask == 1.0

    def test_iter2_fields_round_trip(self) -> None:
        cfg = VideoJEPAConfig(
            mask_prediction_enabled=False,
            mask_ratio=0.75,
            lambda_next_frame=0.5,
            lambda_mask=2.0,
        )
        cfg2 = VideoJEPAConfig.from_dict(cfg.to_dict())
        assert cfg2 == cfg
        assert cfg2.mask_prediction_enabled is False
        assert cfg2.mask_ratio == 0.75
        assert cfg2.lambda_next_frame == 0.5
        assert cfg2.lambda_mask == 2.0

    def test_iter2_mask_ratio_bounds(self) -> None:
        # Strict upper bound at 1.0.
        with pytest.raises(ValueError, match="mask_ratio"):
            VideoJEPAConfig(mask_ratio=1.0)
        with pytest.raises(ValueError, match="mask_ratio"):
            VideoJEPAConfig(mask_ratio=-0.1)
        # Edge: zero is allowed (regression-guard path).
        cfg = VideoJEPAConfig(mask_ratio=0.0)
        assert cfg.mask_ratio == 0.0

    def test_iter2_lambda_non_negative(self) -> None:
        with pytest.raises(ValueError, match="lambda_next_frame"):
            VideoJEPAConfig(lambda_next_frame=-0.01)
        with pytest.raises(ValueError, match="lambda_mask"):
            VideoJEPAConfig(lambda_mask=-0.01)


# ============================================================================
# TestTubeMaskGenerator — iter-2 (unchanged by iter-3).
# ============================================================================


class TestTubeMaskGenerator:
    """Unit tests for :class:`TubeMaskGenerator` (iter-2, C8/C9/C12)."""

    def test_mask_ratio_exact(self) -> None:
        """C8 — exact per-row cardinality across 50 random batches+seeds."""
        H_p = 8
        mask_ratio = 0.6
        expected_K = round(mask_ratio * H_p * H_p)  # = 38
        gen = TubeMaskGenerator(mask_ratio=mask_ratio, patches_per_side=H_p)

        for trial in range(50):
            B = 2 + (trial % 5)
            keras.utils.set_random_seed(trial)
            mask = np.asarray(gen(B))
            assert mask.shape == (B, H_p, H_p), mask.shape
            assert mask.dtype == np.float32
            unique = np.unique(mask)
            assert set(unique.tolist()) <= {0.0, 1.0}, unique
            per_row = mask.reshape(B, -1).sum(axis=-1)
            assert np.all(per_row == expected_K), (
                f"trial={trial}, per_row={per_row}, expected {expected_K}"
            )

    def test_per_sample_independence(self) -> None:
        gen = TubeMaskGenerator(mask_ratio=0.6, patches_per_side=8)
        keras.utils.set_random_seed(0)
        mask = np.asarray(gen(8))
        pairwise_eq = []
        for i in range(mask.shape[0]):
            for j in range(i + 1, mask.shape[0]):
                pairwise_eq.append(np.array_equal(mask[i], mask[j]))
        assert not all(pairwise_eq), (
            "All masks identical across batch — likely missing per-row sampling."
        )

    def test_tube_structure_by_broadcast(self) -> None:
        gen = TubeMaskGenerator(mask_ratio=0.5, patches_per_side=4)
        keras.utils.set_random_seed(1)
        mask = np.asarray(gen(2))
        T = 5
        broadcast = np.broadcast_to(mask[:, None, :, :], (2, T, 4, 4))
        for t in range(T):
            np.testing.assert_array_equal(broadcast[:, t], mask)

    def test_ratio_zero_all_visible(self) -> None:
        gen = TubeMaskGenerator(mask_ratio=0.0, patches_per_side=8)
        mask = np.asarray(gen(4))
        assert np.all(mask == 0.0)
        assert mask.shape == (4, 8, 8)
        assert gen.num_masked == 0

    def test_ratio_high(self) -> None:
        gen = TubeMaskGenerator(mask_ratio=0.75, patches_per_side=4)
        keras.utils.set_random_seed(2)
        mask = np.asarray(gen(3))
        per_row = mask.reshape(3, -1).sum(axis=-1)
        assert np.all(per_row == 12), per_row

    def test_serialization_round_trip(self) -> None:
        gen = TubeMaskGenerator(mask_ratio=0.6, patches_per_side=8)
        config = gen.get_config()
        gen2 = TubeMaskGenerator.from_config(config)
        assert gen2.mask_ratio == gen.mask_ratio
        assert gen2.patches_per_side == gen.patches_per_side
        assert gen2.num_masked == gen.num_masked

    def test_rejects_bad_args(self) -> None:
        with pytest.raises(ValueError, match="mask_ratio"):
            TubeMaskGenerator(mask_ratio=-0.01, patches_per_side=8)
        with pytest.raises(ValueError, match="mask_ratio"):
            TubeMaskGenerator(mask_ratio=1.01, patches_per_side=8)
        with pytest.raises(ValueError, match="patches_per_side"):
            TubeMaskGenerator(mask_ratio=0.5, patches_per_side=0)


# ============================================================================
# TestEncoder — hybrid PatchEmbedding2D + PE2D + CliffordNetBlock stack (C5)
# ============================================================================


def _default_encoder_kwargs() -> dict:
    return dict(
        embed_dim=32, patch_size=8, img_size=32, img_channels=3,
        depth=1, shifts=(1, 2), dropout=0.0,
    )


class TestEncoder:
    """Shape + serialization tests for :class:`VideoJEPACliffordEncoder`."""

    def test_forward_shape(self) -> None:
        enc = VideoJEPACliffordEncoder(**_default_encoder_kwargs())
        B_total, H, W, C = 4, 32, 32, 3
        x = np.random.rand(B_total, H, W, C).astype("float32")
        y = enc(x, training=False)
        Hp = 32 // 8
        assert tuple(y.shape) == (B_total, Hp, Hp, 32), y.shape

    def test_forward_preserves_finiteness(self) -> None:
        enc = VideoJEPACliffordEncoder(**_default_encoder_kwargs())
        x = np.random.rand(4, 32, 32, 3).astype("float32")
        y = np.asarray(enc(x, training=False))
        assert np.all(np.isfinite(y))

    def test_rejects_bad_rank(self) -> None:
        enc = VideoJEPACliffordEncoder(**_default_encoder_kwargs())
        with pytest.raises(ValueError, match="4D input"):
            enc.build((4, 32, 3))

    def test_even_embed_dim_enforced(self) -> None:
        with pytest.raises(ValueError, match="even"):
            VideoJEPACliffordEncoder(
                embed_dim=31, patch_size=8, img_size=32, img_channels=3,
            )

    def test_serialization_round_trip(self, tmp_path) -> None:
        kwargs = _default_encoder_kwargs()
        inputs = keras.Input(shape=(32, 32, 3))
        enc = VideoJEPACliffordEncoder(**kwargs, name="enc")
        outputs = enc(inputs)
        model = keras.Model(inputs, outputs, name="enc_wrap")

        x = np.random.rand(4, 32, 32, 3).astype("float32")
        y_before = np.asarray(model(x, training=False))

        path = str(tmp_path / "enc.keras")
        model.save(path)
        del model, enc
        keras.backend.clear_session()
        reloaded = keras.models.load_model(path)
        y_after = np.asarray(reloaded(x, training=False))

        np.testing.assert_allclose(y_after, y_before, atol=1e-5, rtol=1e-5)


# ============================================================================
# TestPredictor — HARDEST FIRST (C1 causality, C3 identity-at-init,
#                                C5 shapes, C2 SIGReg finiteness, C4 serialize)
# ============================================================================


def _make_predictor(
    *, embed_dim: int = 32, T: int = 4, Hp: int = 4, depth: int = 2,
    num_heads: int = 2, dim_head: int = 16, mlp_dim: int = 64,
) -> VideoJEPAPredictor:
    pred = VideoJEPAPredictor(
        embed_dim=embed_dim,
        num_frames_max=T,
        patches_per_side=Hp,
        depth=depth,
        num_heads=num_heads,
        dim_head=dim_head,
        mlp_dim=mlp_dim,
        shifts=(1, 2),
        dropout=0.0,
        name="pred",
    )
    pred.build((2, T, Hp, Hp, embed_dim))
    return pred


class TestPredictor:
    """Predictor invariants. Causality runs first (highest-risk assertion)."""

    # ------------------------------------------------------------------
    # C1 — CAUSALITY (the critical test — runs first)
    # ------------------------------------------------------------------
    def test_causality(self) -> None:
        """Perturbation at frame k must not alter outputs at frames < k."""
        np.random.seed(0)
        keras.utils.set_random_seed(0)

        D, T, Hp, B = 32, 4, 4, 2
        pred = _make_predictor(embed_dim=D, T=T, Hp=Hp, depth=2)

        z_base = np.random.randn(B, T, Hp, Hp, D).astype("float32")

        out_base = np.asarray(pred(z_base, training=False))
        assert out_base.shape == (B, T, Hp, Hp, D)

        for k in range(1, T):
            z_pert = z_base.copy()
            z_pert[:, k:] += np.random.randn(B, T - k, Hp, Hp, D).astype(
                "float32"
            ) * 5.0
            out_pert = np.asarray(pred(z_pert, training=False))

            # Frames < k must be identical to the unperturbed baseline.
            diff = np.max(np.abs(
                out_base[:, :k] - out_pert[:, :k]
            ))
            assert diff < 1e-5, (
                f"Causality violated at perturbation frame k={k}: "
                f"max|Δ| on frames < k = {diff} (tol 1e-5)."
            )

            # Sanity: frames >= k SHOULD differ.
            diff_future = np.max(np.abs(
                out_base[:, k:] - out_pert[:, k:]
            ))
            assert diff_future > 1e-6, (
                f"Predictor appears collapsed at frame k={k}: "
                f"future-frame diff = {diff_future}."
            )

    # ------------------------------------------------------------------
    # C3 — Predictor identity-at-init (replaces AdaLN-zero identity test)
    # ------------------------------------------------------------------
    def test_predictor_identity_init(self) -> None:
        """At init, predictor(z) ≈ z + pos_t (near-identity).

        The new ``CausalSelfAttnMLPBlock`` scales its attn and MLP residual
        paths by LayerScale γ=1e-5, and ``CausalCliffordNetBlock`` +
        ``CliffordNetBlock`` use the same γ=1e-5 trick. So at init the
        predictor is ``z + pos_t + eps`` where ``eps`` comes entirely from
        these γ-scaled residuals.
        """
        np.random.seed(1)
        keras.utils.set_random_seed(1)

        D, T, Hp, B = 32, 4, 4, 2
        pred = _make_predictor(embed_dim=D, T=T, Hp=Hp, depth=2)
        z = np.random.randn(B, T, Hp, Hp, D).astype("float32")
        out = np.asarray(pred(z, training=False))

        # Expected: out ≈ z + pos_t[:T] (broadcast over spatial).
        pos_t = np.asarray(pred.pos_t)[:, :T, :]               # (1, T, D)
        pos_t_bcast = pos_t.reshape(1, T, 1, 1, D)
        expected = z + pos_t_bcast
        diff = np.max(np.abs(out - expected))
        # LayerScale γ=1e-5 on several residual paths ⇒ tolerance ~5e-3.
        assert diff < 5e-3, (
            f"Predictor not near-identity at init: max|out - (z+pos_t)| = "
            f"{diff} (tol 5e-3)."
        )

    # ------------------------------------------------------------------
    # C5 — Shape propagation
    # ------------------------------------------------------------------
    def test_forward_shape(self) -> None:
        D, T, Hp, B = 32, 4, 4, 2
        pred = _make_predictor(embed_dim=D, T=T, Hp=Hp, depth=2)
        z = np.random.randn(B, T, Hp, Hp, D).astype("float32")
        y = pred(z, training=False)
        assert tuple(y.shape) == (B, T, Hp, Hp, D), y.shape

    def test_rejects_list_input(self) -> None:
        pred = _make_predictor()
        z = np.random.randn(2, 4, 4, 4, 32).astype("float32")
        c = np.random.randn(2, 4, 32).astype("float32")
        with pytest.raises(ValueError, match="single tensor"):
            pred([z, c])

    # ------------------------------------------------------------------
    # C2 — SIGReg finiteness smoke (reshape to (B*T, N, D))
    # ------------------------------------------------------------------
    def test_sigreg_finite(self) -> None:
        """SIGReg on reshape-to-(B*T, N, D) must be finite at init."""
        D, T, Hp, B = 32, 4, 4, 2
        pred = _make_predictor(embed_dim=D, T=T, Hp=Hp, depth=2)
        z = np.random.randn(B, T, Hp, Hp, D).astype("float32")
        y = pred(z, training=False)
        y_r = np.asarray(y).reshape(B * T, Hp * Hp, D)

        sig = SIGRegLayer(knots=17, num_proj=64, name="sigreg_smoke")
        loss = sig(y_r)
        loss_val = float(np.asarray(loss))
        assert np.isfinite(loss_val), f"SIGReg non-finite at init: {loss_val}"
        assert loss_val < 100.0, f"SIGReg > 100 at init: {loss_val}"

    # ------------------------------------------------------------------
    # C4 — Serialization round-trip
    # ------------------------------------------------------------------
    def test_serialization_round_trip(self, tmp_path) -> None:
        D, T, Hp, B = 32, 4, 4, 2
        z_in = keras.Input(shape=(T, Hp, Hp, D), name="z")
        pred = VideoJEPAPredictor(
            embed_dim=D, num_frames_max=T, patches_per_side=Hp,
            depth=2, num_heads=2, dim_head=16, mlp_dim=64,
            shifts=(1, 2), dropout=0.0, name="pred",
        )
        out = pred(z_in)
        model = keras.Model(z_in, out, name="pred_wrap")

        z = np.random.randn(B, T, Hp, Hp, D).astype("float32")
        y_before = np.asarray(model(z, training=False))

        path = str(tmp_path / "pred.keras")
        model.save(path)
        del model, pred
        keras.backend.clear_session()
        reloaded = keras.models.load_model(path)
        y_after = np.asarray(reloaded(z, training=False))
        np.testing.assert_allclose(y_after, y_before, atol=1e-5, rtol=1e-5)


# ============================================================================
# TestVideoJEPA — top-level model: forward, save/load, streaming, T=1 edge
# ============================================================================


def _small_config(**overrides) -> VideoJEPAConfig:
    defaults = dict(
        img_size=32, img_channels=3, patch_size=8, embed_dim=32,
        num_frames=4, history_size_k=4,
        encoder_clifford_depth=1, encoder_shifts=(1, 2),
        predictor_depth=1, predictor_num_heads=2, predictor_dim_head=16,
        predictor_mlp_dim=64, predictor_shifts=(1, 2),
        sigreg_knots=17, sigreg_num_proj=8, sigreg_weight=0.09,
        dropout=0.0,
    )
    defaults.update(overrides)
    return VideoJEPAConfig(**defaults)


class TestVideoJEPA:
    def test_forward_shape_and_losses(self) -> None:
        cfg = _small_config()
        model = VideoJEPA(config=cfg)
        B, T = 2, cfg.num_frames
        pixels = np.random.rand(B, T, cfg.img_size, cfg.img_size,
                                cfg.img_channels).astype("float32")
        pred = model({"pixels": pixels}, training=False)
        Hp = cfg.patches_per_side
        assert tuple(pred.shape) == (B, T, Hp, Hp, cfg.embed_dim), pred.shape

        assert len(model.losses) >= 1
        for loss in model.losses:
            assert np.isfinite(float(np.asarray(loss)))

    def test_forward_t1_edge(self) -> None:
        """T=1 (iter-1 semantics, masking off): MSE next-frame loss skipped."""
        cfg = _small_config(
            num_frames=1, history_size_k=1, mask_prediction_enabled=False,
        )
        model = VideoJEPA(config=cfg)
        pixels = np.random.rand(2, 1, cfg.img_size, cfg.img_size,
                                cfg.img_channels).astype("float32")
        pred = model({"pixels": pixels}, training=False)
        assert tuple(pred.shape) == (
            2, 1, cfg.patches_per_side, cfg.patches_per_side, cfg.embed_dim,
        )
        assert len(model.losses) == 1
        assert np.isfinite(float(np.asarray(model.losses[0])))

    def test_rejects_missing_pixels_key(self) -> None:
        """Input-dict guard: missing 'pixels' key raises ValueError."""
        cfg = _small_config()
        model = VideoJEPA(config=cfg)
        with pytest.raises(ValueError, match="pixels"):
            model({"not_pixels": np.zeros(
                (2, cfg.num_frames, cfg.img_size, cfg.img_size,
                 cfg.img_channels), dtype="float32",
            )}, training=False)

    def test_fit_one_step_no_nan(self) -> None:
        """Smoke: one .fit step produces finite losses and no NaN."""
        cfg = _small_config()
        model = VideoJEPA(config=cfg)
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
            loss=None, jit_compile=False,
        )
        B, T = 2, cfg.num_frames

        def gen():
            for _ in range(2):
                yield (
                    {
                        "pixels": np.random.rand(
                            B, T, cfg.img_size, cfg.img_size, cfg.img_channels
                        ).astype("float32"),
                    },
                    np.float32(0.0),
                )

        import tensorflow as tf
        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                {
                    "pixels": tf.TensorSpec(
                        shape=(B, T, cfg.img_size, cfg.img_size,
                               cfg.img_channels), dtype=tf.float32,
                    ),
                },
                tf.TensorSpec(shape=(), dtype=tf.float32),
            ),
        )
        h = model.fit(ds, epochs=1, steps_per_epoch=2, verbose=0)
        loss_val = h.history["loss"][-1]
        assert np.isfinite(loss_val), f"Training loss non-finite: {loss_val}"

    def test_save_load_round_trip(self, tmp_path) -> None:
        """Iter-1 deterministic round-trip (masking off)."""
        cfg = _small_config(mask_prediction_enabled=False)
        model = VideoJEPA(config=cfg)
        B, T = 2, cfg.num_frames
        pixels = np.random.rand(
            B, T, cfg.img_size, cfg.img_size, cfg.img_channels
        ).astype("float32")

        y_before = np.asarray(
            model({"pixels": pixels}, training=False)
        )

        path = str(tmp_path / "vj.keras")
        model.save(path)
        del model
        keras.backend.clear_session()
        reloaded = keras.models.load_model(path)
        y_after = np.asarray(
            reloaded({"pixels": pixels}, training=False)
        )
        np.testing.assert_allclose(y_after, y_before, atol=1e-5, rtol=1e-5)

    def test_stream_step_shape(self) -> None:
        """stream_step emits (B, H_p, W_p, D) per call."""
        cfg = _small_config()
        model = VideoJEPA(config=cfg)
        _ = model({
            "pixels": np.random.rand(
                2, cfg.num_frames, cfg.img_size, cfg.img_size,
                cfg.img_channels,
            ).astype("float32"),
        }, training=False)

        model.stream_reset(B=2)
        for _ in range(cfg.history_size_k + 2):
            frame = np.random.rand(
                2, cfg.img_size, cfg.img_size, cfg.img_channels,
            ).astype("float32")
            out = model.stream_step(frame)
            Hp = cfg.patches_per_side
            assert tuple(out.shape) == (2, Hp, Hp, cfg.embed_dim), out.shape
            assert np.all(np.isfinite(np.asarray(out)))

    def test_stream_step_timing(self) -> None:
        """Rough O(1)-per-step check: frames K..2K within ±60% of K..K+5."""
        import time
        cfg = _small_config()
        model = VideoJEPA(config=cfg)
        _ = model({
            "pixels": np.random.rand(
                2, cfg.num_frames, cfg.img_size, cfg.img_size,
                cfg.img_channels,
            ).astype("float32"),
        }, training=False)

        K = cfg.history_size_k
        model.stream_reset(B=2)
        timings = []
        for i in range(2 * K + 5):
            frame = np.random.rand(
                2, cfg.img_size, cfg.img_size, cfg.img_channels,
            ).astype("float32")
            t0 = time.perf_counter()
            _ = model.stream_step(frame)
            timings.append(time.perf_counter() - t0)

        fill = float(np.mean(timings[K : K + 5]))
        roll = float(np.mean(timings[2 * K : 2 * K + 5]))
        ratio = roll / fill if fill > 0 else float("inf")
        assert 0.4 < ratio < 2.5, (
            f"stream_step not O(1)-ish: fill={fill*1e3:.2f}ms, "
            f"roll={roll*1e3:.2f}ms, ratio={ratio:.2f}"
        )


# ============================================================================
# TestVideoJEPAIter2 — iter-2 mask-prediction branch (C10/C11/C12)
# ============================================================================


class TestVideoJEPAIter2:
    """Iter-2 additions: mask-loss finiteness, disabled-flag regression,
    serialization round-trip with masking on."""

    def test_mask_loss_finite(self) -> None:
        """C10 — with masking on, 3 finite losses appear via add_loss."""
        cfg = _small_config(mask_prediction_enabled=True, mask_ratio=0.6)
        model = VideoJEPA(config=cfg)
        B, T = 2, cfg.num_frames
        pixels = np.random.rand(B, T, cfg.img_size, cfg.img_size,
                                cfg.img_channels).astype("float32")
        _ = model({"pixels": pixels}, training=False)

        assert len(model.losses) == 3, (
            f"expected 3 losses (next-frame + mask + sigreg), "
            f"got {len(model.losses)}"
        )
        values = [float(np.asarray(x)) for x in model.losses]
        for v in values:
            assert np.isfinite(v), f"non-finite loss: {values}"
        assert values[1] > 1e-5, (
            f"mask loss should be > 1e-5 at init (mask_token zero-init "
            f"replaces z values at masked positions), got {values[1]}"
        )

    def test_mask_disabled_matches_iter1_behavior(self) -> None:
        """C11 — with masking off, len(losses)==2 (iter-1 shape). Outputs
        deterministic across two forwards (no random sampling)."""
        cfg = _small_config(mask_prediction_enabled=False)
        model = VideoJEPA(config=cfg)
        B, T = 2, cfg.num_frames
        pixels = np.random.rand(B, T, cfg.img_size, cfg.img_size,
                                cfg.img_channels).astype("float32")
        y1 = np.asarray(
            model({"pixels": pixels}, training=False)
        )
        assert len(model.losses) == 2, len(model.losses)
        y2 = np.asarray(
            model({"pixels": pixels}, training=False)
        )
        np.testing.assert_allclose(y2, y1, atol=1e-6, rtol=1e-6)

    def test_serialization_roundtrip_with_masking(self, tmp_path) -> None:
        """C12 — save/load with masking enabled."""
        cfg = _small_config(mask_prediction_enabled=True, mask_ratio=0.6)
        model = VideoJEPA(config=cfg)
        B, T = 2, cfg.num_frames
        pixels = np.random.rand(B, T, cfg.img_size, cfg.img_size,
                                cfg.img_channels).astype("float32")
        _ = model({"pixels": pixels}, training=False)

        before_weights = {
            w.name: np.asarray(w).copy() for w in model.weights
        }

        path = str(tmp_path / "vj_iter2.keras")
        model.save(path)
        del model
        keras.backend.clear_session()
        reloaded = keras.models.load_model(path)

        assert reloaded.config.mask_prediction_enabled is True
        assert reloaded.config.mask_ratio == 0.6

        after_weights = {w.name: np.asarray(w) for w in reloaded.weights}
        assert set(after_weights.keys()) == set(before_weights.keys()), (
            set(before_weights) ^ set(after_weights)
        )
        for name, v0 in before_weights.items():
            v1 = after_weights[name]
            np.testing.assert_allclose(
                v1, v0, atol=1e-7, rtol=1e-7,
                err_msg=f"weight mismatch after load: {name}",
            )

        y = np.asarray(
            reloaded({"pixels": pixels}, training=False)
        )
        Hp = cfg.patches_per_side
        assert y.shape == (B, T, Hp, Hp, cfg.embed_dim), y.shape
        assert np.all(np.isfinite(y))

    def test_fit_one_step_with_masking(self) -> None:
        """Smoke: one fit step with masking on, three finite losses."""
        cfg = _small_config(mask_prediction_enabled=True, mask_ratio=0.6)
        model = VideoJEPA(config=cfg)
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
            loss=None, jit_compile=False,
        )
        B, T = 2, cfg.num_frames

        def gen():
            for _ in range(2):
                yield (
                    {
                        "pixels": np.random.rand(
                            B, T, cfg.img_size, cfg.img_size, cfg.img_channels
                        ).astype("float32"),
                    },
                    np.float32(0.0),
                )

        import tensorflow as tf
        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                {
                    "pixels": tf.TensorSpec(
                        shape=(B, T, cfg.img_size, cfg.img_size,
                               cfg.img_channels), dtype=tf.float32,
                    ),
                },
                tf.TensorSpec(shape=(), dtype=tf.float32),
            ),
        )
        h = model.fit(ds, epochs=1, steps_per_epoch=2, verbose=0)
        loss_val = h.history["loss"][-1]
        assert np.isfinite(loss_val), (
            f"Training loss non-finite under masking: {loss_val}"
        )


# ============================================================================
# TestSyntheticDataset — shape + finiteness of synthetic_drone_video_dataset
# ============================================================================


class TestSyntheticDataset:
    def test_one_batch_shapes_and_finite(self) -> None:
        from dl_techniques.datasets.synthetic_drone_video import (
            synthetic_drone_video_dataset,
        )
        ds = synthetic_drone_video_dataset(
            batch_size=2, num_batches=2, T=4, img_size=32, img_channels=3,
            seed=0,
        )
        for inputs, y in ds.take(1):
            pixels = inputs["pixels"].numpy()
            assert pixels.shape == (2, 4, 32, 32, 3), pixels.shape
            assert np.all(np.isfinite(pixels))
            assert np.all((pixels >= 0.0) & (pixels <= 1.0))
            assert np.all(y.numpy() == 0.0)
            # Iter-3: 'telemetry' key must NOT be present.
            assert "telemetry" not in inputs

    def test_rejects_batch_size_1(self) -> None:
        from dl_techniques.datasets.synthetic_drone_video import (
            synthetic_drone_video_dataset,
        )
        with pytest.raises(ValueError, match="batch_size must be >= 2"):
            synthetic_drone_video_dataset(batch_size=1)
