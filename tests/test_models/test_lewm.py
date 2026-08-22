"""Tests for the LeWM model.

Covers:
- forward pass shapes + finite loss
- serialization round-trip
- rollout shape
- identity-at-init of the predictor (via LeWM forward path)
"""

import os

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.lewm.config import LeWMConfig
from dl_techniques.models.lewm.model import LeWM


def _small_cfg() -> LeWMConfig:
    """A small CPU-friendly config — still exercises every code path."""
    return LeWMConfig(
        img_size=56, patch_size=14, encoder_scale="tiny",
        embed_dim=192, projector_hidden_dim=192,
        history_size=2, num_preds=1,  # num_frames derived = 3
        depth=2, heads=4, dim_head=48, mlp_dim=256,
        dropout_rate=0.0, emb_dropout_rate=0.0,
        action_dim=2, smoothed_dim=10, mlp_scale=4,
        sigreg_weight=0.09, sigreg_knots=17, sigreg_num_proj=32,
    )


@pytest.fixture
def cfg():
    return _small_cfg()


@pytest.fixture
def rng():
    return np.random.default_rng(0)


class TestLeWM:
    def test_forward_pass_shapes(self, cfg, rng):
        model = LeWM(config=cfg)
        B, T = 2, cfg.history_size + cfg.num_preds
        pixels = rng.standard_normal((B, T, cfg.img_size, cfg.img_size, 3)).astype("float32")
        action = rng.standard_normal((B, T - 1, cfg.action_dim)).astype("float32")

        out = model({"pixels": pixels, "action": action}, training=True)
        assert tuple(out.shape) == (B, T, cfg.embed_dim)
        # Losses accumulated.
        assert len(model.losses) >= 2, f"expected >=2 losses, got {len(model.losses)}"
        for loss_val in model.losses:
            loss_np = float(ops.convert_to_numpy(loss_val))
            assert np.isfinite(loss_np), f"Non-finite loss: {loss_val}"

    def test_serialization_round_trip(self, cfg, rng, tmp_path):
        model = LeWM(config=cfg)
        B, T = 2, cfg.history_size + cfg.num_preds
        pixels = rng.standard_normal((B, T, cfg.img_size, cfg.img_size, 3)).astype("float32")
        action = rng.standard_normal((B, T - 1, cfg.action_dim)).astype("float32")

        # Build + forward.
        y1 = ops.convert_to_numpy(model(
            {"pixels": pixels, "action": action}, training=False
        ))

        path = str(tmp_path / "lewm.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        y2 = ops.convert_to_numpy(loaded(
            {"pixels": pixels, "action": action}, training=False
        ))

        # SIGReg resamples A each call, but it's NOT in the returned tensor
        # (only in losses). The returned pred_emb must be deterministic given
        # deterministic weights.
        max_diff = float(np.max(np.abs(y1 - y2)))
        assert max_diff < 1e-4, (
            f"Round-trip mismatch: max|y1 - y2| = {max_diff}"
        )

    def test_rollout_shape(self, cfg, rng):
        model = LeWM(config=cfg)
        B = 2
        # rollout encodes only pixels_history[:, 0]; the model contract requires
        # S == 1 (distinct per-S histories must be tiled / called individually).
        S = 1
        T_rollout = 4

        # Build the model with a matching training forward (T = num_frames
        # so pos_embedding covers the sequence). Rollout uses truncated
        # windows of length history_size, so it's safe regardless of T_rollout.
        T_build = cfg.num_frames
        pixels = rng.standard_normal(
            (B, T_build, cfg.img_size, cfg.img_size, 3)
        ).astype("float32")
        action = rng.standard_normal(
            (B, T_build - 1, cfg.action_dim)
        ).astype("float32")
        _ = model({"pixels": pixels, "action": action}, training=False)

        ph = rng.standard_normal(
            (B, S, cfg.history_size, cfg.img_size, cfg.img_size, 3)
        ).astype("float32")
        aseq = rng.standard_normal((B, S, T_rollout, cfg.action_dim)).astype("float32")
        res = model.rollout(ph, aseq)

        # T_full = history_size + n_steps + 1 where n_steps = T_rollout - HS
        # so T_full = T_rollout + 1.
        expected_T = T_rollout + 1
        assert tuple(res["predicted_emb"].shape) == (B, S, expected_T, cfg.embed_dim)

    def test_predictor_identity_at_init(self, cfg, rng):
        """At init, predictor blocks are identity — so pred_emb (before
        pred_proj) should preserve the input embeddings up to the learned
        pos embedding and optional input_proj. Hard to assert strict
        identity through the full model; we just check pred_emb is finite
        and has the right shape. Component-level identity is covered in
        test_adaln_zero.py."""
        model = LeWM(config=cfg)
        B, T = 2, cfg.history_size + cfg.num_preds
        pixels = rng.standard_normal((B, T, cfg.img_size, cfg.img_size, 3)).astype("float32")
        action = rng.standard_normal((B, T - 1, cfg.action_dim)).astype("float32")
        out = model({"pixels": pixels, "action": action}, training=False)
        out_np = ops.convert_to_numpy(out)
        assert np.all(np.isfinite(out_np))

    def test_default_config_forward(self, rng):
        """Regression for BUG-1: a config that does NOT set num_frames must
        still produce a model whose forward pass works. Before the fix the
        num_frames default (3) omitted num_preds, so T=4 inputs crashed the
        predictor's positional-embedding add."""
        cfg = LeWMConfig(
            img_size=56, patch_size=14, encoder_scale="tiny",
            depth=2, heads=4, dim_head=48, mlp_dim=256,
            sigreg_num_proj=32,
            # history_size=3, num_preds=1 (defaults); num_frames intentionally
            # left unset — must be derived to 4.
        )
        assert cfg.num_frames == cfg.history_size + cfg.num_preds == 4
        model = LeWM(config=cfg)
        B, T = 2, cfg.history_size + cfg.num_preds
        pixels = rng.standard_normal(
            (B, T, cfg.img_size, cfg.img_size, 3)
        ).astype("float32")
        action = rng.standard_normal((B, T - 1, cfg.action_dim)).astype("float32")
        out = model({"pixels": pixels, "action": action}, training=True)
        assert tuple(out.shape) == (B, T, cfg.embed_dim)

    def test_num_frames_validation(self):
        """num_frames derives when unset, and an explicit too-small value is
        rejected at config construction."""
        assert LeWMConfig().num_frames == 4  # history 3 + preds 1
        with pytest.raises(ValueError, match="num_frames"):
            LeWMConfig(history_size=3, num_preds=1, num_frames=2)
        # An explicit value >= history+preds is accepted as-is.
        assert LeWMConfig(history_size=2, num_preds=1, num_frames=8).num_frames == 8

    def test_rollout_rejects_short_horizon(self, cfg, rng):
        """GAP-5b: rollout must raise on an action horizon shorter than the
        history window instead of silently returning a too-short result."""
        model = LeWM(config=cfg)
        T_build = cfg.num_frames
        pixels = rng.standard_normal(
            (2, T_build, cfg.img_size, cfg.img_size, 3)
        ).astype("float32")
        action = rng.standard_normal((2, T_build - 1, cfg.action_dim)).astype("float32")
        _ = model({"pixels": pixels, "action": action}, training=False)

        B, S = 2, 2
        ph = rng.standard_normal(
            (B, S, cfg.history_size, cfg.img_size, cfg.img_size, 3)
        ).astype("float32")
        # action horizon 1 < history_size (2).
        aseq = rng.standard_normal((B, S, 1, cfg.action_dim)).astype("float32")
        with pytest.raises(ValueError, match="history_size"):
            model.rollout(ph, aseq)

    def test_loss_metrics_tracked(self, cfg, rng):
        """OPP-7: the MSE prediction loss and the SIGReg loss are exposed as
        separate tracked metrics, not just folded into the summed loss."""
        model = LeWM(config=cfg)
        B, T = 2, cfg.history_size + cfg.num_preds
        pixels = rng.standard_normal(
            (B, T, cfg.img_size, cfg.img_size, 3)
        ).astype("float32")
        action = rng.standard_normal((B, T - 1, cfg.action_dim)).astype("float32")
        _ = model({"pixels": pixels, "action": action}, training=True)

        metric_names = {m.name for m in model.metrics}
        assert {"pred_loss", "sigreg_loss"} <= metric_names
        # `val >= 0.0` was vacuous: both losses are non-negative by
        # construction, and 0.0 is exactly what an UNUPDATED `Mean` tracker
        # reports -- the failure this test exists to catch. A strict positive
        # floor is the instrument.
        values = {}
        for name, tracker in (
            ("pred_loss", model.pred_loss_tracker),
            ("sigreg_loss", model.sigreg_loss_tracker),
        ):
            val = float(ops.convert_to_numpy(tracker.result()))
            assert np.isfinite(val), f"{name} tracker holds {val}"
            assert val > 0.0, (
                f"{name} tracker reads exactly {val}; an un-updated Mean "
                f"tracker reads 0.0, so the forward pass never fed it"
            )
            values[name] = val
        # ...and the two must be separately tracked, not the same number
        # written twice (OPP-7's actual claim).
        assert values["pred_loss"] != values["sigreg_loss"], (
            f"both trackers hold {values['pred_loss']}: the two losses are not "
            "being tracked separately"
        )


# ---------------------------------------------------------------------
# Gradient flow (plan-2026-08-19-a616f581 step 11)
# ---------------------------------------------------------------------
#
# MEASURED 2026-08-19 at `_small_cfg()`, under LeWM's REAL objective
# (`sum(model.losses)` -- the prediction MSE plus the weighted SIGReg term, both
# registered by `add_loss` inside `call`; there is no external label loss):
#
#   at init              : 199 weights, 30 with an identically-ZERO gradient
#   after 1 real fit step: 199 weights,  0 dead
#
# The 30 are the 6 `action_encoder` weights and all 24 weights of the two
# `predictor/block_*` AdaLN-zero blocks. This is the mechanism the step-10
# report hypothesized, CONFIRMED by measurement and cited:
# `layers/transformers/adaln_zero.py` builds the modulation Dense with
# `kernel_initializer="zeros", bias_initializer="zeros"` under the comment
# "identity at init. This is the 'Zero' of AdaLN-Zero", and the module docstring
# states the consequence -- "At init therefore `shift=scale=gate=0`, giving
# `gate * attn(...) = 0` and `gate * mlp(...) = 0`, i.e. the block is identity
# in `x`". With gate == 0 the attention and MLP branch weights receive exactly
# zero gradient, and because the modulation kernel is itself zero, no gradient
# reaches `cond` either -- which is why `action_encoder`, whose only consumer is
# the conditioning path, is dead too.
#
# Note the distinction the hypothesis had to clear: a zero-INITIALIZED weight
# normally still has a NONZERO gradient. The zero here is a zero GRADIENT, and
# it is a consequence of the gate multiplying the branch, not of the weight's
# value. The `adaln/linear` weights themselves are among the 169 LIVE ones,
# which is exactly why one optimizer step clears all 30.
#
# So the adoption asserts the oracle AFTER one real fit step with NO waivers,
# and the companion test pins the init state so the mechanism cannot change
# silently. See GF-06 in
# plans/plan-2026-08-19T070627-a616f581/findings/gradient-flow-adoption-findings.md

from .gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    gradient_report,
)

GF_N_WEIGHTS = 199
GF_N_DEAD_AT_INIT = 30


def _gf_inputs(cfg: LeWMConfig, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    B, T = 2, cfg.history_size + cfg.num_preds
    return {
        "pixels": rng.standard_normal(
            (B, T, cfg.img_size, cfg.img_size, 3)).astype("float32"),
        "action": rng.standard_normal(
            (B, T - 1, cfg.action_dim)).astype("float32"),
    }


def _gf_objective(model):
    """LeWM's actual training objective: the sum of its `add_loss` terms."""
    return lambda _outputs: ops.sum(ops.stack(model.losses))


class TestLeWMGradientFlow:
    """Every trainable weight is on the backward graph once training starts."""

    def test_gradients_reach_every_trainable_weight_after_one_step(self, cfg):
        # DECISION plan-2026-08-19T070627-a616f581/D-012
        # One real fit step, then the oracle with NO waivers. Do NOT replace
        # this with a 30-entry `expect_zero`: waiving the entire predictor would
        # leave this suite asserting nothing about the component LeWM exists to
        # train, and any FUTURE dead predictor weight would land inside the
        # waiver unseen. The zero is a documented init transient (AdaLN-zero),
        # and the companion test below is what makes that a claim rather than an
        # excuse. See D-012 in
        # plans/plan-2026-08-19T070627-a616f581/decisions.md
        model = LeWM(config=cfg)
        x = _gf_inputs(cfg)
        model(x, training=True)

        model.compile(optimizer=keras.optimizers.Adam(1e-3))
        model.fit(x, epochs=1, batch_size=2, verbose=0)

        report = assert_gradients_reach_every_trainable_weight(
            model, x, loss_fn=_gf_objective(model)
        )

        assert len(report) == len(model.trainable_weights)
        assert len(report) == GF_N_WEIGHTS, (
            "LeWM's weight set changed shape; re-measure the init-time count "
            "in the companion test below"
        )

    def test_at_init_adaln_zero_makes_the_predictor_dead(self, cfg):
        """The mechanism, pinned; and the dead set named, not just counted."""
        model = LeWM(config=cfg)
        x = _gf_inputs(cfg)
        model(x, training=True)

        report = gradient_report(model, x, loss_fn=_gf_objective(model))
        dead = sorted(p for p, v in report.items() if v is None or v == 0.0)

        assert len(dead) == GF_N_DEAD_AT_INIT, (
            f"expected {GF_N_DEAD_AT_INIT} identically-zero gradients at init, "
            f"measured {len(dead)}:\n  " + "\n  ".join(dead)
        )
        # Named, not just counted: every one of them must be in the two places
        # the AdaLN-zero gate explains. A dead weight anywhere ELSE is a finding
        # this test must not absorb.
        for path in dead:
            assert "action_encoder/" in path or "/predictor/block_" in path, (
                f"{path} is dead at init but is NOT downstream of the AdaLN-zero "
                f"gate -- that is a new finding, not the documented transient"
            )
        # The gates themselves must be LIVE, which is what makes the transient
        # self-clearing rather than permanent.
        gates = [p for p in report if "adaln" in p.lower()]
        assert gates, "no AdaLN modulation weight found -- has the block changed?"
        assert all(report[p] is not None and report[p] > 0.0 for p in gates), (
            f"the AdaLN gates are dead too, so nothing will ever turn the "
            f"predictor on: { {p: report[p] for p in gates} }"
        )
