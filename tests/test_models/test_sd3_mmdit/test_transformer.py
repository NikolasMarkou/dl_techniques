"""Tests for the SD3MMDiT model + create_sd3_mmdit factory (step 6).

Uses the TINY preset throughout to stay fast (latent 16x16x16, depth 4, dim 192,
6 heads, patch_size 2 -> 8x8 = 64 patch tokens; block 0 uses dual attention).
Verifies forward velocity shape, compute_output_shape (pre/post build), variable
batch + variable text seq len, get_config / from_config round-trip, and the
full ``.keras`` save/load deterministic-velocity round-trip.
"""

import os
import keras
import numpy as np
import pytest

from dl_techniques.models.sd3_mmdit.config import get_sd3_config
from dl_techniques.models.sd3_mmdit.transformer import (
    SD3MMDiT,
    create_sd3_mmdit,
)


# ---------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------

LATENT_HW = 16
IN_CH = 16
JOINT_DIM = 512
POOLED_DIM = 256


def _make_batch(batch: int = 2, txt_len: int = 7, seed: int = 0) -> dict:
    """Build a valid TINY input dict."""
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(
        (batch, LATENT_HW, LATENT_HW, IN_CH)
    ).astype("float32")
    enc = rng.standard_normal((batch, txt_len, JOINT_DIM)).astype("float32")
    pooled = rng.standard_normal((batch, POOLED_DIM)).astype("float32")
    timestep = rng.uniform(0.0, 1000.0, size=(batch,)).astype("float32")
    return {
        "latent": keras.ops.convert_to_tensor(latent),
        "encoder_hidden_states": keras.ops.convert_to_tensor(enc),
        "pooled_projections": keras.ops.convert_to_tensor(pooled),
        "timestep": keras.ops.convert_to_tensor(timestep),
    }


def _input_shapes(batch: int = 2, txt_len: int = 7) -> dict:
    return {
        "latent": (batch, LATENT_HW, LATENT_HW, IN_CH),
        "encoder_hidden_states": (batch, txt_len, JOINT_DIM),
        "pooled_projections": (batch, POOLED_DIM),
        "timestep": (batch,),
    }


# ---------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------


class TestSD3MMDiT:
    def test_factory_builds(self):
        model = create_sd3_mmdit("tiny")
        assert isinstance(model, SD3MMDiT)
        assert model.config.embedding_size == 192
        assert model.config.depth == 4

    def test_forward_shape(self):
        model = create_sd3_mmdit("tiny")
        batch = _make_batch(batch=2, txt_len=7)
        out = model(batch)
        # Velocity must exactly match the latent shape (in==out channels).
        assert tuple(out.shape) == (2, LATENT_HW, LATENT_HW, IN_CH)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_compute_output_shape_matches_and_pre_build(self):
        # Pre-build: a fresh, unbuilt instance must answer compute_output_shape.
        fresh = create_sd3_mmdit("tiny")
        shapes = _input_shapes(batch=2, txt_len=7)
        pred = fresh.compute_output_shape(shapes)
        assert tuple(pred) == (2, LATENT_HW, LATENT_HW, IN_CH)

        # Post-build: matches the actual forward output.
        batch = _make_batch(batch=2, txt_len=7)
        out = fresh(batch)
        assert tuple(out.shape) == tuple(pred)

    @pytest.mark.parametrize("batch", [1, 3])
    @pytest.mark.parametrize("txt_len", [5, 11])
    def test_variable_batch_and_seqlen(self, batch, txt_len):
        model = create_sd3_mmdit("tiny")
        b = _make_batch(batch=batch, txt_len=txt_len, seed=batch + txt_len)
        out = model(b)
        assert tuple(out.shape) == (batch, LATENT_HW, LATENT_HW, IN_CH)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_get_config_has_config_dict(self):
        model = create_sd3_mmdit("tiny")
        cfg = model.get_config()
        assert "config" in cfg
        assert isinstance(cfg["config"], dict)
        assert cfg["config"]["embedding_size"] == 192

    def test_from_config_reconstructs(self):
        model = create_sd3_mmdit("tiny")
        rebuilt = SD3MMDiT.from_config(model.get_config())
        batch = _make_batch(batch=2, txt_len=7)
        out = rebuilt(batch)
        assert tuple(out.shape) == (2, LATENT_HW, LATENT_HW, IN_CH)

    def test_keras_round_trip(self, tmp_path):
        """The serialization gate: save/reload yields IDENTICAL velocity."""
        model = create_sd3_mmdit("tiny")
        batch = _make_batch(batch=2, txt_len=7, seed=42)
        out_before = keras.ops.convert_to_numpy(model(batch))

        path = os.path.join(str(tmp_path), "sd3_mmdit_tiny.keras")
        model.save(path)
        # Registration handles deserialization -- no custom_objects needed.
        reloaded = keras.models.load_model(path)
        out_after = keras.ops.convert_to_numpy(reloaded(batch))

        try:
            np.testing.assert_allclose(out_before, out_after, atol=1e-6)
        except AssertionError:
            np.testing.assert_allclose(out_before, out_after, atol=1e-5)


# ---------------------------------------------------------------------
# Gradient flow (plan-2026-08-19-a616f581 step 11)
# ---------------------------------------------------------------------
#
# MEASURED 2026-08-19 on the TINY preset, one tape step, `default_loss`:
#
#   at init                        : 146 weights, 132 dead
#   after 1 real SD3FlowTrainer step:              30 dead
#   after 2 steps                  :               3 dead
#   after 3, 4, 5 steps            :               3 dead   <-- PERMANENT
#
# The 129 that clear are AdaLN-zero at init and are NOT a defect:
# `layers/transformers/sd3_adaln.py` builds every modulation Dense with
# `kernel_initializer="zeros", bias_initializer="zeros"` ("The modulation
# Dense(6*dim) is zero-initialized so at init the ... block is an exact
# identity"), and the transformer's module docstring states the intent --
# "Gating the residual branch (rather than the pre-norm input) is what allows a
# zero-initialized gate to make a fresh block an exact identity, so depth can be
# added without destabilizing an already-trained trunk." With every gate at 0
# the attention/FFN branches contribute nothing (zero gradient), and because the
# modulation KERNEL is zero, nothing propagates back into `time_embed`,
# `pooled_proj_*` or `context_embedder` either. Two optimizer steps clear all of
# it, because the modulation Denses themselves are live from step 0.
#
# The 3 that do NOT clear are PRODUCT FINDING GF-04. See
# plans/plan-2026-08-19T070627-a616f581/findings/gradient-flow-adoption-findings.md

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    gradient_report,
)

#: Warm-up steps needed to get past the documented AdaLN-zero identity.
GF_WARMUP_STEPS = 2

#: PRODUCT FINDING GF-04 -- permanently dead, in EVERY state after warm-up.
#: The final block is built with `context_pre_only=True`, so its text-stream
#: output is discarded (`mmdit_joint_attention.py`: "context_pre_only=True:
#: to_add_out is never created and call() returns image_out alone"). But
#: `add_q_proj` and `norm_added_q` ARE still created and still applied in
#: `call()` -- the text QUERY is computed, concatenated, attended, and then
#: thrown away with the text rows. Nothing consumes it, so these three tensors
#: can never receive a gradient in any state of training.
GF04_PERMANENTLY_DEAD = (
    "block_3/attn/add_q_proj/kernel",
    "block_3/attn/add_q_proj/bias",
    "block_3/attn/norm_added_q/scale",
)


def _gf_warm_transformer():
    """A TINY SD3MMDiT taken `GF_WARMUP_STEPS` real training steps past init.

    Uses the SHIPPED trainer (`train.sd3_mmdit.train_sd3_mmdit`), the same one
    `test_trainer.py` in this directory uses, rather than a hand-rolled
    optimizer loop: the state being asserted about has to be a state the real
    pipeline reaches.
    """
    from train.sd3_mmdit.train_sd3_mmdit import (
        TrainingConfig,
        build_trainer,
        make_synthetic_dataset,
    )

    config = TrainingConfig(
        variant="tiny", batch_size=2, steps_per_epoch=1, epochs=1,
        learning_rate=1e-3, num_text_tokens=7, seed=123,
    )
    trainer, sd3_config = build_trainer(config)
    trainer.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))
    dataset = make_synthetic_dataset(config, sd3_config)
    for _ in range(GF_WARMUP_STEPS):
        trainer.fit(dataset, epochs=1, steps_per_epoch=1, verbose=0)
    return trainer.transformer


class TestSD3MMDiTGradientFlow:
    """Gradient flow through the MMDiT trunk, after the AdaLN-zero transient."""

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "PRODUCT FINDING GF-04: the final block is context_pre_only, so its "
            "text-stream output is discarded, but add_q_proj / norm_added_q are "
            "still created, still trainable=True and still applied in call() -- "
            "the text query is computed and thrown away. 3 of 146 weights can "
            "never receive a gradient. Not fixed here: this is an adoption step. "
            "See GF-04 in plans/plan-2026-08-19T070627-a616f581/findings/"
            "gradient-flow-adoption-findings.md"
        ),
    )
    def test_gradients_reach_every_trainable_weight(self):
        model = _gf_warm_transformer()
        inputs = _make_batch()
        assert_gradients_reach_every_trainable_weight(model, inputs)

    def test_the_dead_set_is_exactly_the_three_context_query_tensors(self):
        """The complement of the xfail above, and the reason it is not vacuous.

        An `xfail(strict=True)` is satisfied by ANY failure, including a
        totally dead model. This pins BOTH halves by name: exactly 3 weights are
        dead and they are exactly the final block's discarded context query;
        the other 143 are live. If a future change kills a fourth weight, this
        test -- not the xfail -- is what reports it.
        """
        model = _gf_warm_transformer()
        inputs = _make_batch()
        report = gradient_report(model, inputs)

        dead = sorted(p for p, v in report.items() if v is None or v == 0.0)
        live = [p for p, v in report.items() if v is not None and v > 0.0]

        assert len(report) == len(model.trainable_weights)
        assert len(dead) == len(GF04_PERMANENTLY_DEAD), (
            f"expected exactly {len(GF04_PERMANENTLY_DEAD)} dead weights after "
            f"{GF_WARMUP_STEPS} training steps, measured {len(dead)}:\n  "
            + "\n  ".join(dead)
        )
        for suffix in GF04_PERMANENTLY_DEAD:
            assert any(p.endswith(suffix) for p in dead), (
                f"{suffix} is no longer dead -- if GF-04 was fixed, delete the "
                f"xfail above (it is strict and will fail on an unexpected pass)"
            )
        assert len(live) == len(report) - len(GF04_PERMANENTLY_DEAD)

    def test_the_adaln_zero_identity_is_an_init_transient_not_a_dead_component(self):
        """132 dead at init, 3 dead after warm-up: the two states, compared.

        Without this the warm-up in the tests above looks like a way to dodge a
        result. It is the opposite -- it is what SEPARATES the 129 documented
        transients from the 3 permanent ones, which a 132-entry `expect_zero`
        would have buried.
        """
        cold = create_sd3_mmdit("tiny")
        inputs = _make_batch()
        cold(inputs, training=False)
        n_dead_cold = sum(
            1 for v in gradient_report(cold, inputs).values()
            if v is None or v == 0.0
        )

        warm = _gf_warm_transformer()
        n_dead_warm = sum(
            1 for v in gradient_report(warm, inputs).values()
            if v is None or v == 0.0
        )

        assert n_dead_cold > 100, (
            f"only {n_dead_cold} weights are dead at init; the AdaLN-zero "
            f"identity documented in layers/transformers/sd3_adaln.py may have "
            f"changed, which would make the warm-up above unnecessary"
        )
        assert n_dead_warm == len(GF04_PERMANENTLY_DEAD), n_dead_warm
        assert n_dead_cold - n_dead_warm >= 100
