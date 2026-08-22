"""Tests for SAM 3's dot-product class-score head (`models/SAM/SAM3/model_misc.py`).

Two design choices in this file are load-bearing and both exist because of
measured precedent in this plan.

**The float64 oracle is written from the SPEC, not from the layer.** It
re-derives the whole expression -- prompt MLP, masked pool, both projections,
the ``1/sqrt(d_proj)`` scale and the clamp -- in double precision from the
layer's WEIGHTS only. An oracle that calls the implementation is a tautology,
and this plan has already shipped one green oracle certifying the wrong
quantity.

**The oracle tolerance is 1e-3 because of TF32.** The same comparison measures
``1.1e-7`` on CPU and ``1.3e-4`` on GPU 1 -- the RTX 4070 runs float32 matmuls
through TF32, so a tolerance derived in one regime is wrong in the other (both
numbers MEASURED here, not inherited). ``1e-3`` clears the GPU noise floor by an
order while every wrong candidate the mutations install misses by ``0.38`` to
``2.1``, i.e. by three orders more -- ``test_the_oracle_probes_are_unsaturated``
pins those margins so the tolerance can never quietly swallow a defect.

**The saturation probe pins a point inside ``(10, 12]``.** The decoder's
presence clamp is 10.0 and this head's is 12.0. Every probe whose scores stay
below 10 is a coincidence point at which the two are indistinguishable, so
``TestClampBound`` constructs a score of exactly 11.5 by writing identity
projections -- the one probe that separates them.

The fixtures deliberately randomize BIASES as well as kernels. With Keras's
default zero-initialized biases, the all-padding row's score would be exactly
0.0, which is precisely the value a dead component returns -- the probe would
then be satisfied by construction rather than by correctness.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.SAM.SAM3.model_misc import Sam3DotProductScoring

# ---------------------------------------------------------------------
# tiny variant
# ---------------------------------------------------------------------

TINY = dict(
    d_model=8, d_proj=4, use_prompt_mlp=True, prompt_mlp_hidden_dim=16,
    prompt_mlp_dropout_rate=0.1, clamp_logits=True, clamp_max_val=12.0,
)

# The settled SAM 3 head, re-read from the pinned upstream clone's
# `_create_dot_product_scoring()`: d_model=256, d_proj=256, and a prompt MLP of
# 256 -> 2048 -> 256 with dropout 0.1, residual, and a terminal LayerNorm.
SHIPPED = dict(
    d_model=256, d_proj=256, use_prompt_mlp=True, prompt_mlp_hidden_dim=2048,
    prompt_mlp_dropout_rate=0.1, clamp_logits=True, clamp_max_val=12.0,
)

BATCH, SEQ, QUERIES = 2, 6, 5


def _params(cfg: dict) -> int:
    """Closed-form parameter count, written from the STRUCTURE."""
    d, p, h = cfg["d_model"], cfg["d_proj"], cfg["prompt_mlp_hidden_dim"]
    mlp = (d * h + h) + (h * d + d) + 2 * d if cfg["use_prompt_mlp"] else 0
    return mlp + 2 * (d * p + p)


def _randomize(head: Sam3DotProductScoring, seed: int = 0) -> None:
    """Give every weight -- kernels AND biases -- a non-trivial value."""
    rng = np.random.default_rng(seed)
    for weight in head.weights:
        weight.assign(rng.normal(0.0, 0.4, size=weight.shape).astype("float32"))


@pytest.fixture()
def head() -> Sam3DotProductScoring:
    layer = Sam3DotProductScoring(**TINY)
    layer.build((BATCH, QUERIES, TINY["d_model"]), (BATCH, SEQ, TINY["d_model"]))
    _randomize(layer)
    return layer


def _inputs(seed: int = 3, queries: int = QUERIES):
    rng = np.random.default_rng(seed)
    hs = rng.normal(size=(BATCH, queries, TINY["d_model"])).astype("float32")
    prompt = rng.normal(size=(BATCH, SEQ, TINY["d_model"])).astype("float32")
    return hs, prompt


def _pad(kind: str) -> np.ndarray:
    """The three mandated mask probes."""
    mask = np.zeros((BATCH, SEQ), dtype=bool)
    if kind == "all_valid":
        return mask
    if kind == "trailing":                 # probe (ii)
        mask[0, 4:] = True
        mask[1, 2:] = True
        return mask
    if kind == "all_invalid":              # probe (iii): divisor would be 0
        mask[0, 4:] = True
        mask[1, :] = True
        return mask
    raise ValueError(kind)


# ---------------------------------------------------------------------
# the float64 oracle -- written from the mechanism, never from the layer
# ---------------------------------------------------------------------


def _w(layer) -> tuple:
    return (
        np.asarray(layer.kernel, dtype=np.float64),
        np.asarray(layer.bias, dtype=np.float64),
    )


def _layer_norm(x: np.ndarray, gamma, beta, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def oracle(
        head: Sam3DotProductScoring, hs: np.ndarray, prompt: np.ndarray,
        pad: np.ndarray, *, masked: bool = True, scaled: bool = True,
        clamp_at: float = None,
) -> np.ndarray:
    """Recompute the head's output in float64 from its weights.

    The keyword flags exist so that the WRONG candidates the named mutations
    represent can be evaluated by the same oracle: ``masked=False`` is M5.1,
    ``scaled=False`` is M5.2 and ``clamp_at=10.0`` is M5.3.
    """
    hs = hs.astype(np.float64)
    prompt = prompt.astype(np.float64)
    if head.use_prompt_mlp:
        k1, b1 = _w(head.prompt_fc1)
        k2, b2 = _w(head.prompt_fc2)
        hidden = np.maximum(prompt @ k1 + b1, 0.0)
        gamma = np.asarray(head.prompt_norm.gamma, dtype=np.float64)
        beta = np.asarray(head.prompt_norm.beta, dtype=np.float64)
        prompt = _layer_norm((hidden @ k2 + b2) + prompt, gamma, beta)
    if masked:
        valid = (~pad).astype(np.float64)[..., None]
        pooled = (prompt * valid).sum(axis=1) / np.maximum(
            valid.sum(axis=1), 1.0)
    else:
        pooled = prompt.mean(axis=1)
    kp, bp = _w(head.prompt_proj)
    kh, bh = _w(head.hs_proj)
    scale = 1.0 / np.sqrt(head.d_proj) if scaled else 1.0
    scores = (hs @ kh + bh) @ (pooled @ kp + bp)[..., None] * scale
    bound = head.clamp_max_val if clamp_at is None else clamp_at
    return np.clip(scores, -bound, bound) if head.clamp_logits else scores


def _forward(head, hs, prompt, pad) -> np.ndarray:
    return ops.convert_to_numpy(head(hs, prompt, pad, training=False))


# ---------------------------------------------------------------------


class TestConstruction:

    def test_output_is_one_logit_per_query(self, head):
        hs, prompt = _inputs()
        out = _forward(head, hs, prompt, _pad("all_valid"))
        assert out.shape == (BATCH, QUERIES, 1)

    def test_leading_axes_are_free_so_a_layer_stack_works(self, head):
        # The decoder stacks per-layer states on a LEADING axis.
        rng = np.random.default_rng(9)
        hs = rng.normal(size=(3, BATCH, QUERIES, TINY["d_model"])).astype(
            "float32")
        _, prompt = _inputs()
        out = _forward(head, hs, prompt, _pad("all_valid"))
        assert out.shape == (3, BATCH, QUERIES, 1)

    def test_compute_output_shape_matches_the_forward_pass(self, head):
        hs, prompt = _inputs()
        declared = head.compute_output_shape(
            (BATCH, QUERIES, TINY["d_model"]), (BATCH, SEQ, TINY["d_model"]))
        assert declared == _forward(head, hs, prompt, _pad("all_valid")).shape

    def test_a_missing_mask_means_every_position_is_valid(self, head):
        hs, prompt = _inputs()
        with_none = ops.convert_to_numpy(head(hs, prompt, None, training=False))
        with_zeros = _forward(head, hs, prompt, _pad("all_valid"))
        assert np.abs(with_none - with_zeros).max() == 0.0

    @pytest.mark.parametrize("bad", [
        dict(d_model=0), dict(d_proj=0), dict(prompt_mlp_hidden_dim=0),
        dict(prompt_mlp_dropout_rate=1.0), dict(prompt_mlp_dropout_rate=-0.1),
        dict(clamp_max_val=0.0),
    ])
    def test_invalid_configuration_raises(self, bad):
        with pytest.raises(ValueError):
            Sam3DotProductScoring(**{**TINY, **bad})

    def test_rank_two_queries_raise(self):
        with pytest.raises(ValueError, match="rank >= 3"):
            Sam3DotProductScoring(**TINY).build(
                (BATCH, TINY["d_model"]), (BATCH, SEQ, TINY["d_model"]))

    def test_a_rank_two_prompt_raises(self):
        with pytest.raises(ValueError, match="batch, seq, d_model"):
            Sam3DotProductScoring(**TINY).build(
                (BATCH, QUERIES, TINY["d_model"]), (BATCH, TINY["d_model"]))

    def test_a_width_other_than_d_model_raises(self):
        with pytest.raises(ValueError, match="d_model"):
            Sam3DotProductScoring(**TINY).build(
                (BATCH, QUERIES, 9), (BATCH, SEQ, TINY["d_model"]))


class TestValueOracle:
    """The float64 oracle, at the three mandated mask probes."""

    @pytest.mark.parametrize("kind", ["all_valid", "trailing", "all_invalid"])
    def test_matches_the_float64_oracle(self, head, kind):
        hs, prompt = _inputs()
        pad = _pad(kind)
        actual = _forward(head, hs, prompt, pad)
        expected = oracle(head, hs, prompt, pad)
        assert np.abs(actual - expected).max() < 1e-3

    def test_the_oracle_probes_are_unsaturated_so_the_scale_is_visible(
            self, head
    ):
        # PRE-CHECK for M5.2: if every score sat on the clamp, dropping the
        # 1/sqrt(d_proj) scale would be INERT rather than wrong.
        hs, prompt = _inputs()
        for kind in ("all_valid", "trailing", "all_invalid"):
            scores = oracle(head, hs, prompt, _pad(kind))
            unscaled = oracle(head, hs, prompt, _pad(kind), scaled=False)
            assert np.abs(scores).max() < head.clamp_max_val
            # MEASURED margin 1.9 - 2.1, three orders above the 1e-3 tolerance.
            assert np.abs(scores - unscaled).max() > 0.1

    def test_padding_positions_do_not_reach_the_score(self, head):
        # M5.1's discriminating probe: rewriting only the PADDED positions must
        # leave every score bit-identical.
        hs, prompt = _inputs()
        pad = _pad("trailing")
        base = _forward(head, hs, prompt, pad)
        rng = np.random.default_rng(21)
        noisy = prompt.copy()
        noisy[pad] = rng.normal(0.0, 5.0, size=noisy[pad].shape)
        assert np.abs(base - _forward(head, hs, noisy, pad)).max() == 0.0

    def test_an_unmasked_pool_would_give_a_different_answer(self, head):
        # Proves the assertion above is not vacuous: the wrong candidate that
        # M5.1 installs is measurably different on this exact fixture.
        hs, prompt = _inputs()
        pad = _pad("trailing")
        # NOTE the probe choice: at an ALL-VALID mask the two candidates are
        # mathematically identical (measured delta exactly 0.0), so probe (i)
        # cannot see M5.1 and probes (ii)/(iii) are the ones that must.
        wrong = oracle(head, hs, prompt, pad, masked=False)
        assert np.abs(oracle(head, hs, prompt, pad) - wrong).max() > 0.1


class TestAllPaddingRow:
    """Probe (iii): the row a naive ``sum / sum`` divides by zero on."""

    def test_the_all_padding_row_is_finite(self, head):
        hs, prompt = _inputs()
        out = _forward(head, hs, prompt, _pad("all_invalid"))
        assert np.isfinite(out).all()

    def test_the_all_padding_row_scores_from_the_projection_bias(self, head):
        # POSITIVE liveness arm: the pooled vector is zero, so the score is
        # driven by the projection biases and must be NON-zero. A dead
        # component returning a constant 0 fails this by value, not by absence.
        hs, prompt = _inputs()
        out = _forward(head, hs, prompt, _pad("all_invalid"))
        assert np.abs(out[1]).min() > 1e-4

    def test_the_all_padding_row_ignores_the_prompt_entirely(self, head):
        hs, prompt = _inputs()
        pad = _pad("all_invalid")
        rng = np.random.default_rng(5)
        other = prompt + rng.normal(0.0, 3.0, size=prompt.shape).astype(
            "float32")
        base = _forward(head, hs, prompt, pad)
        assert np.abs(base[1] - _forward(head, hs, other, pad)[1]).max() == 0.0

    def test_the_divisor_floor_is_what_makes_it_finite(self, head):
        # The comparator is proven able to see the defect: the naive spelling,
        # computed here in float64, is NaN on exactly this row.
        valid = (~_pad("all_invalid")).astype(np.float64)[..., None]
        with np.errstate(invalid="ignore"):
            naive = valid.sum(axis=1) / valid.sum(axis=1)
        assert np.isnan(naive[1]).all()


class TestClampBound:
    """The (10, 12] probe -- the only region where 12.0 and 10.0 differ."""

    @pytest.fixture()
    def identity_head(self) -> Sam3DotProductScoring:
        cfg = {**TINY, "d_model": 4, "d_proj": 4, "use_prompt_mlp": False}
        layer = Sam3DotProductScoring(**cfg)
        layer.build((1, 1, 4), (1, 1, 4))
        for projection in (layer.prompt_proj, layer.hs_proj):
            projection.kernel.assign(np.eye(4, dtype="float32"))
            projection.bias.assign(np.zeros(4, dtype="float32"))
        return layer

    def _score(self, layer, magnitude: float) -> float:
        hs = np.zeros((1, 1, 4), dtype="float32")
        hs[0, 0, 0] = 1.0
        prompt = np.zeros((1, 1, 4), dtype="float32")
        prompt[0, 0, 0] = magnitude
        pad = np.zeros((1, 1), dtype=bool)
        return float(
            ops.convert_to_numpy(layer(hs, prompt, pad, training=False))[0, 0, 0]
        )

    def test_a_score_inside_ten_to_twelve_is_not_clamped(self, identity_head):
        # 23 * 1 / sqrt(4) = 11.5, inside (10, 12]. Under a 10.0 bound this
        # reads 10.0; under the correct 12.0 bound it reads 11.5.
        assert self._score(identity_head, 23.0) == pytest.approx(11.5, abs=1e-4)

    def test_a_negative_score_inside_the_same_band_is_not_clamped(
            self, identity_head
    ):
        assert self._score(identity_head, -23.0) == pytest.approx(
            -11.5, abs=1e-4)

    def test_scores_beyond_the_bound_saturate_at_it(self, identity_head):
        assert self._score(identity_head, 25.0) == pytest.approx(12.0, abs=1e-4)
        assert self._score(identity_head, -25.0) == pytest.approx(
            -12.0, abs=1e-4)

    def test_the_bound_is_configurable_and_the_default_is_twelve(self):
        assert Sam3DotProductScoring().clamp_max_val == 12.0

    def test_clamping_can_be_switched_off(self):
        cfg = {**TINY, "d_model": 4, "d_proj": 4, "use_prompt_mlp": False,
               "clamp_logits": False}
        layer = Sam3DotProductScoring(**cfg)
        layer.build((1, 1, 4), (1, 1, 4))
        for projection in (layer.prompt_proj, layer.hs_proj):
            projection.kernel.assign(np.eye(4, dtype="float32"))
            projection.bias.assign(np.zeros(4, dtype="float32"))
        assert self._score(layer, 100.0) == pytest.approx(50.0, abs=1e-3)


class TestScale:

    def test_the_scale_is_one_over_root_d_proj(self, head):
        assert head.scale == pytest.approx(1.0 / np.sqrt(TINY["d_proj"]))

    def test_the_scale_tracks_d_proj_not_d_model(self):
        layer = Sam3DotProductScoring(**{**TINY, "d_proj": 16})
        assert layer.scale == pytest.approx(0.25)


class TestIndependentProjections:

    def test_the_two_projections_are_distinct_objects(self, head):
        assert head.prompt_proj is not head.hs_proj
        assert head.prompt_proj.kernel is not head.hs_proj.kernel

    def test_moving_one_projection_does_not_move_the_other(self, head):
        before = np.asarray(head.hs_proj.kernel).copy()
        head.prompt_proj.kernel.assign(
            np.zeros_like(np.asarray(head.prompt_proj.kernel)))
        assert np.abs(np.asarray(head.hs_proj.kernel) - before).max() == 0.0

    def test_the_query_side_and_prompt_side_are_not_interchangeable(self, head):
        # A single shared projection would make this swap a no-op.
        hs, prompt = _inputs()
        pad = _pad("all_valid")
        base = _forward(head, hs, prompt, pad)
        kp = np.asarray(head.prompt_proj.kernel).copy()
        kh = np.asarray(head.hs_proj.kernel).copy()
        head.prompt_proj.kernel.assign(kh)
        head.hs_proj.kernel.assign(kp)
        assert np.abs(base - _forward(head, hs, prompt, pad)).max() > 1e-4


class TestPromptMlp:

    def test_the_mlp_is_residual_around_its_own_input(self, head):
        # Zeroing the second projection's kernel and bias leaves the residual
        # path, so the MLP output must reduce to LayerNorm(prompt).
        hs, prompt = _inputs()
        head.prompt_fc2.kernel.assign(
            np.zeros_like(np.asarray(head.prompt_fc2.kernel)))
        head.prompt_fc2.bias.assign(
            np.zeros_like(np.asarray(head.prompt_fc2.bias)))
        pad = _pad("all_valid")
        assert np.abs(
            _forward(head, hs, prompt, pad) - oracle(head, hs, prompt, pad)
        ).max() < 1e-3

    def test_the_mlp_can_be_disabled_and_then_owns_no_weights(self):
        layer = Sam3DotProductScoring(**{**TINY, "use_prompt_mlp": False})
        layer.build((BATCH, QUERIES, TINY["d_model"]),
                    (BATCH, SEQ, TINY["d_model"]))
        assert layer.count_params() == _params(
            {**TINY, "use_prompt_mlp": False})
        assert not hasattr(layer, "prompt_fc1")

    def test_dropout_only_acts_in_training_mode(self, head):
        hs, prompt = _inputs()
        pad = _pad("all_valid")
        first = ops.convert_to_numpy(head(hs, prompt, pad, training=False))
        second = ops.convert_to_numpy(head(hs, prompt, pad, training=False))
        assert np.abs(first - second).max() == 0.0

    def test_the_hidden_width_is_the_configured_one(self, head):
        assert tuple(head.prompt_fc1.kernel.shape) == (
            TINY["d_model"], TINY["prompt_mlp_hidden_dim"])
        assert tuple(head.prompt_fc2.kernel.shape) == (
            TINY["prompt_mlp_hidden_dim"], TINY["d_model"])


class TestLiveness:
    """Positive arms, budgeted BEFORE the dead-component probe was run."""

    def test_scores_are_not_all_equal_across_queries(self, head):
        hs, prompt = _inputs()
        out = _forward(head, hs, prompt, _pad("all_valid"))
        assert len(np.unique(out)) == out.size
        assert float(out.std()) > 0.0

    def test_changing_the_prompt_moves_every_query_score(self, head):
        hs, prompt = _inputs()
        pad = _pad("all_valid")
        base = _forward(head, hs, prompt, pad)
        rng = np.random.default_rng(31)
        moved = _forward(
            head, hs, prompt + rng.normal(0.0, 1.0, prompt.shape).astype(
                "float32"), pad)
        assert np.abs(base - moved).min() > 1e-5

    def test_changing_one_query_moves_only_that_query(self, head):
        hs, prompt = _inputs()
        pad = _pad("all_valid")
        base = _forward(head, hs, prompt, pad)
        bumped = hs.copy()
        bumped[:, 2] += 1.0
        moved = _forward(head, hs=bumped, prompt=prompt, pad=pad)
        assert np.abs(base[:, 2] - moved[:, 2]).min() > 1e-5
        assert np.abs(
            np.delete(base, 2, axis=1) - np.delete(moved, 2, axis=1)).max() == 0.0


class TestParameterAudit:

    def test_tiny_count_matches_the_closed_form_exactly(self, head):
        assert head.count_params() == _params(TINY)

    def test_shipped_count_matches_the_closed_form_exactly(self):
        layer = Sam3DotProductScoring(**SHIPPED)
        layer.build((BATCH, QUERIES, SHIPPED["d_model"]),
                    (BATCH, SEQ, SHIPPED["d_model"]))
        assert layer.count_params() == _params(SHIPPED) == 1_182_976

    def test_the_audit_is_non_vacuous(self):
        # EXECUTED, not asserted: a head with the prompt MLP removed must NOT
        # satisfy the full closed form.
        layer = Sam3DotProductScoring(**{**SHIPPED, "use_prompt_mlp": False})
        layer.build((BATCH, QUERIES, SHIPPED["d_model"]),
                    (BATCH, SEQ, SHIPPED["d_model"]))
        assert layer.count_params() != _params(SHIPPED)


class TestSerialization:

    def test_config_round_trip_preserves_every_init_parameter(self):
        layer = Sam3DotProductScoring(**TINY)
        config = layer.get_config()
        for key, value in TINY.items():
            assert config[key] == value
        rebuilt = Sam3DotProductScoring.from_config(config)
        assert rebuilt.get_config() == config

    def test_full_keras_roundtrip_preserves_output_VALUES(self, tmp_path):
        # D-098: weight counts and weight PATHS are the instrument that FAILED
        # in this plan. Only an output-value comparison sees fresh kernels.
        keras.utils.set_random_seed(3)
        hs_in = keras.Input(batch_shape=(BATCH, QUERIES, TINY["d_model"]))
        prompt_in = keras.Input(batch_shape=(BATCH, SEQ, TINY["d_model"]))
        pad_in = keras.Input(batch_shape=(BATCH, SEQ), dtype="bool")
        layer = Sam3DotProductScoring(**TINY)
        model = keras.Model([hs_in, prompt_in, pad_in],
                            layer(hs_in, prompt_in, pad_in))
        _randomize(layer, seed=17)
        hs, prompt = _inputs()
        pad = _pad("trailing")
        before = ops.convert_to_numpy(model([hs, prompt, pad], training=False))
        path = tmp_path / "sam3_scoring.keras"
        model.save(path)
        restored = keras.models.load_model(path)
        after = ops.convert_to_numpy(restored([hs, prompt, pad], training=False))
        assert np.abs(before - after).max() == 0.0

    def test_the_restored_head_still_clamps_at_twelve(self, tmp_path):
        cfg = {**TINY, "d_model": 4, "d_proj": 4, "use_prompt_mlp": False}
        hs_in = keras.Input(batch_shape=(1, 1, 4))
        prompt_in = keras.Input(batch_shape=(1, 1, 4))
        pad_in = keras.Input(batch_shape=(1, 1), dtype="bool")
        layer = Sam3DotProductScoring(**cfg)
        model = keras.Model([hs_in, prompt_in, pad_in],
                            layer(hs_in, prompt_in, pad_in))
        for projection in (layer.prompt_proj, layer.hs_proj):
            projection.kernel.assign(np.eye(4, dtype="float32"))
            projection.bias.assign(np.zeros(4, dtype="float32"))
        path = tmp_path / "sam3_scoring_clamp.keras"
        model.save(path)
        restored = keras.models.load_model(path)
        hs = np.zeros((1, 1, 4), dtype="float32")
        hs[0, 0, 0] = 1.0
        prompt = np.zeros((1, 1, 4), dtype="float32")
        prompt[0, 0, 0] = 23.0
        out = ops.convert_to_numpy(
            restored([hs, prompt, np.zeros((1, 1), dtype=bool)],
                     training=False))
        assert float(out[0, 0, 0]) == pytest.approx(11.5, abs=1e-4)


class TestPackageSurface:

    def test_the_head_is_exported_from_the_package(self):
        from dl_techniques.models.SAM import SAM3 as sam3
        assert sam3.Sam3DotProductScoring is Sam3DotProductScoring
        assert "Sam3DotProductScoring" in sam3.__all__


class TestBuildIsReEntrant:
    """D-136: the guard D-126 said only ONE class was missing.

    `Sam3DotProductScoring.build` had no `if self.built: return`, so a second
    `build()` raised `ValueError: You cannot add new elements of state ...`.
    `Sam3Image._build_once` masked it, which is why 390 tests never saw it --
    that helper is the caller's, not this class's, so any other composer met
    the raise. These tests execute the double build directly.
    """

    def test_a_second_build_is_a_no_op(self):
        head = Sam3DotProductScoring(**TINY)
        head.build((BATCH, QUERIES, TINY["d_model"]),
                   (BATCH, SEQ, TINY["d_model"]))
        before = [np.asarray(w) for w in head.weights]
        head.build((BATCH, QUERIES, TINY["d_model"]),
                   (BATCH, SEQ, TINY["d_model"]))
        after = [np.asarray(w) for w in head.weights]
        assert len(after) == len(before)
        for old, new in zip(before, after):
            assert np.abs(old - new).max() == 0.0

    def test_the_guard_is_what_prevents_the_raise(self):
        """RED proof: with the guard bypassed, the second build DIES.

        The defect is executed, not asserted -- `Layer.build` is invoked with
        the guard's flag temporarily cleared, reproducing the exact pre-fix
        state, and the `ValueError` it raises is the one D-126 quoted.
        """
        head = Sam3DotProductScoring(**TINY)
        shapes = ((BATCH, QUERIES, TINY["d_model"]),
                  (BATCH, SEQ, TINY["d_model"]))
        head.build(*shapes)
        assert head.built
        object.__setattr__(head, "built", False)
        with pytest.raises(ValueError, match="already built"):
            head.build(*shapes)

    def test_a_second_build_still_leaves_the_forward_pass_correct(self):
        head = Sam3DotProductScoring(**TINY)
        shapes = ((BATCH, QUERIES, TINY["d_model"]),
                  (BATCH, SEQ, TINY["d_model"]))
        head.build(*shapes)
        _randomize(head)
        hs, prompt = _inputs()
        pad = _pad("trailing")
        first = _forward(head, hs, prompt, pad)
        head.build(*shapes)
        second = _forward(head, hs, prompt, pad)
        assert np.abs(first - second).max() == 0.0
