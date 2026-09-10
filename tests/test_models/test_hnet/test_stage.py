"""Guards for H-Net's recursive stage.

What this file pins
-------------------
1. **The straight-through gate's exact spelling.** ``x + stop_gradient(hard - x)`` written
   as ONE grouped expression is bit-exactly ``1.0``; the algebraically identical
   ungrouped rewrite is not. Measured in float64 below, plus a source guard on the
   spelling, because the two forms are indistinguishable to any tolerance-based test.
2. **``residual_proj`` is exactly zero at init and NOT dead after one real optimizer
   step.** A zero-initialised weight looks dead at init by construction, so liveness is
   asserted after the step and reported BY NAME, never as a count.
3. **The composition order** -- ``out * gate + residual``, the gate multiplying the
   dechunked branch only. Both orderings are forward-identical while the gate returns
   exactly 1.0, so the guard substitutes a non-unit gate to make the order visible.
4. **The recursion**: routing records propagate outward, outermost first, one per
   chunking level; ``build()`` materialises exactly the tree ``call()`` runs; and the
   whole nest round-trips through ``get_config``/``from_config`` and through ``.keras``
   on VALUES.
5. **``pad_dimension``** widens on entry and is sliced back off on exit.

Numerical tolerances
--------------------
Every ``assert_allclose`` passes ``rtol=0`` and every bound is derived at its call site.
``tests/numerics.reassociation_atol`` is deliberately NOT used: D-012 measured it
UNDER-counting against a float64 oracle. Both non-zero-bound comparisons here are
recompositions of the subject's own sub-layers in the subject's own order, so the bound
is an exact ``0.0`` and ``0.0`` is attained; each has a twin proving it can fail.

Everything runs on CPU with deliberately small dimensions -- this is a correctness step.
"""

import inspect
import re

import keras
import numpy as np
import pytest

from dl_techniques.layers.dynamic_chunking.chunk_layer import ChunkLayer
from dl_techniques.layers.dynamic_chunking.dechunk_layer import DeChunkLayer
from dl_techniques.layers.dynamic_chunking.routing_module import RoutingModule
from dl_techniques.models.language.hnet import stage as stage_module
from dl_techniques.models.language.hnet.components import HNetIsotropic
from dl_techniques.models.language.hnet.config import (
    AttnSpec,
    HNetArchConfig,
    SSMSpec,
)
from dl_techniques.models.language.hnet.stage import (
    ROUTING_RECORD_KEYS,
    HNetStage,
    straight_through_ones,
)

from ..gradient_flow_oracle import assert_gradients_reach_every_trainable_weight
from ..test_sam.dead_component_oracle import fit_one_step_moved_variables


# ---------------------------------------------------------------------
# Small architectures. Every dimension here is tiny on purpose.
# ---------------------------------------------------------------------

SSM = SSMSpec(d_conv=4, expand=2, d_state=8)


def one_stage_config(d_model=16):
    """``["m1", ["T1"], "m1"]`` -- ONE chunking level, encoder/decoder + innermost."""
    return HNetArchConfig(
        arch_layout=["m1", ["T1"], "m1"],
        d_model=[d_model, d_model],
        d_intermediate=[0, 0],
        ssm_cfg=SSM,
        attn_cfg=AttnSpec(
            num_heads=(2, 2), rotary_emb_dim=(4, 4), window_size=(-1, -1)
        ),
    )


def two_stage_config(widths=(16, 16, 32)):
    """``["m1", ["T1m1", ["T1"], "m1T1"], "m1"]`` -- TWO chunking levels, and the
    innermost stage is WIDER than its parent so ``pad_dimension`` is exercised."""
    return HNetArchConfig(
        arch_layout=["m1", ["T1m1", ["T1"], "m1T1"], "m1"],
        d_model=list(widths),
        d_intermediate=[0, 0, 0],
        ssm_cfg=SSM,
        attn_cfg=AttnSpec(
            num_heads=(2, 2, 2),
            rotary_emb_dim=(4, 4, 8),
            window_size=(-1, -1, -1),
        ),
    )


def asymmetric_config(d_model=16):
    """Encoder and decoder deliberately DIFFERENT, so a swap is observable."""
    return HNetArchConfig(
        arch_layout=["m2", ["T1"], "T1"],
        d_model=[d_model, d_model],
        d_intermediate=[0, 0],
        ssm_cfg=SSM,
        attn_cfg=AttnSpec(
            num_heads=(2, 2), rotary_emb_dim=(4, 4), window_size=(-1, -1)
        ),
    )


def make_stage(config=None, max_chunks=(4,), seed=13, name=None, **kwargs):
    """Build a deterministic outermost stage."""
    keras.utils.set_random_seed(seed)
    config = one_stage_config() if config is None else config
    stage = HNetStage(
        config,
        stage_idx=0,
        max_chunks=max_chunks,
        max_seq_len=64,
        headdim=8,
        name=name,
        **kwargs,
    )
    stage.build((None, None, config.d_model[0]))
    return stage


def wrap(stage, seq_len=12):
    """A functional model whose single output is the stage's hidden states."""
    inputs = keras.Input(shape=(seq_len, stage.parent_d_model))
    hidden, _ = stage(inputs)
    return keras.Model(inputs, hidden)


# ---------------------------------------------------------------------
# 1. The straight-through gate
# ---------------------------------------------------------------------


class TestStraightThroughGate:
    """The gate's exact floating-point spelling, pinned two independent ways."""

    def test_the_grouped_form_is_bit_exactly_one_in_float64(self):
        """100000 float64 draws, all bit-exactly 1.0. Attained: 100000/100000, max|d|=0.0.

        The bound is ``atol=0.0`` and it is not aspirational: for ``x >= 0.5`` Sterbenz's
        lemma makes ``1 - x`` exact, and for ``x < 0.5`` the rounding error of ``1 - x``
        is at most half an ulp of 1.0, which the second addition rounds back to 1.0.
        """
        draws = np.random.default_rng(0).uniform(0.0, 1.0, size=100000)
        assert draws.dtype == np.float64

        gate = np.array(straight_through_ones(keras.ops.convert_to_tensor(draws)))

        assert gate.dtype == np.float64, gate.dtype
        exact = int((gate == 1.0).sum())
        assert exact == draws.size, (
            f"the grouped STE is not the identity on {draws.size - exact} of "
            f"{draws.size} float64 draws; max|gate - 1| = {np.max(np.abs(gate - 1.0)):.6e}"
        )

    def test_the_ungrouped_rewrite_is_measurably_a_different_function(self):
        """TWIN: the 0.0 above is a property of the GROUPING, not of the draws.

        Attained: the ungrouped form is exact on 75042 of 100000 draws; the other 24958
        (24.96%) land exactly one ulp away, max|u - 1| = 1.110223e-16 = 2^-53.
        """
        draws = np.random.default_rng(0).uniform(0.0, 1.0, size=100000)
        x = keras.ops.convert_to_tensor(draws)

        ungrouped = np.array(
            x
            + keras.ops.stop_gradient(keras.ops.ones_like(x))
            - keras.ops.stop_gradient(x)
        )

        moved = int((ungrouped != 1.0).sum())
        assert moved > 0, (
            "the ungrouped rewrite reproduced 1.0 exactly on every draw, so this twin "
            "proves nothing about the grouping -- the measurement it is built on "
            "(24958/100000 moved) no longer holds"
        )
        assert np.max(np.abs(ungrouped - 1.0)) < 1e-15

    def test_the_source_writes_the_gate_as_one_grouped_expression(self):
        """SOURCE guard, and it is here by necessity.

        Both forms return 1.0 in float32 on most draws and differ only in the low bit,
        so no value assertion on the assembled stage can distinguish them reliably. The
        text of the shipped expression is therefore the claim.
        """
        source = inspect.getsource(straight_through_ones)
        assert re.search(
            r"return\s+x\s*\+\s*keras\.ops\.stop_gradient\(\s*hard\s*-\s*x\s*\)",
            source,
        ), (
            "straight_through_ones must return the single grouped expression "
            "`x + stop_gradient(hard - x)`; got:\n" + source
        )

    def test_the_gradient_passes_straight_through(self):
        """The whole point of the STE: value 1.0, gradient ``d/dx = 1``."""
        import tensorflow as tf

        variable = tf.Variable(np.array([0.1, 0.3, 0.7, 0.9]))
        weights = tf.constant([2.0, 3.0, 4.0, 5.0], dtype=tf.float64)
        with tf.GradientTape() as tape:
            out = keras.ops.sum(straight_through_ones(variable) * weights)
        gradient = np.array(tape.gradient(out, variable))

        np.testing.assert_allclose(gradient, np.array([2.0, 3.0, 4.0, 5.0]),
                                   rtol=0, atol=0.0)


# ---------------------------------------------------------------------
# 2. The residual projection
# ---------------------------------------------------------------------


class TestResidualProjection:

    def test_it_starts_at_exactly_zero_at_every_stage(self):
        """Kernel AND bias, everywhere in the nest. ``atol = 0.0``, attained."""
        stage = make_stage(two_stage_config(), max_chunks=(6, 3))

        seen = 0
        node = stage
        while node.residual_proj is not None:
            for weight in node.residual_proj.weights:
                value = np.array(weight)
                assert np.max(np.abs(value)) == 0.0, (
                    f"{weight.path} is not exactly zero at init: "
                    f"max|w| = {np.max(np.abs(value)):.6e}"
                )
                seen += 1
            node = node.main_network
        assert seen == 4, f"expected 2 stages x (kernel, bias), saw {seen}"

    def test_it_is_pinned_to_float32(self):
        """The reference's "do the residual in fp32" (``hnet.py:102-105``)."""
        stage = make_stage()
        assert stage.residual_proj.dtype == "float32"
        assert stage.residual_proj.compute_dtype == "float32"

    def test_it_is_not_dead_after_one_real_optimizer_step(self):
        """Liveness AFTER a step, never at init.

        A weight initialised to exactly zero is indistinguishable from a dead one at
        init: any init-time probe would report a false defect. The instrument returns a
        NAME SET, so a dead sub-layer is named rather than counted.
        """
        stage = make_stage()
        model = wrap(stage)
        model.compile(optimizer=keras.optimizers.SGD(1.0), loss="mse")
        x = np.array(keras.random.normal((4, 12, 16), seed=1))
        y = np.array(keras.random.normal((4, 12, 16), seed=2))

        report = fit_one_step_moved_variables(model, x, y)

        assert report.unmoved == (), report.summary()
        residual_weights = [
            label for label in report.moved if "residual_proj" in label
        ]
        assert len(residual_weights) == 2, (
            f"expected the residual projection's kernel and bias among the moved "
            f"variables, found {residual_weights}"
        )

    def test_every_trainable_weight_is_on_the_backward_graph(self):
        """Per-weight, by name -- no aggregate norm can hide one dead tensor."""
        stage = make_stage(two_stage_config(), max_chunks=(6, 3))
        model = wrap(stage)
        x = np.array(keras.random.normal((2, 12, 16), seed=4))

        assert_gradients_reach_every_trainable_weight(model, x, training=False)


# ---------------------------------------------------------------------
# 3. The composition order
# ---------------------------------------------------------------------


class TestCompositionOrder:
    """``out * gate + residual`` -- and why this needs a substituted gate.

    While the gate returns exactly 1.0, ``out * gate + residual`` and
    ``(out + residual) * gate`` are the SAME numbers in the forward pass, and at init
    ``residual`` is exactly zero so even the gradients agree. The order is real
    (it changes what the router's gradient sees once the residual is non-zero) but it is
    invisible to a direct value test. Substituting a non-unit gate makes it visible.
    """

    @staticmethod
    def _stage_with_a_live_residual(seed=21):
        stage = make_stage(seed=seed)
        kernel, bias = stage.residual_proj.weights
        kernel.assign(np.eye(16, dtype="float32") * 0.5)
        bias.assign(np.full((16,), 0.25, dtype="float32"))
        return stage

    def test_the_gate_multiplies_the_dechunked_branch_only(self, monkeypatch):
        """With a gate of 2.0, the two orderings differ by ``2 * residual``.

        Bound: ``atol = 0.0``. The expected value is recomputed from the stage's own
        sub-layers in the stage's own order, so the two computations are the identical
        op sequence and 0.0 is attained.
        """
        stage = self._stage_with_a_live_residual()
        x = np.array(keras.random.normal((2, 12, 16), seed=8))

        monkeypatch.setattr(
            stage_module,
            "straight_through_ones",
            lambda t: keras.ops.ones_like(t) * 2.0,
        )
        actual = np.array(stage(x, training=False)[0])

        # Hand recomposition, gate applied to the dechunked branch ONLY.
        hidden = stage.encoder(x, training=False)
        residual = stage.residual_proj(hidden)
        prob, mask, selected = stage.routing_module(hidden, training=False)
        inner, inner_mask = stage.chunk_layer(hidden, boundary_mask=mask,
                                              training=False)
        inner, _ = stage.main_network(inner, padding_mask=inner_mask, training=False)
        dechunked = stage.dechunk_layer(
            inner, boundary_prob=prob, boundary_mask=mask, training=False
        )
        expected = np.array(
            stage.decoder(
                dechunked * (keras.ops.ones_like(selected) * 2.0) + residual,
                training=False,
            )
        )

        np.testing.assert_allclose(actual, expected, rtol=0, atol=0.0)

    def test_gating_the_sum_instead_would_be_a_different_model(self, monkeypatch):
        """TWIN: the 0.0 above is not vacuous -- the rejected order differs, a lot.

        Attained separation: max|difference| = 1.317e+00 at this seed, against a
        ``> 1e-3`` bar four orders above the float32 recomposition noise floor.
        """
        stage = self._stage_with_a_live_residual()
        x = np.array(keras.random.normal((2, 12, 16), seed=8))

        monkeypatch.setattr(
            stage_module,
            "straight_through_ones",
            lambda t: keras.ops.ones_like(t) * 2.0,
        )
        actual = np.array(stage(x, training=False)[0])

        hidden = stage.encoder(x, training=False)
        residual = stage.residual_proj(hidden)
        prob, mask, selected = stage.routing_module(hidden, training=False)
        inner, inner_mask = stage.chunk_layer(hidden, boundary_mask=mask,
                                              training=False)
        inner, _ = stage.main_network(inner, padding_mask=inner_mask, training=False)
        dechunked = stage.dechunk_layer(
            inner, boundary_prob=prob, boundary_mask=mask, training=False
        )
        wrong_order = np.array(
            stage.decoder(
                (dechunked + residual) * (keras.ops.ones_like(selected) * 2.0),
                training=False,
            )
        )

        assert np.max(np.abs(actual - wrong_order)) > 1e-3

    def test_the_unsubstituted_forward_is_the_hand_recomposition(self):
        """The shipped gate, no substitution: the whole pipeline, atol = 0.0 attained."""
        stage = make_stage()
        x = np.array(keras.random.normal((2, 12, 16), seed=9))
        actual = np.array(stage(x, training=False)[0])

        hidden = stage.encoder(x, training=False)
        residual = stage.residual_proj(hidden)
        prob, mask, selected = stage.routing_module(hidden, training=False)
        inner, inner_mask = stage.chunk_layer(hidden, boundary_mask=mask,
                                              training=False)
        inner, _ = stage.main_network(inner, padding_mask=inner_mask, training=False)
        dechunked = stage.dechunk_layer(
            inner, boundary_prob=prob, boundary_mask=mask, training=False
        )
        expected = np.array(
            stage.decoder(
                dechunked * straight_through_ones(selected) + residual, training=False
            )
        )

        np.testing.assert_allclose(actual, expected, rtol=0, atol=0.0)

    def test_the_router_reads_the_ENCODER_output_not_the_raw_input(self):
        """A router fed the raw input picks different boundaries.

        Attained: 11 of 24 positions disagree at this seed. The claim is a genuine
        disagreement, not a tolerance.
        """
        stage = make_stage()
        x = np.array(keras.random.normal((2, 12, 16), seed=12))

        _, records = stage(x, training=False)
        from_encoder = np.array(records[0]["boundary_mask"])
        _, from_raw, _ = stage.routing_module(x, training=False)

        assert not np.array_equal(from_encoder, np.array(from_raw)), (
            "the routing module's boundaries are identical whether it reads the "
            "encoder output or the raw input, so this guard cannot see the swap"
        )


# ---------------------------------------------------------------------
# 4. Encoder / decoder identity
# ---------------------------------------------------------------------


class TestEncoderAndDecoderAreNotInterchangeable:

    def test_each_slot_gets_its_own_layout(self):
        """``["m2", ["T1"], "T1"]``: encoder is ``m2``, decoder is ``T1``."""
        stage = make_stage(asymmetric_config())
        assert stage.encoder.layout == "m2"
        assert stage.decoder.layout == "T1"

    def test_the_slots_hold_different_weight_populations(self):
        """A swap is not merely a naming difference at this layout."""
        stage = make_stage(asymmetric_config())
        assert len(stage.encoder.weights) != len(stage.decoder.weights), (
            "encoder and decoder hold the same number of weights at this layout, so "
            "the structural guard above could pass a swap"
        )


# ---------------------------------------------------------------------
# 5. pad_dimension
# ---------------------------------------------------------------------


class TestPadDimension:

    def test_a_widening_stage_owns_a_zero_initialised_pad_vector(self):
        """Innermost width 32 against a parent's 16 -> a learned ``(16,)`` vector."""
        stage = make_stage(two_stage_config((16, 16, 32)), max_chunks=(6, 3))
        innermost = stage.main_network.main_network

        assert innermost.is_innermost
        assert innermost.pad_dimension is not None
        assert tuple(innermost.pad_dimension.shape) == (16,)
        assert np.max(np.abs(np.array(innermost.pad_dimension))) == 0.0
        assert innermost.pad_dimension.trainable

    def test_a_same_width_stage_owns_no_pad_vector(self):
        """No delta, no parameter -- ``hnet.py:117-118``'s ``else: None``."""
        stage = make_stage(two_stage_config((16, 16, 16)), max_chunks=(6, 3))
        assert stage.pad_dimension is None
        assert stage.main_network.pad_dimension is None
        assert stage.main_network.main_network.pad_dimension is None

    def test_the_pad_is_sliced_back_off_on_exit(self):
        """A widened stage hands its PARENT's width back, not its own."""
        stage = make_stage(two_stage_config((16, 16, 32)), max_chunks=(6, 3))
        innermost = stage.main_network.main_network

        inner_in = keras.random.normal((2, 6, 16), seed=3)
        out, records = innermost(inner_in, training=False)

        assert records == []
        assert tuple(out.shape) == (2, 6, 16), (
            f"the innermost stage returned {tuple(out.shape)}; its parent hands it "
            f"width 16 and must get width 16 back"
        )

    def test_the_pad_vector_is_actually_read(self):
        """Move the vector, move the output. Otherwise it is a dead parameter.

        Attained: max|delta| = 2.70e+00 at this seed, against a ``> 1e-3`` bar.
        """
        stage = make_stage(two_stage_config((16, 16, 32)), max_chunks=(6, 3))
        innermost = stage.main_network.main_network
        inner_in = np.array(keras.random.normal((2, 6, 16), seed=3))

        before = np.array(innermost(inner_in, training=False)[0])
        innermost.pad_dimension.assign(np.full((16,), 0.75, dtype="float32"))
        after = np.array(innermost(inner_in, training=False)[0])

        assert np.max(np.abs(after - before)) > 1e-3

    def test_a_widening_NON_innermost_stage_also_slices_back(self):
        """The middle stage widens too, and it takes the OTHER exit path.

        The innermost branch of ``call`` has its own slice; a suite whose only widening
        stage is the innermost one cannot see the non-innermost slice go missing. This
        arm was added after a mutation that deleted exactly that line SURVIVED.
        """
        stage = make_stage(two_stage_config((16, 32, 32)), max_chunks=(6, 3))
        middle = stage.main_network

        assert not middle.is_innermost
        assert middle.pad_dimension is not None
        assert tuple(middle.pad_dimension.shape) == (16,)

        out, records = middle(keras.random.normal((2, 6, 16), seed=3), training=False)

        assert tuple(out.shape) == (2, 6, 16), (
            f"the middle stage returned {tuple(out.shape)}; its parent hands it width "
            f"16 and must get width 16 back"
        )
        assert len(records) == 1

    def test_a_narrowing_hierarchy_is_refused(self):
        """The reference has no defined behaviour here; this port says so loudly."""
        with pytest.raises(ValueError, match="NARROWER than its parent"):
            HNetStage(two_stage_config((16, 32, 16)), stage_idx=0,
                      max_chunks=(6, 3), headdim=8)


# ---------------------------------------------------------------------
# 6. The recursion and the routing records
# ---------------------------------------------------------------------


class TestRecursion:

    @pytest.mark.parametrize(
        "config, max_chunks, expected_levels",
        [
            (one_stage_config(), (4,), 1),
            (two_stage_config(), (6, 3), 2),
        ],
        ids=["one_chunking_level", "two_chunking_levels"],
    )
    def test_one_routing_record_per_chunking_level(
        self, config, max_chunks, expected_levels
    ):
        """A 1-level suite cannot see a recursion defect; both layouts run."""
        stage = make_stage(config, max_chunks=max_chunks)
        x = np.array(keras.random.normal((2, 12, config.d_model[0]), seed=6))

        hidden, records = stage(x, training=False)

        assert tuple(hidden.shape) == (2, 12, config.d_model[0])
        assert len(records) == expected_levels, (
            f"expected {expected_levels} routing records, got {len(records)}; the "
            f"inner stage's records are not being propagated outward"
        )

    def test_every_record_carries_exactly_the_declared_keys(self):
        stage = make_stage(two_stage_config(), max_chunks=(6, 3))
        x = np.array(keras.random.normal((2, 12, 16), seed=6))
        _, records = stage(x, training=False)

        for record in records:
            assert tuple(record) == ROUTING_RECORD_KEYS, tuple(record)

    def test_the_records_are_ordered_outermost_first(self):
        """Record 0 is THIS stage's; record 1 lives at the inner resolution.

        The widths separate them without ambiguity: the outer router runs at ``L = 12``,
        the inner one at the outer chunk cap ``C = 6``.
        """
        stage = make_stage(two_stage_config(), max_chunks=(6, 3))
        x = np.array(keras.random.normal((2, 12, 16), seed=6))
        _, records = stage(x, training=False)

        assert tuple(records[0]["boundary_mask"].shape) == (2, 12)
        assert tuple(records[1]["boundary_mask"].shape) == (2, 6)

    def test_the_outer_record_is_the_outer_routers_own_output(self):
        """Recomposed from the stage's own sub-layers. ``atol = 0.0``, attained."""
        stage = make_stage()
        x = np.array(keras.random.normal((2, 12, 16), seed=7))
        _, records = stage(x, training=False)

        prob, mask, selected = stage.routing_module(
            stage.encoder(x, training=False), training=False
        )
        np.testing.assert_allclose(
            np.array(records[0]["boundary_prob"]), np.array(prob), rtol=0, atol=0.0
        )
        np.testing.assert_array_equal(
            np.array(records[0]["boundary_mask"]), np.array(mask)
        )
        np.testing.assert_allclose(
            np.array(records[0]["selected_probs"]), np.array(selected),
            rtol=0, atol=0.0,
        )

    def test_the_innermost_stage_returns_no_records(self):
        stage = make_stage(two_stage_config(), max_chunks=(6, 3))
        innermost = stage.main_network.main_network
        assert innermost.is_innermost
        assert innermost.routing_module is None
        assert innermost.chunk_layer is None
        assert innermost.residual_proj is None
        assert innermost(keras.random.normal((2, 6, 16), seed=3), training=False)[1] == []

    def test_the_nest_holds_the_expected_layer_types(self):
        stage = make_stage(two_stage_config(), max_chunks=(6, 3))

        assert isinstance(stage.encoder, HNetIsotropic)
        assert isinstance(stage.routing_module, RoutingModule)
        assert isinstance(stage.chunk_layer, ChunkLayer)
        assert isinstance(stage.dechunk_layer, DeChunkLayer)
        assert isinstance(stage.main_network, HNetStage)
        assert isinstance(stage.main_network.main_network, HNetStage)
        assert isinstance(stage.main_network.main_network.main_network, HNetIsotropic)

    def test_each_stage_takes_its_own_chunk_cap(self):
        stage = make_stage(two_stage_config(), max_chunks=(6, 3))
        assert stage.chunk_layer.max_chunks == 6
        assert stage.main_network.chunk_layer.max_chunks == 3

    def test_a_wrong_length_max_chunks_is_refused(self):
        with pytest.raises(ValueError, match="one entry per non-innermost stage"):
            HNetStage(two_stage_config(), stage_idx=0, max_chunks=(6,), headdim=8)


# ---------------------------------------------------------------------
# 7. build() materialises exactly the tree call() runs
# ---------------------------------------------------------------------


def weight_paths(layer):
    return sorted(weight.path for weight in layer.weights)


class TestBuildMaterializesTheTree:

    def test_explicit_build_reaches_the_same_weights_a_forward_pass_does(self):
        """The defect this guards is SILENT: identical shapes, identical totals.

        A ``build()`` that misses a sub-layer produces a model that trains and saves and
        reloads with no error and different numbers. The comparison is on the weight
        NAME SET, both directions, never on a count.
        """
        config = two_stage_config()
        keras.utils.set_random_seed(13)
        explicit = HNetStage(config, stage_idx=0, max_chunks=(6, 3),
                             max_seq_len=64, headdim=8, name="stage")
        explicit.build((None, None, 16))

        keras.utils.set_random_seed(13)
        lazy = HNetStage(config, stage_idx=0, max_chunks=(6, 3),
                         max_seq_len=64, headdim=8, name="stage")
        lazy(keras.random.normal((2, 12, 16), seed=1), training=False)

        assert weight_paths(explicit) == weight_paths(lazy)
        assert len(weight_paths(explicit)) > 0

    def test_the_inner_stage_is_materialised_too(self):
        """Named explicitly: a build that stops at the outer stage is the easy defect."""
        config = two_stage_config()
        stage = HNetStage(config, stage_idx=0, max_chunks=(6, 3),
                          max_seq_len=64, headdim=8)
        stage.build((None, None, 16))

        assert stage.built
        assert stage.main_network.built
        assert stage.main_network.main_network.built
        assert len(stage.main_network.main_network.weights) > 0

    def test_a_wrong_input_width_is_refused_at_build(self):
        config = two_stage_config()
        stage = HNetStage(config, stage_idx=0, max_chunks=(6, 3), headdim=8)
        with pytest.raises(ValueError, match="expects an input width of 16"):
            stage.build((None, 12, 24))


# ---------------------------------------------------------------------
# 8. Symbolic build
# ---------------------------------------------------------------------


class TestSymbolicBuild:

    def test_it_traces_at_a_fully_dynamic_shape(self):
        """``(None, None, D)``. The recursion needs it and step 5 made ``C > L`` legal."""
        stage = make_stage(two_stage_config(), max_chunks=(6, 3))
        hidden, records = stage(keras.KerasTensor((None, None, 16)))

        assert tuple(hidden.shape) == (None, None, 16)
        assert len(records) == 2

    def test_the_routing_records_keep_their_dtypes(self):
        """Guards the deliberate ABSENCE of ``compute_output_shape`` (D-020).

        With a shape method present Keras builds the output specs from shapes alone and
        types every one of them float32, so a symbolically built model would hand the
        chunk/dechunk layers a float tensor where they expect a boolean one.
        """
        stage = make_stage(two_stage_config(), max_chunks=(6, 3))
        _, records = stage(keras.KerasTensor((None, None, 16)))

        for record in records:
            assert record["boundary_mask"].dtype == "bool", record["boundary_mask"].dtype
            assert record["padding_mask"].dtype == "bool"
            assert record["boundary_prob"].dtype == "float32"

    def test_a_functional_model_wraps_it(self):
        stage = make_stage()
        model = wrap(stage)
        out = model(np.array(keras.random.normal((3, 12, 16), seed=2)), training=False)
        assert tuple(out.shape) == (3, 12, 16)


# ---------------------------------------------------------------------
# 9. Padding
# ---------------------------------------------------------------------


class TestPaddingMask:

    def test_the_mask_reaches_the_router(self):
        """No padded position may be selected as a boundary."""
        stage = make_stage()
        x = np.array(keras.random.normal((2, 12, 16), seed=15))
        mask = np.ones((2, 12), dtype=bool)
        mask[:, 8:] = False

        _, records = stage(x, padding_mask=mask, training=False)
        selected = np.array(records[0]["boundary_mask"])

        assert not selected[:, 8:].any(), selected
        assert np.array_equal(np.array(records[0]["padding_mask"]), mask)

    def test_an_absent_mask_records_an_all_true_one(self):
        """The record never carries ``None``: the ratio loss reads it unconditionally."""
        stage = make_stage()
        x = np.array(keras.random.normal((2, 12, 16), seed=15))
        _, records = stage(x, training=False)
        assert np.array(records[0]["padding_mask"]).all()

    def test_the_inner_stage_is_masked_by_the_chunk_layers_validity_mask(self):
        """Columns past a row's real chunk count are marked invalid inward.

        The outer cap is 10, not 6, and that is load-bearing: at a cap of 6 these rows
        carry 8 boundaries each, ``inner_mask`` is all-True, and the comparison below
        is satisfied by an inner stage that received NO mask at all. A mutation passing
        ``padding_mask=None` inward survived exactly that vacuity. The anti-vacuity
        assertion is kept in the test so the cap cannot be tuned back silently.
        """
        stage = make_stage(two_stage_config(), max_chunks=(10, 3))
        x = np.array(keras.random.normal((2, 12, 16), seed=16))

        _, records = stage(x, training=False)
        hidden = stage.encoder(x, training=False)
        _, boundary_mask, _ = stage.routing_module(hidden, training=False)
        _, inner_mask = stage.chunk_layer(hidden, boundary_mask=boundary_mask,
                                          training=False)

        inner_mask = np.array(inner_mask)
        assert inner_mask.any() and not inner_mask.all(), (
            f"inner_mask is uniform ({inner_mask}), so this comparison cannot "
            f"distinguish a masked inner stage from an unmasked one"
        )
        np.testing.assert_array_equal(
            np.array(records[1]["padding_mask"]), inner_mask
        )


# ---------------------------------------------------------------------
# 10. Serialization, through the recursion
# ---------------------------------------------------------------------


class TestSerialization:

    def test_config_round_trip_reproduces_the_nest_by_value(self):
        """``get_config``/``from_config`` through TWO levels of recursion.

        The comparison is on VALUES with ``training=False`` explicit on both sides -- a
        shape comparison passes for a nest that rebuilt the wrong layouts, and an
        implicit ``training`` differs between a direct call and a reloaded graph.
        Weights are copied across because ``from_config`` rebuilds at a fresh init.
        ``atol = 0.0``, attained.
        """
        stage = make_stage(two_stage_config(), max_chunks=(6, 3))
        x = np.array(keras.random.normal((2, 12, 16), seed=17))
        before = np.array(stage(x, training=False)[0])

        rebuilt = HNetStage.from_config(stage.get_config())
        rebuilt.build((None, None, 16))
        rebuilt.set_weights(stage.get_weights())
        after = np.array(rebuilt(x, training=False)[0])

        np.testing.assert_allclose(after, before, rtol=0, atol=0.0)

    def test_the_rebuilt_nest_has_the_same_structure(self):
        stage = make_stage(two_stage_config(), max_chunks=(6, 3))
        rebuilt = HNetStage.from_config(stage.get_config())
        rebuilt.build((None, None, 16))

        assert weight_paths(rebuilt) == weight_paths(stage)
        assert rebuilt.main_network.chunk_layer.max_chunks == 3
        assert rebuilt.main_network.main_network.is_innermost

    def test_the_config_is_json_safe(self):
        """A ``.keras`` archive JSON-serializes the config; tuples and dataclasses are not."""
        import json

        config = make_stage(two_stage_config(), max_chunks=(6, 3)).get_config()
        json.dumps(config)

    def test_keras_round_trip_preserves_VALUES(self, tmp_path):
        """Saved and reloaded, the wrapped model returns the same numbers.

        ``atol = 0.0`` and 0.0 is attained: the reload reconstructs the identical graph
        and reads back the identical weights, so the bound is exact, not statistical.
        """
        stage = make_stage(two_stage_config(), max_chunks=(6, 3))
        model = wrap(stage)
        x = np.array(keras.random.normal((3, 12, 16), seed=18))
        before = np.array(model(x, training=False))

        path = tmp_path / "hnet_stage.keras"
        model.save(path)
        reloaded = keras.models.load_model(path)
        after = np.array(reloaded(x, training=False))

        np.testing.assert_allclose(after, before, rtol=0, atol=0.0)

    def test_the_round_trip_comparison_can_fail(self, tmp_path):
        """TWIN: the 0.0 above is not an artifact of comparing a tensor with itself."""
        stage = make_stage(two_stage_config(), max_chunks=(6, 3))
        model = wrap(stage)
        x = np.array(keras.random.normal((3, 12, 16), seed=18))
        other = np.array(keras.random.normal((3, 12, 16), seed=19))

        path = tmp_path / "hnet_stage.keras"
        model.save(path)
        reloaded = keras.models.load_model(path)

        assert np.max(
            np.abs(np.array(reloaded(other, training=False))
                   - np.array(model(x, training=False)))
        ) > 1e-3


# ---------------------------------------------------------------------
# 11. Constructor contract
# ---------------------------------------------------------------------


class TestConstructorContract:

    def test_registration_strips_the_family_directory(self):
        registered = keras.saving.get_registered_name(HNetStage)
        assert registered == "dl_techniques.models.hnet.stage>HNetStage", registered

    @pytest.mark.parametrize("stage_idx", [-1, 2])
    def test_an_out_of_range_stage_index_is_refused(self, stage_idx):
        with pytest.raises(ValueError, match="stage_idx must lie in"):
            HNetStage(one_stage_config(), stage_idx=stage_idx, max_chunks=(4,),
                      headdim=8)

    def test_a_non_config_is_refused(self):
        with pytest.raises(TypeError, match="arch_config must be an HNetArchConfig"):
            HNetStage({"arch_layout": ["m1"]}, stage_idx=0)

    def test_the_inner_stage_inherits_the_shared_knobs(self):
        stage = make_stage(two_stage_config(), max_chunks=(6, 3))
        assert stage.main_network.headdim == stage.headdim == 8
        assert stage.main_network.max_seq_len == stage.max_seq_len == 64
        assert stage.main_network.max_chunks == stage.max_chunks == (6, 3)

    def test_the_per_stage_attention_knobs_are_indexed_by_stage(self):
        """``attn_cfg`` is a per-stage LIST; a stage reading index 0 everywhere is the
        defect this catches."""
        stage = make_stage(two_stage_config(), max_chunks=(6, 3))
        innermost = stage.main_network.main_network

        assert stage.encoder.rotary_emb_dim == 4
        assert innermost.main_network.rotary_emb_dim == 8
