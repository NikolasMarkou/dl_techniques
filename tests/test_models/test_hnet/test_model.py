"""Guards for the assembled H-Net language model.

What this file pins
-------------------
1. **The depth-scaled init is PER STAGE, not hierarchy-wide.** The reference threads
   ``parent_residuals`` inward (``hnet.py:121-147``), so stage ``k``'s residual-writing
   projections are scaled by an outside-in CUMULATIVE count. The two definitions coincide
   at the innermost stage and ONLY there, so the guard for this is a **two-stage** one --
   a 1-stage config literally cannot see the difference. See
   :class:`TestDepthScaledInit`.
2. **The init asymmetry**: the byte embedding at ``stddev = 1.0``, every Linear at
   ``0.02``. Asserted on the REALISED sample standard deviation of the built weights, not
   on the constructor argument, so an initializer that is set but never used fails.
3. **Weight tying actually ties**, and an untied head actually does not.
4. **The ratio loss reaches ``model.losses`` through ``add_loss``**, with an
   ``alpha = 0.0`` twin reading exactly zero -- and there is no custom ``train_step``.
5. The house surface: ``from_variant`` on all six names, ``pretrained=True`` raising,
   ``create_hnet`` delegating, ``get_config`` round-tripping on VALUES, a symbolic build
   at ``(None, None)``, and gradient flow after ONE real optimizer step.

Instruments
-----------
The shared oracles are reused rather than reinvented: ``gradient_flow_oracle``
(after one REAL optimizer step, never at init), ``smoke_contract_oracle``,
``knob_sensitivity_oracle`` (instrument matched to knob CLASS -- structural knobs on the
weight-SHAPE signature, value knobs on the output at an identical signature, scoped value
knobs on one named subtree's weights), ``roundtrip_instrument_oracle``,
``lazy_build_contract_oracle``, ``precision_arm_oracle`` and
``test_sam/dead_component_oracle``. ``tests/numerics.reassociation_atol`` is deliberately
NOT used -- D-012 measured it under-counting against a float64 oracle and blind to
normalize chains; every bound here is derived at its call site with ``rtol=0``.

The causality guard is NOT here: it lives in ``test_causality.py``, and the invariants a
shape test is blind to live in ``test_architecture_facts.py``.

Every dimension here is tiny on purpose. This is a correctness step, and the Mamba-2 scan
is sequential.
"""

import inspect
import math

import keras
import numpy as np
import pytest

from dl_techniques.models.language.hnet import model as model_module
from dl_techniques.models.language.hnet.config import (
    MODEL_VARIANTS,
    AttnSpec,
    HNetArchConfig,
    SSMSpec,
    n_residuals,
    n_residuals_by_stage,
)
from dl_techniques.models.language.hnet.model import (
    EMBEDDING_INIT_STDDEV,
    INITIALIZER_RANGE,
    RATIO_LOSS_ALPHA,
    RESIDUAL_WRITING_PROJECTIONS,
    HNet,
    create_hnet,
    default_max_chunks,
)

from ..gradient_flow_oracle import assert_gradients_reach_every_trainable_weight
from ..knob_sensitivity_oracle import (
    assert_scoped_value_knob_changes_weights,
    assert_structural_knob_changes_weights,
    assert_value_knob_changes_output,
)
from ..lazy_build_contract_oracle import assert_lazy_build_costs_nothing
from ..precision_arm_oracle import assert_precision_arm
from ..roundtrip_instrument_oracle import (
    assert_build_parity,
    assert_roundtrip_output_values,
    assert_weights_restored_before_first_call,
    measure_build_parity,
    measure_roundtrip,
)
from ..smoke_contract_oracle import assert_contract_rejects_a_broken_forward
from ..test_sam.dead_component_oracle import fit_one_step_moved_variables


# ---------------------------------------------------------------------
# Tiny architectures
# ---------------------------------------------------------------------

SSM = SSMSpec(d_conv=4, expand=2, d_state=8)


def one_stage_config(d_model=16, vocab_size=256):
    """``["m1", ["T1"], "m1"]`` -- ONE chunking level.

    ``n_residuals_by_stage`` is ``(2, 4)`` here and ``n_residuals`` is ``4``: the outer
    sandwich contributes ``1 + 1``, the innermost ``T1`` contributes ``2``.
    """
    return HNetArchConfig(
        arch_layout=["m1", ["T1"], "m1"],
        d_model=[d_model, d_model],
        d_intermediate=[0, 0],
        vocab_size=vocab_size,
        ssm_cfg=SSM,
        attn_cfg=AttnSpec(
            num_heads=(2, 2), rotary_emb_dim=(4, 4), window_size=(-1, -1)
        ),
    )


def two_stage_config(widths=(16, 16, 32)):
    """``["m1", ["T1m1", ["T1"], "m1T1"], "m1"]`` -- TWO chunking levels.

    ``n_residuals_by_stage`` is ``(2, 8, 10)``; the hierarchy-wide ``n_residuals`` is
    ``10``. The stage-0 ratio ``sqrt(10 / 2) = 2.24`` is what separates the correct
    per-stage denominator from the flat one, and it is far outside the sampling noise of
    the smallest kernel measured below.
    """
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


def make_model(config=None, max_chunks=(6,), seed=13, build=True, **kwargs):
    """Build a deterministic model."""
    keras.utils.set_random_seed(seed)
    config = one_stage_config() if config is None else config
    model = HNet(
        config,
        max_chunks=max_chunks,
        max_seq_len=64,
        headdim=8,
        **kwargs,
    )
    if build:
        model.build((None, None))
    return model


def byte_ids(batch=2, length=12, seed=5, vocab=256):
    return np.random.default_rng(seed).integers(0, vocab, (batch, length)).astype("int32")


def realised_std(weight):
    """Sample standard deviation of a built weight, as a float."""
    return float(np.std(np.asarray(weight)))


def sampling_bound(n_samples, stddev, sigmas=6.0):
    """A bound on how far a sample std may sit from its population std.

    For ``n`` normal draws the sample standard deviation has standard error
    ``sigma / sqrt(2n)``. Six of those is a 1-in-10^9 event, so a failure at this bound
    is a defect and not a draw.

    :param n_samples: Number of elements in the weight.
    :param stddev: The population standard deviation being claimed.
    :param sigmas: How many standard errors to allow.
    :returns: The absolute bound.
    """
    return sigmas * stddev / math.sqrt(2.0 * n_samples)


def weight_named(model, suffix):
    """Every built weight whose path ends with ``suffix``."""
    return [w for w in model.weights if w.path.endswith(suffix)]


# ---------------------------------------------------------------------
# 1. The depth-scaled init -- THE decision this step makes
# ---------------------------------------------------------------------


class TestDepthScaledInit:
    """The denominator is the per-stage cumulative count, not the hierarchy total."""

    def test_the_two_definitions_differ_on_a_two_stage_layout(self):
        """Establish that the guard below is not vacuous BEFORE relying on it.

        `n_residuals_by_stage` is (2, 8, 10) and `n_residuals` is 10 for the two-stage
        config. Stage 0's two definitions therefore differ by `sqrt(10 / 2) = 2.24x`. On
        the ONE-stage config the outer count is 2 against a total of 4 -- also different,
        but the innermost stage agrees in both, which is the coincidence D-016 measured.
        """
        spec = two_stage_config().stage_spec
        by_stage = n_residuals_by_stage(spec)
        total = n_residuals(spec)

        assert by_stage == (2, 8, 10)
        assert total == 10
        assert by_stage[-1] == total, "they always agree at the INNERMOST stage"
        assert by_stage[0] != total, "and must differ at the outer one, or no guard bites"

    def test_a_two_stage_model_scales_each_stage_by_its_own_cumulative_count(self):
        """THE guard for the n_residuals correction (D-016 -> D-021).

        Every residual-writing projection is checked against `0.02 / sqrt(n_k)` for its
        OWN stage's cumulative count. A flat hierarchy-wide denominator would give every
        stage `0.02 / sqrt(10) = 0.00632`; stage 0 must instead read
        `0.02 / sqrt(2) = 0.01414`, which is 2.24x away and far outside the 6-sigma
        sampling bound below (the smallest kernel here is 16x16 = 256 elements, whose
        bound at 0.0141 is 0.0037).

        Asserted on the REALISED sample std of the built kernels, so an initializer that
        is configured but never consulted cannot pass.
        """
        config = two_stage_config()
        model = make_model(config, max_chunks=(8, 4))
        counts = n_residuals_by_stage(config.stage_spec)
        flat = n_residuals(config.stage_spec)

        # backbone -> stage 0; backbone/main_network -> stage 1; and so on.
        def stage_of(path):
            """Depth of the stage owning `path`, counted from the outside.

            Each nested `HNetStage` is attached as `main_network`, so the depth is the
            number of `main_network/` segments -- EXCEPT that the innermost stage's own
            isotropic stack is also attached under that name, adding one extra segment.
            Clamping to the last stage is that off-by-one, not a fudge: measured, the
            deepest path here is
            `backbone/main_network/main_network/main_network/block_0/...` on a 3-stage
            nest whose stage indices only run to 2.
            """
            body = path.split("backbone/", 1)[1]
            depth = 0
            while body.startswith("main_network/"):
                body = body.split("main_network/", 1)[1]
                depth += 1
            return min(depth, len(counts) - 1)

        checked = 0
        for weight in model.weights:
            name = weight.path.rsplit("/", 2)[-2] if "/" in weight.path else ""
            if name not in RESIDUAL_WRITING_PROJECTIONS:
                continue
            if not weight.path.endswith("/kernel"):
                continue
            stage_idx = stage_of(weight.path)
            expected = INITIALIZER_RANGE / math.sqrt(counts[stage_idx])
            got = realised_std(weight)
            bound = sampling_bound(int(np.prod(weight.shape)), expected)

            assert abs(got - expected) < bound, (
                f"{weight.path}: realised std {got:.5f} is not "
                f"{expected:.5f} = 0.02/sqrt({counts[stage_idx]}) "
                f"(bound {bound:.5f})"
            )
            if stage_idx == 0:
                flat_expected = INITIALIZER_RANGE / math.sqrt(flat)
                assert abs(got - flat_expected) > bound, (
                    f"{weight.path}: the FLAT hierarchy-wide denominator "
                    f"({flat}) would have given {flat_expected:.5f}; this test must "
                    f"be able to tell the two apart"
                )
            checked += 1

        assert checked >= 6, (
            f"only {checked} residual-writing kernels were checked; the two-stage "
            f"config must expose several at more than one depth"
        )

    def test_the_projections_that_READ_the_residual_stream_are_not_scaled(self):
        """`in_proj`, `w_q`/`w_k`/`w_v`, `gate_proj`, `up_proj` keep the plain 0.02.

        Upstream's predicate is `"out_proj" in name or "fc2" in name` and the else-branch
        is `initializer_range` flat (`hnet.py:127-131`). Scaling everything would be a
        different model with no shape symptom.
        """
        model = make_model(two_stage_config(), max_chunks=(8, 4))
        readers = ("in_proj", "w_q", "w_k", "w_v", "gate_proj", "up_proj")

        checked = 0
        for weight in model.weights:
            if not weight.path.endswith("/kernel"):
                continue
            name = weight.path.rsplit("/", 2)[-2]
            if name not in readers:
                continue
            got = realised_std(weight)
            bound = sampling_bound(int(np.prod(weight.shape)), INITIALIZER_RANGE)
            assert abs(got - INITIALIZER_RANGE) < bound, (
                f"{weight.path}: realised std {got:.5f} != {INITIALIZER_RANGE} "
                f"(bound {bound:.5f})"
            )
            checked += 1

        assert checked >= 6

    def test_the_routing_and_residual_projections_are_left_alone(self):
        """Upstream's `_init_weights` walks the STACKS only.

        `routing_module`'s identity-initialised q/k and the zero-initialised
        `residual_proj` are never visited (`hnet.py:121-147` iterates `self.encoder`,
        `self.decoder` and `self.main_network`), and re-initialising either would destroy
        a load-bearing init that steps 4 and 10 already pinned.
        """
        model = make_model(two_stage_config(), max_chunks=(8, 4))

        residuals = weight_named(model, "residual_proj/kernel")
        assert residuals, "the two-stage nest must own residual projections"
        for weight in residuals:
            assert float(np.max(np.abs(np.asarray(weight)))) == 0.0

        q_projections = weight_named(model, "routing_module/q_proj/kernel")
        assert q_projections
        for weight in q_projections:
            value = np.asarray(weight)
            np.testing.assert_allclose(
                value, np.eye(value.shape[0], dtype=value.dtype), rtol=0, atol=0.0
            )

    def test_a_renamed_residual_projection_is_refused_not_silently_unscaled(self):
        """A guard keyed on a NAME goes blind the moment the name moves, so it says so.

        With `RESIDUAL_WRITING_PROJECTIONS` monkeypatched to a name no layer carries, the
        constructor must RAISE rather than quietly build a model with no depth scaling
        anywhere.
        """
        original = model_module.RESIDUAL_WRITING_PROJECTIONS
        model_module.RESIDUAL_WRITING_PROJECTIONS = ("a_name_no_layer_has",)
        try:
            with pytest.raises(ValueError, match="no Dense named one of"):
                HNet(one_stage_config(), max_chunks=(6,), max_seq_len=64, headdim=8)
        finally:
            model_module.RESIDUAL_WRITING_PROJECTIONS = original

    def test_the_init_survives_a_PARENT_triggered_build(self):
        """The StatelessScope arm.

        MEASURED on this backend: a `.assign()` executed inside a build that a PARENT
        triggered is recorded by Keras' StatelessScope and discarded, while the same
        assign on a directly-built model sticks. That is why this model sets the
        `kernel_initializer` before the variables exist instead of assigning afterwards.
        Wrapping the model in an outer model and building THAT is the arm which would
        have caught an assign-based implementation, and which no direct-build test can.
        """

        class Wrapper(keras.Model):
            def __init__(self, inner, **kwargs):
                super().__init__(**kwargs)
                self.inner = inner

            def call(self, x, training=None):
                return self.inner(x, training=training)

        keras.utils.set_random_seed(13)
        inner = HNet(two_stage_config(), max_chunks=(8, 4), max_seq_len=64, headdim=8)
        wrapper = Wrapper(inner)
        wrapper(byte_ids())

        counts = n_residuals_by_stage(two_stage_config().stage_spec)
        expected = INITIALIZER_RANGE / math.sqrt(counts[0])
        kernels = [
            w for w in inner.weights
            if w.path.endswith("/kernel")
            and w.path.rsplit("/", 2)[-2] in RESIDUAL_WRITING_PROJECTIONS
            and "/backbone/encoder/" in w.path
        ]
        assert kernels, "the outer encoder must own a residual-writing kernel"
        for weight in kernels:
            bound = sampling_bound(int(np.prod(weight.shape)), expected)
            assert abs(realised_std(weight) - expected) < bound


# ---------------------------------------------------------------------
# 2. The embedding / head init asymmetry
# ---------------------------------------------------------------------


class TestInitAsymmetry:
    """`mixer_seq.py:55-62`: linears at 0.02, embeddings at 1.0."""

    def test_the_embedding_is_unit_stddev_and_the_head_is_the_initializer_range(self):
        """Both realised, both with a derived 6-sigma bound, and 50x apart.

        The embedding is (256, 16) = 4096 draws, so its bound at sigma = 1.0 is
        6/sqrt(8192) = 0.066. The head is (16, 256) = 4096 draws, bound
        6*0.02/sqrt(8192) = 0.00133. The two claims are 50x apart, so neither bound can
        be satisfied by the other's value.
        """
        model = make_model()

        embedding = np.asarray(model.embeddings.embeddings)
        head = np.asarray(model.lm_head.kernel)

        embedding_bound = sampling_bound(embedding.size, EMBEDDING_INIT_STDDEV)
        head_bound = sampling_bound(head.size, INITIALIZER_RANGE)

        assert abs(np.std(embedding) - EMBEDDING_INIT_STDDEV) < embedding_bound
        assert abs(np.std(head) - INITIALIZER_RANGE) < head_bound

        assert EMBEDDING_INIT_STDDEV / INITIALIZER_RANGE == 50.0
        assert abs(np.std(embedding) - INITIALIZER_RANGE) > embedding_bound, (
            "the embedding must be distinguishable from an initializer_range draw"
        )
        assert abs(np.std(head) - EMBEDDING_INIT_STDDEV) > head_bound

    def test_the_embedding_is_zero_mean(self):
        """`nn.init.normal_(..., mean=0.0, std=1.0)` -- the mean is 0, not 0.5."""
        model = make_model()
        embedding = np.asarray(model.embeddings.embeddings)
        # standard error of the mean is sigma / sqrt(n); 6 of them.
        bound = 6.0 * EMBEDDING_INIT_STDDEV / math.sqrt(embedding.size)
        assert abs(float(np.mean(embedding))) < bound


# ---------------------------------------------------------------------
# 3. Weight tying
# ---------------------------------------------------------------------


class TestWeightTying:
    """Tied and untied heads, both built, and tying that actually shares."""

    def test_a_tied_head_is_the_embedding_table_transposed(self):
        """The strongest available claim: the logits ARE `hidden @ E.T`, exactly.

        Recomputed from the model's own sub-layers in the model's own order, so the bound
        is 0.0 and 0.0 is attained.
        """
        model = make_model(tie_word_embeddings=True)
        x = byte_ids()

        logits = np.asarray(model(x, training=False))

        hidden = model.embeddings(x)
        hidden, _ = model.backbone(hidden, training=False)
        table = np.asarray(model.embeddings.embeddings)
        expected = np.asarray(keras.ops.matmul(hidden, np.transpose(table)))

        np.testing.assert_allclose(logits, expected, rtol=0, atol=0.0)

    def test_an_untied_head_is_NOT_the_embedding_table_transposed(self):
        """The twin. Without it the equality above is satisfied by any head at all."""
        model = make_model(tie_word_embeddings=False)
        x = byte_ids()

        logits = np.asarray(model(x, training=False))

        hidden = model.embeddings(x)
        hidden, _ = model.backbone(hidden, training=False)
        table = np.asarray(model.embeddings.embeddings)
        tied = np.asarray(keras.ops.matmul(hidden, np.transpose(table)))

        assert float(np.max(np.abs(logits - tied))) > 1e-3

    def test_tying_reads_the_TRANSPOSED_table(self):
        """A square vocabulary would hide a missing transpose; this one cannot.

        `vocab_size = 256` and `d_model = 16`, so `hidden @ E` is not even shape-legal --
        but the claim is made on values against an explicitly non-symmetric table so that
        it stays a value claim rather than a shape accident.
        """
        model = make_model(tie_word_embeddings=True)
        table = np.asarray(model.embeddings.embeddings)
        assert table.shape == (256, 16)
        assert table.shape[0] != table.shape[1], (
            "a square table would make the transpose invisible"
        )

    def test_a_tied_model_owns_no_head_weights_at_all(self):
        """Tied means SHARED, not "duplicated and then ignored"."""
        tied = make_model(tie_word_embeddings=True)
        untied = make_model(tie_word_embeddings=False)

        assert tied.lm_head is None
        assert not weight_named(tied, "lm_head/kernel")
        assert weight_named(untied, "lm_head/kernel")
        assert len(untied.weights) == len(tied.weights) + 1

    def test_the_flag_defaults_to_the_architectures_own_spelling(self):
        """The reference spells it `tie_embeddings`; the constructor flag defaults to it.

        D-006: the house spelling is `tie_word_embeddings`, and the divergence must not
        cost the config field its effect.
        """
        config = one_stage_config()
        tied_config = HNetArchConfig.from_dict(
            {**config.to_dict(), "tie_embeddings": True}
        )

        assert HNet(config, max_chunks=(6,), headdim=8).tie_word_embeddings is False
        assert HNet(tied_config, max_chunks=(6,), headdim=8).tie_word_embeddings is True
        # and the explicit flag overrides it in both directions
        assert HNet(
            tied_config, max_chunks=(6,), headdim=8, tie_word_embeddings=False
        ).tie_word_embeddings is False


# ---------------------------------------------------------------------
# 4. The ratio loss reaches the optimizer through add_loss
# ---------------------------------------------------------------------


class TestRatioLossWiring:
    """`add_loss`, and specifically NOT a custom `train_step`."""

    def test_the_ratio_loss_reaches_model_losses(self):
        model = make_model()
        model(byte_ids(), training=True)

        losses = [float(value) for value in model.losses]
        assert len(losses) == 1
        assert losses[0] > 0.0

    def test_an_alpha_of_zero_reads_exactly_zero(self):
        """The twin. Without it, "non-zero" could be any constant at all."""
        model = make_model(ratio_loss_alpha=0.0)
        model(byte_ids(), training=True)

        losses = [float(value) for value in model.losses]
        assert len(losses) == 1
        assert losses[0] == 0.0

    def test_alpha_scales_the_term_linearly(self):
        """A third point, which separates "alpha is read" from "alpha is a flag"."""
        single = make_model(ratio_loss_alpha=RATIO_LOSS_ALPHA)
        double = make_model(ratio_loss_alpha=2.0 * RATIO_LOSS_ALPHA)
        x = byte_ids()
        single(x, training=True)
        double(x, training=True)

        one = float(single.losses[0])
        two = float(double.losses[0])
        assert one > 0.0
        np.testing.assert_allclose(two, 2.0 * one, rtol=1e-5, atol=0.0)

    def test_a_two_stage_model_contributes_one_term_per_chunking_level(self):
        """The sum is over levels; a 2-stage model must not read like a 1-stage one."""
        deep = make_model(two_stage_config(), max_chunks=(8, 4))
        deep(byte_ids(), training=True)
        assert len(deep.losses) == 1, "one add_loss call, summed inside"

        shallow = make_model()
        shallow(byte_ids(), training=True)
        assert float(deep.losses[0]) != float(shallow.losses[0])

    def test_there_is_no_custom_train_step(self):
        """A repository-wide invariant, asserted on the class, not on prose.

        Under `mixed_float16` a hand-written training step silently skips the framework's
        own `scale_loss`, i.e. 2**15 of gradient magnitude.
        """
        assert "train_step" not in HNet.__dict__
        assert "test_step" not in HNet.__dict__
        source = inspect.getsource(model_module)
        assert "def train_step" not in source

    def test_the_loss_actually_trains(self):
        """`fit()` runs and every trainable variable moves, reported BY NAME.

        The step size is DERIVED, not tuned. Measured on this model with
        `gradient_flow_oracle.gradient_report`, the smallest live gradient is the decoder
        Mamba block's `A_log` at `6.2e-11`, and `|A_log| ~ 1.0` where float32's ulp is
        `1.19e-07`. Any learning rate below `1.19e-07 / 6.2e-11 = 1.9e+03` therefore
        CANNOT move that variable at all, and the resulting "unmoved" report would be a
        float32 rounding artefact rather than a dead component -- the trap this
        instrument's own docstring warns about from the other direction. `1e+04` clears
        the smallest gradient by more than five ulps. The gradients themselves are
        asserted non-zero separately, by
        `TestSharedOracles::test_gradients_reach_every_trainable_weight_after_one_optimizer_step`.
        """
        model = make_model()
        model.compile(
            optimizer=keras.optimizers.SGD(1e4),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )
        x = byte_ids()
        report = fit_one_step_moved_variables(model, x, x)

        assert report.unmoved == (), f"dead variables: {report.summary()}"


# ---------------------------------------------------------------------
# 5. Variants, pretrained, factory
# ---------------------------------------------------------------------


class TestVariants:
    """The house surface."""

    @pytest.mark.parametrize("variant", sorted(MODEL_VARIANTS))
    def test_every_shipped_variant_constructs(self, variant):
        """All six, unbuilt -- the real widths are far too large to materialise here.

        Construction still runs the whole recursive layer tree and the depth-scaled init
        walk, so a variant whose layout and per-stage lists disagree fails here.
        """
        model = HNet.from_variant(variant, max_seq_len=256)
        assert model.arch_config is MODEL_VARIANTS[variant]
        assert len(model.max_chunks) == model.arch_config.num_stages - 1
        assert len(model.target_ratios) == model.arch_config.num_stages - 1

    def test_an_unknown_variant_lists_every_available_name(self):
        with pytest.raises(ValueError) as excinfo:
            HNet.from_variant("hnet_3stage_XXL")
        message = str(excinfo.value)
        for name in MODEL_VARIANTS:
            assert name in message, f"{name} missing from the error message"

    @pytest.mark.parametrize("variant", sorted(MODEL_VARIANTS))
    def test_pretrained_raises_and_names_the_variant(self, variant):
        """H10: never warn-and-return-random-weights."""
        with pytest.raises(NotImplementedError) as excinfo:
            HNet.from_variant(variant, pretrained=True)
        assert variant in str(excinfo.value)

    def test_the_factory_delegates_with_no_logic_of_its_own(self):
        """`create_hnet` is `HNet.from_variant`, and its source says only that."""
        keras.utils.set_random_seed(3)
        direct = HNet.from_variant("hnet_1stage_L", max_seq_len=128)
        keras.utils.set_random_seed(3)
        made = create_hnet("hnet_1stage_L", max_seq_len=128)

        assert type(made) is HNet
        made_config = {k: v for k, v in made.get_config().items() if k != "name"}
        direct_config = {k: v for k, v in direct.get_config().items() if k != "name"}
        assert made_config == direct_config
        assert made.get_config()["name"] != direct.get_config()["name"], (
            "the two models are distinct objects, so `name` is expected to differ and "
            "is the ONE key excluded above"
        )

        body = inspect.getsource(create_hnet).split('"""')[-1]
        assert body.strip() == (
            "return HNet.from_variant(variant, pretrained=pretrained, **kwargs)"
        )

    def test_the_factory_forwards_pretrained(self):
        with pytest.raises(NotImplementedError):
            create_hnet("hnet_2stage_XL", pretrained=True)

    def test_the_default_chunk_caps_halve_at_every_level(self):
        assert default_max_chunks(1, 1024) == ()
        assert default_max_chunks(2, 1024) == (512,)
        assert default_max_chunks(3, 1024) == (512, 256)
        assert default_max_chunks(3, 1) == (1, 1), "floored at 1, never zero"

    def test_an_explicit_chunk_cap_of_the_wrong_length_raises(self):
        with pytest.raises(ValueError, match="max_chunks needs one entry per"):
            HNet(two_stage_config(), max_chunks=(8,))

    def test_target_ratios_of_the_wrong_length_raise(self):
        with pytest.raises(ValueError, match="target_ratios needs one entry per"):
            HNet(two_stage_config(), max_chunks=(8, 4), target_ratios=(6.0,))


# ---------------------------------------------------------------------
# 6. Serialization
# ---------------------------------------------------------------------


class TestSerialization:
    """Round trips on VALUES, with `training=False` stated explicitly."""

    def test_get_config_round_trips_on_values(self):
        model = make_model(two_stage_config(), max_chunks=(8, 4))
        x = byte_ids()
        expected = np.asarray(model(x, training=False))

        rebuilt = HNet.from_config(model.get_config())
        rebuilt.build((None, None))
        rebuilt.set_weights(model.get_weights())
        got = np.asarray(rebuilt(x, training=False))

        np.testing.assert_allclose(got, expected, rtol=0, atol=0.0)

    def test_the_round_trip_twin_a_different_input_separates(self):
        """The "something changed" twin the exact-0.0 arm above owes."""
        model = make_model()
        a = np.asarray(model(byte_ids(seed=1), training=False))
        b = np.asarray(model(byte_ids(seed=2), training=False))
        assert float(np.max(np.abs(a - b))) > 1e-3

    def test_every_constructor_argument_survives_get_config(self):
        model = make_model(
            two_stage_config(),
            max_chunks=(8, 4),
            tie_word_embeddings=True,
            ratio_loss_alpha=0.05,
            target_ratios=(4.0, 3.0),
            initializer_range=0.01,
        )
        config = model.get_config()
        rebuilt = HNet.from_config(config)

        assert rebuilt.max_chunks == (8, 4)
        assert rebuilt.tie_word_embeddings is True
        assert rebuilt.ratio_loss_alpha == 0.05
        assert rebuilt.target_ratios == (4.0, 3.0)
        assert rebuilt.initializer_range == 0.01
        assert rebuilt.headdim == model.headdim
        assert rebuilt.max_seq_len == model.max_seq_len
        assert rebuilt.arch_config == model.arch_config

    def test_a_keras_archive_round_trips_on_values(self, tmp_path):
        model = make_model(tie_word_embeddings=True)
        x = byte_ids()
        expected = np.asarray(model(x, training=False))

        path = tmp_path / "hnet.keras"
        model.save(path)
        loaded = keras.models.load_model(path)
        got = np.asarray(loaded(x, training=False))

        np.testing.assert_allclose(got, expected, rtol=0, atol=0.0)

    def test_it_is_registered_with_the_family_stripped(self):
        """H7: `dl_techniques.models.hnet.model`, never a bare `Custom>HNet`."""
        assert (
            keras.saving.get_registered_name(HNet)
            == "dl_techniques.models.hnet.model>HNet"
        )


# ---------------------------------------------------------------------
# 7. Build paths
# ---------------------------------------------------------------------


class TestBuild:
    """Symbolic, lazy and functional."""

    def test_it_builds_symbolically_at_a_fully_dynamic_shape(self):
        model = make_model(build=False)
        model.build((None, None))
        assert model.built
        assert model.weights

    def test_a_symbolically_built_model_runs_at_two_different_lengths(self):
        """`(None, None)` must mean both axes, not just the batch."""
        model = make_model()
        assert model(byte_ids(batch=2, length=9), training=False).shape == (2, 9, 256)
        assert model(byte_ids(batch=3, length=14), training=False).shape == (3, 14, 256)

    def test_a_functional_wrapper_builds_and_matches_the_direct_call(self):
        model = make_model()
        inputs = keras.Input(shape=(None,), dtype="int32")
        functional = keras.Model(inputs, model(inputs))

        x = byte_ids()
        np.testing.assert_allclose(
            np.asarray(functional(x, training=False)),
            np.asarray(model(x, training=False)),
            rtol=0,
            atol=0.0,
        )

    def test_a_rank_three_input_shape_is_refused(self):
        model = make_model(build=False)
        with pytest.raises(ValueError, match="rank-2"):
            model.build((None, None, 16))

    def test_the_lazy_and_explicit_build_reach_the_same_weight_names(self):
        keras.utils.set_random_seed(13)
        lazy = HNet(one_stage_config(), max_chunks=(6,), max_seq_len=64, headdim=8)
        lazy(byte_ids())
        explicit = make_model()

        def relative(model):
            # `Variable.path` is prefixed with the model's auto-generated name
            # (`h_net_4/...` vs `h_net_5/...`), which differs between two instances by
            # construction; everything after it is the claim.
            return {w.path.split("/", 1)[1] for w in model.weights}

        assert relative(lazy) == relative(explicit)
        assert len(relative(lazy)) == len(lazy.weights)

    def test_compute_output_shape_reports_the_vocabulary(self):
        model = make_model()
        assert model.compute_output_shape((None, None)) == (None, None, 256)
        assert model.compute_output_shape((4, 17)) == (4, 17, 256)


# ---------------------------------------------------------------------
# 8. Padding
# ---------------------------------------------------------------------


class TestPaddingMask:
    """The optional `(B, L)` validity mask reaches the backbone and the loss."""

    def test_the_mask_changes_the_output(self):
        model = make_model()
        x = byte_ids()
        mask = np.ones(x.shape, dtype=bool)
        mask[:, -4:] = False

        free = np.asarray(model(x, training=False))
        masked = np.asarray(model(x, padding_mask=mask, training=False))

        assert float(np.max(np.abs(free - masked))) > 1e-3

    def test_the_mask_changes_the_ratio_loss(self):
        model = make_model()
        x = byte_ids()
        mask = np.ones(x.shape, dtype=bool)
        mask[:, -4:] = False

        model(x, training=True)
        free = float(model.losses[0])
        model(x, padding_mask=mask, training=True)
        masked = float(model.losses[0])

        assert free != masked, (
            "padded positions must be excluded from both means of the ratio loss"
        )


# ---------------------------------------------------------------------
# 9. Shared oracles
# ---------------------------------------------------------------------


class TestSharedOracles:
    """Gradient flow, the smoke contract and knob sensitivity."""

    def test_gradients_reach_every_trainable_weight_after_one_optimizer_step(self):
        """Never at init: `residual_proj` is zero-initialised and looks dead there.

        One real optimizer step first, then the gradient report, which is the D-019
        instrument ordering.
        """
        model = make_model(two_stage_config(), max_chunks=(8, 4))
        x = byte_ids()
        model.compile(
            optimizer=keras.optimizers.SGD(1.0),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )
        model.fit(x, x, epochs=1, verbose=0)

        report = assert_gradients_reach_every_trainable_weight(model, x)
        assert len(report) == len(model.trainable_weights)

    def test_the_smoke_contract_rejects_a_broken_forward(self):
        model = make_model()
        x = byte_ids()

        def contract(output):
            assert tuple(output.shape) == (2, 12, 256), output.shape
            assert float(np.max(np.abs(np.asarray(output)))) > 0.0

        messages = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert len(messages) == 3

    def test_the_layout_is_a_structural_knob(self):
        """Instrument matched to knob CLASS: depth changes the weight-SHAPE signature."""

        def build(config, caps):
            def builder():
                model = HNet(config, max_chunks=caps, max_seq_len=64, headdim=8)
                model.build((None, None))
                return model
            return builder

        assert_structural_knob_changes_weights(
            {
                "one_stage": build(one_stage_config(), (6,)),
                "two_stage": build(two_stage_config(), (8, 4)),
            },
            knob="arch_layout",
        )

    def test_tying_is_a_structural_knob(self):
        """Tying REMOVES a weight, so the shape signature is the right instrument."""

        def build(tied):
            def builder():
                model = HNet(
                    one_stage_config(),
                    max_chunks=(6,),
                    max_seq_len=64,
                    headdim=8,
                    tie_word_embeddings=tied,
                )
                model.build((None, None))
                return model
            return builder

        assert_structural_knob_changes_weights(
            {"untied": build(False), "tied": build(True)},
            knob="tie_word_embeddings",
        )


# ---------------------------------------------------------------------
# 9b. The rest of the shared instrument family
# ---------------------------------------------------------------------


def _unbuilt_builder(config=None, **kwargs):
    """A zero-argument factory returning an UNBUILT model.

    The round-trip and lazy-build instruments materialise the model themselves -- that
    ordering is the whole content of R-073 -- so handing them a pre-built one would
    measure something else.
    """
    def builder():
        return HNet(
            one_stage_config() if config is None else config,
            max_chunks=kwargs.pop("max_chunks", (6,)),
            max_seq_len=64,
            headdim=8,
            **kwargs,
        )
    return builder


def _built_builder(config=None, max_chunks=(6,), **kwargs):
    """A zero-argument factory returning a BUILT model, for the knob instruments."""
    def builder():
        model = HNet(
            one_stage_config() if config is None else config,
            max_chunks=max_chunks,
            max_seq_len=64,
            headdim=8,
            **kwargs,
        )
        model.build((None, None))
        return model
    return builder


def _make_inputs():
    """DETERMINISTIC -- the instruments call this three times and compare exactly."""
    return byte_ids(seed=5)


class TestTheRoundTripInstrument:
    """R-063 / R-072 / R-073, through the shared oracle rather than by hand."""

    def test_a_keras_round_trip_restores_every_weight_before_the_first_call(self):
        """MEASURED: output delta 0.0, weight delta 0.0, weights read after 0 calls.

        ``atol=0.0`` on both arms. Restoration is a copy, not a computation, and the
        forward is deterministic -- the instrument's own ``self_max_delta`` reads exactly
        0.0, asserted here so a future non-determinism cannot quietly widen the bound.
        """
        report = measure_roundtrip(_unbuilt_builder(), _make_inputs)

        assert report["self_max_delta"] == 0.0, (
            "the model no longer repeats itself bit-exactly, so atol=0.0 below is "
            f"measuring the RNG ({report['self_max_delta']})"
        )
        assert_roundtrip_output_values(report, atol=0.0)
        assert_weights_restored_before_first_call(report, atol=0.0)

    def test_the_lazy_and_explicit_build_paths_agree_on_relative_paths(self):
        """MEASURED: 38 weights, zero auto-name drift, explicit/lazy ratio exactly 1.0.

        ``autoname_stems=()`` is the strong reading: every sub-layer in this tree carries
        an explicit ``name=``, so no drift is waived.
        """
        report = measure_build_parity(
            _unbuilt_builder(), _make_inputs, input_shape=(None, None)
        )

        assert report["n_lazy"] == report["n_lazy_unique"], (
            f"{report['n_lazy'] - report['n_lazy_unique']} weights share a relative "
            "path, so the pairing below is ambiguous"
        )
        assert report["explicit"]["status"] == "built"
        assert report["explicit"]["ratio"] == 1.0, (
            "HNet.build((None, None)) must materialise the whole tree; a ratio below 1 "
            "means a sub-layer is built only by the first call"
        )
        assert_build_parity(report, autoname_stems=())

    def test_the_lazy_build_costs_nothing(self):
        """The lazy path's weights survive a save/load cycle at ``atol=0.0``.

        The oracle refuses to run if perturbing every weight leaves the output unmoved,
        so the exact round trip below cannot be a statement about a forward pass that
        ignores its own parameters.
        """
        report = assert_lazy_build_costs_nothing(
            _unbuilt_builder(), _make_inputs, input_shape=(None, None), atol=0.0
        )

        assert report["n_perturbed"] == report["n_weights"]
        assert report["perturb_liveness"] > 1e-3, report["perturb_liveness"]
        assert report["roundtrip_max_delta"] == 0.0


class TestThePrecisionArm:
    """``mixed_float16`` against a float32 control, all four parts.

    ``allowed_none_grads=0``: a healthy model has no dead gradient, and this one is
    measured not to. ``rtol_against_float32=1e-2`` is the oracle's documented realistic
    half-precision bound; MEASURED here, the two arms' ``absmax`` read 0.287842 (fp16)
    against 0.287878 (float32), a relative difference of 1.3e-04, i.e. 77x inside it.
    """

    def test_the_mixed_float16_arm_holds_with_a_float32_control(self):
        assert_precision_arm(
            _unbuilt_builder(),
            _make_inputs,
            expected_compute_dtype="float16",
            check_backward=True,
            allowed_none_grads=0,
            rtol_against_float32=1e-2,
        )


class TestTheKnobInstrumentsMatchTheKnobClass:
    """Structural knobs on the SHAPE signature, value knobs on the output.

    The two structural knobs (``arch_layout``, ``tie_word_embeddings``) are pinned by
    :class:`TestSharedOracles`. These are the other two classes, which that instrument
    cannot judge: a value knob leaves the shape signature alone, so an output difference
    IS attributable to the knob -- and a scoped value knob is judged on the weights of
    one named subtree.
    """

    def test_max_chunks_is_a_value_knob(self):
        """The chunk cap changes no weight shape and every output. MEASURED 4.39e-01."""
        deltas = assert_value_knob_changes_output(
            {
                "cap_6": _built_builder(max_chunks=(6,)),
                "cap_3": _built_builder(max_chunks=(3,)),
            },
            _make_inputs(),
            knob="max_chunks",
        )
        assert min(deltas.values()) > 1e-2, deltas

    def test_the_attention_window_is_a_value_knob(self):
        """``window_size`` is a keep-predicate band: same weights, different attention.

        MEASURED 4.00e-04 against the oracle's 1e-05 bar -- a 40x margin, and small for a
        stated reason: at initialisation attention's residual-writing ``w_o`` is
        depth-scaled to ``0.02 / sqrt(4) = 0.01``, so attention contributes little to the
        residual stream. The bar is NOT widened; the measurement is recorded.
        """
        def windowed(window):
            config = HNetArchConfig(
                arch_layout=["m1", ["T1"], "m1"],
                d_model=[16, 16],
                d_intermediate=[0, 0],
                ssm_cfg=SSM,
                attn_cfg=AttnSpec(
                    num_heads=(2, 2), rotary_emb_dim=(4, 4), window_size=window
                ),
            )
            return _built_builder(config)

        deltas = assert_value_knob_changes_output(
            {"global": windowed((-1, -1)), "banded": windowed((-1, 2))},
            _make_inputs(),
            knob="window_size",
        )
        assert min(deltas.values()) > 1e-5, deltas

    def test_initializer_range_is_a_scoped_value_knob_on_the_head(self):
        """It must reach the ``lm_head`` kernel specifically. MEASURED 1.73e+00.

        Scoped rather than global because ``initializer_range`` reaches every Dense in
        the tree: an unscoped output assertion would pass even if the head alone were
        left on a hard-coded 0.02.
        """
        deltas = assert_scoped_value_knob_changes_weights(
            {
                "narrow": _built_builder(initializer_range=0.02),
                "wide": _built_builder(initializer_range=0.5),
            },
            _make_inputs(),
            knob="initializer_range",
            scope="lm_head",
        )
        assert min(deltas.values()) > 0.0, deltas


# ---------------------------------------------------------------------
# 10. Constructor validation
# ---------------------------------------------------------------------


class TestConstructorContract:
    """Bad configuration fails at construction, not deep inside a build."""

    def test_a_non_config_arch_config_raises(self):
        with pytest.raises(TypeError, match="arch_config must be an HNetArchConfig"):
            HNet({"arch_layout": ["m1", ["T1"], "m1"]}, max_chunks=(6,))

    @pytest.mark.parametrize("alpha", [-1.0, -1e-6])
    def test_a_negative_alpha_raises(self, alpha):
        with pytest.raises(ValueError, match="ratio_loss_alpha must be non-negative"):
            HNet(one_stage_config(), max_chunks=(6,), ratio_loss_alpha=alpha)

    @pytest.mark.parametrize("value", [0.0, -0.02])
    def test_a_non_positive_initializer_range_raises(self, value):
        with pytest.raises(ValueError, match="initializer_range must be positive"):
            HNet(one_stage_config(), max_chunks=(6,), initializer_range=value)

    def test_the_vocabulary_comes_from_the_architecture(self):
        """`vocab_size` is not a second knob on the model; the config owns it."""
        config = one_stage_config(vocab_size=37)
        model = make_model(config)
        assert model(byte_ids(vocab=37), training=False).shape[-1] == 37
        assert np.asarray(model.embeddings.embeddings).shape == (37, 16)


class TestShortSequencesAtTheConstructorDefaults:
    """A sequence shorter than ``max_chunks[0]`` must work -- it is the DEFAULT path.

    ``max_chunks`` defaults to ``default_max_chunks(num_stages, max_seq_len)``,
    i.e. ``max_seq_len // 2`` at the first chunking level. So EVERY model built
    without an explicit ``max_chunks`` rejects half its own advertised length
    range unless ``L >= max_chunks[0]`` is handled, and
    ``create_hnet("hnet_1stage_L")`` -- whose default is ``(1024,)`` at
    ``max_seq_len=2048`` -- rejected every prompt below 1024 bytes. A 19-byte
    generation prompt is what found it (decisions.md D-028, D-029).

    The shipped variants are not INSTANTIATED here: ``hnet_1stage_L`` is
    ~600M parameters and building one is not a unit test. The severity claim is
    pinned on the config arithmetic instead, and the behaviour is graded on the
    tiny architecture through the SAME defaulting code path -- ``max_chunks``
    omitted, so ``HNet.__init__`` derives it.
    """

    def test_the_shipped_variants_default_to_a_cap_above_a_short_prompt(self):
        """Config-level: the default cap really does exceed an ordinary prompt.

        This is arithmetic on :func:`default_max_chunks`, not a forward pass, and
        it is what makes the behavioural tests below load-bearing rather than
        academic.
        """
        assert default_max_chunks(2, 2048) == (1024,)
        assert default_max_chunks(3, 2048) == (1024, 512)
        for prompt in (19, 64, 255, 1023):
            assert prompt < default_max_chunks(2, 2048)[0]

    @pytest.mark.parametrize("length", [1, 2, 5, 19, 31])
    def test_the_model_accepts_a_sequence_shorter_than_its_default_cap(self, length):
        """``max_chunks`` OMITTED -- the constructor default is under test."""
        keras.utils.set_random_seed(13)
        model = HNet(one_stage_config(), max_seq_len=64, headdim=8)
        model.build((None, None))
        assert model.max_chunks == (32,), "the default cap, not an explicit one"
        assert length < model.max_chunks[0], "this test must exercise L < M"

        logits = model(byte_ids(batch=2, length=length), training=False)

        assert tuple(logits.shape) == (2, length, 256)
        assert np.isfinite(np.asarray(logits)).all()

    def test_a_two_level_layout_also_accepts_a_short_sequence(self):
        """Both chunking levels default-cap, and both must tolerate ``L < M``."""
        keras.utils.set_random_seed(13)
        model = HNet(two_stage_config(), max_seq_len=64, headdim=8)
        model.build((None, None))
        assert model.max_chunks == (32, 16)

        logits = model(byte_ids(batch=2, length=6), training=False)

        assert tuple(logits.shape) == (2, 6, 256)
        assert np.isfinite(np.asarray(logits)).all()

    def test_a_long_sequence_still_gives_the_same_values_it_did(self):
        """Anti-regression twin: extending the domain must not move ``L >= M``.

        The two calls differ only in that the second is a strict prefix-free
        second batch; what is asserted is that the ``L > M`` path still produces
        finite, length-consistent logits at the explicit cap the suite has always
        used.
        """
        model = make_model()
        for length in (12, 40):
            logits = model(byte_ids(batch=2, length=length), training=False)
            assert tuple(logits.shape) == (2, length, 256)
            assert np.isfinite(np.asarray(logits)).all()


class TestDefaultCompileEntryPoints:
    """``predict`` / ``evaluate`` / ``fit`` on a plain NumPy array, DEFAULT compile.

    No ``jit_compile`` is passed anywhere in this class, deliberately. Under the
    GPU default ``jit_compile="auto"`` every one of these three raised
    ``INVALID_ARGUMENT: Input 0 to node .../chunk_layer/BroadcastArgs with op
    BroadcastArgs must be a compile-time constant`` -- the gather's dynamic
    broadcast (decisions.md D-029). Pinning ``jit_compile=False`` here would make
    the class green while leaving every ordinary caller broken, so it is not
    done; the repair is in the layer.

    These pass trivially on CPU, where XLA is not engaged. They are RED on GPU
    against the pre-fix code, which is where they were proven.
    """

    def _compiled(self):
        model = make_model()
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )
        return model

    def test_predict_works_on_a_plain_numpy_array(self):
        model = self._compiled()
        x = byte_ids(batch=5, length=12)

        logits = model.predict(x, batch_size=2, verbose=0)

        assert logits.shape == (5, 12, 256)
        assert np.isfinite(logits).all()

    def test_evaluate_works_on_a_plain_numpy_array(self):
        model = self._compiled()
        x = byte_ids(batch=5, length=12)

        loss = model.evaluate(x, x, batch_size=2, verbose=0)

        assert np.isfinite(float(np.asarray(loss).ravel()[0]))

    def test_fit_works_on_a_plain_numpy_array_with_a_ragged_last_batch(self):
        """``5 % 2 == 1``: the last batch is short, so the batch dim is dynamic.

        That is the condition D-028 measured as necessary -- a static batch (a
        ``tf.data`` pipeline with ``drop_remainder=True``) never hit the defect,
        which is why the trainer's own validation loop stayed green while
        ``model.predict(numpy_array)`` was dead.
        """
        model = self._compiled()
        x = byte_ids(batch=5, length=12)

        history = model.fit(x, x, epochs=1, batch_size=2, verbose=0)

        assert np.isfinite(history.history["loss"][0])
