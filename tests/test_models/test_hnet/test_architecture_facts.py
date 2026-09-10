"""One single-claim guard per H-Net invariant that has no shape symptom.

Every claim here is made on the **assembled** :class:`HNet` -- the object a caller
actually builds -- rather than on a hand-constructed component. That is the whole reason
the file exists beside ``test_components.py``, ``test_stage.py`` and
``test_arch_layout.py``: those pin one layer in isolation, and a model can wire a
correct layer at a wrong epsilon, forget to reach a stage's projection, or grow a second
copy of a constant. A fact that holds in a component test and not in the assembly is a
defect the component test cannot see.

Where an existing test already makes the same claim on an isolated component, this file
says so and makes the STRICTLY different claim instead of restating it:

============================== ============================================ ==================================================
invariant                      already pinned in isolation by               the strictly different claim made here
============================== ============================================ ==================================================
hard threshold ``p > 0.5``     ``test_routing_module.py`` (its own layer)   the routing module the ASSEMBLED model owns
STE returns exactly 1.0        ``test_stage.py`` (float64, synthetic draws)  the model's OWN realised ``selected_probs``
RMSNorm ``epsilon = 1e-5``     ``test_components.py`` (one block, one stack) EVERY norm site in a 3-stage assembly, counted
embedding 1.0 vs head 0.02     ``test_model.py`` (realised, one tiny config) realised at 4x the sample size + all six variants
``residual_proj`` zero-init    ``test_stage.py`` (one stage)                 EVERY stage of a 3-stage assembly, kernel + bias
tied vs untied signature       ``test_model.py`` (the knob oracle)           the exact shape the tied signature is missing
``pretrained=True`` raises     ``test_model.py`` (``True``, per variant)     every TRUTHY value, and never a warn-and-return
``from_variant`` names         ``test_model.py`` (the error message)         the table itself: six keys, one object, no copy
no ``chunk_size``              ``test_arch_layout.py`` (``SSMSpec``)         the whole config surface + every public signature
============================== ============================================ ==================================================

Nothing here trains and every dimension is tiny.
"""

import inspect
import json
import math

import keras
import numpy as np
import pytest

from dl_techniques.layers import dynamic_chunking
from dl_techniques.models.language import hnet as hnet_package
from dl_techniques.models.language.hnet import components as components_module
from dl_techniques.models.language.hnet import config as config_module
from dl_techniques.models.language.hnet import model as model_module
from dl_techniques.models.language.hnet import stage as stage_module
from dl_techniques.models.language.hnet.components import NORM_EPSILON
from dl_techniques.models.language.hnet.config import (
    MODEL_VARIANTS,
    AttnSpec,
    HNetArchConfig,
    SSMSpec,
)
from dl_techniques.models.language.hnet.model import (
    EMBEDDING_INIT_STDDEV,
    INITIALIZER_RANGE,
    HNet,
)
from dl_techniques.models.language.hnet.stage import straight_through_ones

SEED = 3
SSM = SSMSpec(d_conv=4, expand=2, d_state=8)


def _config(layout, widths):
    n = len(widths)
    return HNetArchConfig(
        arch_layout=layout,
        d_model=list(widths),
        d_intermediate=[0] * n,
        ssm_cfg=SSM,
        attn_cfg=AttnSpec(
            num_heads=(2,) * n, rotary_emb_dim=(4,) * n, window_size=(-1,) * n
        ),
    )


def one_stage(d_model=16):
    """``["m1", ["T1"], "m1"]`` -- one chunking level."""
    return _config(["m1", ["T1"], "m1"], (d_model, d_model))


def three_stage():
    """``["m1", ["T1m1", ["T1"], "m1T1"], "m1"]`` -- TWO chunking levels, three stages.

    Used wherever the claim is "at EVERY site": a one-stage assembly has exactly one of
    everything, so it cannot tell "the model sets this once" from "the model sets this
    everywhere".
    """
    return _config(["m1", ["T1m1", ["T1"], "m1T1"], "m1"], (16, 16, 32))


def built(config=None, max_chunks=(6,), seed=SEED, **kwargs):
    keras.utils.set_random_seed(seed)
    model = HNet(
        one_stage() if config is None else config,
        max_chunks=max_chunks,
        max_seq_len=64,
        headdim=8,
        **kwargs,
    )
    model.build((None, None))
    return model


def byte_ids(batch=2, length=12, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (batch, length)).astype("int32")


def weight_shape_signature(model):
    return tuple(tuple(w.shape) for w in model.weights)


# ---------------------------------------------------------------------
# 1. The boundary decision is STRICTLY `p > 0.5`
# ---------------------------------------------------------------------


class TestTheBoundaryThresholdIsStrict:
    """``p > 0.5``, never ``>=`` -- on the routing module the model itself built.

    ``p = 0.5`` is reachable BIT-EXACTLY, not merely approachable: ``p = (1 - cos)/2`` and
    two orthogonal adjacent hidden states give ``cos = 0`` exactly, which under the
    reference's identity initialisation of ``q_proj``/``k_proj`` needs nothing more than
    two distinct one-hot rows. So this is a live branch, not a measure-zero curiosity.
    """

    def test_an_exactly_one_half_probability_is_not_a_boundary(self):
        routing = built().backbone.routing_module
        assert np.allclose(np.asarray(routing.q_proj.kernel), np.eye(16)), (
            "the tie below is constructed from the identity init; without it "
            "cos = 0 is no longer reachable by two one-hot rows and the test would be "
            "measuring something else"
        )

        # Four mutually orthogonal one-hot rows: every adjacent pair has cos = 0.
        hidden = np.zeros((1, 4, 16), dtype="float32")
        for position in range(4):
            hidden[0, position, position] = 1.0

        probs, mask, _ = routing(hidden)
        p = np.asarray(probs)[..., 1]
        boundary = np.asarray(mask)

        # Anti-vacuity: the tie must actually be exact, or "not a boundary" below is
        # about an ordinary p < 0.5 and says nothing about the comparison operator.
        assert np.all(p[0, 1:] == 0.5), f"p is not exactly 0.5: {p!r}"
        assert not boundary[0, 1:].any(), (
            "a position at exactly p = 0.5 was selected as a boundary, so the "
            "comparison is `>=`; `argmax([1-p, p])` breaks this tie LOW"
        )
        assert boundary[0, 0], (
            "position 0 is force-fed p = 1.0 and must stay a boundary; if it did not, "
            "the assertion above would be satisfied by a layer that selects nothing"
        )

    def test_a_probability_one_ulp_above_one_half_IS_a_boundary(self):
        """The twin. Without it, "not selected at 0.5" is satisfied by ``p > 1.0``."""
        routing = built().backbone.routing_module
        just_above = np.nextafter(np.float32(0.5), np.float32(1.0))
        assert just_above > 0.5

        # cos = 1 - 2p, so p just above 0.5 needs cos just below 0.
        hidden = np.zeros((1, 2, 16), dtype="float32")
        hidden[0, 0, 0] = 1.0
        hidden[0, 1, 0] = -1e-3
        hidden[0, 1, 1] = 1.0

        probs, mask, _ = routing(hidden)
        p = float(np.asarray(probs)[0, 1, 1])

        assert p > 0.5, f"the probe did not clear the tie: p = {p!r}"
        assert bool(np.asarray(mask)[0, 1]), (
            f"p = {p!r} is above 0.5 and was still not selected"
        )


# ---------------------------------------------------------------------
# 2. The straight-through gate is grouped
# ---------------------------------------------------------------------


class TestTheStraightThroughGateIsGrouped:
    """``x + stop_gradient(1 - x)`` returns bit-exactly 1.0 on the model's OWN probs.

    ``test_stage.py`` pins this in float64 over 100000 synthetic uniform draws (D-018).
    The claim here is narrower and different: on the ``selected_probs`` this ASSEMBLED
    model actually produces, the grouped form is bit-exactly 1.0 while the reassociated
    form is not -- so the grouping is load-bearing at the values the model really sees,
    not only at values chosen to expose it.
    """

    def test_the_gate_is_bit_exactly_one_on_the_models_own_selected_probs(self):
        model = built()
        hidden = model.embeddings(byte_ids())
        encoded = model.backbone.encoder(hidden, training=False)
        _, _, selected = model.backbone.routing_module(encoded)

        gate = np.asarray(straight_through_ones(selected))

        assert gate.size > 0
        assert np.all(gate == 1.0), (
            f"{int((gate != 1.0).sum())} of {gate.size} gate entries are not exactly "
            "1.0; the forward pass is being SCALED by the routing confidence instead of "
            "receiving it as a pure gradient conduit"
        )
        # Anti-vacuity: a gate of all-ones is trivially 1.0 if the probs were all 1.0.
        probs = np.asarray(selected)
        assert float(np.min(probs)) < 1.0, (
            f"every selected_prob is 1.0 ({float(np.min(probs))!r}), so the assertion "
            "above cannot distinguish the grouped form from any other spelling"
        )

    def test_the_reassociated_form_is_NOT_bit_exact_on_those_same_probs(self):
        """The twin, in float32 -- and it must stay in float32.

        MEASURED on these exact values: the ungrouped ``x + 1 - x`` misses 1.0 on
        **2 of 24** entries in float32 and on **0 of 24** in float64. Promoting this twin
        to float64 "for precision" would turn it into a guard that cannot fail. The
        float64 statement of the same fact lives in ``test_stage.py``, where the sample
        is 100000 draws rather than 24.
        """
        model = built()
        hidden = model.embeddings(byte_ids())
        encoded = model.backbone.encoder(hidden, training=False)
        _, _, selected = model.backbone.routing_module(encoded)

        x = np.asarray(selected).astype("float32")
        ungrouped = x + np.float32(1.0) - x

        n_missed = int((ungrouped != 1.0).sum())
        assert n_missed > 0, (
            f"the reassociated form was exact on all {x.size} of the model's own "
            "selected_probs, so this twin is vacuous here; re-derive it (test_stage.py's "
            "100000-draw float64 arm remains the primary guard)"
        )


# ---------------------------------------------------------------------
# 3. RMSNorm epsilon at EVERY site
# ---------------------------------------------------------------------


class TestEveryNormalizationSiteUsesTheReferenceEpsilon:
    """``1e-5`` (``block.py:41``, ``isotropic.py:96``), not the factory default ``1e-6``.

    A 100x epsilon error has no shape symptom and no exception; it shifts every
    normalised activation slightly and the model trains, worse, in silence.
    """

    def test_every_rms_norm_in_a_three_stage_assembly_is_at_1e_minus_5(self):
        model = built(three_stage(), max_chunks=(8, 4))

        epsilons = [
            (sub.name, float(sub.epsilon))
            for sub in model._flatten_layers(include_self=True)
            if type(sub).__name__ == "RMSNorm"
        ]

        assert len(epsilons) >= 10, (
            f"only {len(epsilons)} RMSNorm sites were found in a three-stage assembly; "
            "the walk is not reaching the nested stages and 'every site' would be a "
            "claim about a handful"
        )
        offenders = [pair for pair in epsilons if pair[1] != NORM_EPSILON]
        assert not offenders, f"RMSNorm sites off {NORM_EPSILON}: {offenders}"

    def test_the_mamba_mixers_carry_the_same_epsilon(self):
        """``Mamba2Layer`` normalises internally and takes its OWN epsilon argument.

        It is not an ``RMSNorm`` instance, so the walk above cannot see it; a separate
        single claim rather than a widened predicate.
        """
        model = built(three_stage(), max_chunks=(8, 4))

        mixers = [
            (sub.name, float(sub.norm_epsilon))
            for sub in model._flatten_layers(include_self=True)
            if type(sub).__name__ == "Mamba2Layer"
        ]

        assert mixers, "no Mamba2Layer was found; this assembly has m stages"
        assert all(value == NORM_EPSILON for _, value in mixers), mixers

    def test_the_constant_is_not_the_factory_default(self):
        """If ``create_normalization_layer`` ever defaults to 1e-5, the two tests above
        would pass without H-Net asking for anything -- so the difference is asserted."""
        from dl_techniques.layers.norms.factory import create_normalization_layer

        default = create_normalization_layer("rms_norm", name="default_probe")
        assert float(default.epsilon) != NORM_EPSILON, (
            "the factory default now equals the reference epsilon; the explicit "
            "epsilon= arguments in components.py have become unobservable"
        )
        assert NORM_EPSILON == 1e-5


# ---------------------------------------------------------------------
# 4. The init asymmetry, at a larger sample and across all six variants
# ---------------------------------------------------------------------


class TestTheInitAsymmetryHoldsEverywhereItIsShipped:
    """Embedding at ``stddev = 1.0``, every Linear at ``0.02`` (``mixer_seq.py:55-62``).

    ``test_model.py`` asserts the REALISED sample std once, on a ``d_model = 16`` model.
    The two claims added here are (a) the same realised measurement at 4x the sample
    size, where the 6-sigma bound is half as wide, and (b) that the CONFIGURED
    initializers carry the same two numbers at all six shipped variants, whose real
    widths are far too large to build here.
    """

    def test_the_realised_standard_deviations_are_50x_apart(self):
        """Derivation of the bound, so it is not a guess.

        For ``n`` normal draws the sample standard deviation has standard error
        ``sigma / sqrt(2n)``. The embedding here is ``(256, 64) = 16384`` draws, so at
        ``sigma = 1.0`` six standard errors is ``6 / sqrt(32768) = 3.3e-02``; the head is
        the same 16384 draws at ``sigma = 0.02``, giving ``6.6e-04``. Six standard errors
        is a 1-in-10^9 event, so a failure at this bound is a defect and not a draw. The
        two claims are 50x apart and each bound is ~30x narrower than that gap, so
        neither can be satisfied by the other's value -- asserted below rather than
        asserted about.
        """
        model = built(_config(["m1", ["T1"], "m1"], (64, 64)))

        embedding = np.asarray(model.embeddings.embeddings)
        head = np.asarray(model.lm_head.kernel)
        assert embedding.size == 16384 and head.size == 16384

        def bound(size, sigma):
            return 6.0 * sigma / math.sqrt(2.0 * size)

        embedding_bound = bound(embedding.size, EMBEDDING_INIT_STDDEV)
        head_bound = bound(head.size, INITIALIZER_RANGE)

        assert abs(float(np.std(embedding)) - EMBEDDING_INIT_STDDEV) < embedding_bound
        assert abs(float(np.std(head)) - INITIALIZER_RANGE) < head_bound

        # The two bounds cannot overlap the other claim.
        assert abs(EMBEDDING_INIT_STDDEV - INITIALIZER_RANGE) > 10.0 * (
            embedding_bound + head_bound
        )

    @pytest.mark.parametrize("variant", sorted(MODEL_VARIANTS))
    def test_every_shipped_variant_configures_the_same_two_numbers(self, variant):
        """Constructed, never built -- the real widths would be gigabytes."""
        model = HNet.from_variant(variant, max_seq_len=256)

        assert float(model.embeddings.embeddings_initializer.stddev) == (
            EMBEDDING_INIT_STDDEV
        )
        if model.lm_head is not None:
            assert float(model.lm_head.kernel_initializer.stddev) == INITIALIZER_RANGE
        else:
            # A tied head owns no weights, so there is no head initializer to check --
            # stated rather than skipped, so a variant silently turning tied does not
            # quietly drop this row.
            assert model.tie_word_embeddings is True


# ---------------------------------------------------------------------
# 5. `residual_proj` is exactly zero at EVERY stage
# ---------------------------------------------------------------------


class TestEveryResidualProjectionStartsAtExactlyZero:
    """Kernel AND bias, at every chunking level, in float32.

    A non-zero residual at step 0 imposes the identity branch instead of letting the
    model learn it (``hnet.py:106-107`` flags the weight ``_no_reinit`` precisely so the
    depth-scaled init pass cannot overwrite it). MEASURED consequence, recorded in
    ``test_causality.py``: a zero residual makes a non-boundary position's own byte reach
    nothing at all, which is why liveness for this projection is asserted after a real
    optimizer step and never at init.
    """

    def test_both_stages_of_a_three_stage_model_start_at_zero(self):
        model = built(three_stage(), max_chunks=(8, 4))

        weights = [w for w in model.weights if "residual_proj" in w.path]
        kernels = [w for w in weights if w.path.endswith("kernel")]
        biases = [w for w in weights if w.path.endswith("bias")]

        assert len(kernels) == 2, (
            f"a three-stage layout has two chunking levels and so two residual "
            f"projections; found {len(kernels)}: {[w.path for w in weights]}"
        )
        assert len(biases) == 2
        for weight in kernels + biases:
            assert float(np.max(np.abs(np.asarray(weight)))) == 0.0, weight.path

    def test_the_projection_is_pinned_to_float32_UNDER_MIXED_PRECISION(self):
        """``hnet.py:102-105`` -- the residual accumulator stays fp32 whatever the policy.

        Under ``mixed_float16`` this is the one place where the gate's small corrections
        would otherwise be rounded away.

        **This test has to run inside the policy, and the first draft did not.** At the
        default ``float32`` policy a ``Dense`` built WITHOUT ``dtype="float32"`` gets
        float32 anyway, and the mutation that deletes the pin was MEASURED to leave a
        46/46 green suite: a guard that cannot fail (the pattern D-020 recorded twice in
        this plan). Under ``mixed_float16`` the two spellings finally separate --
        ``compute_dtype`` reads ``float32`` with the pin and ``float16`` without it. The
        VARIABLE dtype is float32 either way, which is why the weight-dtype form of this
        assertion is the one that was blind.
        """
        previous = keras.config.dtype_policy()
        keras.config.set_dtype_policy("mixed_float16")
        try:
            model = built(three_stage(), max_chunks=(8, 4))
            residual = model.backbone.residual_proj
            others = [
                sub for sub in model.backbone.encoder._flatten_layers()
                if isinstance(sub, keras.layers.Dense)
            ]

            assert model.dtype_policy.name == "mixed_float16"
            assert others, "no comparison Dense was found; the contrast below is vacuous"
            assert all(sub.compute_dtype == "float16" for sub in others), (
                "the surrounding Denses are not computing in float16, so 'residual_proj "
                "is float32' says nothing about the policy"
            )
            assert residual.compute_dtype == "float32", (
                "the residual accumulator is following the mixed-precision policy; the "
                'dtype="float32" pin has been dropped'
            )
            assert residual.variable_dtype == "float32"
        finally:
            keras.config.set_dtype_policy(previous)

    def test_the_depth_scaled_init_does_not_overwrite_it(self):
        """``initializer_range`` reaches every other Dense; this one must stay zero."""
        loud = built(three_stage(), max_chunks=(8, 4), initializer_range=0.5)

        residual = [
            w for w in loud.weights
            if "residual_proj" in w.path and w.path.endswith("kernel")
        ]
        other = [
            w for w in loud.weights
            if w.path.endswith("out_proj/kernel")
        ]

        assert residual and other, "one of the two scopes matched nothing"
        assert all(float(np.max(np.abs(np.asarray(w)))) == 0.0 for w in residual)
        assert all(float(np.max(np.abs(np.asarray(w)))) > 0.0 for w in other), (
            "the comparison scope is itself zero, so 'residual_proj stayed zero' is "
            "satisfied by an init that did nothing at all"
        )


# ---------------------------------------------------------------------
# 6. Tying changes the weight-SHAPE signature
# ---------------------------------------------------------------------


class TestTyingChangesTheWeightShapeSignature:
    """Structural, not a value knob -- and by exactly one known tensor."""

    def test_the_tied_signature_is_the_untied_one_minus_the_head_kernel(self):
        untied = built(tie_word_embeddings=False)
        tied = built(tie_word_embeddings=True)

        untied_signature = weight_shape_signature(untied)
        tied_signature = weight_shape_signature(tied)

        assert tied_signature != untied_signature
        head_shape = tuple(untied.lm_head.kernel.shape)
        assert head_shape == (16, 256)

        remaining = list(untied_signature)
        remaining.remove(head_shape)
        assert tuple(remaining) == tied_signature, (
            "tying removed something other than exactly the lm_head kernel: "
            f"{sorted(set(untied_signature) ^ set(tied_signature))}"
        )

    def test_a_tied_model_still_produces_full_vocabulary_logits(self):
        """The twin: removing a weight must not remove the output."""
        tied = built(tie_word_embeddings=True)
        assert tuple(tied(byte_ids(), training=False).shape) == (2, 12, 256)


# ---------------------------------------------------------------------
# 7. `pretrained` never warns and returns random weights
# ---------------------------------------------------------------------


class TestPretrainedAlwaysRaises:
    """H10. The alternative -- warning and returning random weights -- lets a caller
    publish numbers from an untrained model."""

    @pytest.mark.parametrize("truthy", [True, 1, "local.keras", ["path"], 0.5])
    def test_every_truthy_value_raises(self, truthy):
        """``if pretrained:`` must be a truth test, not ``is True``: a caller who passes
        a path is asking for weights just as loudly as one who passes ``True``."""
        with pytest.raises(NotImplementedError):
            HNet.from_variant("hnet_1stage_L", pretrained=truthy)

    @pytest.mark.parametrize("falsy", [False, 0, None, ""])
    def test_every_falsy_value_builds(self, falsy):
        """The twin. Without it, ``raise NotImplementedError`` unconditionally passes."""
        model = HNet.from_variant("hnet_1stage_L", pretrained=falsy, max_seq_len=256)
        assert isinstance(model, HNet)

    def test_the_module_contains_no_warn_and_return_path(self):
        source = inspect.getsource(model_module)
        assert "logger.warning" not in source
        assert "warnings.warn" not in source


# ---------------------------------------------------------------------
# 8. The variant table
# ---------------------------------------------------------------------


class TestTheVariantTableIsOneObjectWithSixRows:
    """Six keys, aliased not copied. A second literal table is a copy that drifts."""

    EXPECTED = {
        "hnet_1stage_L",
        "hnet_1stage_XL",
        "hnet_2stage_L",
        "hnet_2stage_XL",
        "hnet_2stage_XL_chinese",
        "hnet_2stage_XL_code",
    }

    def test_there_are_exactly_these_six_names(self):
        assert set(MODEL_VARIANTS) == self.EXPECTED
        assert len(MODEL_VARIANTS) == 6

    def test_the_class_attribute_is_the_same_object_not_a_copy(self):
        assert HNet.MODEL_VARIANTS is config_module.MODEL_VARIANTS
        assert model_module.MODEL_VARIANTS is config_module.MODEL_VARIANTS

    def test_the_model_module_defines_no_second_table(self):
        """A textual claim, because ``is`` above is satisfied by an alias that shadows a
        literal defined elsewhere in the same file."""
        source = inspect.getsource(model_module)
        assert "MODEL_VARIANTS: Dict[str, HNetArchConfig] = MODEL_VARIANTS" in source
        assert source.count("MODEL_VARIANTS = {") == 0

    @pytest.mark.parametrize("variant", sorted(EXPECTED))
    def test_every_name_resolves_through_from_variant(self, variant):
        model = HNet.from_variant(variant, max_seq_len=256)
        assert model.arch_config is MODEL_VARIANTS[variant]


# ---------------------------------------------------------------------
# 9. `chunk_size` exists nowhere in the config surface
# ---------------------------------------------------------------------


class TestNoChunkSizeAnywhereInTheConfigSurface:
    """H17: ``Mamba2Layer`` has no chunked scan, so a ``chunk_size`` knob would be a
    declared field that nothing reads -- the ``test_config_fields_are_live`` defect class.

    ``test_arch_layout.py`` pins the absence on ``SSMSpec``. The claims here are the two
    places a re-introduction would actually surface: a serialized model config, and the
    public callable signatures a user reads.
    """

    def test_a_serialized_model_config_contains_the_string_nowhere(self):
        model = built(three_stage(), max_chunks=(8, 4))
        payload = json.dumps(model.get_config(), default=str)
        assert "chunk_size" not in payload

    def test_max_chunks_is_present_so_the_search_above_is_not_blind(self):
        """Anti-vacuity: a ``get_config`` that returned ``{}`` would pass the test above.
        The chunk cap H-Net DOES have must be findable by the same search."""
        model = built(three_stage(), max_chunks=(8, 4))
        payload = json.dumps(model.get_config(), default=str)
        assert "max_chunks" in payload
        assert '"max_chunks": [8, 4]' in payload

    @pytest.mark.parametrize(
        "module",
        [config_module, components_module, stage_module, model_module,
         hnet_package, dynamic_chunking],
        ids=["config", "components", "stage", "model", "package", "dynamic_chunking"],
    )
    def test_no_public_callable_declares_a_chunk_size_parameter(self, module):
        offenders = []
        n_checked = 0
        for name in getattr(module, "__all__", []) or dir(module):
            if name.startswith("_"):
                continue
            member = getattr(module, name, None)
            if not (inspect.isfunction(member) or inspect.isclass(member)):
                continue
            target = member.__init__ if inspect.isclass(member) else member
            try:
                signature = inspect.signature(target)
            except (TypeError, ValueError):
                continue
            n_checked += 1
            if "chunk_size" in signature.parameters:
                offenders.append(f"{module.__name__}.{name}")

        assert n_checked > 0, (
            f"{module.__name__}: no public callable was inspected at all, so this "
            "sweep is vacuous"
        )
        assert not offenders, offenders
