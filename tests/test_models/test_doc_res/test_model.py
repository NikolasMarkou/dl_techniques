"""The comprehensive ``doc_res`` model suite -- v2 guide section 16.3.

Companion files in this directory, and the split between them:

===============================  =============================================
``test_model_smoke.py``          shape/parameter contract, the variant API,
                                 the six documented asymmetries read off
                                 CONFIG, and the divisibility refusal
``test_components.py``           the three composites in isolation, plus the
                                 D-007 pixel-shuffle bracketing predicates
``test_architecture_facts.py``   one single-claim guard per invariant a shape
                                 test is blind to, each RED-proven by
                                 injection into the shipped ``model.py``
``test_precision_arm.py``        the ``mixed_float16`` and ``float64`` arms
**this file**                    the shared-instrument arms: round trip,
                                 build parity, gradient flow, knob
                                 sensitivity, XLA agreement, both variant
                                 rows, and the assembled-model D-007 adjacency
===============================  =============================================

Every arm here is an ADOPTION of an existing shared instrument
(``tests/test_models/*_oracle.py``). None of them is re-implemented locally;
where this file has its own code it is a subject builder, a contract function
passed to an oracle, or the recorded-forward instrument that had no equivalent
(``forward_order_oracle.py``, whose docstring states what it rejected).

On ``tests/numerics.reassociation_atol``, and why it is NOT used here
--------------------------------------------------------------------
Deliberate, and measured rather than assumed. Every float comparison in this
file is either EXACT (the round-trip and adjacency arms run at ``atol=0.0``,
where a derived bound would only add slack) or self-calibrating (the XLA arm
takes its own eager-vs-eager spread as the floor, inside the shared oracle).
The one arm that would need a derived bound -- batch-2 versus two batch-1
calls -- was written, measured and then dropped: ``reassociation_atol`` is
expressed in ``eps_f32 = 1.19e-07`` and its own docstring records that TF32
truncates the matmul mantissa to 10 bits. MEASURED on GPU 1 (RTX 4070,
``tf.config.experimental.tensor_float_32_execution_enabled() == True``, which
is the regime this suite actually runs in): batch delta 2.94e-03 against a
derived bound of 3.94e-04, i.e. 7.5x over. That assertion would have been
measuring the TF32 toggle, not the model, and the ``tf32_disabled`` fixture
that ``test_mdta.py`` opts into lives in ``tests/test_layers/conftest.py`` and
is not visible from ``tests/test_models/``. Copying it here would be a second
copy of a fixture whose whole point is that there be one.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.image_restoration.doc_res.model import (
    DocRes,
    create_doc_res,
)

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    stop_all_gradients,
)
from ..knob_sensitivity_oracle import assert_structural_knob_changes_weights
from ..lazy_build_contract_oracle import assert_lazy_build_costs_nothing
from ..precision_arm_oracle import assert_xla_equivalence
from ..roundtrip_instrument_oracle import (
    assert_build_parity,
    assert_roundtrip_output_values,
    assert_weights_restored_before_first_call,
    measure_build_parity,
    measure_roundtrip,
)
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
)
from ..test_sam.dead_component_oracle import fit_one_step_moved_variables
from .forward_order_oracle import (
    RecordedForward,
    assert_consumes,
    assert_every_leaf_is_recorded,
    record_forward_ops,
)
from .test_components import (
    assert_pixel_reshuffle_sites_are_bracketed,
    assert_pixel_reshuffle_sites_have_a_per_channel_successor,
)

# ---------------------------------------------------------------------
# The subject.
#
# `dim=8` with the SHIPPED `heads=[1, 2, 4, 8]`, not a flattened head
# schedule: the head counts are what hold the per-head channel width constant
# across levels, and a subject whose MDTA never sees more than one head would
# be blind to a per-head temperature that failed to serialize or to receive a
# gradient. 32x32 is the smallest legal spatial size that is still a multiple
# of 8 after three downsampling stages (D-016 refuses anything else).
# ---------------------------------------------------------------------

_SMALL = dict(dim=8, num_blocks=[1, 1, 1, 1], num_refinement_blocks=1,
              heads=[1, 2, 4, 8])
# BATCH is 2, not 1, and that is load-bearing for exactly one arm: the shared
# `slice_leading_axis` breaker slices each output leaf to `leaf[:1]`, which is
# a NO-OP at batch 1. The smoke-contract meta-test would then report "the
# contract accepts a broken forward" about a breaker that broke nothing.
BATCH, HEIGHT, WIDTH = 2, 32, 32
IN_CHANNELS, OUT_CHANNELS = 6, 3
INPUT_SHAPE = (None, None, None, IN_CHANNELS)

#: Leaf sub-layers of the ``_SMALL`` subject: 43 Conv2D + 16 DepthwiseConv2D +
#: 16 LayerNormalization + 3 PixelUnshuffle2D + 3 PixelShuffle2D. Pinned so the
#: recorded-forward arms cannot go quietly vacuous if the assembly shrinks.
EXPECTED_LEAVES = 81

#: Pixel-reshuffle sites in the assembled model: one per resampler, three down
#: and three up.
EXPECTED_RESHUFFLE_SITES = 6


def _build() -> DocRes:
    """The subject, unbuilt. Oracles that need a built model call it."""
    return DocRes(**_SMALL)


def _inputs() -> np.ndarray:
    """A deterministic input. Called repeatedly by the oracles, which compare
    across calls, so it must not draw fresh values."""
    return np.random.RandomState(0).randn(
        BATCH, HEIGHT, WIDTH, IN_CHANNELS).astype("float32")


def _built() -> DocRes:
    model = _build()
    model.build(INPUT_SHAPE)
    return model


@pytest.fixture(scope="module")
def built_model() -> DocRes:
    """One built subject shared by the read-only arms."""
    keras.utils.set_random_seed(0)
    return _built()


@pytest.fixture(scope="module")
def records(built_model):
    """One recorded forward pass, shared by the three arms that read it."""
    return record_forward_ops(built_model, _inputs())


# ---------------------------------------------------------------------
# 1. The `.keras` round trip, on VALUES
# ---------------------------------------------------------------------


def test_the_round_trip_reproduces_the_output_values_exactly():
    """R-063: output VALUES at ``rtol=0``, with ``training=False`` explicit.

    ``training`` is passed by the oracle, not left to default, because a
    subclassed ``keras.Model`` whose ``call`` takes ``training=None`` resolves
    it from the enclosing scope; "the default happened to be inference" is not
    the same claim as "the inference path round-trips".
    """
    report = measure_roundtrip(_build, _inputs, training=False)
    # DocRes has no sampling anywhere in its forward, so its self-spread is
    # exactly 0.0 and the round trip is held to the same. A non-zero reading
    # here would mean the model became stochastic, which is a finding.
    assert report["self_max_delta"] == 0.0, (
        f"DocRes is not deterministic (self spread "
        f"{report['self_max_delta']:.6e}); the exact round-trip claim below "
        "would then be measuring the sampler")
    assert_roundtrip_output_values(report, atol=0.0)


def test_the_weights_are_restored_before_the_loaded_model_is_called():
    """R-073: weight VALUES at ``atol=0.0``, read BEFORE the first call.

    The order is the whole assertion. A subclassed model that reloaded nothing
    would fill its gap with fresh random weights on first call, at which point
    the weight COUNT is identical for the correct and the broken variant.
    """
    report = measure_roundtrip(_build, _inputs, training=False)
    assert report["call_count_before_weight_read"] == 0, (
        "the loaded model was called before its weights were read; that "
        "comparison cannot distinguish restored from re-initialized")
    assert_weights_restored_before_first_call(report, atol=0.0)


def test_the_lazy_build_costs_nothing():
    """Perturb, save, reload: the output must not move by any amount.

    The oracle's own anti-vacuity arms carry this test: it asserts the
    perturbation was LIVE (the forward is sensitive to its own weights) before
    it asserts the reload was exact, which is the shape that let one package
    here pass 3/3 while 464 tensors were never saved at all.
    """
    report = assert_lazy_build_costs_nothing(
        _build, _inputs, input_shape=INPUT_SHAPE, atol=0.0)
    assert report["n_weights"] == report["n_weights_reloaded"]
    assert report["perturb_liveness"] > 0.0


# ---------------------------------------------------------------------
# 2. Build parity -- both directions
# ---------------------------------------------------------------------


def test_the_explicit_build_matches_the_lazy_build_by_weight_path():
    """R-072(a): ``build(shape)`` and a first call produce the same tree.

    No waiver is passed: every sub-layer in ``doc_res`` carries an explicit
    ``name=`` (``_make_stage`` exists for that reason), so an ``AUTONAME_STEMS``
    entry would be papering over a naming regression rather than recording one.
    """
    report = measure_build_parity(_build, _inputs, input_shape=INPUT_SHAPE)
    assert_build_parity(report, autoname_stems=(), expect_path_collisions=0)


def test_build_materializes_exactly_the_tree_that_call_runs(built_model, records):
    """Both halves of build parity that a weight-path diff cannot see.

    Under-build and over-build are both defects and they fail differently:

    * **Under-build** -- a sub-layer ``build()`` skipped would be created
      lazily on the first call and would not be in the archive. The weight-path
      arm above catches that.
    * **Over-build** -- a sub-layer that EXISTS and is built but that ``call``
      never applies is dead weight that every parameter count, every
      round-trip and every path diff agrees about, because both sides build it.
      Only the forward path can see it, which is what this reads.
    """
    assert len(records) == EXPECTED_LEAVES, (
        f"recorded {len(records)} leaf ops, expected {EXPECTED_LEAVES}")
    assert_every_leaf_is_recorded(built_model, records)


# ---------------------------------------------------------------------
# 3. Gradients, after ONE real optimizer step
# ---------------------------------------------------------------------


def test_gradients_reach_every_trainable_weight():
    """Every trainable weight gets a finite, not-identically-zero gradient.

    No ``expect_zero`` waiver: a waived weight here would be a weight the
    architecture cannot train, and DocRes has none.
    """
    keras.utils.set_random_seed(0)
    model = _built()
    report = assert_gradients_reach_every_trainable_weight(model, _inputs())
    assert len(report) == len(model.trainable_weights)


def test_the_gradient_assertion_can_fail():
    """RED proof for the arm above, using the oracle's OWN shared injection.

    ``stop_all_gradients`` is imported rather than re-typed as a local lambda:
    a hand-written detach is a hand-written chance to detach only part of the
    output, which convicts nothing while looking exactly like a proof.
    """
    keras.utils.set_random_seed(0)
    model = _built()
    with broken_forward(model, stop_all_gradients):
        with pytest.raises(AssertionError, match="received NO gradient"):
            assert_gradients_reach_every_trainable_weight(model, _inputs())


def test_one_real_optimizer_step_moves_every_trainable_variable():
    """``fit()`` once and require the moved NAME SET to be the full set.

    A count would pass while a named component sat still; the oracle returns
    names for exactly that reason. The model is compiled and BUILT before the
    step, so nothing can be reported as "moved" merely by coming into
    existence.
    """
    keras.utils.set_random_seed(0)
    model = _built()
    model.compile(optimizer=keras.optimizers.Adam(1e-2), loss="mse")

    x = _inputs()
    y = np.zeros((BATCH, HEIGHT, WIDTH, OUT_CHANNELS), dtype="float32")
    report = fit_one_step_moved_variables(model, x, y, batch_size=BATCH)

    assert report.total == len(model.trainable_variables)
    assert set(report.unmoved) == set(), (
        f"{len(report.unmoved)} of {report.total} trainable variables did not "
        f"move under one Adam step: {sorted(report.unmoved)}")


# ---------------------------------------------------------------------
# 4. The smoke contract, proven falsifiable
# ---------------------------------------------------------------------


def _contract(output) -> None:
    """The forward contract. Shared with the meta-test so it is falsifiable."""
    assert not isinstance(output, (dict, list, tuple)), (
        f"DocRes returns a single restored image, got {type(output)}")
    assert tuple(output.shape) == (BATCH, HEIGHT, WIDTH, OUT_CHANNELS), (
        tuple(output.shape))
    assert_finite(output)


def test_the_forward_satisfies_its_contract(built_model):
    _contract(built_model(_inputs(), training=False))


def test_the_smoke_contract_rejects_a_broken_forward(built_model):
    """RED proof: the contract above raises ``AssertionError`` on every
    degenerate forward the shared breaker set produces.

    The oracle requires ``AssertionError`` SPECIFICALLY. A ``TypeError`` would
    be the contract CRASHING rather than judging -- that is what made one
    earlier meta-test in this tree vacuous -- which is why the shape assertion
    is preceded by the container-type assertion above.
    """
    rejections = assert_contract_rejects_a_broken_forward(
        built_model, _inputs(), _contract)
    assert len(rejections) >= 3, rejections


# ---------------------------------------------------------------------
# 5. Knob sensitivity -- STRUCTURAL knobs, pinned on the weight signature
# ---------------------------------------------------------------------
#
# `num_blocks`, `dim` and `heads` are structural: they change what weights
# EXIST. They are therefore asserted on the weight-SHAPE signature, never on an
# output difference. Two models with different shapes consume different numbers
# of RNG draws from the same seed, so their outputs differ whether or not the
# knob reached anything -- an output-difference assertion on a structural knob
# is satisfied by random-init luck alone.


def _variant_of(**overrides):
    def build():
        model = DocRes(**{**_SMALL, **overrides})
        model.build(INPUT_SHAPE)
        return model
    return build


def test_num_blocks_changes_the_parameterisation():
    signatures = assert_structural_knob_changes_weights(
        {
            "[1,1,1,1]": _variant_of(num_blocks=[1, 1, 1, 1]),
            "[2,1,1,1]": _variant_of(num_blocks=[2, 1, 1, 1]),
            "[2,3,3,4]": _variant_of(num_blocks=[2, 3, 3, 4]),
        },
        knob="num_blocks",
    )
    # Stronger than "different": more blocks is strictly more weights.
    counts = [len(signatures[k]) for k in ("[1,1,1,1]", "[2,1,1,1]", "[2,3,3,4]")]
    assert counts == sorted(counts) and counts[0] < counts[-1], counts


def test_dim_changes_the_parameterisation():
    assert_structural_knob_changes_weights(
        {8: _variant_of(dim=8), 16: _variant_of(dim=16)},
        knob="dim",
    )


def test_heads_changes_the_parameterisation():
    """``heads`` is structural here even though it moves no weight SHAPE...

    ...except one: the MDTA temperature is a ``(heads, 1, 1)`` weight, so the
    head count is visible in the signature. That is the only place it shows,
    which is exactly why this arm is worth having -- a ``heads`` value that
    never reached MDTA would leave every other shape identical.
    """
    signatures = assert_structural_knob_changes_weights(
        {
            "[1,2,4,8]": _variant_of(heads=[1, 2, 4, 8]),
            "[1,2,2,2]": _variant_of(heads=[1, 2, 2, 2]),
        },
        knob="heads",
    )
    temperatures = {
        key: sorted(shape for shape in signature if len(shape) == 3)
        for key, signature in signatures.items()
    }
    assert temperatures["[1,2,4,8]"] != temperatures["[1,2,2,2]"], temperatures


# ---------------------------------------------------------------------
# 6. XLA agrees with eager
# ---------------------------------------------------------------------


def test_the_traced_forward_agrees_with_the_eager_one():
    """``fit()`` on a GPU defaults to ``jit_compile='auto'``; an eager-only
    result is not a result. The oracle takes its own eager-vs-eager spread as
    the floor, so a stochastic forward could not be misread as an XLA defect.
    """
    report = assert_xla_equivalence(_build, _inputs, training=False, rtol=1e-2)
    # Per-output, not aggregated: the oracle reports one spread per output
    # tensor and a mean would hide a single stochastic head.
    assert report["eager_spread"] == [0.0], (
        f"DocRes' eager forward is not deterministic: {report['eager_spread']}")


# ---------------------------------------------------------------------
# 7. Both MODEL_VARIANTS rows really run
# ---------------------------------------------------------------------


@pytest.mark.parametrize("variant", sorted(DocRes.MODEL_VARIANTS))
def test_every_variant_row_builds_and_runs(variant):
    """A table row nobody ever instantiates is documentation, not a variant.

    Both rows are built at their FULL shipped configuration -- no ``dim``
    override -- because the row IS the configuration; overriding it would test
    a model the table does not describe.
    """
    keras.utils.set_random_seed(0)
    model = create_doc_res(variant)
    x = np.random.RandomState(1).randn(1, 32, 32, 6).astype("float32")
    outputs = keras.ops.convert_to_numpy(model(x, training=False))

    assert outputs.shape == (1, 32, 32, 3)
    assert np.isfinite(outputs).all()
    expected = DocRes.MODEL_VARIANTS[variant]
    assert model.num_blocks == expected["num_blocks"]
    assert model.heads == expected["heads"]


# ---------------------------------------------------------------------
# 8. The D-007 adjacency, closed over the ASSEMBLED model
# ---------------------------------------------------------------------
#
# CARRIED from step 5. `test_components.py` could prove only the "preceded by"
# half: both resamplers END at their pixel op, so their successor is wired in
# `model.py` and is unanswerable from a component
# (`test_a_resampler_alone_cannot_answer_the_successor_half` asserts exactly
# that). Both halves are re-run here over the recorded forward order of the
# assembled model, using the SAME two predicates rather than model-level
# copies of them.


def test_every_assembled_pixel_reshuffle_site_is_bracketed_on_both_sides(records):
    """The full D-007 claim: per-channel-parameterised op on each side."""
    recorded = RecordedForward(records)
    preceded = assert_pixel_reshuffle_sites_are_bracketed(recorded)
    followed = assert_pixel_reshuffle_sites_have_a_per_channel_successor(recorded)

    assert preceded == followed == EXPECTED_RESHUFFLE_SITES, (
        f"expected {EXPECTED_RESHUFFLE_SITES} pixel-reshuffle sites in the "
        f"assembled model, found {preceded} preceded / {followed} followed")


def test_the_successor_of_every_site_really_consumes_it(records):
    """Temporal adjacency upgraded to DATA-FLOW adjacency, by value.

    ``keras.ops.concatenate`` is a function, not a layer, so no recorder can
    see it and "the next op that ran" is not automatically "the op that read
    this tensor". Three of the six sites (the upsamplers) do have a
    concatenation in between, and the mechanism is asserted per site rather
    than accepted silently -- if a downsampler ever grew one, or an upsampler
    lost one, this fires.
    """
    from dl_techniques.layers.pooling.pixel_unshuffle import (
        PixelShuffle2D,
        PixelUnshuffle2D,
    )

    mechanisms = []
    for index, record in enumerate(records):
        if not isinstance(record.layer, (PixelShuffle2D, PixelUnshuffle2D)):
            continue
        assert index < len(records) - 1
        mechanisms.append((
            type(record.layer).__name__,
            assert_consumes(record, records[index + 1]),
        ))

    assert mechanisms == [
        # Down: the unshuffled tensor is fed straight into the next stage's
        # first LayerNormalization.
        ("PixelUnshuffle2D", "direct"),
        ("PixelUnshuffle2D", "direct"),
        ("PixelUnshuffle2D", "direct"),
        # Up: the shuffled tensor is concatenated with the skip connection
        # first, and the reduce/decoder op reads the concatenation.
        ("PixelShuffle2D", "concat(axis=-1)"),
        ("PixelShuffle2D", "concat(axis=-1)"),
        ("PixelShuffle2D", "concat(axis=-1)"),
    ], mechanisms
