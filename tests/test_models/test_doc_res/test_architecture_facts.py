"""Six single-claim guards for the ``doc_res`` invariants nothing else can see.

Each of the six is an architecture FACT: a property of the assembled network
that a shape test, a parameter count and a serialization round trip all agree
about whether it holds or not. One test per fact, no test asserting two of
them, so a failure names the invariant rather than "the model changed".

Why these six and not others -- what the neighbouring instruments are blind to
-----------------------------------------------------------------------------
* **The shape contract is blind to all six.** ``(B,H,W,6) -> (B,H,W,3)`` is
  produced just as well by a network with an input residual, with a symmetric
  ``reduce_chan_level1``, with a dead ``skip_conv``, with biased convolutions
  and with Keras' default ``1e-3`` epsilon.
* **The parameter-count oracle
  (``test_model_smoke.test_the_parameter_count_matches_the_independent_derivation``)
  is blind to exactly two of them, and this is measured, not assumed.**
  Invariant 1 (no input residual) adds ZERO parameters, so no count can see
  it. Invariant 6 (``num_blocks == [2,3,3,4]``) is worse: the oracle reads
  ``DocRes.MODEL_VARIANTS[variant]["num_blocks"]`` to build its own
  expectation, so editing the table moves BOTH sides of the comparison and the
  test stays green -- and its anti-vacuity companion
  ``test_the_shipped_docres_count_is_pinned_absolutely`` pins
  ``analytic_parameter_count()`` at its own function default, never at the
  table, so it does not close the gap either. Guard 6 below is the only thing
  in the tree that reads the shipped table and refuses to move with it.
* **The remaining four are visible to a parameter count only as a number.**
  A biased conv or a missing ``reduce_chan_level1`` shifts ``count_params()``,
  but the count test then reports "15,203,680 != 15,207,552", which names
  nothing. These guards name the invariant.

RED proof
---------
Every guard here was proven RED by INJECTION into the shipped source -- the
real ``model.py`` / ``components.py`` edited, the suite run, the verbatim
failure captured, the file restored, and the restoration verified by
``sha256sum``. The injection and its output are recorded per guard in the
plan's ``decisions.md`` (iter-1/step-8) and summarised in each docstring. A
guard that will not go red under its own injection is a comment.
"""

import numpy as np
import keras
import pytest

from dl_techniques.models.vision.image_restoration.doc_res.model import (
    DocRes,
    create_doc_res,
)

from tests.norm_epsilon_oracle import assert_every_block_norm_uses
from .forward_order_oracle import (
    leaf_layers,
    record_forward,
    record_forward_ops,
)

# ---------------------------------------------------------------------
# Subjects. Two, and the split is deliberate.
#
# Guards 2 and 6 are claims about the SHIPPED `docres` configuration -- "96
# channels, not 48" and "[2,3,3,4]" are literal values of that row -- so they
# are asserted on the real thing at dim=48. The other four are claims about
# every DocRes, so they run on a small subject.
# ---------------------------------------------------------------------

_SMALL = dict(dim=8, num_blocks=[1, 1, 1, 1], num_refinement_blocks=1,
              heads=[1, 2, 4, 8])
HEIGHT = WIDTH = 32

#: 2 LayerNormalizations per transformer block; the ``_SMALL`` subject has
#: 4 encoder/latent + 3 decoder + 1 refinement = 8 blocks. Anti-vacuity for
#: guard 5: a walk that found zero norms would satisfy "all of them are 1e-5".
SMALL_NORM_COUNT = 16

#: Convolutions in the ``_SMALL`` subject: 43 ``Conv2D`` + 16
#: ``DepthwiseConv2D``. Anti-vacuity for guard 4.
SMALL_CONV_COUNT = 59


def _small() -> DocRes:
    keras.utils.set_random_seed(0)
    model = DocRes(**_SMALL)
    model.build((None, None, None, 6))
    return model


def _x(batch: int = 1, seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).randn(
        batch, HEIGHT, WIDTH, 6).astype("float32")


@pytest.fixture(scope="module")
def shipped() -> DocRes:
    """The real ``docres`` row at ``dim=48`` -- 15.2M parameters."""
    keras.utils.set_random_seed(0)
    model = create_doc_res("docres")
    model.build((None, None, None, 6))
    return model


# ---------------------------------------------------------------------
# 1. There is NO input residual
# ---------------------------------------------------------------------


# DECISION plan-2026-09-08T111844-de235227/D-018
# Do NOT restore the "perturb the input by a constant and assert the output
# does not shift by that constant" form of this guard, however much more
# directly it reads as the invariant. MEASURED: with the residual injected it
# PASSES (tracked 1.387 against a 1e-3 threshold), because DocRes is not
# shift-invariant and the backbone's own response to the shift is 2.75x the
# residual's contribution. See D-018 in the plan's decisions.md.
def test_nothing_is_added_after_the_output_projection():
    """The returned tensor IS ``output_conv``'s output, bit for bit.

    The usual image-restoration convention is ``return f(x) + x``, and the
    reference DocRes does not follow it (``restormer_arch.py:278-280`` ends at
    ``return self.output(...)``). Several DocRes tasks are not perturbations of
    their input at all -- binarization emits logits, dewarping emits a
    coordinate field -- so an input residual would be actively wrong there,
    not merely unfaithful.

    **The mechanism the plan specified for this guard -- "perturb the input by
    a constant and assert the output does not shift by that constant" -- was
    written first, and it does not work on this architecture.** MEASURED under
    the injection ``return self.output_conv(x) + inputs[..., :3]`` (dim=8
    subject, shift 0.5): the output moved by 1.376 in response to the shift and
    the residual contributed 0.5 of it, so ``max|out(x+c) - (out(x) + c)|`` read
    1.387 -- indistinguishable from the 1.376 the un-injected model produces.
    The backbone is not shift-invariant (``patch_embed`` is bias-free, so a
    constant input shift becomes a constant per-channel offset that the first
    ``LayerNormalization`` only partially removes), and its own response
    swamps the residual by 2.8x. That formulation is not falsifiable here, and
    a guard that cannot go red is a comment.

    What replaces it is strictly stronger: the model's returned tensor is
    compared, at ``atol=0``, with what ``output_conv`` actually emitted during
    the same forward. Any post-projection addition reds it -- an input
    residual, a residual on a DIFFERENT tensor (which a zeroed-``output_conv``
    test would miss), a clamp, a rescale.

    RED proof (iter-1/step-8): ``return self.output_conv(x) + inputs[..., :3]``
    in ``model.py:call`` -> ``AssertionError: DocRes returned a tensor that is
    not output_conv's output: max|delta| = 3.74e+00. Something is applied after
    the projection -- an input residual is the usual culprit``.
    """
    model = _small()
    trace = record_forward(model, _x())

    last = trace.records[-1]
    assert last.layer is model.output_conv, (
        f"the last leaf op to run was {last.layer.name}, not output_conv; "
        "this guard assumes the projection is final")

    delta = float(np.abs(trace.outputs - last.outputs).max())
    assert delta == 0.0, (
        f"DocRes returned a tensor that is not output_conv's output: "
        f"max|delta| = {delta:.2e}. Something is applied after the "
        "projection -- an input residual is the usual culprit")
    # Anti-vacuity: the compared tensors are not two all-zero arrays.
    assert np.abs(trace.outputs).max() > 0.0


# ---------------------------------------------------------------------
# 2. `decoder_level1` and `refinement` run at 96 channels, not 48
# ---------------------------------------------------------------------


def test_the_last_third_of_the_network_runs_at_double_the_embedding_width(
        shipped):
    """Read the channel width of the tensors that actually FLOWED.

    Upstream omits the ``reduce_chan_level1`` that levels 3 and 2 each have
    (``restormer_arch.py:231``: "NO 1x1 conv to reduce channels"), so the
    concatenated ``2 * dim`` width is carried through ``decoder_level1``, the
    four ``refinement`` blocks and into ``output_conv``. Adding the missing
    1x1 "for symmetry" would change the width of the last third of the network.

    ``test_model_smoke.test_asymmetry_2/3`` assert the same fact off the
    CONFIG (``block.dim``). This one asserts it off the runtime tensors, which
    is the stronger claim: a block that stored ``dim=96`` but was handed a
    48-channel tensor would pass there and fail here.

    RED proof (iter-1/step-8): ``self.decoder_level1``/``self.refinement``
    built at ``level_dims[0]`` instead of ``level_dims[1]`` in
    ``model.py:__init__`` -> ``AssertionError: decoder_level1/refinement ran
    at {48} channels, expected {96}`` (the forward raises first if the widths
    are inconsistent, so the injection also had to add the missing
    ``reduce_chan_level1``; both are recorded in decisions.md).
    """
    records = record_forward_ops(shipped, _x())
    by_id = {id(record.layer): record for record in records}

    widths = set()
    for block in list(shipped.decoder_level1) + list(shipped.refinement):
        # `norm1` is the first op of every transformer block, so its INPUT is
        # the tensor the stage was handed.
        record = by_id[id(block.norm1)]
        widths.add(int(record.inputs.shape[-1]))

    assert widths == {96}, (
        f"decoder_level1/refinement ran at {widths} channels, expected {{96}} "
        f"(= 2 * dim at dim={shipped.dim})")
    # And the width survives all the way into the projection.
    assert int(by_id[id(shipped.output_conv)].inputs.shape[-1]) == 96
    # Anti-vacuity: the subject really is the 48-wide-embedding model, so
    # "96, not 48" is a distinction and not a tautology.
    assert shipped.dim == 48
    assert int(by_id[id(shipped.patch_embed)].outputs.shape[-1]) == 48


# ---------------------------------------------------------------------
# 3. `skip_conv` is live
# ---------------------------------------------------------------------


def test_zeroing_skip_conv_moves_the_output():
    """The unconditional skip projection is really in the forward path.

    Upstream builds ``skip_conv`` only when ``dual_pixel_task=True`` but uses
    it with no guard at all in ``forward``, so the flag has exactly one
    working setting; this port therefore has no flag. That makes "is it
    applied?" a question with no config answer -- only the forward can say.

    Zeroing the KERNEL, not deleting the layer: the layer keeps its place in
    the weight list, so this cannot pass by changing the model's structure.

    RED proof (iter-1/step-8): ``x = x + self.skip_conv(enc1_in)`` replaced by
    ``x = x`` in ``model.py:call`` -> ``AssertionError: zeroing skip_conv's
    kernel changed the output by 0.00e+00; the projection is not in the
    forward path``. (The `materialize_sublayers` build then also leaves
    `skip_conv` unbuilt, which is what `test_asymmetry_4` sees -- but that
    test reads `.built`, and a `skip_conv` applied and then MULTIPLIED BY ZERO
    would satisfy it.)
    """
    model = _small()
    x = _x()

    before = np.asarray(keras.ops.convert_to_numpy(model(x, training=False)))
    model.skip_conv.kernel.assign(keras.ops.zeros_like(model.skip_conv.kernel))
    after = np.asarray(keras.ops.convert_to_numpy(model(x, training=False)))

    delta = float(np.abs(after - before).max())
    assert delta > 0.0, (
        f"zeroing skip_conv's kernel changed the output by {delta:.2e}; the "
        "projection is not in the forward path")


# ---------------------------------------------------------------------
# 4. Every backbone convolution is bias-free
# ---------------------------------------------------------------------


def test_every_convolution_in_the_backbone_is_bias_free():
    """``use_bias=False`` everywhere, asserted by walking the built tree.

    Not a config read: ``DocRes.use_bias`` is one flag and it does NOT govern
    every convolution -- ``patch_embed``, ``output_conv`` and both resamplers
    hard-code ``use_bias=False`` regardless of it, matching upstream. Only a
    walk over the actual layers can state the invariant for all of them.

    RED proof (iter-1/step-8): ``use_bias=True`` on ``patch_embed`` in
    ``model.py:__init__`` -> ``AssertionError: 1 of 59 convolutions carries a
    bias: ['patch_embed']``.
    """
    model = _small()
    convs = [
        layer for layer in leaf_layers(model)
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D))
    ]
    assert len(convs) == SMALL_CONV_COUNT, (
        f"expected {SMALL_CONV_COUNT} convolutions in the subject, walked "
        f"{len(convs)}; the count is pinned so this guard cannot go vacuous")

    biased = sorted(layer.name for layer in convs if layer.use_bias)
    assert not biased, (
        f"{len(biased)} of {len(convs)} convolutions carries a bias: {biased}")
    # The flag is not the weight. A layer could declare `use_bias=False` and
    # still own a bias variable if something re-created it.
    with_bias_weight = sorted(
        layer.name for layer in convs if len(layer.weights) != 1)
    assert not with_bias_weight, (
        f"convolutions declaring no bias but owning more than a kernel: "
        f"{with_bias_weight}")


# ---------------------------------------------------------------------
# 5. Every LayerNormalization runs at epsilon 1e-5
# ---------------------------------------------------------------------


def test_every_layer_normalization_uses_the_restormer_epsilon():
    """``epsilon=1e-5``, set EXPLICITLY -- Keras' default is ``1e-3``.

    A 100x difference in a denominator with no symptom: no raise, no NaN, no
    shape change. Dropping the explicit kwarg in ``components.py`` would
    silently inherit ``1e-3`` and nothing else in this tree would notice.

    The walk is ``tests/norm_epsilon_oracle.assert_every_block_norm_uses``,
    the shared instrument six other packages use, rather than a local
    ``for layer in ...`` -- its ``expected_count`` arm is what keeps the
    assertion from being vacuous. Its liveness companion
    ``assert_epsilon_tracks_the_knob`` is deliberately NOT used: DocRes has no
    epsilon knob to move, which is the very thing being pinned here.

    RED proof (iter-1/step-8): ``epsilon=1e-5`` deleted from ``norm1`` in
    ``components.py`` -> ``AssertionError: 8 of 16 in-block norms do not use
    the model's own epsilon 1e-05``.
    """
    model = _small()
    found = assert_every_block_norm_uses(
        [model], expected=1e-5, expected_count=SMALL_NORM_COUNT)
    assert {row[2] for row in found} == {1e-5}
    # State the alternative explicitly: this is the value Keras would have
    # given us for free, and it is NOT the one shipped.
    assert keras.layers.LayerNormalization().epsilon == 1e-3


# ---------------------------------------------------------------------
# 6. `num_blocks == [2, 3, 3, 4]` for the shipped `docres` variant
# ---------------------------------------------------------------------


def test_the_docres_variant_ships_the_two_three_three_four_schedule(shipped):
    """The depth schedule, read off the BUILT model and pinned as a literal.

    ``[2,3,3,4]`` is the DocRes configuration at all three upstream call sites;
    ``[4,6,6,8]`` is the Restormer paper default that the upstream class
    carries and that every DocRes call site overrides. Confusing the two is a
    26.1M-parameter model wearing a 15.2M-parameter name.

    **This guard exists because the parameter-count oracle cannot see this
    fact.** That oracle derives its expectation from
    ``DocRes.MODEL_VARIANTS[variant]["num_blocks"]``, so an edit to the table
    moves the expectation with the model and the comparison stays green; its
    absolute companion pins ``analytic_parameter_count()`` at that function's
    OWN default, which the table cannot reach either. The literal below is the
    only place in the tree that would have to be edited a second time.

    RED proof (iter-1/step-8): ``MODEL_VARIANTS["docres"]["num_blocks"]``
    changed to ``[4, 6, 6, 8]`` -> this guard reds with
    ``AssertionError: the docres variant ships num_blocks=[4, 6, 6, 8],
    expected [2, 3, 3, 4]`` while both parameter-count tests stay GREEN. That
    green is the measurement this docstring is making, not an aside.
    """
    assert DocRes.MODEL_VARIANTS["docres"]["num_blocks"] == [2, 3, 3, 4], (
        f"the docres variant ships "
        f"num_blocks={DocRes.MODEL_VARIANTS['docres']['num_blocks']}, "
        "expected [2, 3, 3, 4]")
    # The table is one thing; what the built model runs is another.
    assert shipped.num_blocks == [2, 3, 3, 4]
    assert [len(stage) for stage in (shipped.encoder_level1,
                                     shipped.encoder_level2,
                                     shipped.encoder_level3,
                                     shipped.latent)] == [2, 3, 3, 4]
    # And the row it must not be confused with is still the other one.
    assert DocRes.MODEL_VARIANTS["restormer_base"]["num_blocks"] == [4, 6, 6, 8]
