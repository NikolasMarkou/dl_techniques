"""Contract and mechanism guards for ``doc_res/components.py``.

Three composites are under test: :class:`RestormerTransformerBlock`,
:class:`RestormerDownsample` and :class:`RestormerUpsample`. Shape tests are
blind to every claim that actually matters here, so each of the following gets
its own guard, and each guard was proven RED by injection before this file was
committed (the injections are recorded in the plan's ``decisions.md``):

1. The NET resampling effect. ``RestormerDownsample``'s convolution *halves*
   the channel count; the composite *doubles* it. Reading the convolution
   alone gets the sign wrong, so the doubling is asserted, not described.
2. **The D-007 pixel-shuffle bracketing claim.** This repo's
   ``PixelShuffle2D`` / ``PixelUnshuffle2D`` use a different channel-block
   ordering from ``torch.nn.PixelShuffle`` / ``PixelUnshuffle``, and the port
   ships anyway on the argument that every such site is bracketed by learnable
   per-channel-parameterised ops which absorb a fixed channel permutation. An
   anchor stating that is a comment; the two tests below are the guard. They
   are deliberately NOT greps over the source text:

   * :func:`test_the_pixel_reshuffle_site_is_preceded_by_a_per_channel_op`
     reads ``op_sequence``, the list ``call()`` itself iterates, so it inspects
     the real forward path; a hand-built decoy component whose ops are in the
     wrong order is asserted to be REJECTED, which is what makes the checker
     discriminating rather than always-true.
   * :func:`test_the_torch_vs_repo_ordering_is_a_channel_permutation` derives
     BOTH index maps from their published closed forms and measures that the
     repo's unshuffle output is a fixed channel permutation of torch's (10 of
     12 channels displaced at ``C=3``, reproducing D-007's table), and
     :func:`test_a_channel_permutation_is_absorbed_by_the_following_ops` then
     feeds THAT permutation through the exact op chain the model builds --
     ``LayerNormalization`` then the attention's ``qkv`` ``Conv2D`` -- and
     measures that permuting their per-channel weights restores the identical
     output. The control in the same test shows the un-permuted follower does
     NOT restore it, so the comparison is not passing for a trivial reason.

   The *structural* "and is followed by" half of the adjacency lives across a
   component boundary -- ``RestormerDownsample``'s consumer is wired in
   ``doc_res/model.py`` -- so it is UNANSWERABLE here, and
   :func:`test_a_resampler_alone_cannot_answer_the_successor_half` asserts
   exactly that rather than leaving it as prose. Both
   :func:`assert_pixel_reshuffle_sites_are_bracketed` and
   :func:`assert_pixel_reshuffle_sites_have_a_per_channel_successor` are
   exported for ``test_model.py`` to re-run over the ASSEMBLED model's
   recorded forward order, which is where the successor half was closed
   (iter-1/step-8).
3. Both residual adds are live. Zeroing the two output projections must make
   the whole block the IDENTITY at ``rtol=0``; if either ``+`` were dropped
   the output would be zero instead.
4. ``epsilon == 1e-5`` on both normalizations. Keras defaults to ``1e-3``, a
   100x difference with no shape symptom, so dropping the explicit kwarg is a
   silent defect this guard catches.
5. ``use_bias=False`` on every convolution, including the two resampler convs
   that hard-code it regardless of any outer flag.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.pooling.pixel_unshuffle import (
    PixelShuffle2D,
    PixelUnshuffle2D,
)
from dl_techniques.models.vision.image_restoration.doc_res.components import (
    RESTORMER_LAYERNORM_EPSILON,
    RestormerDownsample,
    RestormerTransformerBlock,
    RestormerUpsample,
)

from ..precision_arm_oracle import assert_float64_arm, assert_precision_arm


DIM = 48
RNG = np.random.default_rng(20260908)


def _x(shape):
    return RNG.normal(size=shape).astype("float32")


# ---------------------------------------------------------------------
# 1. Shape contracts
# ---------------------------------------------------------------------


def test_downsample_doubles_channels_and_halves_resolution():
    """``(2,32,32,48) -> (2,16,16,96)``: the NET effect, not the conv's."""
    layer = RestormerDownsample(n_feat=48)
    out = layer(_x((2, 32, 32, 48)))
    assert tuple(out.shape) == (2, 16, 16, 96)
    # The convolution alone goes the OTHER way -- 48 -> 24. Pinning that here
    # is what makes the composite's doubling a measured claim rather than a
    # restatement of the output shape.
    assert layer.conv.filters == 24
    assert layer.compute_output_shape((2, 32, 32, 48)) == (2, 16, 16, 96)


def test_upsample_halves_channels_and_doubles_resolution():
    """``(2,16,16,96) -> (2,32,32,48)``."""
    layer = RestormerUpsample(n_feat=96)
    out = layer(_x((2, 16, 16, 96)))
    assert tuple(out.shape) == (2, 32, 32, 48)
    assert layer.conv.filters == 192
    assert layer.compute_output_shape((2, 16, 16, 96)) == (2, 32, 32, 48)


@pytest.mark.parametrize("heads", [1, 2, 4, 8])
def test_transformer_block_is_shape_preserving(heads):
    """The block refines in place at every DocRes head count."""
    block = RestormerTransformerBlock(dim=DIM, num_heads=heads)
    out = block(_x((2, 16, 16, DIM)))
    assert tuple(out.shape) == (2, 16, 16, DIM)
    assert block.compute_output_shape((2, 16, 16, DIM)) == (2, 16, 16, DIM)


def test_the_resamplers_reject_an_odd_channel_count():
    with pytest.raises(ValueError, match="even"):
        RestormerDownsample(n_feat=47)
    with pytest.raises(ValueError, match="even"):
        RestormerUpsample(n_feat=47)


def test_the_components_reject_a_rank_3_input():
    """DocRes is NHWC-only; a sequence tensor must raise, not broadcast."""
    for layer in (
        RestormerDownsample(n_feat=48),
        RestormerUpsample(n_feat=48),
        RestormerTransformerBlock(dim=DIM, num_heads=2),
    ):
        with pytest.raises(ValueError, match="4D"):
            layer(_x((2, 16, DIM)))


# ---------------------------------------------------------------------
# 2. What round-trips, and what does not
# ---------------------------------------------------------------------


def test_the_bare_pixel_ops_are_exact_inverses_but_the_resamplers_are_not():
    """D-007's round-trip claim, and the claim it must NOT be confused with.

    ``PixelShuffle2D(PixelUnshuffle2D(x)) == x`` EXACTLY -- that is D-007's
    measured round-trip and it is a property of the two pooling layers alone.
    ``RestormerUpsample(RestormerDownsample(x))`` is NOT the identity and
    never could be: each composite owns an independent, randomly initialised
    ``3x3`` convolution. Asserting the pair round-trips would be asserting
    something false; the control below asserts it does not.
    """
    x = _x((2, 8, 8, 6))
    round_tripped = PixelShuffle2D(block_size=2)(PixelUnshuffle2D(scale=2)(x))
    assert np.array_equal(np.asarray(keras.ops.convert_to_numpy(round_tripped)), x)

    down = RestormerDownsample(n_feat=48)
    up = RestormerUpsample(n_feat=96)
    y = _x((2, 16, 16, 48))
    back = np.asarray(keras.ops.convert_to_numpy(up(down(y))))
    assert back.shape == y.shape
    # Two independent random convolutions: nowhere near an inverse.
    assert np.abs(back - y).max() > 1e-3


# ---------------------------------------------------------------------
# 3. The D-007 bracketing guard
# ---------------------------------------------------------------------


def _is_pixel_reshuffle(op) -> bool:
    return isinstance(op, (PixelShuffle2D, PixelUnshuffle2D))


def _is_per_channel_parameterised(op) -> bool:
    """Does ``op`` own a learnable parameter dedicated to each channel it reads?

    That is the property the D-007 absorption argument needs: a fixed
    permutation of the channel axis can be undone by permuting the op's own
    per-channel weights, so the two orderings span the same function family.

    * ``Conv2D`` / ``DepthwiseConv2D``: the kernel has a distinct slice per
      input channel and per output channel -- it absorbs a permutation of
      either side exactly.
    * ``LayerNormalization`` with ``scale`` and ``center``: ``gamma`` and
      ``beta`` are per-channel, and the normalization itself is
      permutation-equivariant, so a permutation survives it unchanged and is
      absorbed by the next weighted op.

    Anything else -- a pointwise activation, an add, a reshape -- is
    per-channel BLIND and returns ``False``.
    """
    if isinstance(op, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
        return True
    if isinstance(op, keras.layers.LayerNormalization):
        return bool(op.scale and op.center)
    return False


def assert_pixel_reshuffle_sites_are_bracketed(layer) -> int:
    """Assert every pixel-reshuffle in ``layer.op_sequence`` is preceded by a
    per-channel-parameterised op, and return how many sites were checked.

    ``op_sequence`` is the list the component's ``call()`` iterates, so this
    reads the real forward path rather than the source text.

    :param layer: A component exposing ``op_sequence``.
    :return: Number of pixel-reshuffle sites found (0 means the guard had no
        subject at this layer; the caller must check that).
    :rtype: int
    :raises AssertionError: If a site is first in the sequence or its
        predecessor is per-channel blind.
    """
    ops_ = list(layer.op_sequence)
    sites = 0
    for index, op in enumerate(ops_):
        if not _is_pixel_reshuffle(op):
            continue
        sites += 1
        assert index > 0, (
            f"{type(layer).__name__}: {type(op).__name__} is the FIRST op in "
            "op_sequence, so nothing precedes it to absorb the D-007 channel "
            "permutation"
        )
        previous = ops_[index - 1]
        assert _is_per_channel_parameterised(previous), (
            f"{type(layer).__name__}: {type(op).__name__} is preceded by "
            f"{type(previous).__name__}, which owns no per-channel learnable "
            "parameter; the D-007 absorption argument does not hold here"
        )
    return sites


def assert_pixel_reshuffle_sites_have_a_per_channel_successor(layer) -> int:
    """The OTHER half of the bracketing claim: what runs *after* each site.

    Split from :func:`assert_pixel_reshuffle_sites_are_bracketed` rather than
    folded into it, because within a single component the two halves are not
    both answerable: both resamplers end at their pixel op
    (``op_sequence == [conv, shuffle]``), so the successor lives in
    ``doc_res/model.py``. Applying this function to a resampler alone would
    always fail; applying it to the ASSEMBLED model's recorded forward order
    (``tests/test_models/test_doc_res/test_model.py``) is what closes the half
    that step 5 could only argue functionally.

    Both halves share :func:`_is_per_channel_parameterised`, deliberately -- a
    second copy of that predicate is a second chance to weaken one of them.

    :param layer: Anything exposing ``op_sequence``.
    :return: Number of pixel-reshuffle sites found (0 means no subject; the
        caller must check that).
    :rtype: int
    :raises AssertionError: If a site is LAST in the sequence or its successor
        is per-channel blind.
    """
    ops_ = list(layer.op_sequence)
    sites = 0
    for index, op in enumerate(ops_):
        if not _is_pixel_reshuffle(op):
            continue
        sites += 1
        assert index < len(ops_) - 1, (
            f"{type(layer).__name__}: {type(op).__name__} is the LAST op in "
            "op_sequence, so nothing follows it to absorb the D-007 channel "
            "permutation"
        )
        following = ops_[index + 1]
        assert _is_per_channel_parameterised(following), (
            f"{type(layer).__name__}: {type(op).__name__} is followed by "
            f"{type(following).__name__}, which owns no per-channel learnable "
            "parameter; the D-007 absorption argument does not hold here"
        )
    return sites


class _DecoyResampler(keras.layers.Layer):
    """A component with the ops in the WRONG order: the negative control.

    Its ``op_sequence`` puts the pixel-unshuffle FIRST, so nothing precedes it
    to absorb a channel permutation. If
    :func:`assert_pixel_reshuffle_sites_are_bracketed` accepted this, the guard
    on the real components would be proving nothing.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shuffle = PixelUnshuffle2D(scale=2)
        self.act = keras.layers.Activation("relu")

    @property
    def op_sequence(self):
        return [self.shuffle, self.act]


class _BlindBracketResampler(keras.layers.Layer):
    """Ops in the right ORDER but the predecessor is per-channel BLIND.

    A ``ReLU`` before the unshuffle preserves the sequence shape the structural
    check looks at while owning no weight at all, so it cannot absorb a
    permutation. This separates "the checker looks at position" from "the
    checker looks at the op's parameterisation" -- without it, a checker that
    only asserted ``index > 0`` would pass.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.act = keras.layers.Activation("relu")
        self.shuffle = PixelUnshuffle2D(scale=2)

    @property
    def op_sequence(self):
        return [self.act, self.shuffle]


def test_the_pixel_reshuffle_site_is_preceded_by_a_per_channel_op():
    """D-007's bracketing claim, read off the list ``call()`` iterates."""
    sites = 0
    for layer in (RestormerDownsample(n_feat=48), RestormerUpsample(n_feat=96)):
        sites += assert_pixel_reshuffle_sites_are_bracketed(layer)
    # Anti-vacuity: the guard must have had subjects. Exactly two
    # pixel-reshuffle sites exist in this module, one per resampler.
    assert sites == 2, f"expected 2 pixel-reshuffle sites, found {sites}"


def test_the_bracketing_checker_rejects_both_ways_of_breaking_the_claim():
    """The negative controls: position AND parameterisation each matter."""
    with pytest.raises(AssertionError, match="FIRST op"):
        assert_pixel_reshuffle_sites_are_bracketed(_DecoyResampler())
    with pytest.raises(AssertionError, match="no per-channel learnable"):
        assert_pixel_reshuffle_sites_are_bracketed(_BlindBracketResampler())


def test_the_successor_checker_rejects_both_ways_of_breaking_the_claim():
    """The same two decoys, judged by the successor half.

    They swap roles, which is the point of running both: ``_DecoyResampler``
    (``[shuffle, relu]``) has a successor but a per-channel BLIND one, and
    ``_BlindBracketResampler`` (``[relu, shuffle]``) has no successor at all.
    A checker that only asserted ``index < len - 1`` would pass the first, and
    one that only inspected the successor would crash on the second.
    """
    with pytest.raises(AssertionError, match="no per-channel learnable"):
        assert_pixel_reshuffle_sites_have_a_per_channel_successor(_DecoyResampler())
    with pytest.raises(AssertionError, match="LAST op"):
        assert_pixel_reshuffle_sites_have_a_per_channel_successor(
            _BlindBracketResampler())


def test_a_resampler_alone_cannot_answer_the_successor_half():
    """Why the assembled-model re-run in ``test_model.py`` exists at all.

    Both real resamplers END at their pixel op, so this assertion FAILS on
    each of them in isolation. Recording that as a passing test rather than a
    prose note keeps the claim honest: the successor half is not "also true
    here", it is unanswerable here.
    """
    for layer in (RestormerDownsample(n_feat=48), RestormerUpsample(n_feat=96)):
        with pytest.raises(AssertionError, match="LAST op"):
            assert_pixel_reshuffle_sites_have_a_per_channel_successor(layer)


def _torch_pixel_unshuffle_nhwc(x: np.ndarray, r: int) -> np.ndarray:
    """``torch.nn.PixelUnshuffle`` transcribed into NHWC, from its closed form.

    Torch maps source channel ``c`` at intra-block offset ``(di, dj)`` to
    output channel ``c * r * r + di * r + dj``. Written index-by-index on
    purpose: deriving it with the same reshape/transpose the repo layer uses
    would make this a second copy of the subject rather than a reference.
    """
    b, h, w, c = x.shape
    out = np.empty((b, h // r, w // r, c * r * r), dtype=x.dtype)
    for ch in range(c):
        for di in range(r):
            for dj in range(r):
                out[..., ch * r * r + di * r + dj] = x[:, di::r, dj::r, ch]
    return out


def _repo_ordering_permutation(c: int, r: int) -> np.ndarray:
    """``perm`` such that ``repo_out[..., k] == torch_out[..., perm[k]]``.

    The repo maps ``(di, dj, ch)`` to output channel ``(di * r + dj) * c + ch``;
    torch maps the same triple to ``ch * r * r + di * r + dj``.
    """
    perm = np.empty(c * r * r, dtype=np.int64)
    for ch in range(c):
        for di in range(r):
            for dj in range(r):
                perm[(di * r + dj) * c + ch] = ch * r * r + di * r + dj
    return perm


def test_the_torch_vs_repo_ordering_is_a_channel_permutation():
    """Reproduce D-007's table, and pin the difference as a PERMUTATION.

    The whole absorption argument rests on the difference being a fixed
    permutation of the channel axis -- if it were anything else (a mixing, a
    scaling) no downstream weight relabelling could undo it. Measured here at
    ``C=3, r=2``: 10 of 12 channels displaced, exactly D-007's count, and the
    repo output equals the torch output gathered through ``perm`` at
    ``max|diff| == 0``.
    """
    r, c = 2, 3
    x = _x((2, 8, 8, c))
    repo = np.asarray(keras.ops.convert_to_numpy(PixelUnshuffle2D(scale=r)(x)))
    torch = _torch_pixel_unshuffle_nhwc(x, r)
    perm = _repo_ordering_permutation(c, r)

    displaced = int((perm != np.arange(perm.size)).sum())
    assert displaced == 10, (
        f"expected D-007's 10 of 12 displaced channels, measured {displaced}"
    )
    # The orderings really do disagree -- without this the next assertion
    # would pass for a model where nothing had to be absorbed at all.
    assert not np.array_equal(repo, torch)
    assert np.array_equal(repo, torch[..., perm])


def test_a_channel_permutation_is_absorbed_by_the_following_ops():
    """The D-007 absorption claim, MEASURED on the real downstream chain.

    In the assembled model a ``RestormerDownsample`` output is consumed by a
    ``RestormerTransformerBlock``: first its ``norm1``
    (``LayerNormalization``), then its attention's ``qkv`` ``Conv2D``. This
    test permutes the downsample output by the exact torch-vs-repo permutation
    and permutes those two ops' per-channel weights to match; the composite
    output must come back unchanged. The CONTROL leaves the weights
    un-permuted and must NOT come back unchanged -- otherwise the chain would
    be permutation-insensitive and this test would prove nothing.
    """
    n_feat, r = 24, 2
    down = RestormerDownsample(n_feat=n_feat)
    y = np.asarray(keras.ops.convert_to_numpy(down(_x((2, 8, 8, n_feat)))))
    channels = y.shape[-1]              # n_feat * 2 == 48
    perm = _repo_ordering_permutation(n_feat // 2, r)
    assert perm.size == channels

    block = RestormerTransformerBlock(dim=channels, num_heads=2)
    block.build((None, 4, 4, channels))

    def chain(tensor, gamma, beta, kernel):
        """LayerNorm(axis=-1, eps=1e-5) then the qkv 1x1 conv, weights explicit."""
        mean = keras.ops.mean(tensor, axis=-1, keepdims=True)
        var = keras.ops.var(tensor, axis=-1, keepdims=True)
        unit = (tensor - mean) / keras.ops.sqrt(var + RESTORMER_LAYERNORM_EPSILON)
        return keras.ops.conv(unit * gamma + beta, kernel, strides=1,
                              padding="same")

    gamma = np.asarray(keras.ops.convert_to_numpy(block.norm1.gamma))
    beta = np.asarray(keras.ops.convert_to_numpy(block.norm1.beta))
    # A constant gamma/beta would make the LayerNorm permutation-blind and the
    # control below vacuous, so give them distinct per-channel values.
    gamma = gamma + RNG.normal(size=gamma.shape).astype("float32")
    beta = beta + RNG.normal(size=beta.shape).astype("float32")
    kernel = np.asarray(keras.ops.convert_to_numpy(block.attn.qkv.kernel))

    reference = np.asarray(keras.ops.convert_to_numpy(
        chain(y, gamma, beta, kernel)
    ))
    absorbed = np.asarray(keras.ops.convert_to_numpy(
        chain(y[..., perm], gamma[perm], beta[perm], kernel[:, :, perm, :])
    ))
    control = np.asarray(keras.ops.convert_to_numpy(
        chain(y[..., perm], gamma, beta, kernel)
    ))

    scale = float(np.abs(reference).max())
    assert np.abs(absorbed - reference).max() <= 1e-5 * scale, (
        "permuting the following ops' per-channel weights did NOT restore the "
        f"output: max|diff| = {np.abs(absorbed - reference).max():.3e}"
    )
    # Anti-vacuity: without the weight relabelling the permutation is visible.
    assert np.abs(control - reference).max() > 1e-2 * scale, (
        "the downstream chain is insensitive to a channel permutation, so the "
        "absorption assertion above proves nothing"
    )


# ---------------------------------------------------------------------
# 4. LayerNorm epsilon
# ---------------------------------------------------------------------


def test_both_layer_normalizations_use_the_upstream_epsilon():
    """Keras defaults to ``1e-3``; upstream Restormer uses ``1e-5``.

    Dropping the explicit ``epsilon=`` kwarg is a 100x change in every
    denominator with no shape symptom and no warning, which is exactly the
    failure ``layers/CLAUDE.md`` rule 5 exists for.
    """
    assert RESTORMER_LAYERNORM_EPSILON == 1e-5
    block = RestormerTransformerBlock(dim=DIM, num_heads=2)
    norms = [block.norm1, block.norm2]
    assert len(norms) == 2
    for norm in norms:
        assert norm.epsilon == 1e-5, (
            f"{norm.name} has epsilon={norm.epsilon}; Keras' 1e-3 default has "
            "leaked back in"
        )
        assert norm.axis == -1


# ---------------------------------------------------------------------
# 5. Bias-free convolutions
# ---------------------------------------------------------------------


def _all_convolutions(layer):
    return [
        sub for sub in layer._flatten_layers(include_self=False)
        if isinstance(sub, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D))
    ]


def test_every_convolution_is_bias_free():
    """Upstream passes ``bias=False`` everywhere, and the two resampler convs
    hard-code it regardless of any outer flag."""
    subjects = [
        RestormerTransformerBlock(dim=DIM, num_heads=2),
        RestormerDownsample(n_feat=48),
        RestormerUpsample(n_feat=96),
    ]
    total = 0
    for layer in subjects:
        layer.build((None, 16, 16, layer.dim if hasattr(layer, "dim")
                     else layer.n_feat))
        convs = _all_convolutions(layer)
        assert convs, f"{type(layer).__name__} exposed no convolutions to judge"
        total += len(convs)
        for conv in convs:
            assert conv.use_bias is False, (
                f"{type(layer).__name__}.{conv.name} carries a bias"
            )
            assert getattr(conv, "bias", None) is None
    # Anti-vacuity: 3 (MDTA) + 3 (GDFN) in the block, 1 in each resampler.
    assert total == 8, f"expected 8 convolutions across the three, found {total}"


def test_the_resamplers_expose_no_use_bias_knob():
    """The hard-coded ``bias=False`` must not be re-opened as a parameter.

    Adding one would make the port configurable where the reference is not and
    would silently change every level's parameter count.
    """
    import inspect

    for cls in (RestormerDownsample, RestormerUpsample):
        params = inspect.signature(cls.__init__).parameters
        assert "use_bias" not in params, (
            f"{cls.__name__} grew a use_bias parameter; upstream's "
            "Downsample/Upsample hard-code bias=False"
        )


# ---------------------------------------------------------------------
# 6. Both residual adds are live
# ---------------------------------------------------------------------


def test_both_residual_adds_are_live():
    """Zero the two output projections; the block must become the IDENTITY.

    With ``attn`` and ``ffn`` both silenced the block computes
    ``x + 0 + 0``. If either ``+ x`` were dropped the output would be zero (or
    the FFN branch alone), so this single ``rtol=0`` comparison covers both
    adds. The two controls beneath it show each branch is otherwise live.
    """
    x = _x((2, 8, 8, DIM))
    block = RestormerTransformerBlock(dim=DIM, num_heads=2)
    block(x)  # build

    live = np.asarray(keras.ops.convert_to_numpy(block(x, training=False)))

    block.attn.project_out.kernel.assign(
        keras.ops.zeros(block.attn.project_out.kernel.shape)
    )
    attn_only_silenced = np.asarray(
        keras.ops.convert_to_numpy(block(x, training=False))
    )
    block.ffn.project_out.kernel.assign(
        keras.ops.zeros(block.ffn.project_out.kernel.shape)
    )
    both_silenced = np.asarray(
        keras.ops.convert_to_numpy(block(x, training=False))
    )

    np.testing.assert_array_equal(both_silenced, x)
    # Controls: each branch really was contributing before it was silenced.
    assert np.abs(attn_only_silenced - x).max() > 1e-6, (
        "silencing only MDTA already made the block the identity, so the FFN "
        "residual branch is dead"
    )
    assert np.abs(live - attn_only_silenced).max() > 1e-6, (
        "silencing MDTA changed nothing, so the attention branch is dead"
    )


# ---------------------------------------------------------------------
# 7. Serialization round-trip on VALUES
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,shape",
    [
        (lambda: RestormerTransformerBlock(dim=DIM, num_heads=2), (2, 8, 8, DIM)),
        (lambda: RestormerDownsample(n_feat=48), (2, 8, 8, 48)),
        (lambda: RestormerUpsample(n_feat=96), (2, 8, 8, 96)),
    ],
    ids=["block", "downsample", "upsample"],
)
def test_serialization_round_trip_on_values(factory, shape, tmp_path):
    """Save, load, and compare OUTPUTS at ``rtol=0`` with explicit ``training``.

    A ``get_config`` equality check would pass against a layer that reloaded
    with different weights; only the values catch that.
    """
    inputs = keras.Input(shape=shape[1:])
    model = keras.Model(inputs, factory()(inputs))
    x = _x(shape)
    before = np.asarray(keras.ops.convert_to_numpy(model(x, training=False)))

    path = tmp_path / "component.keras"
    model.save(path)
    reloaded = keras.models.load_model(path)
    after = np.asarray(keras.ops.convert_to_numpy(reloaded(x, training=False)))

    np.testing.assert_allclose(after, before, rtol=0, atol=0)


# ---------------------------------------------------------------------
# 8. Dynamic spatial dims
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,channels,sizes,expected",
    [
        (lambda: RestormerTransformerBlock(dim=DIM, num_heads=2), DIM,
         [(1, 8, 8), (1, 16, 24)], lambda h, w: (h, w)),
        (lambda: RestormerDownsample(n_feat=48), 48,
         [(1, 8, 8), (1, 16, 24)], lambda h, w: (h // 2, w // 2)),
        (lambda: RestormerUpsample(n_feat=96), 96,
         [(1, 8, 8), (1, 16, 24)], lambda h, w: (h * 2, w * 2)),
    ],
    ids=["block", "downsample", "upsample"],
)
def test_builds_symbolically_with_unknown_spatial_dims(
        factory, channels, sizes, expected
):
    """DocRes runs at arbitrary page sizes; ``(None, None, None, C)`` must build."""
    inputs = keras.Input(shape=(None, None, channels))
    model = keras.Model(inputs, factory()(inputs))
    for batch, h, w in sizes:
        out = model(_x((batch, h, w, channels)))
        assert tuple(out.shape)[1:3] == expected(h, w)


# ---------------------------------------------------------------------
# 9. Precision arms
# ---------------------------------------------------------------------


_ARMS = [
    ("block", lambda: RestormerTransformerBlock(dim=16, num_heads=2), 16),
    ("downsample", lambda: RestormerDownsample(n_feat=16), 16),
    ("upsample", lambda: RestormerUpsample(n_feat=16), 16),
]


@pytest.mark.parametrize("name,build,channels", _ARMS, ids=[a[0] for a in _ARMS])
def test_mixed_float16_arm(name, build, channels):
    """Part 1-4 of the fp16 arm, through the shared oracle."""
    reports = assert_precision_arm(
        build=build,
        make_inputs=lambda: np.random.RandomState(0).randn(
            1, 8, 8, channels
        ).astype("float32"),
        rtol_against_float32=5e-2,
    )
    assert reports["mixed_float16"]["dtypes"] == ["float16"]
    assert reports["mixed_float16"]["n_nan"] == [0]


@pytest.mark.parametrize("name,build,channels", _ARMS, ids=[a[0] for a in _ARMS])
def test_float64_arm(name, build, channels):
    """The float64 arm: policy, input and output all really are float64."""
    report = assert_float64_arm(
        build=build,
        make_inputs=lambda: np.random.RandomState(0).randn(
            1, 8, 8, channels
        ).astype("float32"),
    )
    assert report["dtypes"] == ["float64"]
    assert report["n_nan"] == [0]
    assert report["n_inf"] == [0]
