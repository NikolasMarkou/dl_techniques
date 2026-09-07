"""Per-site guards for `tripse_attention.py`'s composite initializer fan-out.

Every class in `tripse_attention.py` resolves ONE initializer instance in
`__init__` (`self.kernel_initializer = initializers.get(kernel_initializer)`)
and hands that same instance to every sub-layer it constructs. A seedless
`Initializer` INSTANCE replays the same underlying sample at every later site,
so those sub-layer kernels start life as the same random numbers -- a spatial
branch gate and a channel gate, which differ by nothing but their
architectural role, come out bit-identical.

MEASURED before the fix (plan-2026-09-07T183458-be1c267e step 6, input
`(2, 8, 8, 4)`): 14 of 14 source sites live, 37 aliased weight tensors --
`TripletAttentionBranch` 1, `TripSE1` 5, `TripSE2` 9, `TripSE3` 9, `TripSE4`
11, `_SEWeights` standalone 2. Every one bit-identical to a replay from the
still-shared instance at that weight's own shape.

Oracle (plan.md § S-1)
----------------------
Build the layer, then draw from the layer's OWN still-shared
`self.kernel_initializer` at that weight's OWN shape and assert the created
weight is not bit-equal to that replay. One assertion per source site, so a
one-line revert reddens exactly that site. A pairwise comparison between two
sites is deliberately NOT the guard -- cloning either member decorrelates the
pair, so a single-line revert would stay green (re-measured in this plan's
step 3).

Why the four `SqueezeExcitation` construction sites are guarded differently
---------------------------------------------------------------------------
Four of the 14 sites hand the shared instance to the SHARED
`SqueezeExcitation` layer (`TripSE1.se_block`, `TripSE2.se_layers`,
`TripSE3.se_layers`, `TripSE4.final_se`), which lives in
`layers/conv_blocks/squeeze_excitation.py` and has its own, separate internal
fan-out between `conv_reduce` and `conv_restore`. That internal fan-out is a
DIFFERENT defect with a DIFFERENT owner and is fixed in its own commit (plan
step 7 / decision D-002).

So those four sites are guarded on the INSTANCE that crosses the boundary --
`se.kernel_initializer is not layer.kernel_initializer`, and mutually distinct
across the three per-branch SE blocks -- not on the weight values inside the
SE block. That keeps each guard RED-provable by reverting exactly its own
line while making it immune to the step-7 change: step 7 alters what
`SqueezeExcitation` does with the instance it received, never which instance
it received. A value guard on an SE-internal weight would move under step 7
and make an unrelated commit look as though it had reddened this one.

`TripSE4.se_logit_layers` is NOT in that set. `_SEWeights` is defined in
`tripse_attention.py` itself, so its `conv_reduce`/`conv_restore` are
tripse-owned and are guarded on their values like every other local site.

Scope of the claim, exactly
---------------------------
Copied from the corrected canonical wording in
`src/dl_techniques/initializers/clone.py`'s module docstring: independence
holds for a RANDOM SEEDLESS initializer. Three exemptions, all correct
behaviour, none a defect --

1. a caller-supplied SEEDED instance (e.g. `GlorotUniform(seed=7)`) replays
   deliberately and by contract, ACROSS DIFFERING SHAPES TOO;
2. a DETERMINISTIC initializer (`'zeros'`/`'ones'`/`Constant`, and `Identity`
   only where the weight is 2-D -- it raises on rank 3+) holds no random
   state, so every site is bit-identical and that is what it is meant to do;
3. a CUSTOM initializer whose `get_config()`/`from_config()` round trip raises
   falls back to `copy.deepcopy`, which copies the ALREADY-RESOLVED seed
   rather than drawing a new one, so such a site silently stays tied.

Exemption 1 is asserted below as a positive control, so this module never
states an absolute it has not measured. There is deliberately NO bias arm, and
that is a MEASURED verdict rather than an omission: none of the six classes
exposes a `bias_initializer` parameter at all, so every bias comes from Keras'
stock `'zeros'` (exemption 2) and there is no shared bias instance to fan out.
`test_no_bias_initializer_fans_out_because_there_is_no_bias_initializer` pins
that reasoning instead of shipping a guard that could not fail.
"""

import inspect

import pytest
import numpy as np
import keras

from dl_techniques.layers.attention.tripse_attention import (
    TripletAttentionBranch,
    TripSE1,
    TripSE2,
    TripSE3,
    TripSE4,
    _SEWeights,
)

# ---------------------------------------------------------------------

_SHAPE = (2, 8, 8, 4)
_SUFFIXES = ('hw', 'cw', 'hc')

_ALL_CLASSES = (TripletAttentionBranch, TripSE1, TripSE2, TripSE3, TripSE4, _SEWeights)


def _np(x):
    return keras.ops.convert_to_numpy(x)


def _built(cls, **kwargs):
    layer = cls(kernel_initializer=keras.initializers.GlorotUniform(), **kwargs)
    layer(np.zeros(_SHAPE, dtype='float32'))
    return layer


def _assert_is_not_the_shared_replay(layer, weight, label):
    """The per-site oracle, drawn at the weight's OWN shape."""
    replay = _np(layer.kernel_initializer(tuple(weight.shape), dtype='float32'))
    assert not np.array_equal(replay, _np(weight)), (
        f"{label} {tuple(weight.shape)} is bit-identical to a fresh draw from "
        f"the layer's shared kernel_initializer instance -- the site was not cloned"
    )


# ---------------------------------------------------------------------


class TestTheTripletBranchInitializerDoesNotFanOut:
    """`TripletAttentionBranch`, 1 site (`tripse_attention.py` `self.conv`)."""

    def test_the_branch_conv_kernel_is_not_the_shared_replay(self):
        layer = _built(TripletAttentionBranch)
        _assert_is_not_the_shared_replay(layer, layer.conv.kernel, 'conv/kernel')


class TestTheTripSE1InitializerDoesNotFanOut:
    """`TripSE1`, 4 sites: three `TripletAttentionBranch` + one `SqueezeExcitation`.

    The three branches are the sharpest group here: H-W, C-W and H-C are the
    same convolution applied to three DIFFERENT axis pairs, so their only
    architectural difference is which pair they gate. Sharing one initializer
    made all three start as the same kernel.
    """

    @pytest.mark.parametrize("suffix", _SUFFIXES)
    def test_the_branch_conv_kernel_is_not_the_shared_replay(self, suffix):
        layer = _built(TripSE1)
        branch = getattr(layer, f'branch_{suffix}')
        _assert_is_not_the_shared_replay(
            layer, branch.conv.kernel, f'branch_{suffix}/conv/kernel'
        )

    def test_the_se_block_did_not_receive_the_shared_instance(self):
        """The SE boundary site: guarded on the INSTANCE, not on SE's weights.

        `SqueezeExcitation` has its own internal `conv_reduce`/`conv_restore`
        fan-out, fixed separately (D-002 / step 7). Asserting a value inside
        the SE block would make that commit look like a regression here.
        """
        layer = _built(TripSE1)
        assert layer.se_block.kernel_initializer is not layer.kernel_initializer


class TestTheTripSE2InitializerDoesNotFanOut:
    """`TripSE2`, 2 sites, each inside the 3-iteration branch loop."""

    @pytest.mark.parametrize("index,suffix", list(enumerate(_SUFFIXES)))
    def test_the_branch_conv_kernel_is_not_the_shared_replay(self, index, suffix):
        layer = _built(TripSE2)
        _assert_is_not_the_shared_replay(
            layer, layer.conv_layers[index].kernel, f'conv_{suffix}/kernel'
        )

    def test_no_se_block_received_the_shared_instance(self):
        """The SE boundary site. Instance-level, for the reason in the docstring."""
        layer = _built(TripSE2)
        received = [se.kernel_initializer for se in layer.se_layers]
        assert len(received) == 3
        for instance in received:
            assert instance is not layer.kernel_initializer
        assert len({id(instance) for instance in received}) == 3


class TestTheTripSE3InitializerDoesNotFanOut:
    """`TripSE3`, 2 sites, each inside the 3-iteration branch loop."""

    @pytest.mark.parametrize("index,suffix", list(enumerate(_SUFFIXES)))
    def test_the_branch_conv_kernel_is_not_the_shared_replay(self, index, suffix):
        layer = _built(TripSE3)
        _assert_is_not_the_shared_replay(
            layer, layer.conv_layers[index].kernel, f'conv_{suffix}/kernel'
        )

    def test_no_se_block_received_the_shared_instance(self):
        layer = _built(TripSE3)
        received = [se.kernel_initializer for se in layer.se_layers]
        assert len(received) == 3
        for instance in received:
            assert instance is not layer.kernel_initializer
        assert len({id(instance) for instance in received}) == 3


class TestTheTripSE4InitializerDoesNotFanOut:
    """`TripSE4`, 3 sites: `_SEWeights` x3, `conv_{suffix}` x3, one final SE.

    `_SEWeights` is defined in `tripse_attention.py`, so unlike the shared
    `SqueezeExcitation` its two convolutions ARE tripse-owned and get value
    guards.
    """

    @pytest.mark.parametrize("index,suffix", list(enumerate(_SUFFIXES)))
    def test_the_branch_conv_kernel_is_not_the_shared_replay(self, index, suffix):
        layer = _built(TripSE4)
        _assert_is_not_the_shared_replay(
            layer, layer.conv_layers[index].kernel, f'conv_{suffix}/kernel'
        )

    @pytest.mark.parametrize("index,suffix", list(enumerate(_SUFFIXES)))
    @pytest.mark.parametrize("conv_name", ['conv_reduce', 'conv_restore'])
    def test_the_se_logits_kernel_is_not_the_shared_replay(self, conv_name, index, suffix):
        layer = _built(TripSE4)
        conv = getattr(layer.se_logit_layers[index], conv_name)
        _assert_is_not_the_shared_replay(
            layer, conv.kernel, f'se_logits_{suffix}/{conv_name}/kernel'
        )

    def test_the_final_se_did_not_receive_the_shared_instance(self):
        """The SE boundary site. Instance-level, for the reason in the docstring."""
        layer = _built(TripSE4)
        assert layer.final_se.kernel_initializer is not layer.kernel_initializer


class TestTheSEWeightsInitializerDoesNotFanOut:
    """`_SEWeights`, 2 sites: `conv_reduce` and `conv_restore`.

    The sharpest pair in this file that is NOT behind the shared
    `SqueezeExcitation`: a bottleneck DOWN-projection and its matching UP-
    projection, whose entire reason to exist is that they do opposite things.
    They are created in `build()`, not `__init__` (an accepted deviation --
    the bottleneck width needs the input channel count), which is why the
    guard builds the layer before looking.
    """

    @pytest.mark.parametrize("conv_name", ['conv_reduce', 'conv_restore'])
    def test_the_bottleneck_conv_kernel_is_not_the_shared_replay(self, conv_name):
        layer = _built(_SEWeights)
        conv = getattr(layer, conv_name)
        _assert_is_not_the_shared_replay(layer, conv.kernel, f'{conv_name}/kernel')


class TestTheTripseExemptionsThisModuleDoesNotClaimAway:
    """Positive controls. Cloning must not break any exemption."""

    @pytest.mark.parametrize("cls", _ALL_CLASSES, ids=lambda c: c.__name__)
    def test_no_bias_initializer_fans_out_because_there_is_no_bias_initializer(self, cls):
        """MEASURED: no class in this file exposes a `bias_initializer`.

        Every bias therefore comes from Keras' stock `'zeros'` -- exemption 2
        (DETERMINISTIC: identical at every site and correctly so). There is no
        shared bias instance, so a bias fan-out guard could not fail. This
        records the measurement instead of silently omitting a bias arm.
        """
        params = inspect.signature(cls.__init__).parameters
        assert 'bias_initializer' not in params
        assert params['use_bias'].default is False

        layer = cls(
            kernel_initializer=keras.initializers.GlorotUniform(),
            use_bias=True,
        )
        layer(np.zeros(_SHAPE, dtype='float32'))
        biases = [w for w in layer.weights if w.path.endswith('/bias')]
        assert biases, f"{cls.__name__} created no bias at use_bias=True"
        for bias in biases:
            assert float(np.max(np.abs(_np(bias)))) == 0.0

    @pytest.mark.parametrize("cls", _ALL_CLASSES, ids=lambda c: c.__name__)
    def test_a_seeded_initializer_stays_reproducible_across_two_instances(self, cls):
        """Exemption 1 + invariant I-2: an explicit seed still reproduces exactly.

        This is what forces the clone to sit AT THE SITE rather than at the
        `initializers.get(...)` line in `__init__`: cloning once there would
        hand every sub-layer the same clone and restore the tie, while cloning
        per site keeps a seeded caller bit-reproducible.

        Compared by weight ORDER rather than by path, because the `_SEWeights`
        naming defect this same commit fixes is what makes paths comparable in
        the first place -- `test_the_se_weights_carry_explicit_names.py` owns
        the path claim, and this control must not silently depend on it.
        """
        snapshots = []
        for _ in range(2):
            layer = cls(kernel_initializer=keras.initializers.GlorotUniform(seed=7))
            layer(np.zeros(_SHAPE, dtype='float32'))
            snapshots.append([_np(w) for w in layer.weights])
        first, second = snapshots
        assert len(first) == len(second)
        for position, (a, b) in enumerate(zip(first, second)):
            assert np.array_equal(a, b), (
                f"{cls.__name__} weight #{position} is not reproducible under an "
                f"explicit seed"
            )

    @pytest.mark.parametrize("cls", _ALL_CLASSES, ids=lambda c: c.__name__)
    def test_cloning_changed_neither_the_config_keys_nor_the_weight_shapes(self, cls):
        """Cloning happens AT THE SITE, never at the `get_config` boundary."""
        layer = cls(kernel_initializer='glorot_uniform')
        layer(np.zeros(_SHAPE, dtype='float32'))
        config = layer.get_config()
        assert 'kernel_initializer' in config

        restored = cls.from_config(config)
        restored(np.zeros(_SHAPE, dtype='float32'))
        assert (
            [tuple(w.shape) for w in layer.weights]
            == [tuple(w.shape) for w in restored.weights]
        )
