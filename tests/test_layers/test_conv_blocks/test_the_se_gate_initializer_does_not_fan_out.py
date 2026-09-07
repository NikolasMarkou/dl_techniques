"""Per-site guards for `SqueezeExcitation`'s internal composite initializer fan-out.

`SqueezeExcitation.__init__` resolves ONE `kernel_initializer` instance and ONE
`bias_initializer` instance (`squeeze_excitation.py`
`self.kernel_initializer = keras.initializers.get(...)`) and hands both of them
to BOTH bottleneck convolutions built in `build()` -- `conv_reduce`
(`C -> C * reduction_ratio`) and `conv_restore` (`C * reduction_ratio -> C`). A
seedless `Initializer` INSTANCE replays the same underlying sample at every
later site, so the DOWN-projection and the UP-projection -- whose entire reason
to exist is that they do opposite things -- start life as the same random
numbers. Shape-matching is NOT the criterion: the two kernels here have
different shapes ((1, 1, C, B) vs (1, 1, B, C)) and were still bit-identical to
the shared instance's replay at their own shape.

MEASURED before the fix (plan-2026-09-07T183458-be1c267e step 7, input
`(2, 8, 8, 8)`): **4 of 4 sites live**, not the 2 the plan text carried --
`conv_reduce`/`conv_restore` kernel AND `conv_reduce`/`conv_restore` bias. The
bias pair is INVISIBLE at the class defaults, because `use_bias` defaults to
`False` and `bias_initializer` defaults to `'zeros'`; at `use_bias=True` with a
caller-supplied random `bias_initializer` both biases came back bit-identical to
the shared replay (`max|bias|` 0.0123 and 0.0878, so the values are real, not a
zero-vs-zero coincidence).

Oracle (plan.md § S-1)
----------------------
Build the layer, then draw from the layer's OWN still-shared
`self.kernel_initializer` / `self.bias_initializer` at that weight's OWN shape
and assert the created weight is not bit-equal to that replay. One assertion per
source site, so a one-line revert reddens exactly that site. A pairwise
comparison between `conv_reduce` and `conv_restore` is deliberately NOT the
guard -- cloning either member decorrelates the pair, so a single-line revert
would stay green (measured in this plan's step 3).

Why this layer is guarded here and not from `tripse_attention.py`
-----------------------------------------------------------------
`SqueezeExcitation` is shared by seven source consumers (`res_path`,
`mobile_one_block`, `reparam_large_kernel_conv`, `yolo12_heads`,
`tripse_attention`, `hanc_block`, `multi_level_feature_compilation`), so its
internal fan-out is a defect with its own owner and its own commit (decision
D-002). `tripse_attention.py`'s four SE-boundary guards deliberately assert on
the INSTANCE crossing the boundary rather than on any weight inside an SE block
(D-010), precisely so that this commit -- which changes what `SqueezeExcitation`
does with the instance it received, never which instance it received -- cannot
look like a regression over there.
`test_the_layer_still_stores_the_caller_s_own_instance` below pins the half of
that contract this file owns.

Scope of the claim, exactly
---------------------------
Copied from the corrected canonical wording in
`src/dl_techniques/initializers/clone.py`'s module docstring: independence holds
for a RANDOM SEEDLESS initializer. Three exemptions, all correct behaviour, none
a defect --

1. a caller-supplied SEEDED instance (e.g. `GlorotUniform(seed=7)`) replays
   deliberately and by contract, ACROSS DIFFERING SHAPES TOO;
2. a DETERMINISTIC initializer (`'zeros'`/`'ones'`/`Constant`, and `Identity`
   only where the weight is 2-D -- it raises on rank 3+) holds no random state,
   so every site is bit-identical and that is what it is meant to do;
3. a CUSTOM initializer whose `get_config()`/`from_config()` round trip raises
   falls back to `copy.deepcopy`, which copies the ALREADY-RESOLVED seed rather
   than drawing a new one, so such a site silently stays tied.

Exemptions 1 and 2 are both asserted below as positive controls, so this module
never states an absolute it has not measured. Unlike every class in
`tripse_attention.py`, this one DOES expose a `bias_initializer` parameter, so
the bias arm is a real guard here rather than a guard that could not fail -- and
the `'zeros'` default gets the exemption-2 control beside it.
"""

import pytest
import numpy as np
import keras

from dl_techniques.layers.conv_blocks.squeeze_excitation import SqueezeExcitation

# ---------------------------------------------------------------------

_SHAPE = (2, 8, 8, 8)
_CONV_NAMES = ('conv_reduce', 'conv_restore')


def _np(x):
    return keras.ops.convert_to_numpy(x)


def _built(**kwargs):
    kwargs.setdefault('kernel_initializer', keras.initializers.GlorotUniform())
    layer = SqueezeExcitation(**kwargs)
    layer(np.zeros(_SHAPE, dtype='float32'))
    return layer


def _assert_is_not_the_shared_replay(shared, weight, label):
    """The per-site oracle, drawn at the weight's OWN shape."""
    replay = _np(shared(tuple(weight.shape), dtype='float32'))
    assert not np.array_equal(replay, _np(weight)), (
        f"{label} {tuple(weight.shape)} is bit-identical to a fresh draw from "
        f"the layer's shared initializer instance -- the site was not cloned"
    )


# ---------------------------------------------------------------------


class TestTheSEKernelInitializerDoesNotFanOut:
    """The two `kernel_initializer=` sites, one guard each.

    These are the sharpest pair in the layer: a bottleneck DOWN-projection and
    the matching UP-projection. Sharing one initializer instance made the
    gate's two halves start as the same numbers even though their shapes are
    transposes of each other.
    """

    @pytest.mark.parametrize("conv_name", _CONV_NAMES)
    def test_the_bottleneck_conv_kernel_is_not_the_shared_replay(self, conv_name):
        layer = _built()
        conv = getattr(layer, conv_name)
        _assert_is_not_the_shared_replay(
            layer.kernel_initializer, conv.kernel, f'{conv_name}/kernel'
        )


class TestTheSEBiasInitializerDoesNotFanOut:
    """The two `bias_initializer=` sites, one guard each.

    Invisible at the defaults (`use_bias=False`), which is why the plan text
    counted 2 sites and the measurement found 4. Driven explicitly here with a
    caller-supplied random `bias_initializer`, which is the configuration in
    which the fan-out is a live defect rather than exemption 2.
    """

    @pytest.mark.parametrize("conv_name", _CONV_NAMES)
    def test_the_bottleneck_conv_bias_is_not_the_shared_replay(self, conv_name):
        layer = _built(
            use_bias=True,
            bias_initializer=keras.initializers.RandomNormal(stddev=0.05),
        )
        conv = getattr(layer, conv_name)
        assert conv.bias is not None
        assert float(np.max(np.abs(_np(conv.bias)))) > 0.0, (
            "the bias arm degenerated to zeros -- it would pass for the wrong reason"
        )
        _assert_is_not_the_shared_replay(
            layer.bias_initializer, conv.bias, f'{conv_name}/bias'
        )


class TestTheSEExemptionsThisModuleDoesNotClaimAway:
    """Positive controls. Cloning must not break any exemption."""

    def test_the_zeros_default_bias_is_identical_at_both_sites_and_that_is_correct(self):
        """Exemption 2 (DETERMINISTIC), asserted as CORRECT rather than as a bug.

        `bias_initializer` defaults to `'zeros'`, which holds no random state,
        so both biases are the same vector of zeros at every site by design.
        Cloning must not change that, and this arm exists so the module never
        implies the identical-bias case is itself the defect.
        """
        layer = _built(use_bias=True)
        for conv_name in _CONV_NAMES:
            bias = getattr(layer, conv_name).bias
            assert float(np.max(np.abs(_np(bias)))) == 0.0
            replay = _np(layer.bias_initializer(tuple(bias.shape), dtype='float32'))
            assert np.array_equal(replay, _np(bias))

    def test_a_seeded_initializer_replays_at_both_sites_by_contract(self):
        """Exemption 1 (SEEDED), including ACROSS THE TWO DIFFERING SHAPES.

        `conv_reduce/kernel` is (1, 1, C, B) and `conv_restore/kernel` is
        (1, 1, B, C). A seeded instance still reproduces its own draw at each
        of those shapes after cloning, because `clone_initializer` carries the
        seed through `get_config()`/`from_config()`.
        """
        seeded = keras.initializers.GlorotUniform(seed=7)
        layer = _built(kernel_initializer=seeded)
        for conv_name in _CONV_NAMES:
            kernel = getattr(layer, conv_name).kernel
            replay = _np(seeded(tuple(kernel.shape), dtype='float32'))
            assert np.array_equal(replay, _np(kernel)), (
                f"{conv_name}/kernel no longer replays under an explicit seed"
            )

    def test_a_seeded_initializer_stays_reproducible_across_two_instances(self):
        """Invariant I-2: an explicit seed still reproduces exactly.

        This is what forces the clone to sit AT THE SITE rather than at the
        `keras.initializers.get(...)` line in `__init__`: cloning once there
        would hand both convolutions the same clone and restore the tie, while
        cloning per site keeps a seeded caller bit-reproducible.
        """
        snapshots = []
        for _ in range(2):
            layer = _built(
                kernel_initializer=keras.initializers.GlorotUniform(seed=7),
                use_bias=True,
                bias_initializer=keras.initializers.RandomNormal(stddev=0.05, seed=11),
            )
            snapshots.append([_np(w) for w in layer.weights])
        first, second = snapshots
        assert len(first) == len(second) == 4
        for position, (a, b) in enumerate(zip(first, second)):
            assert np.array_equal(a, b), (
                f"weight #{position} is not reproducible under an explicit seed"
            )

    def test_the_layer_still_stores_the_caller_s_own_instance(self):
        """The half of the D-010 cross-step contract that this file owns.

        `tripse_attention.py`'s SE-boundary guards assert on the INSTANCE the
        SE block received. They are only immune to this commit while the clone
        stays inside `build()`: the layer must keep storing exactly the object
        the caller handed it.
        """
        kernel_init = keras.initializers.GlorotUniform()
        bias_init = keras.initializers.RandomNormal(stddev=0.05)
        layer = SqueezeExcitation(
            kernel_initializer=kernel_init,
            use_bias=True,
            bias_initializer=bias_init,
        )
        layer(np.zeros(_SHAPE, dtype='float32'))
        assert layer.kernel_initializer is kernel_init
        assert layer.bias_initializer is bias_init

    def test_cloning_changed_neither_the_config_keys_nor_the_weight_shapes(self):
        """Cloning happens AT THE SITE, never at the `get_config` boundary."""
        layer = SqueezeExcitation(
            kernel_initializer='glorot_uniform',
            use_bias=True,
            bias_initializer='random_normal',
        )
        layer(np.zeros(_SHAPE, dtype='float32'))
        config = layer.get_config()
        assert 'kernel_initializer' in config
        assert 'bias_initializer' in config

        restored = SqueezeExcitation.from_config(config)
        restored(np.zeros(_SHAPE, dtype='float32'))
        assert (
            [tuple(w.shape) for w in layer.weights]
            == [tuple(w.shape) for w in restored.weights]
        )
