"""Delegation-contract tests for `models/bias_free_denoisers/bfconvunext.py`.

`create_convunext_denoiser` is a THIN WRAPPER over
`models/convunext/model.create_convunext`. What this file pins is the *delegation*
itself -- the half that had no coverage at all before it existed:

(A) An unknown keyword is LOUD at both wrappers. `create_convunext` declares 42
    explicit parameters and NO `**kwargs`, so a typo'd keyword can never be
    swallowed. That language guarantee is what replaces the old `dict(locals())`
    forwarding idiom (`# DECISION plan-2026-08-14T092357-0e3d792d/D-011`), and this
    is the test that supersede note names. If it ever goes green-by-silence, a
    misspelled parameter starts being ignored at every call site in the repo.
(B) The wrapper's pinned keywords cannot be overridden. `use_bias` is the one that
    matters: a silent override would build a BIASED network out of the bias-free
    entry point, destroying the degree-1 homogeneity the whole package is named for
    (and with it the Miyasawa residual-as-score reading that
    `applications/bias_free_denoiser/denoiser_prior.py` depends on).
(C) `include_top=False` reaches the bias-free path AND changes the output contract.
(D) `output_channels=N` reaches the bias-free path AND sets the output width.
(E) Every kwarg NAME SET a live caller actually passes still binds. The sets are not
    guessed: they were harvested at RUNTIME by wrapping both factories while the four
    relevant suites ran (111 calls, 38 distinct denoiser sets). A static harvest could
    not see inside the seven `**splat` helpers the callers use.
(F) A bias-free ConvUNext survives a real `.keras` save-to-disk +
    `keras.models.load_model` round-trip with IDENTICAL weight values, and both
    registry keys a saved graph names still resolve. No historical checkpoint exists
    in this repo (findings O-11), so this round-trip is the only instrument that can
    say anything about checkpoint compatibility at all.

Deliberately a NEW file: plan invariant I-1 freezes `test_bfconvunext_denoiser.py`
and `test_bfconvunext_gabor.py` at 78 assertions, and the registry-key EXACT-match
pin lives in `test_bfconvunext_wrappers.py` beside the fresh-subprocess fixture it
needs (decisions.md D-011).

Message-matching policy, measured rather than assumed: CPython produces the two
TypeErrors through different call-machinery paths and formats them differently --
`create_convunext() got an unexpected keyword argument 'dpeth'` (BARE name) versus
`dl_techniques.models.convunext.model.create_convunext() got multiple values for
keyword argument 'use_bias'` (fully DOTTED name). Every assertion below therefore
matches a SUBSTRING plus the offending parameter name. Pinning a function-name
prefix or a full message would pin a CPython implementation detail.
"""

import inspect
from typing import Any, Dict, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.models.bias_free_denoisers.bfconvunext import (
    create_convunext_denoiser,
    create_convunext_variant as bf_create_convunext_variant,
)
from dl_techniques.models.convunext.model import create_convunext

# Small everywhere: these tests measure WIRING, not capacity.
INPUT_SHAPE: Tuple[int, int, int] = (32, 32, 1)
SMALL: Dict[str, Any] = dict(depth=2, initial_filters=8, blocks_per_level=1)

# The two keys any saved bias-free ConvUNext graph names. Duplicated here as
# LITERALS on purpose -- the exact-match registry pin lives in
# `test_bfconvunext_wrappers.py`; here they are used only to give the round-trip its
# checkpoint meaning (a same-process save/load would happily resolve a MOVED key and
# prove nothing about a graph saved yesterday).
STEM_REGISTRY_KEY = 'dl_techniques.bias_free_denoisers>ConvUNextStem'
ATTENTION_REGISTRY_KEY = 'Custom>SpatialLinearAttention'

# Substring alternation covering BOTH TypeError shapes. The contract is "it raises",
# not "it raises this particular way": against the pre-rewrite wrapper `use_bias` is
# simply an UNKNOWN keyword (the hand-copied signature omits it), while against the
# `**kwargs` delegator it is a DUPLICATED one. Both are TypeError; only the wording
# differs, and no caller depends on the wording.
RAISES_EITHER_WAY = r"unexpected keyword argument|got multiple values"

# One legal, cheap value per parameter name observed at runtime (31 names). Used to
# turn a harvested NAME SET into a callable kwarg dict.
VALUES: Dict[str, Any] = {
    'input_shape': INPUT_SHAPE,
    'depth': 2,
    'initial_filters': 8,
    'blocks_per_level': 1,
    'filter_multiplier': 2.0,
    'convnext_version': 'v2',
    'block_activation': 'gelu',
    'block_normalization': 'batchnorm',
    'stem_activation': 'gelu',
    'drop_path_rate': 0.0,
    'dropout_rate': 0.0,
    'final_activation': 'linear',
    'enable_deep_supervision': False,
    'expose_bottleneck': False,
    'extra_zero_output_channels': False,
    'zero_pad_channels': False,
    'use_gabor_stem': False,
    'gabor_filters': 8,
    'gabor_kernel_size': 5,
    'gabor_activation': None,
    'gabor_stem_projection': True,
    'use_laplacian_pyramid': False,
    'high_freq_blocks': 0,
    'bottleneck_attention_blocks': 0,
    'bottleneck_attention_heads': 2,
    'downsample_pool_type': 'max',
    'final_projection_groups': 1,
    'depthwise_initializer': None,
    'depthwise_regularizer': None,
    'supervision_activation': 'gelu',
    'model_name': 'delegation_probe',
}

# The 38 DISTINCT kwarg name sets recorded at runtime across 105 real calls to
# `create_convunext_denoiser` made by `src/train/bfunet/train_convunext_denoiser.py`,
# `src/dl_techniques/utils/multiplicative_miyasawa.py` and the six test modules that
# call it. Order is sorted-within-set and sets are sorted, so a diff of this block is
# readable. NOT hand-copied from prose: generated from the recorder's JSON.
CALLER_KWARG_SETS = [
    ('block_activation', 'block_normalization', 'blocks_per_level', 'bottleneck_attention_blocks', 'bottleneck_attention_heads', 'convnext_version', 'depth', 'depthwise_initializer', 'depthwise_regularizer', 'downsample_pool_type', 'drop_path_rate', 'dropout_rate', 'enable_deep_supervision', 'expose_bottleneck', 'extra_zero_output_channels', 'filter_multiplier', 'final_activation', 'final_projection_groups', 'gabor_activation', 'gabor_filters', 'gabor_kernel_size', 'gabor_stem_projection', 'high_freq_blocks', 'initial_filters', 'input_shape', 'model_name', 'stem_activation', 'supervision_activation', 'use_gabor_stem', 'use_laplacian_pyramid', 'zero_pad_channels'),
    ('block_activation', 'block_normalization', 'blocks_per_level', 'convnext_version', 'depth', 'drop_path_rate', 'filter_multiplier', 'final_activation', 'initial_filters', 'input_shape', 'use_gabor_stem'),
    ('block_activation', 'block_normalization', 'blocks_per_level', 'depth', 'drop_path_rate', 'initial_filters', 'input_shape', 'stem_activation'),
    ('block_activation', 'blocks_per_level', 'convnext_version', 'depth', 'drop_path_rate', 'extra_zero_output_channels', 'filter_multiplier', 'final_activation', 'initial_filters', 'input_shape', 'stem_activation', 'use_gabor_stem'),
    ('block_activation', 'blocks_per_level', 'convnext_version', 'depth', 'drop_path_rate', 'filter_multiplier', 'final_activation', 'initial_filters', 'input_shape', 'stem_activation', 'use_gabor_stem', 'zero_pad_channels'),
    ('block_activation', 'blocks_per_level', 'convnext_version', 'depth', 'drop_path_rate', 'filter_multiplier', 'final_activation', 'initial_filters', 'input_shape', 'use_gabor_stem'),
    ('block_activation', 'blocks_per_level', 'depth', 'drop_path_rate', 'initial_filters', 'input_shape', 'stem_activation'),
    ('block_activation', 'blocks_per_level', 'depth', 'enable_deep_supervision', 'initial_filters', 'input_shape', 'stem_activation', 'supervision_activation', 'use_gabor_stem'),
    ('block_activation', 'blocks_per_level', 'depth', 'initial_filters', 'input_shape'),
    ('block_normalization', 'blocks_per_level', 'convnext_version', 'depth', 'drop_path_rate', 'filter_multiplier', 'final_activation', 'initial_filters', 'input_shape', 'use_gabor_stem'),
    ('block_normalization', 'blocks_per_level', 'depth', 'initial_filters', 'input_shape'),
    ('blocks_per_level', 'convnext_version', 'depth', 'depthwise_initializer', 'depthwise_regularizer', 'drop_path_rate', 'initial_filters', 'input_shape'),
    ('blocks_per_level', 'convnext_version', 'depth', 'drop_path_rate', 'dropout_rate', 'initial_filters', 'input_shape'),
    ('blocks_per_level', 'convnext_version', 'depth', 'drop_path_rate', 'enable_deep_supervision', 'extra_zero_output_channels', 'filter_multiplier', 'final_activation', 'initial_filters', 'input_shape'),
    ('blocks_per_level', 'convnext_version', 'depth', 'drop_path_rate', 'enable_deep_supervision', 'initial_filters', 'input_shape'),
    ('blocks_per_level', 'convnext_version', 'depth', 'drop_path_rate', 'extra_zero_output_channels', 'filter_multiplier', 'final_activation', 'initial_filters', 'input_shape'),
    ('blocks_per_level', 'convnext_version', 'depth', 'drop_path_rate', 'extra_zero_output_channels', 'filter_multiplier', 'final_activation', 'initial_filters', 'input_shape', 'zero_pad_channels'),
    ('blocks_per_level', 'convnext_version', 'depth', 'drop_path_rate', 'filter_multiplier', 'final_activation', 'initial_filters', 'input_shape'),
    ('blocks_per_level', 'convnext_version', 'depth', 'drop_path_rate', 'filter_multiplier', 'final_activation', 'initial_filters', 'input_shape', 'use_gabor_stem'),
    ('blocks_per_level', 'convnext_version', 'depth', 'drop_path_rate', 'filter_multiplier', 'final_activation', 'initial_filters', 'input_shape', 'zero_pad_channels'),
    ('blocks_per_level', 'convnext_version', 'depth', 'drop_path_rate', 'initial_filters', 'input_shape'),
    ('blocks_per_level', 'convnext_version', 'depth', 'extra_zero_output_channels', 'final_projection_groups', 'initial_filters', 'input_shape'),
    ('blocks_per_level', 'convnext_version', 'depth', 'final_projection_groups', 'initial_filters', 'input_shape'),
    ('blocks_per_level', 'convnext_version', 'depth', 'gabor_filters', 'gabor_kernel_size', 'gabor_stem_projection', 'initial_filters', 'input_shape', 'use_gabor_stem'),
    ('blocks_per_level', 'convnext_version', 'depth', 'gabor_filters', 'gabor_kernel_size', 'initial_filters', 'input_shape', 'use_gabor_stem'),
    ('blocks_per_level', 'convnext_version', 'depth', 'initial_filters', 'input_shape'),
    ('blocks_per_level', 'depth', 'drop_path_rate', 'initial_filters', 'input_shape'),
    ('blocks_per_level', 'depth', 'enable_deep_supervision', 'initial_filters', 'input_shape'),
    ('blocks_per_level', 'depth', 'enable_deep_supervision', 'initial_filters', 'input_shape', 'supervision_activation'),
    ('blocks_per_level', 'depth', 'initial_filters', 'input_shape'),
    ('blocks_per_level', 'depth', 'initial_filters', 'input_shape', 'stem_activation', 'use_gabor_stem'),
    ('blocks_per_level', 'depth', 'initial_filters', 'input_shape', 'use_gabor_stem'),
    ('blocks_per_level', 'input_shape'),
    ('convnext_version', 'input_shape'),
    ('depth', 'input_shape'),
    ('filter_multiplier', 'input_shape'),
    ('initial_filters', 'input_shape'),
    ('input_shape',),]

# The three sets that are actually BUILT (E2). The other 35 are bind-checked only
# (E1): building all 38 real Keras graphs costs ~20 minutes for no extra information
# -- every one of them differs from these three only in the VALUE of a parameter
# `create_convunext` already declares, and the delegation contract is about NAMES.
# What building buys, and binding cannot, is proof that the pinned `use_bias=False` /
# `stem_normalization=...` arguments do not collide with a real caller's set at
# call time. Three sets exercise that at the extremes: the widest set any caller
# passes, a mid-size production set, and the bare minimum.
_LARGEST = max(CALLER_KWARG_SETS, key=len)
BUILD_SUBSET = [
    _LARGEST,
    ('blocks_per_level', 'depth', 'drop_path_rate', 'initial_filters', 'input_shape'),
    ('input_shape',),
]


def _kwargs_for(names: Tuple[str, ...]) -> Dict[str, Any]:
    """Turn a harvested NAME SET into a callable kwarg dict of small legal values."""
    missing = [n for n in names if n not in VALUES]
    assert not missing, (
        f"harvested caller kwarg set names {missing} have no value in VALUES; the "
        f"recorder saw them at runtime, so this table is stale"
    )
    return {n: VALUES[n] for n in names}


def _weight_values(model: keras.Model) -> list:
    """Snapshot every weight VALUE, in `model.weights` order.

    Ordinal, not by-name: a rename that shuffles the list is itself a finding. Read
    eagerly so the caller can sample BEFORE the model's first call -- calling a model
    can update BatchNorm moving statistics and mask a real difference.
    """
    return [np.asarray(keras.ops.convert_to_numpy(w)) for w in model.weights]


class TestUnknownKwargIsLoud:
    """A misspelled keyword must raise TypeError at BOTH wrappers, naming the typo.

    This is the contract that REPLACES the `dict(locals())` forwarding idiom: the
    forward was written to guarantee that a parameter could never be silently
    dropped, and `create_convunext`'s lack of `**kwargs` guarantees the same thing
    at the language level, for the signature as well as the forward. If either of
    these ever stops raising, a typo'd parameter starts being silently ignored --
    the exact failure the anchored decision exists to prevent -- and the caller gets
    a model built with a default it never asked for.
    """

    def test_unknown_kwarg_raises_at_the_denoiser_wrapper(self) -> None:
        with pytest.raises(TypeError) as excinfo:
            create_convunext_denoiser(
                input_shape=INPUT_SHAPE, dpeth=4, **SMALL
            )
        assert 'unexpected keyword argument' in str(excinfo.value)
        assert 'dpeth' in str(excinfo.value), (
            f"the TypeError must NAME the offending keyword; got: {excinfo.value}"
        )

    def test_unknown_kwarg_raises_at_the_variant_wrapper(self) -> None:
        """The variant path merges `**kwargs` into a config dict before delegating.

        Measured (plan assumption 2, closed by execution): that merge does NOT
        swallow an unknown key -- it surfaces at `create_convunext()` with the same
        message. Pinned here because "an intermediate `**config` merge eats the
        typo" is the plausible way this contract could hold at one wrapper and fail
        at the other.
        """
        with pytest.raises(TypeError) as excinfo:
            bf_create_convunext_variant('tiny', INPUT_SHAPE, dpeth=4)
        assert 'unexpected keyword argument' in str(excinfo.value)
        assert 'dpeth' in str(excinfo.value)


class TestPinnedKwargsCannotBeOverridden:
    """The wrapper's own pinned arguments must not be silently overridable.

    `use_bias=False` and `stem_normalization='global_response_norm'` are what make
    this a BIAS-FREE builder. A silent override of `use_bias` would return a biased
    network from the bias-free entry point while every existing test still passed,
    breaking degree-1 homogeneity f(ax) = a*f(x) and with it the residual-as-score
    reading the denoiser-prior application is built on.

    The assertion accepts EITHER TypeError shape on purpose. Against the pre-rewrite
    hand-copied signature these names are simply absent, so Python says "unexpected
    keyword argument"; against the `**kwargs` delegator they collide with an
    explicitly-passed argument, so Python says "got multiple values". The contract
    being pinned is "it raises, and the message names the parameter" -- not which of
    the two CPython call paths produced it.
    """

    @pytest.mark.parametrize(
        "name,value",
        [
            ("use_bias", True),
            ("stem_normalization", "layernorm"),
        ],
    )
    def test_pinned_kwarg_raises(self, name: str, value: Any) -> None:
        with pytest.raises(TypeError, match=RAISES_EITHER_WAY) as excinfo:
            create_convunext_denoiser(
                input_shape=INPUT_SHAPE, **{name: value}, **SMALL
            )
        assert name in str(excinfo.value), (
            f"the TypeError must NAME the pinned parameter {name!r}; "
            f"got: {excinfo.value}"
        )

    def test_input_shape_cannot_be_supplied_twice(self) -> None:
        """`input_shape` is a NAMED parameter, never an entry in `**kwargs`.

        Passing it positionally and again by keyword must raise rather than let one
        of the two win. This is what keeps the built graph's input shape equal to
        the shape the caller actually asked for; the companion assertion below
        checks that equality directly, so the test cannot pass by raising alone.
        """
        with pytest.raises(TypeError, match=RAISES_EITHER_WAY) as excinfo:
            create_convunext_denoiser(INPUT_SHAPE, input_shape=(8, 8, 1), **SMALL)
        assert 'input_shape' in str(excinfo.value)

        model = create_convunext_denoiser(input_shape=INPUT_SHAPE, **SMALL)
        assert model.input_shape[1:] == INPUT_SHAPE


class TestIncludeTopIsReachable:
    """`include_top=False` must reach the bias-free path AND change the model.

    Before the rewrite this parameter is simply ABSENT from the wrapper's
    hand-copied 38-parameter signature, so a bias-free ConvUNext feature extractor
    cannot be built through the bias-free entry point at all -- the defect the
    rewrite exists to remove. Asserting reachability alone would be satisfied by a
    wrapper that accepted the argument and dropped it, so the assertion is on the
    DIFFERENCE between the two models: `include_top=False` returns the decoder
    feature map (initial_filters channels) instead of the projected output, and
    owns a strictly smaller weight list.
    """

    def test_include_top_false_builds_and_changes_the_output_contract(self) -> None:
        with_top = create_convunext_denoiser(
            input_shape=INPUT_SHAPE, include_top=True, **SMALL
        )
        without_top = create_convunext_denoiser(
            input_shape=INPUT_SHAPE, include_top=False, **SMALL
        )

        assert with_top.output_shape != without_top.output_shape, (
            f"include_top was accepted but had NO effect: both models output "
            f"{with_top.output_shape}"
        )
        assert with_top.output_shape[-1] == INPUT_SHAPE[-1]
        assert without_top.output_shape[-1] == SMALL['initial_filters']
        assert len(without_top.weights) < len(with_top.weights), (
            "include_top=False must yield a STRICTLY smaller weight list "
            f"(got {len(without_top.weights)} vs {len(with_top.weights)})"
        )


class TestOutputChannelsIsReachable:
    """`output_channels=N` must reach the bias-free path AND set the output width.

    The second of the two parameters absent from the pre-rewrite signature. Two
    values, neither equal to the input's channel count, so the test cannot pass by
    the parameter being ignored and the width defaulting to `input_shape[-1]`.
    """

    @pytest.mark.parametrize("n", [2, 3])
    def test_output_channels_sets_the_output_width(self, n: int) -> None:
        model = create_convunext_denoiser(
            input_shape=INPUT_SHAPE, output_channels=n, **SMALL
        )
        assert model.output_shape[-1] == n, (
            f"output_channels={n} was accepted but the model outputs "
            f"{model.output_shape[-1]} channels (input has {INPUT_SHAPE[-1]}); the "
            f"argument is being dropped"
        )


class TestEveryLiveCallerKwargSetStillBinds:
    """Every kwarg NAME SET a live caller passes must still reach `create_convunext`.

    Under a `**kwargs` delegation a caller's keyword binds if and only if
    `create_convunext` declares it. That is exactly the set of names the wrapper used
    to hand-copy, so the rewrite is supposed to be invisible at every call site -- and
    this is the test that says so rather than assuming it. The sets are the 38
    DISTINCT ones recorded at RUNTIME across 105 real calls; a static harvest could
    not resolve the seven `**splat` helpers the callers use.

    Split deliberately: all 38 are BIND-checked (E1, microseconds each), three are
    actually BUILT (E2). See the comment on BUILD_SUBSET for why building all 38 buys
    no extra information.
    """

    def test_build_subset_is_drawn_from_the_harvested_sets(self) -> None:
        """A guard on the guard: BUILD_SUBSET must not drift off CALLER_KWARG_SETS."""
        known = {tuple(s) for s in CALLER_KWARG_SETS}
        for s in BUILD_SUBSET:
            assert tuple(s) in known, (
                f"{s} is not one of the harvested caller kwarg sets; this test would "
                f"be building a call nobody makes"
            )

    @pytest.mark.parametrize("names", CALLER_KWARG_SETS, ids=lambda s: '+'.join(s))
    def test_caller_signature_binds(self, names: Tuple[str, ...]) -> None:
        """Bind at BOTH ends: the wrapper accepts it, and the callee declares it.

        The wrapper-side bind alone is nearly vacuous under `**kwargs` (everything
        binds). The callee-side bind -- against `create_convunext` with the wrapper's
        pinned arguments ALREADY supplied -- is the one with teeth: it fails on an
        unknown name AND on a collision with a pin.
        """
        kwargs = _kwargs_for(names)

        inspect.signature(create_convunext_denoiser).bind(**kwargs)

        delegated = {k: v for k, v in kwargs.items() if k != 'input_shape'}
        inspect.signature(create_convunext).bind(
            input_shape=INPUT_SHAPE,
            use_bias=False,
            stem_normalization='global_response_norm',
            **delegated,
        )

    @pytest.mark.parametrize("names", BUILD_SUBSET, ids=lambda s: '+'.join(s))
    def test_caller_signature_builds(self, names: Tuple[str, ...]) -> None:
        model = create_convunext_denoiser(**_kwargs_for(names))
        assert isinstance(model, keras.Model)
        offenders = [
            layer.name
            for layer in model._flatten_layers()
            if getattr(layer, "use_bias", False)
        ]
        assert not offenders, (
            f"the bias-free pin did not survive this caller's kwarg set: {offenders}"
        )


class TestKerasRoundTrip:
    """A saved bias-free ConvUNext must reload with IDENTICAL weights.

    No historical `.keras` checkpoint exists anywhere in this repo (findings O-11),
    so a same-process save/load is the only instrument available -- and on its own it
    is a weak one: it would happily resolve a MOVED registry key and prove nothing
    about a graph saved before the move. The registry-key assertions are what give
    this test its checkpoint meaning; the weight comparison is what gives it its
    serialization meaning. Both are needed.

    Weights are compared ORDINALLY at atol=0.0 (exact equality) and sampled BEFORE
    the loaded model's first call: calling it can update BatchNorm moving statistics,
    which would mask a real difference. A weight COUNT match is not accepted as
    evidence -- a nested sublayer list has been measured in this repo to restore
    FRESH kernels while count, paths and parameter total all matched.
    """

    def test_saved_model_reloads_with_identical_weights(self, tmp_path) -> None:
        model = create_convunext_denoiser(
            input_shape=INPUT_SHAPE, block_normalization='batchnorm', **SMALL
        )
        x = np.asarray(
            keras.random.normal((2,) + INPUT_SHAPE, seed=1234), dtype='float32'
        )
        before_out = np.asarray(
            keras.ops.convert_to_numpy(model(x, training=False))
        )
        before = _weight_values(model)
        assert len(before) > 0, (
            "non-vacuity guard: the subject model owns no weights, so an exact "
            "weight comparison would pass on an empty list"
        )

        path = tmp_path / "m.keras"
        model.save(path)
        loaded = keras.models.load_model(path)

        # Sampled BEFORE `loaded` is ever called.
        after = _weight_values(loaded)

        assert len(after) == len(before), (
            f"weight COUNT changed across the round trip: {len(before)} -> "
            f"{len(after)}"
        )
        for i, (w_before, w_after) in enumerate(zip(before, after)):
            assert w_before.shape == w_after.shape, (
                f"weight {i} ({model.weights[i].path}) changed shape: "
                f"{w_before.shape} -> {w_after.shape}"
            )
            np.testing.assert_allclose(
                w_after,
                w_before,
                rtol=0.0,
                atol=0.0,
                err_msg=(
                    f"weight {i} ({model.weights[i].path}) did not survive the "
                    f".keras round trip byte-for-byte"
                ),
            )

        after_out = np.asarray(
            keras.ops.convert_to_numpy(loaded(x, training=False))
        )
        np.testing.assert_allclose(after_out, before_out, rtol=1e-6, atol=1e-6)

    def test_both_registry_keys_a_saved_graph_names_still_resolve(self) -> None:
        """The keys `load_model` looks up, asserted EXACTLY.

        `ConvUNextStem` carries `package='dl_techniques.bias_free_denoisers'`;
        `SpatialLinearAttention` carries a BARE decorator whose key was measured to
        be module-independent. Adding a `package=` argument to the bare one "for
        symmetry" would mint a BRAND-NEW key -- the model would still build, still
        save, and every previously saved graph would fail at `load_model`, silently,
        until someone loaded one. The fresh-subprocess version of this pin lives in
        `test_bfconvunext_wrappers.py::TestRegistrarContract`; this in-process copy
        is what makes the round-trip above mean something about OLD graphs.
        """
        registered = keras.saving.get_custom_objects()
        assert STEM_REGISTRY_KEY in registered, (
            f"{STEM_REGISTRY_KEY!r} is not registered; every saved bias-free "
            f"ConvUNext graph names it"
        )
        assert ATTENTION_REGISTRY_KEY in registered, (
            f"{ATTENTION_REGISTRY_KEY!r} is not registered; a `package=` argument "
            f"on its bare decorator would change the key and break every saved graph"
        )
