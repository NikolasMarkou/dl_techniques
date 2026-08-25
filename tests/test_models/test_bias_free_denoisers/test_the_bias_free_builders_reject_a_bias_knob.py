"""Three guards for the four public bias-free builders in `models/vision/bias_free_denoisers/`.

`create_bfunet_denoiser`, `create_bfunet_variant`, `create_bfcnn_denoiser` and
`create_bfcnn_variant` are bias-free *by construction*: `use_bias` is not a declared parameter of
any of them, and `use_bias=False` is hardcoded inside `BiasFreeConv2D`
(`layers/bias_free_conv2d.py:206`, `:557`). Nothing in `tests/` pinned any of that. This file does,
because the safety is currently un-pinned language behaviour rather than an asserted contract: a
future edit that added a forwarded `**kwargs` path to either denoiser would silently make a biased
network reachable from the bias-free entry point, and every existing test would stay green.

Why this matters beyond tidiness: the whole package is named for degree-1 positive homogeneity,
`f(c*x) == c*f(x)`. A single additive constant anywhere in the forward path destroys it, and with
it the Miyasawa residual-as-score reading that
`applications/bias_free_denoiser/denoiser_prior.py` depends on. The sibling `bfconvunext` builder
carried exactly this defect in its live form -- `kwargs.setdefault('use_bias', False)` honoured a
caller override and returned a model with 54 bias tensors out of the bias-free builder.

The three guards:

* **G1** -- `use_bias=True` RAISES on all four entry points, and the message names `use_bias`.
* **G2** -- a built model has EXACTLY ZERO bias tensors, with anti-vacuity controls: (a) on
  all four default builds, and (b) on the non-default configurations that reach the
  hand-written `use_bias=False` sites in `bfunet.py` (the grouped final projection and the
  Gabor stem), which a defaults-only build cannot touch.
* **G3** -- the two `*_variant` functions really are `**kwargs` delegators, structurally AND
  behaviourally. There was no signature-parity test between any variant/denoiser pair anywhere
  in `tests/` before this file.

**Measured, not predicted (2026-08-24).** The plan predicted `TypeError` for G1; that prediction
held for all four functions. One detail it did NOT predict, recorded because a future reader will
otherwise assume the message names the function they called: the two VARIANT functions surface the
*denoiser's* name, because the `TypeError` is raised by the delegated call, not by the variant --
`create_bfcnn_variant(..., use_bias=True)` reports
``create_bfcnn_denoiser() got an unexpected keyword argument 'use_bias'``. Every assertion below
therefore matches on the parameter name only, never on a function name.

Deliberately a NEW single-claim file rather than an append. Measured before writing: neither
`test_bfunet_denoiser.py`, `test_bfcnn_denoiser.py` nor `conftest.py` carries an I-1-style
"frozen at N assertions" invariant -- their docstrings are plain suite descriptions. But such an
invariant DOES exist in this same package, one directory over: `test_bfconvunext_wrappers.py` and
`test_the_bfconvunext_delegation_contract.py` both record that plan invariant I-1 freezes
`test_bfconvunext_denoiser.py` and `test_bfconvunext_gabor.py` at 78 assertions, so new coverage
for the sibling may not be appended to them. A new single-claim guard file is this package's
established shape for exactly this situation either way.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import inspect

import keras
import pytest

from dl_techniques.models.vision.bias_free_denoisers.bfunet import (
    create_bfunet_denoiser,
    create_bfunet_variant,
)
from dl_techniques.models.vision.bias_free_denoisers.bfcnn import (
    create_bfcnn_denoiser,
    create_bfcnn_variant,
)

# ---------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------

INPUT_SHAPE = (16, 16, 1)

#: The four public entry points, each wrapped so it builds a SMALL model and accepts extra
#: keyword arguments to be forwarded verbatim. The wrappers exist only to supply size
#: overrides -- they add no defaulting and swallow nothing.
ENTRY_POINTS = {
    'create_bfunet_denoiser': lambda **kw: create_bfunet_denoiser(
        input_shape=INPUT_SHAPE, depth=2, initial_filters=4, blocks_per_level=1, **kw),
    'create_bfunet_variant': lambda **kw: create_bfunet_variant(
        'tiny', INPUT_SHAPE, initial_filters=4, blocks_per_level=1, **kw),
    'create_bfcnn_denoiser': lambda **kw: create_bfcnn_denoiser(
        input_shape=INPUT_SHAPE, num_blocks=1, filters=4, **kw),
    'create_bfcnn_variant': lambda **kw: create_bfcnn_variant(
        'tiny', INPUT_SHAPE, filters=4, **kw),
}


def _count_bias_tensors(model: keras.Model) -> int:
    """Number of weights whose path names a bias.

    Substring-based on purpose: a bias tensor can be introduced by any sublayer at any depth,
    so an exhaustive layer-type walk would be the thing that goes stale. The cost is that the
    predicate is only as good as its ability to actually SEE a bias -- which is why every test
    that uses it also runs it against a deliberately-biased control.
    """
    return sum(1 for w in model.weights if 'bias' in w.path.lower())


def _biased_control() -> keras.Model:
    """A model that definitely HAS a bias, built on the same input shape."""
    control = keras.Sequential([keras.layers.Conv2D(4, 3, use_bias=True)])
    control.build((None,) + INPUT_SHAPE)
    return control


# ---------------------------------------------------------------------
# G1 -- use_bias=True raises on all four entry points
# ---------------------------------------------------------------------

@pytest.mark.parametrize('entry_point', sorted(ENTRY_POINTS))
def test_use_bias_true_raises(entry_point):
    """G1: no caller can obtain a biased model by asking for one.

    `use_bias` is not a declared parameter of either denoiser, so CPython refuses the call.
    That is a language guarantee today and an UNPINNED one -- adding `**kwargs` to either
    denoiser signature would silently turn this raise into a swallowed keyword.
    """
    with pytest.raises(TypeError) as excinfo:
        ENTRY_POINTS[entry_point](use_bias=True)

    # The parameter name only. The two variants surface the DENOISER's function name here,
    # because the TypeError comes from the delegated call (measured 2026-08-24).
    assert 'use_bias' in str(excinfo.value), (
        f"{entry_point} raised TypeError but its message does not name 'use_bias': "
        f"{excinfo.value!r}")


# ---------------------------------------------------------------------
# G2 -- a default-built model has exactly zero bias tensors
# ---------------------------------------------------------------------

@pytest.mark.parametrize('entry_point', sorted(ENTRY_POINTS))
def test_default_built_model_has_zero_bias_tensors(entry_point):
    """G2: bias-freeness is structural, so it must hold with no arguments asked for.

    Two anti-vacuity controls run in this same test, because the dominant failure mode of a
    "count is zero" assertion is being unable to count at all:

    1. the model must have SOME weights (a weightless model trivially has no bias);
    2. the predicate must return > 0 for a deliberately-biased control on the same shape.

    `test_scaling_invariance_property` in the two sibling suites proves this only INDIRECTLY,
    via `f(c*x) == c*f(x)` after a fit step. This is the direct count.
    """
    model = ENTRY_POINTS[entry_point]()

    assert len(model.weights) > 0, (
        f"{entry_point} built a model with NO weights -- the zero-bias assertion below would "
        f"be vacuous")

    control = _biased_control()
    assert _count_bias_tensors(control) > 0, (
        "the bias predicate cannot see a bias even on a Conv2D(use_bias=True) control; the "
        "zero-bias assertion below would be vacuous")

    offenders = [w.path for w in model.weights if 'bias' in w.path.lower()]
    assert _count_bias_tensors(model) == 0, (
        f"{entry_point} built a model with {len(offenders)} bias tensor(s): {offenders}")


#: Configurations that reach a hardcoded-bias-freeness site the DEFAULT build never touches.
#:
#: Measured 2026-08-24 on `bfunet.py`: at defaults the layer named ``final_output`` is a
#: `BiasFreeConv2D`, whose bias-freeness lives inside the layer. But `_build_final_projection`
#: has a SECOND branch, selected by ``final_projection_groups > 1``, which builds a raw
#: `keras.layers.Conv2D(..., use_bias=False)` -- bias-freeness written out by hand at the call
#: site. `_build_gabor_stem` has two more such hand-written sites (the frozen Gabor bank and its
#: 1x1 projection), reachable only via ``use_gabor_stem=True``. Flipping any of them to
#: ``use_bias=True`` leaves a defaults-only zero-bias guard entirely GREEN.
#:
#: Each entry carries a `probe` layer that EXISTS ONLY on the branch it is meant to reach, so
#: the case cannot silently degrade into another default build (see the reachability control in
#: the test below). `raw_conv=True` additionally pins that the probe is a plain
#: `keras.layers.Conv2D` and NOT a `BiasFreeConv2D` -- i.e. that this really is the
#: hand-written-`use_bias` branch.
#:
#: `bfcnn.py` has no counterpart: it contains no raw `Conv2D` and no `use_bias` literal at all
#: (every conv is a `BiasFreeConv2D`), so there is no non-default branch to cover there.
#: `bfunet.py`'s remaining `use_bias=False` literal, on `DownsampleAndSkip`, is unreachable as a
#: bias: it only binds a weight under ``pool_type='strided_conv'``, which `_validate_bfunet_args`
#: rejects (``downsample_pool_type`` must be ``'max'`` or ``'average'``). A guard case for it
#: would be vacuous by construction.
BRANCH_REACHING_CONFIGS = {
    'bfunet_grouped_final_projection': dict(
        build=lambda: create_bfunet_denoiser(
            input_shape=(16, 16, 2), depth=2, initial_filters=4, blocks_per_level=1,
            final_projection_groups=2),
        probe='final_output',
        raw_conv=True,
    ),
    'bfunet_variant_grouped_final_projection': dict(
        build=lambda: create_bfunet_variant(
            'tiny', (16, 16, 2), initial_filters=4, blocks_per_level=1,
            final_projection_groups=2),
        probe='final_output',
        raw_conv=True,
    ),
    'bfunet_gabor_stem_projection_on': dict(
        build=lambda: create_bfunet_denoiser(
            input_shape=INPUT_SHAPE, depth=2, initial_filters=4, blocks_per_level=1,
            use_gabor_stem=True, gabor_filters=4, gabor_stem_projection=True),
        probe='gabor_stem_projection',
        raw_conv=True,
    ),
    'bfunet_gabor_stem_projection_off': dict(
        build=lambda: create_bfunet_denoiser(
            input_shape=INPUT_SHAPE, depth=2, initial_filters=4, blocks_per_level=1,
            use_gabor_stem=True, gabor_filters=4, gabor_stem_projection=False),
        probe='gabor_stem',
        raw_conv=False,
    ),
}


@pytest.mark.parametrize('config_id', sorted(BRANCH_REACHING_CONFIGS))
def test_non_default_branches_have_zero_bias_tensors(config_id):
    """G2(b): bias-freeness also holds on the branches a default build never selects.

    The test above builds every entry point with no knobs, which is exactly why it cannot see
    the hand-written `use_bias=False` sites listed in `BRANCH_REACHING_CONFIGS` -- all of them
    sit behind a non-default parameter value. Measured 2026-08-24: flipping
    `_build_final_projection`'s grouped-branch `use_bias=False` to `True` leaves all twelve
    guards in this file green; it turns THIS test red on
    `[bfunet_grouped_final_projection]` and `[bfunet_variant_grouped_final_projection]` while
    the defaults-only cases stay green.

    Three anti-vacuity controls run here:

    1. REACHABILITY -- the built model must contain the probe layer, which exists only on the
       branch under test, and (where `raw_conv`) that layer must be a plain
       `keras.layers.Conv2D` rather than a `BiasFreeConv2D`. Without this, a config that
       stopped selecting the branch would keep passing forever;
    2. the model must have SOME weights;
    3. the bias predicate must return > 0 on a deliberately-biased control.
    """
    spec = BRANCH_REACHING_CONFIGS[config_id]
    model = spec['build']()

    layer_names = [layer.name for layer in model.layers]
    assert spec['probe'] in layer_names, (
        f"{config_id} did not reach its branch: no layer named {spec['probe']!r} in "
        f"{layer_names}. The configuration no longer selects the branch this case exists to "
        f"cover, so the zero-bias assertion below would be a duplicate of the defaults test.")

    if spec['raw_conv']:
        probe_layer = model.get_layer(spec['probe'])
        assert type(probe_layer) is keras.layers.Conv2D, (
            f"{config_id}: layer {spec['probe']!r} is a {type(probe_layer).__name__}, not a "
            f"plain keras.layers.Conv2D. This case exists to cover the branch whose "
            f"bias-freeness is written at the CALL SITE; a BiasFreeConv2D here means the "
            f"branch moved and this case is now redundant with the defaults test.")

    assert len(model.weights) > 0, (
        f"{config_id} built a model with NO weights -- the zero-bias assertion below would be "
        f"vacuous")

    control = _biased_control()
    assert _count_bias_tensors(control) > 0, (
        "the bias predicate cannot see a bias even on a Conv2D(use_bias=True) control; the "
        "zero-bias assertion below would be vacuous")

    offenders = [w.path for w in model.weights if 'bias' in w.path.lower()]
    assert _count_bias_tensors(model) == 0, (
        f"{config_id} built a model with {len(offenders)} bias tensor(s): {offenders}")


# ---------------------------------------------------------------------
# G3 -- variant -> denoiser forwarding parity
# ---------------------------------------------------------------------

#: The variant functions' own plumbing parameters. Everything else MUST arrive through
#: `**kwargs`, so that the variant can never drift out of sync with the denoiser's parameter
#: list the way `bfconvunext` did (a hand-copied 38-of-42 signature).
VARIANT_EXPLICIT_PARAMS = {
    create_bfunet_variant: [
        'variant', 'input_shape', 'enable_deep_supervision', 'pretrained',
        'weights_dataset', 'weights_input_shape', 'cache_dir',
    ],
    create_bfcnn_variant: ['variant', 'input_shape'],
}


@pytest.mark.parametrize('variant_fn', list(VARIANT_EXPLICIT_PARAMS),
                         ids=lambda f: f.__name__)
def test_variant_is_structurally_a_kwargs_delegator(variant_fn):
    """G3(a) STRUCTURAL: the variant does not hand-copy the denoiser's parameter list.

    Pinned now, while it is still true. This is the guard against reintroducing the
    `bfconvunext` defect SHAPE. It is deliberately paired with the behavioural test below,
    because a signature assertion alone is satisfied by a delegator that silently drops every
    keyword it is handed.
    """
    params = inspect.signature(variant_fn).parameters

    var_keyword = [p.name for p in params.values()
                   if p.kind is inspect.Parameter.VAR_KEYWORD]
    assert len(var_keyword) == 1, (
        f"{variant_fn.__name__} has no **kwargs parameter -- it must forward, not hand-copy, "
        f"the denoiser's parameter list. Parameters: {list(params)}")

    explicit = [p.name for p in params.values()
                if p.kind is not inspect.Parameter.VAR_KEYWORD]
    assert explicit == VARIANT_EXPLICIT_PARAMS[variant_fn], (
        f"{variant_fn.__name__}'s explicit parameters drifted: {explicit} != "
        f"{VARIANT_EXPLICIT_PARAMS[variant_fn]}. Anything beyond the variant's own plumbing "
        f"belongs in **kwargs.")


@pytest.mark.parametrize('variant_fn, width_kwarg', [
    (create_bfcnn_variant, 'filters'),
    (create_bfunet_variant, 'initial_filters'),
], ids=lambda x: getattr(x, '__name__', x))
def test_variant_override_reaches_the_built_graph(variant_fn, width_kwarg):
    """G3(b) BEHAVIOURAL: an override passed to the variant actually reaches the graph.

    A `**kwargs` signature proves only that the keyword is ACCEPTED. This proves it is USED:
    an odd, non-default width of 7 must show up as a 7-wide channel axis in the built weights.
    Odd on purpose -- no variant preset and no default uses it, so a passing assertion cannot
    be a coincidence of the preset.
    """
    kwargs = {width_kwarg: 7}
    if variant_fn is create_bfunet_variant:
        kwargs['blocks_per_level'] = 1

    model = variant_fn('tiny', INPUT_SHAPE, **kwargs)

    widths = {tuple(w.shape)[-1] for w in model.weights}
    assert 7 in widths, (
        f"{variant_fn.__name__}('tiny', ..., {width_kwarg}=7) built a graph with NO 7-wide "
        f"channel axis (observed trailing widths: {sorted(widths)}). The override was accepted "
        f"and then dropped.")
