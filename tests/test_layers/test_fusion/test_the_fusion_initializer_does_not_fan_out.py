"""Per-site guards for `MultiModalFusion`'s composite initializer fan-out.

`MultiModalFusion.__init__` resolves ONE `kernel_initializer` instance and ONE
`bias_initializer` instance (`keras.initializers.get(...)`, `multimodal_fusion.py:395-396`)
and hands those same two instances to every child `keras.layers.Dense` created
by whichever `_build_*` strategy builder runs. A seedless `Initializer`
INSTANCE replays the same underlying sample at every later site, so the child
kernels start life as the same random numbers.

MEASURED before the fix (plan-2026-09-07T183458-be1c267e step 5, `dim=8`,
`num_tensor_projections=2`, over all 8 strategies at 2 and 3 modalities): all
8 source sites and both of their weights alias -- 48 aliased weight tensors
observed across the 15 reachable (strategy, num_modalities) arms, every one
bit-identical to a replay from the still-shared instance.

The sharpest pair is `tensor_proj_0` vs `tensor_proj_1`: they share the kernel
AND the input (both read the same concatenation), so gradient descent receives
identical gradients for them and never separates them. See
`TestTheSharpestFusionPair`.

Oracle (plan.md § S-1)
----------------------
Build the layer, then draw from the layer's OWN still-shared
`self.kernel_initializer` / `self.bias_initializer` at that weight's OWN shape
and assert the created weight is not bit-equal to that replay. One assertion
per source site, so a one-line revert reddens exactly one test. A pairwise
comparison between two sites is deliberately NOT the guard -- cloning either
member of a pair decorrelates it, so a single-line revert would stay green
(re-measured in this plan's step 3).

The branch trap this module is built around
-------------------------------------------
A single `MultiModalFusion` instance fires at most ONE of the eight
mutually-exclusive `_build_*` branches, so a single-config test covers at most
3 of the 8 sites. Two further branch facts (both MEASURED, both easy to get
wrong): `align_projection_{i}` is structurally UNREACHABLE at 2 modalities
(`_build_elementwise` only creates it above 2), and `bilinear` RAISES at 3
modalities. `TestEveryFusionStrategyBranchActuallyFires` pins the exact site
inventory of every reachable arm, so a guard that silently stopped executing
its branch fails loudly instead of passing vacuously.

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

Exemptions 1 and 2 are asserted below as positive controls, so this module
never states an absolute it has not measured. `bias_initializer` defaults to
`'zeros'`, which is exemption 2: identical biases at every site under the
default are CORRECT, and the live bias defect only appears under a
caller-supplied RANDOM bias initializer, which is what the bias guards pass.
"""

import pytest
import numpy as np
import keras

from dl_techniques.layers.fusion.multimodal_fusion import MultiModalFusion

# ---------------------------------------------------------------------

_DIM = 8
_SEQ = 4
_BATCH = 2
_NUM_TENSOR_PROJECTIONS = 2

_STRATEGIES = (
    'cross_attention',
    'concatenation',
    'addition',
    'multiplication',
    'gated',
    'attention_pooling',
    'bilinear',
    'tensor_fusion',
)

# MEASURED site inventory: the exact set of Dense sub-layers that receive the
# shared initializer instances, per (strategy, num_modalities). `bilinear` at 3
# modalities is absent because it RAISES; that arm is pinned separately.
_EXPECTED_SITES = {
    ('cross_attention', 2): [],
    ('cross_attention', 3): [],
    ('concatenation', 2): ['concat_projection'],
    ('concatenation', 3): ['concat_projection'],
    ('addition', 2): [],
    ('addition', 3): ['align_projection_0', 'align_projection_1', 'align_projection_2'],
    ('multiplication', 2): [],
    ('multiplication', 3): ['align_projection_0', 'align_projection_1', 'align_projection_2'],
    ('gated', 2): ['gate_0', 'gate_1', 'gated_projection'],
    ('gated', 3): ['gate_0', 'gate_1', 'gate_2', 'gated_projection'],
    ('attention_pooling', 2): ['pool_projection'],
    ('attention_pooling', 3): ['pool_projection'],
    ('bilinear', 2): ['bilinear_projection'],
    ('tensor_fusion', 2): ['tensor_final_proj', 'tensor_proj_0', 'tensor_proj_1'],
    ('tensor_fusion', 3): ['tensor_final_proj', 'tensor_proj_0', 'tensor_proj_1'],
}

_REACHABLE_ARMS = sorted(_EXPECTED_SITES)

# The 11 arms that actually build a fan-out site. The other 4 reachable arms
# (`cross_attention` at either count, `addition`/`multiplication` at 2) build
# none, so a per-site claim there would be vacuous.
_ARMS_WITH_SITES = [arm for arm in _REACHABLE_ARMS if _EXPECTED_SITES[arm]]

# Every (site name, weight kind) pair an arm must produce -- the seeded control
# asserts this set before comparing values, so it cannot pass over a branch
# that stopped firing.
_EXPECTED_SITE_WEIGHT_KEYS = {
    arm: [(name, kind) for name in names for kind in ('kernel', 'bias')]
    for arm, names in _EXPECTED_SITES.items()
}

# The eight SOURCE sites, as `multimodal_fusion.py` writes them, mapped to the
# arms that reach them. `{i}` sites expand to one guard per index.
_SITE_ARMS = {
    'concat_projection': [('concatenation', 2), ('concatenation', 3)],
    'align_projection_0': [('addition', 3), ('multiplication', 3)],
    'align_projection_1': [('addition', 3), ('multiplication', 3)],
    'align_projection_2': [('addition', 3), ('multiplication', 3)],
    'gate_0': [('gated', 2), ('gated', 3)],
    'gate_1': [('gated', 2), ('gated', 3)],
    'gate_2': [('gated', 3)],
    'gated_projection': [('gated', 2), ('gated', 3)],
    'pool_projection': [('attention_pooling', 2), ('attention_pooling', 3)],
    'bilinear_projection': [('bilinear', 2)],
    'tensor_proj_0': [('tensor_fusion', 2), ('tensor_fusion', 3)],
    'tensor_proj_1': [('tensor_fusion', 2), ('tensor_fusion', 3)],
    'tensor_final_proj': [('tensor_fusion', 2), ('tensor_fusion', 3)],
}


def _site_params(site_name):
    """`pytest.mark.parametrize` args covering every arm that reaches ``site_name``."""
    return [pytest.param(s, n, id=f"{s}-{n}mod") for s, n in _SITE_ARMS[site_name]]


def _np(x):
    return keras.ops.convert_to_numpy(x)


def _built(strategy, num_modalities, kernel_initializer, bias_initializer):
    layer = MultiModalFusion(
        dim=_DIM,
        fusion_strategy=strategy,
        num_tensor_projections=_NUM_TENSOR_PROJECTIONS,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )
    layer([
        np.zeros((_BATCH, _SEQ, _DIM), dtype='float32')
        for _ in range(num_modalities)
    ])
    return layer


def _fanout_sites(layer):
    """Every top-level `Dense` that was handed the shared initializer instances.

    Exactly `projection_layers` + `gate_layers`; the attention, norm and FFN
    sub-layers are built by their own factories and are never passed
    `self.kernel_initializer`, so their internal Dense weights are not sites.
    """
    candidates = list(layer.projection_layers) + list(layer.gate_layers)
    return [sub for sub in candidates if isinstance(sub, keras.layers.Dense)]


def _site(layer, name):
    hits = [sub for sub in _fanout_sites(layer) if sub.name == name]
    assert len(hits) == 1, (
        f"expected exactly one fan-out site named {name!r}, found "
        f"{[sub.name for sub in _fanout_sites(layer)]} -- the branch this guard "
        f"targets did not fire, so the guard would have passed vacuously"
    )
    return hits[0]


def _random_kernel_init():
    return keras.initializers.GlorotUniform()


def _random_bias_init():
    """A RANDOM bias initializer.

    The `'zeros'` default is exemption 2 (DETERMINISTIC: identical at every
    site and correctly so), which is asserted as a positive control below. The
    live bias defect is only observable under a caller-supplied random
    instance, so every bias guard passes one.
    """
    return keras.initializers.RandomNormal(stddev=0.05)


def _assert_kernel_is_not_the_shared_replay(layer, name):
    site = _site(layer, name)
    replay = _np(layer.kernel_initializer(tuple(site.kernel.shape), dtype='float32'))
    assert not np.array_equal(replay, _np(site.kernel)), (
        f"{name}/kernel {tuple(site.kernel.shape)} is bit-identical to a fresh "
        f"draw from the layer's shared kernel_initializer instance -- the site "
        f"was not cloned"
    )


def _assert_bias_is_not_the_shared_replay(layer, name):
    site = _site(layer, name)
    replay = _np(layer.bias_initializer(tuple(site.bias.shape), dtype='float32'))
    assert not np.array_equal(replay, _np(site.bias)), (
        f"{name}/bias {tuple(site.bias.shape)} is bit-identical to a fresh "
        f"draw from the layer's shared bias_initializer instance -- the site "
        f"was not cloned"
    )


# ---------------------------------------------------------------------


class TestEveryFusionStrategyBranchActuallyFires:
    """The parametrization's own guard.

    Only one of the eight `_build_*` branches runs per instance, so a guard
    aimed at a branch that never fires passes for the wrong reason. Each arm
    below pins the EXACT site inventory it produces, so a branch that stops
    creating a site is a failure rather than a silent hole in coverage.
    """

    @pytest.mark.parametrize(
        "strategy,num_modalities",
        [pytest.param(s, n, id=f"{s}-{n}mod") for s, n in _REACHABLE_ARMS],
    )
    def test_the_arm_creates_exactly_the_expected_fan_out_sites(self, strategy, num_modalities):
        layer = _built(strategy, num_modalities, _random_kernel_init(), _random_bias_init())
        assert sorted(sub.name for sub in _fanout_sites(layer)) == _EXPECTED_SITES[
            (strategy, num_modalities)
        ]

    def test_bilinear_is_unreachable_at_three_modalities(self):
        """MEASURED: `_build_bilinear` raises, so there is no `bilinear`-3 arm."""
        with pytest.raises(ValueError, match="requires exactly 2 modalities"):
            _built('bilinear', 3, _random_kernel_init(), _random_bias_init())

    def test_elementwise_has_no_alignment_sites_at_two_modalities(self):
        """MEASURED: `align_projection_{i}` is only created above 2 modalities.

        An `align_projection_*` guard written at 2 modalities would target a
        layer that does not exist, which is why `_SITE_ARMS` reaches those
        sites only at 3.
        """
        for strategy in ('addition', 'multiplication'):
            layer = _built(strategy, 2, _random_kernel_init(), _random_bias_init())
            assert _fanout_sites(layer) == []

    def test_the_arms_between_them_reach_all_eight_source_sites(self):
        """Every site named in `_SITE_ARMS` is produced by some arm."""
        produced = set()
        for strategy, num_modalities in _REACHABLE_ARMS:
            produced.update(_EXPECTED_SITES[(strategy, num_modalities)])
        assert produced == set(_SITE_ARMS)
        assert len(produced) == 13  # 8 source lines; `{i}` sites expand to 13 layers


class TestTheFusionInitializerDoesNotFanOut:
    """One guard per site per weight: 13 layers x {kernel, bias}.

    The eight SOURCE sites are `concat_projection`, `align_projection_{i}`,
    `gate_{i}`, `gated_projection`, `pool_projection`, `bilinear_projection`,
    `tensor_proj_{i}` and `tensor_final_proj`; each carries a paired
    `kernel_initializer=` / `bias_initializer=` line, for 16 source lines.
    """

    @pytest.mark.parametrize("strategy,num_modalities", _site_params('concat_projection'))
    def test_the_concat_projection_kernel_is_not_the_shared_replay(self, strategy, num_modalities):
        layer = _built(strategy, num_modalities, _random_kernel_init(), _random_bias_init())
        _assert_kernel_is_not_the_shared_replay(layer, 'concat_projection')

    @pytest.mark.parametrize("strategy,num_modalities", _site_params('concat_projection'))
    def test_the_concat_projection_bias_is_not_the_shared_replay(self, strategy, num_modalities):
        layer = _built(strategy, num_modalities, _random_kernel_init(), _random_bias_init())
        _assert_bias_is_not_the_shared_replay(layer, 'concat_projection')

    @pytest.mark.parametrize("index", [0, 1, 2])
    @pytest.mark.parametrize("strategy,num_modalities", _site_params('align_projection_0'))
    def test_the_align_projection_kernel_is_not_the_shared_replay(
        self, strategy, num_modalities, index
    ):
        layer = _built(strategy, num_modalities, _random_kernel_init(), _random_bias_init())
        _assert_kernel_is_not_the_shared_replay(layer, f'align_projection_{index}')

    @pytest.mark.parametrize("index", [0, 1, 2])
    @pytest.mark.parametrize("strategy,num_modalities", _site_params('align_projection_0'))
    def test_the_align_projection_bias_is_not_the_shared_replay(
        self, strategy, num_modalities, index
    ):
        layer = _built(strategy, num_modalities, _random_kernel_init(), _random_bias_init())
        _assert_bias_is_not_the_shared_replay(layer, f'align_projection_{index}')

    @pytest.mark.parametrize("num_modalities", [2, 3])
    @pytest.mark.parametrize("index", [0, 1])
    def test_the_gate_kernel_is_not_the_shared_replay(self, index, num_modalities):
        layer = _built('gated', num_modalities, _random_kernel_init(), _random_bias_init())
        _assert_kernel_is_not_the_shared_replay(layer, f'gate_{index}')

    @pytest.mark.parametrize("num_modalities", [2, 3])
    @pytest.mark.parametrize("index", [0, 1])
    def test_the_gate_bias_is_not_the_shared_replay(self, index, num_modalities):
        layer = _built('gated', num_modalities, _random_kernel_init(), _random_bias_init())
        _assert_bias_is_not_the_shared_replay(layer, f'gate_{index}')

    def test_the_third_gate_kernel_is_not_the_shared_replay(self):
        """`gate_2` only exists at 3 modalities -- one gate per modality."""
        layer = _built('gated', 3, _random_kernel_init(), _random_bias_init())
        _assert_kernel_is_not_the_shared_replay(layer, 'gate_2')

    def test_the_third_gate_bias_is_not_the_shared_replay(self):
        layer = _built('gated', 3, _random_kernel_init(), _random_bias_init())
        _assert_bias_is_not_the_shared_replay(layer, 'gate_2')

    @pytest.mark.parametrize("strategy,num_modalities", _site_params('gated_projection'))
    def test_the_gated_projection_kernel_is_not_the_shared_replay(self, strategy, num_modalities):
        layer = _built(strategy, num_modalities, _random_kernel_init(), _random_bias_init())
        _assert_kernel_is_not_the_shared_replay(layer, 'gated_projection')

    @pytest.mark.parametrize("strategy,num_modalities", _site_params('gated_projection'))
    def test_the_gated_projection_bias_is_not_the_shared_replay(self, strategy, num_modalities):
        layer = _built(strategy, num_modalities, _random_kernel_init(), _random_bias_init())
        _assert_bias_is_not_the_shared_replay(layer, 'gated_projection')

    @pytest.mark.parametrize("strategy,num_modalities", _site_params('pool_projection'))
    def test_the_pool_projection_kernel_is_not_the_shared_replay(self, strategy, num_modalities):
        layer = _built(strategy, num_modalities, _random_kernel_init(), _random_bias_init())
        _assert_kernel_is_not_the_shared_replay(layer, 'pool_projection')

    @pytest.mark.parametrize("strategy,num_modalities", _site_params('pool_projection'))
    def test_the_pool_projection_bias_is_not_the_shared_replay(self, strategy, num_modalities):
        layer = _built(strategy, num_modalities, _random_kernel_init(), _random_bias_init())
        _assert_bias_is_not_the_shared_replay(layer, 'pool_projection')

    @pytest.mark.parametrize("strategy,num_modalities", _site_params('bilinear_projection'))
    def test_the_bilinear_projection_kernel_is_not_the_shared_replay(
        self, strategy, num_modalities
    ):
        layer = _built(strategy, num_modalities, _random_kernel_init(), _random_bias_init())
        _assert_kernel_is_not_the_shared_replay(layer, 'bilinear_projection')

    @pytest.mark.parametrize("strategy,num_modalities", _site_params('bilinear_projection'))
    def test_the_bilinear_projection_bias_is_not_the_shared_replay(self, strategy, num_modalities):
        layer = _built(strategy, num_modalities, _random_kernel_init(), _random_bias_init())
        _assert_bias_is_not_the_shared_replay(layer, 'bilinear_projection')

    @pytest.mark.parametrize("num_modalities", [2, 3])
    @pytest.mark.parametrize("index", [0, 1])
    def test_the_tensor_proj_kernel_is_not_the_shared_replay(self, index, num_modalities):
        layer = _built('tensor_fusion', num_modalities, _random_kernel_init(), _random_bias_init())
        _assert_kernel_is_not_the_shared_replay(layer, f'tensor_proj_{index}')

    @pytest.mark.parametrize("num_modalities", [2, 3])
    @pytest.mark.parametrize("index", [0, 1])
    def test_the_tensor_proj_bias_is_not_the_shared_replay(self, index, num_modalities):
        layer = _built('tensor_fusion', num_modalities, _random_kernel_init(), _random_bias_init())
        _assert_bias_is_not_the_shared_replay(layer, f'tensor_proj_{index}')

    @pytest.mark.parametrize("strategy,num_modalities", _site_params('tensor_final_proj'))
    def test_the_tensor_final_proj_kernel_is_not_the_shared_replay(self, strategy, num_modalities):
        layer = _built(strategy, num_modalities, _random_kernel_init(), _random_bias_init())
        _assert_kernel_is_not_the_shared_replay(layer, 'tensor_final_proj')

    @pytest.mark.parametrize("strategy,num_modalities", _site_params('tensor_final_proj'))
    def test_the_tensor_final_proj_bias_is_not_the_shared_replay(self, strategy, num_modalities):
        layer = _built(strategy, num_modalities, _random_kernel_init(), _random_bias_init())
        _assert_bias_is_not_the_shared_replay(layer, 'tensor_final_proj')


class TestTheSharpestFusionPair:
    """`tensor_proj_i` vs `tensor_proj_j`: same kernel AND same input.

    `_build_tensor_fusion`'s parallel projections all read the SAME
    concatenation, so before the fix two of them are the same function of the
    same tensor. Their gradients are then identical too and gradient descent
    never separates them -- the textbook identical-hidden-units pathology, and
    the reason this file's fan-out is a training defect rather than a cosmetic
    one.

    This is an extra cross-check on top of the per-site guards, NOT a
    substitute: cloning either projection alone already separates the pair, so
    this test is blind to a one-line revert of exactly one of the two.
    """

    @pytest.mark.parametrize("num_modalities", [2, 3])
    def test_the_parallel_tensor_projections_do_not_start_identical(self, num_modalities):
        layer = _built('tensor_fusion', num_modalities, _random_kernel_init(), _random_bias_init())
        first = _np(_site(layer, 'tensor_proj_0').kernel)
        second = _np(_site(layer, 'tensor_proj_1').kernel)
        assert first.shape == second.shape
        assert not np.array_equal(first, second), (
            "tensor_proj_0 and tensor_proj_1 start as the same random numbers and "
            "read the same input, so they compute the same function forever"
        )


class TestTheFusionExemptionsThisModuleDoesNotClaimAway:
    """Positive controls. Cloning must not break any exemption."""

    @pytest.mark.parametrize(
        "strategy,num_modalities",
        [pytest.param(s, n, id=f"{s}-{n}mod") for s, n in _REACHABLE_ARMS],
    )
    def test_the_default_zeros_bias_is_identical_at_every_site_and_that_is_correct(
        self, strategy, num_modalities
    ):
        """Exemption 2, DETERMINISTIC.

        `bias_initializer` defaults to `'zeros'`, which holds no random state.
        Every site getting the same all-zero bias is what `'zeros'` is FOR, so
        this asserts the identity rather than treating it as the defect. It is
        also why every bias guard above passes a random instance instead.
        """
        layer = MultiModalFusion(
            dim=_DIM,
            fusion_strategy=strategy,
            num_tensor_projections=_NUM_TENSOR_PROJECTIONS,
            kernel_initializer=_random_kernel_init(),
        )
        layer([
            np.zeros((_BATCH, _SEQ, _DIM), dtype='float32')
            for _ in range(num_modalities)
        ])
        for site in _fanout_sites(layer):
            assert float(np.max(np.abs(_np(site.bias)))) == 0.0

    @pytest.mark.parametrize(
        "strategy,num_modalities",
        [pytest.param(s, n, id=f"{s}-{n}mod") for s, n in _ARMS_WITH_SITES],
    )
    def test_a_seeded_initializer_stays_reproducible_across_two_instances(
        self, strategy, num_modalities
    ):
        """Exemption 1 + invariant I-2: an explicit seed still reproduces exactly.

        This is what forces the clone to sit AT THE SITE rather than at the
        `keras.initializers.get(...)` line: cloning once in `__init__` would
        hand every child the same clone and restore the tie, while cloning per
        site keeps a seeded caller bit-reproducible.

        Scoped to the fan-out SITES, and to the arms that HAVE sites. The
        layer's other sub-layers (`create_attention_layer`,
        `create_normalization_layer`, `create_ffn_layer`) are never handed
        `self.kernel_initializer`, so they draw from their own seedless
        defaults and MEASURED differ between two instances --
        `cross_attn_0_0_to_1/q/kernel` does, pre-fix and post-fix alike. That
        is outside the claim `kernel_initializer=` makes, so asserting it here
        would fail for a reason that has nothing to do with this step.
        """
        snapshots = []
        for _ in range(2):
            layer = _built(
                strategy,
                num_modalities,
                keras.initializers.GlorotUniform(seed=7),
                keras.initializers.RandomNormal(stddev=0.05, seed=11),
            )
            snapshots.append({
                (site.name, kind): _np(getattr(site, kind))
                for site in _fanout_sites(layer)
                for kind in ('kernel', 'bias')
            })
        first, second = snapshots
        assert set(first) == set(_EXPECTED_SITE_WEIGHT_KEYS[(strategy, num_modalities)])
        assert set(first) == set(second)
        for key in first:
            assert np.array_equal(first[key], second[key]), (
                f"{key} is not reproducible under an explicit seed"
            )

    @pytest.mark.parametrize(
        "strategy,num_modalities",
        [pytest.param(s, n, id=f"{s}-{n}mod") for s, n in _REACHABLE_ARMS],
    )
    def test_cloning_changed_neither_the_config_keys_nor_the_weight_shapes(
        self, strategy, num_modalities
    ):
        """Cloning happens AT THE SITE, never at the `get_config` boundary."""
        layer = _built(strategy, num_modalities, 'glorot_uniform', 'zeros')
        config = layer.get_config()
        assert 'kernel_initializer' in config
        assert 'bias_initializer' in config

        restored = MultiModalFusion.from_config(config)
        restored([
            np.zeros((_BATCH, _SEQ, _DIM), dtype='float32')
            for _ in range(num_modalities)
        ])
        assert (
            {("/".join(w.path.split("/")[1:]), tuple(w.shape)) for w in layer.weights}
            == {("/".join(w.path.split("/")[1:]), tuple(w.shape)) for w in restored.weights}
        )
