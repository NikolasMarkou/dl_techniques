"""The guard the D-003 acceptance rests on: `MultiModalFusion` build parity.

`MultiModalFusion` creates every sub-layer in `build()`, never in `__init__` --
a literal deviation from the v2 custom-layer guide's §16.1 ("all sub-layers
created in `__init__`, unconditionally"). The deviation is ACCEPTED
(plan-2026-09-07T183458-be1c267e / D-003) because the sub-layer COUNT depends
on `len(input_shape)`, the runtime number of modalities, which cannot be known
before `build()`; re-architecting the eight `_build_*` builders would be a
large behaviour-risking change bought with no measured benefit.

That acceptance rests on ONE property: every `build()`-created sub-layer
carries an explicit `name=`, so an explicitly-built instance and a lazily-built
instance produce the same weight PATHS. Keras appends a `_<n>` disambiguator to
any sub-layer created without an explicit name, so the second instance built in
a process would otherwise silently get different weight paths -- the exact
`_SEWeights` defect measured in `tripse_attention.py`.

An anchor is a comment, not a guard. This file is the guard: it makes the
property the acceptance depends on monitored rather than asserted. MEASURED at
the time of the acceptance: parity holds for all 15 reachable
(strategy, num_modalities) arms, 4 to 54 weights each.

Weight-path SETS are compared, not just counts -- a count is blind to a rename.
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

# `bilinear` raises above 2 modalities, so there is no `bilinear`-3 arm.
_REACHABLE_ARMS = [
    (strategy, num_modalities)
    for strategy in (
        'cross_attention', 'concatenation', 'addition', 'multiplication',
        'gated', 'attention_pooling', 'bilinear', 'tensor_fusion',
    )
    for num_modalities in (2, 3)
    if not (strategy == 'bilinear' and num_modalities == 3)
]

_ARM_PARAMS = [pytest.param(s, n, id=f"{s}-{n}mod") for s, n in _REACHABLE_ARMS]


def _layer(strategy):
    return MultiModalFusion(
        dim=_DIM,
        fusion_strategy=strategy,
        num_tensor_projections=_NUM_TENSOR_PROJECTIONS,
    )


def _shapes(num_modalities):
    return [(_BATCH, _SEQ, _DIM)] * num_modalities


def _inputs(num_modalities):
    return [np.zeros(shape, dtype='float32') for shape in _shapes(num_modalities)]


def _paths(layer):
    """Weight paths with the top-level layer name stripped.

    The OUTER layer name legitimately differs between two instances in one
    process (`multi_modal_fusion` vs `multi_modal_fusion_1`); that is Keras
    naming the layer, not the layer failing to name its children. Everything
    below the first component is what D-003 depends on.
    """
    return sorted("/".join(w.path.split("/")[1:]) for w in layer.weights)


class TestTheFusionBuildMatchesTheLazyBuild:
    """Explicit `.build(shape)` and lazy `__call__` must agree on weight paths."""

    @pytest.mark.parametrize("strategy,num_modalities", _ARM_PARAMS)
    def test_the_weight_paths_agree(self, strategy, num_modalities):
        explicit = _layer(strategy)
        explicit.build(_shapes(num_modalities))

        lazy = _layer(strategy)
        lazy(_inputs(num_modalities))

        assert explicit.built and lazy.built
        assert _paths(explicit) == _paths(lazy), (
            "an explicitly-built and a lazily-built MultiModalFusion disagree on "
            "weight paths, so some build()-created sub-layer is missing an "
            "explicit name= -- D-003 accepts the build()-time creation ONLY "
            "while this holds"
        )

    @pytest.mark.parametrize("strategy,num_modalities", _ARM_PARAMS)
    def test_the_weight_counts_agree_and_are_non_zero(self, strategy, num_modalities):
        """A path-set comparison is the real guard; this pins that the arm ran.

        Every reachable arm builds at least one weight, so an arm whose builder
        silently did nothing fails here instead of passing the path comparison
        with two empty sets.
        """
        explicit = _layer(strategy)
        explicit.build(_shapes(num_modalities))

        lazy = _layer(strategy)
        lazy(_inputs(num_modalities))

        assert len(explicit.weights) == len(lazy.weights)
        assert len(explicit.weights) > 0

    @pytest.mark.parametrize("strategy,num_modalities", _ARM_PARAMS)
    def test_a_third_instance_built_after_the_first_two_still_agrees(
        self, strategy, num_modalities
    ):
        """Keras' auto-increment counter is process-global.

        A sub-layer missing `name=` drifts further with every instance built in
        the process, so the parity claim is checked against an instance created
        third, not just second.
        """
        first = _layer(strategy)
        first(_inputs(num_modalities))
        second = _layer(strategy)
        second.build(_shapes(num_modalities))
        third = _layer(strategy)
        third(_inputs(num_modalities))

        assert _paths(first) == _paths(second) == _paths(third)
