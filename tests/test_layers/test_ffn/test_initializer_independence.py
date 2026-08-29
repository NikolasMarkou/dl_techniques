"""One table-driven guard for the whole shared-initializer family.

Every layer in this package that calls ``keras.initializers.get(...)`` once in
``__init__`` and then hands the resolved INSTANCE to two or more weights draws
those weights bit-identically. The mechanism is written out once, in
``glu_ffn.py``'s construction comment and in ``decisions.md`` D-008: the
resolved object carries a concrete drawn seed, so every draw from it repeats;
``clone_initializer`` rebuilds it from ``get_config()``, which drops the seed.

Each row below names a layer class, a configuration at which two of its
weights COINCIDE IN SHAPE, and the pairs that must not be equal at build.
Rows are unseeded on purpose. A seeded initializer defeats the clone by
design, so a seeded guard passes with and without the fix and proves nothing;
that contract is pinned separately by the labelled control at the bottom.

Bias rows always pass an unseeded ``RandomNormal()`` INSTANCE. The default
``'zeros'`` makes a shared and a cloned initializer indistinguishable, and
probing at the default is how a kernels-only half fix reaches the tree.
"""

from typing import Any, Dict, List, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.layers.ffn.geglu_ffn import GeGLUFFN
from dl_techniques.layers.ffn.glu_ffn import GLUFFN
from dl_techniques.layers.ffn.logic_ffn import LogicFFN
from dl_techniques.layers.ffn.mlp import MLPBlock
from dl_techniques.layers.ffn.mlp_mixer_block import MixerBlock
from dl_techniques.layers.ffn.monarch_ffn import MonarchFFN
from dl_techniques.layers.ffn.squared_relu_ffn import SquaredReLUFFN
from dl_techniques.layers.ffn.swiglu_ffn import SwiGLUFFN
from dl_techniques.layers.ffn.swin_mlp import SwinMLP
from dl_techniques.layers.ffn.tversky_projection import TverskyProjectionLayer


def _random_normal() -> keras.initializers.Initializer:
    """Return an UNSEEDED RandomNormal instance.

    :return: A fresh, seedless ``RandomNormal``.
    :rtype: keras.initializers.Initializer
    """
    return keras.initializers.RandomNormal()


# (row id, layer class, constructor kwargs, input shape, weight-pair paths)
Row = Tuple[str, type, Dict[str, Any], Tuple[Any, ...], List[Tuple[str, str]]]

ROWS: List[Row] = [
    # -- unconditional: no configuration separates these ------------------
    (
        "glu-kernels",
        GLUFFN,
        {"hidden_dim": 16, "output_dim": 16},
        (None, 4, 16),
        [
            ("gate_proj.kernel", "value_proj.kernel"),
            ("gate_proj.kernel", "output_proj.kernel"),
        ],
    ),
    (
        "glu-biases",
        GLUFFN,
        {"hidden_dim": 16, "output_dim": 16, "bias_initializer": _random_normal()},
        (None, 4, 16),
        [
            ("gate_proj.bias", "value_proj.bias"),
            ("gate_proj.bias", "output_proj.bias"),
        ],
    ),
    (
        "tversky-contrast-scalars",
        TverskyProjectionLayer,
        {"units": 4, "num_features": 5, "contrast_initializer": _random_normal()},
        (None, 6),
        [("theta", "alpha"), ("alpha", "beta")],
    ),
    (
        "swiglu-biases",
        SwiGLUFFN,
        {
            "output_dim": 16,
            "hidden_dim": 8,
            "use_bias": True,
            "bias_initializer": _random_normal(),
        },
        (None, 4, 16),
        [("gate_proj.bias", "up_proj.bias")],
    ),
    # -- add_weight members: the call-site spelling differs, not the rule --
    (
        "monarch-factors",
        MonarchFFN,
        {"hidden_dim": 16, "output_dim": 16, "nblocks": 4},
        (None, 4, 16),
        [
            ("expand_l", "expand_r"),
            ("expand_l", "contract_l"),
            ("expand_r", "contract_r"),
        ],
    ),
    (
        "monarch-biases",
        MonarchFFN,
        {
            "hidden_dim": 16,
            "output_dim": 16,
            "nblocks": 4,
            "bias_initializer": _random_normal(),
        },
        (None, 4, 16),
        [("expand_bias", "contract_bias")],
    ),
    # -- conditional on coinciding shapes ---------------------------------
    (
        "geglu-kernels",
        GeGLUFFN,
        {"hidden_dim": 8, "output_dim": 16},
        (None, 4, 8),
        [("input_proj.kernel", "output_proj.kernel")],
    ),
    (
        "geglu-biases",
        GeGLUFFN,
        {"hidden_dim": 8, "output_dim": 16, "bias_initializer": _random_normal()},
        (None, 4, 8),
        [("input_proj.bias", "output_proj.bias")],
    ),
    (
        "mixer-kernels",
        MixerBlock,
        {"tokens_mlp_dim": 8, "channels_mlp_dim": 8},
        (None, 8, 8),
        [
            ("token_mlp_hidden.kernel", "channel_mlp_hidden.kernel"),
            ("token_mlp_out.kernel", "channel_mlp_out.kernel"),
        ],
    ),
    (
        "mixer-biases",
        MixerBlock,
        {
            "tokens_mlp_dim": 8,
            "channels_mlp_dim": 8,
            "bias_initializer": _random_normal(),
        },
        (None, 8, 8),
        [
            ("token_mlp_hidden.bias", "channel_mlp_hidden.bias"),
            ("token_mlp_out.bias", "channel_mlp_out.bias"),
        ],
    ),
    (
        "squared-relu-kernels",
        SquaredReLUFFN,
        {"hidden_dim": 16, "output_dim": 16},
        (None, 4, 16),
        [("fc1.kernel", "fc2.kernel")],
    ),
    (
        "squared-relu-biases",
        SquaredReLUFFN,
        {"hidden_dim": 16, "output_dim": 16, "bias_initializer": _random_normal()},
        (None, 4, 16),
        [("fc1.bias", "fc2.bias")],
    ),
    (
        "swin-kernels",
        SwinMLP,
        {"hidden_dim": 16, "output_dim": 16},
        (None, 4, 16),
        [("fc1.kernel", "fc2.kernel")],
    ),
    (
        "swin-biases",
        SwinMLP,
        {"hidden_dim": 16, "output_dim": 16, "bias_initializer": _random_normal()},
        (None, 4, 16),
        [("fc1.bias", "fc2.bias")],
    ),
    (
        "logic-kernels",
        LogicFFN,
        {"output_dim": 3, "logic_dim": 3},
        (None, 4, 3),
        [("gate_projection.kernel", "output_projection.kernel")],
    ),
    (
        "logic-biases",
        LogicFFN,
        {"output_dim": 3, "logic_dim": 3, "bias_initializer": _random_normal()},
        (None, 4, 3),
        [("gate_projection.bias", "output_projection.bias")],
    ),
    (
        "mlp-kernels",
        MLPBlock,
        {"hidden_dim": 16, "output_dim": 16},
        (None, 4, 16),
        [("fc1.kernel", "fc2.kernel")],
    ),
    (
        "mlp-biases",
        MLPBlock,
        {"hidden_dim": 16, "output_dim": 16, "bias_initializer": _random_normal()},
        (None, 4, 16),
        [("fc1.bias", "fc2.bias")],
    ),
]


def _weight(layer: keras.layers.Layer, path: str) -> Any:
    """Resolve a dotted weight path such as ``fc1.kernel`` on a built layer.

    :param layer: A built layer.
    :type layer: keras.layers.Layer
    :param path: Dotted attribute path to a weight variable.
    :type path: str
    :return: The weight variable named by ``path``.
    :rtype: Any
    """
    obj: Any = layer
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _max_abs_delta(first: Any, second: Any) -> float:
    """Return ``max|first - second|`` as a Python float.

    :param first: First weight variable.
    :type first: Any
    :param second: Second weight variable.
    :type second: Any
    :return: The largest absolute element-wise difference.
    :rtype: float
    """
    return float(np.max(np.abs(np.array(first) - np.array(second))))


class TestInitializerIndependence:
    """Every weight pair that can coincide in shape must differ at build."""

    @pytest.mark.parametrize(
        "row_id, layer_cls, kwargs, input_shape, pairs",
        ROWS,
        ids=[row[0] for row in ROWS],
    )
    def test_the_pair_is_not_bit_identical_at_build(
        self,
        row_id: str,
        layer_cls: type,
        kwargs: Dict[str, Any],
        input_shape: Tuple[Any, ...],
        pairs: List[Tuple[str, str]],
    ) -> None:
        """Build the layer and assert every named pair separates.

        The shape assertion is not decoration: a pair that does not coincide
        in shape cannot be bit-identical, so a row whose configuration
        stopped triggering the collision would pass vacuously.
        """
        layer = layer_cls(**kwargs)
        layer.build(input_shape)

        for left, right in pairs:
            first = _weight(layer, left)
            second = _weight(layer, right)
            assert tuple(first.shape) == tuple(second.shape), (
                f"{row_id}: {left} and {right} no longer coincide in shape "
                f"({tuple(first.shape)} vs {tuple(second.shape)}), so this "
                f"row can no longer detect a shared initializer"
            )
            delta = _max_abs_delta(first, second)
            assert delta > 0.0, (
                f"{row_id}: {left} and {right} are bit-identical "
                f"(max|delta| = {delta}). One resolved initializer instance "
                f"is reaching both weights; clone it per weight."
            )


class TestTheGuardIsNotVacuous:
    """Controls. These pin the instrument, not the fix."""

    def test_a_seeded_initializer_ties_the_pair_even_with_the_clone(
        self,
    ) -> None:
        """CONTROL. Passes with and without the fix, on purpose.

        ``clone_initializer`` rebuilds an initializer from its own config,
        which carries the seed. Two clones of ``GlorotUniform(seed=7)``
        therefore still draw the same tensor. This is the documented
        contract, and it is why every row above is unseeded.
        """
        layer = GLUFFN(
            hidden_dim=16,
            output_dim=16,
            kernel_initializer=keras.initializers.GlorotUniform(seed=7),
        )
        layer.build((None, 4, 16))

        delta = _max_abs_delta(layer.gate_proj.kernel, layer.value_proj.kernel)
        assert delta == 0.0

    def test_two_distinct_initializer_objects_were_never_the_problem(
        self,
    ) -> None:
        """CONTROL. ``TverskyProjectionLayer``'s other two weights.

        ``prototypes`` and ``feature_bank`` come from two DIFFERENT
        initializer arguments, so they never shared an instance and are not
        part of this family. They are asserted here so that a future change
        that funnels them through one object is caught by this module too.
        """
        layer = TverskyProjectionLayer(units=4, num_features=4)
        layer.build((None, 6))

        assert tuple(layer.prototypes.shape) == tuple(layer.feature_bank.shape)
        assert _max_abs_delta(layer.prototypes, layer.feature_bank) > 0.0
