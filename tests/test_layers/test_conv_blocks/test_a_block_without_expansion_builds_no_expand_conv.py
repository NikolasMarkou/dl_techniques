"""``UniversalInvertedBottleneck(use_expansion=False)`` builds no expansion.

The references build an expansion-ratio-1 row without the ``1x1`` expansion
conv, its norm and its activation (Keras Applications
``_inverted_res_block``'s ``if block_id:``, torchvision's
``if expand_ratio != 1``). Before plan-2026-10-14T062616-81938e81 step 10.3 the
universal block always built it, so MobileNetV2's and V3's first rows carried
an extra ``C x C`` conv and BatchNorm. A width that the flag cannot keep is
refused, never ignored.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.conv_blocks.universal_inverted_bottleneck import (
    UniversalInvertedBottleneck,
)

C = 16


def _block(**kw) -> UniversalInvertedBottleneck:
    kw.setdefault("filters", C)
    kw.setdefault("expanded_channels", C)
    kw.setdefault("use_expansion", False)
    kw.setdefault("use_squeeze_excitation", True)
    kw.setdefault("se_divisor", 8)
    return UniversalInvertedBottleneck(**kw)


def _paths(layer) -> list:
    return sorted(w.path.split("/", 1)[1] for w in layer.weights)


def test_no_expansion_weight_and_the_forward_is_the_hand_composition():
    keras.utils.set_random_seed(0)
    x = np.random.default_rng(0).standard_normal((2, 8, 8, C)).astype("float32")
    block = _block()
    out = np.asarray(block(x, training=False))
    assert block.expand_conv is None
    assert block.expand_norm is None and block.expand_activation is None
    assert not any("expand" in p for p in _paths(block)), _paths(block)
    # depthwise -> norm -> act -> SE -> project -> norm, plus the residual.
    h = block.dw1_activation(block.dw1_norm(block.dw1(x), training=False))
    h = block.se(h, training=False)
    h = block.project_norm(block.project_conv(h), training=False)
    np.testing.assert_allclose(out, x + np.asarray(h), rtol=0, atol=1e-5)


def test_the_count_is_the_expanded_block_minus_its_expansion():
    with_exp = _block(use_expansion=True)
    without = _block()
    with_exp.build((None, 8, 8, C))
    without.build((None, 8, 8, C))
    # A C x C 1x1 kernel (no bias) and a BatchNorm of 4 x C.
    assert with_exp.count_params() - without.count_params() == C * C + 4 * C


@pytest.mark.parametrize("kw", [
    {"expanded_channels": 2 * C},
    {"expanded_channels": None, "expansion_factor": 4},
])
def test_a_width_the_flag_cannot_keep_is_refused(kw):
    with pytest.raises(ValueError, match="use_expansion=False"):
        _block(**kw).build((None, 8, 8, C))


def test_expansion_factor_one_needs_no_explicit_width():
    block = _block(expanded_channels=None, expansion_factor=1)
    block.build((None, 8, 8, C))
    assert block.expand_conv is None


def test_the_flag_survives_the_config_round_trip():
    config = _block().get_config()
    assert config["use_expansion"] is False
    rebuilt = UniversalInvertedBottleneck.from_config(config)
    assert rebuilt.use_expansion is False and rebuilt.expand_conv is None


def test_the_default_still_expands():
    block = UniversalInvertedBottleneck(filters=C)
    block.build((None, 8, 8, C))
    assert block.use_expansion is True
    assert int(block.expand_conv.kernel.shape[-1]) == 4 * C
