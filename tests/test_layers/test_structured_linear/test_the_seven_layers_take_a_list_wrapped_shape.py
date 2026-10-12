"""compute_output_shape on a list-wrapped single shape, for the seven non-crystal layers.

The three crystal layers already route through
`dl_techniques.utils.tensors.canonical_binary_input_shape`; the other seven
used the un-canonicalized `list(input_shape)` / `tuple(input_shape[:-1])`
spelling, which rewrote the *batch* axis for a `[[None, 8]]`-style input
(MIGRATION.md "Known-open gap"). All seven now route through the helper.
"""

import pytest

from dl_techniques.layers.structured_linear.bitlinear_layer import BitLinear
from dl_techniques.layers.structured_linear.harmonic_dense import HarmonicDense
from dl_techniques.layers.structured_linear.hierarchical_harmonic import (
    HierarchicalHarmonicHead,
)
from dl_techniques.layers.structured_linear.mps_layer import MPSLayer
from dl_techniques.layers.structured_linear.orthogonal_butterfly import (
    OrthogonalButterfly,
)
from dl_techniques.layers.structured_linear.orthoblock import OrthoBlock
from dl_techniques.layers.structured_linear.random_fourier_features import (
    RFFKernelLayer,
)


def _cases():
    return [
        ("HarmonicDense", HarmonicDense(units=4), (None, 4)),
        ("HierarchicalHarmonicHead", HierarchicalHarmonicHead(num_classes=8), (None, 8)),
        ("OrthoBlock", OrthoBlock(units=4), (None, 4)),
        ("BitLinear", BitLinear(units=4), (None, 4)),
        ("OrthogonalButterfly", OrthogonalButterfly(), (None, 8)),
        ("MPSLayer", MPSLayer(output_dim=4), (None, 4)),
        ("RFFKernelLayer", RFFKernelLayer(input_dim=8), (None, 8)),
    ]


@pytest.mark.parametrize("form", [[[None, 8]], [(None, 8)], [None, 8]])
def test_the_seven_layers_survive_every_list_wrapped_single_shape(form):
    for name, layer, expected in _cases():
        assert layer.compute_output_shape(form) == expected, name


def test_the_tuple_form_is_unchanged():
    for name, layer, expected in _cases():
        assert layer.compute_output_shape((None, 8)) == expected, name
