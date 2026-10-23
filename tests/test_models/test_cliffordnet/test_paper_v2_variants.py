"""The CliffordNet presets are the architectures of paper v2 and its authors' code.

Guard for plan-2026-10-23T100412-e7eaaa02 (D-007, D-008). Sources: arXiv:2601.06793v2
tables 2 and 3, and the factories in github.com/ParaMind2025/CAN `networks.py`
(read through a fetch summary on 2026-10-23, not byte-checked).

The paper truncates its parameter figures (our Nano is 1,436,516, written 1.43M),
so the paper-table check truncates to the paper's decimals. Presets with no
paper figure (cn_12_3, cn_18_5, cn_32_5) are regression pins of the count
measured here, not paper values; cn_12_3 (patch size 1, 160 channels) has a
stem that was not compared with the authors'.
"""

import math

import pytest

from dl_techniques.models.vision.cliffordnet import CliffordNet

S5 = [1, 2, 4, 8, 16]

# name: (channels, depth, patch_size, shifts, cli_mode, use_global_context, count)
PRESETS = {
    "nano": (128, 12, 2, [1, 2], "full", False, 1_436_516),
    "lite": (128, 12, 2, S5, "full", False, 2_616_164),
    "lite_g": (128, 12, 2, S5, "full", True, 3_404_132),
    "nano_g": (128, 12, 2, [1, 2], "full", True, 2_224_484),
    "cn_12_3": (160, 12, 1, [1, 2, 4], "full", False, 2_953_300),
    "cn_18_5": (128, 18, 2, S5, "full", False, 3_915_620),
    "cn_32_3": (128, 32, 2, [1, 2, 4], "full", False, 4_850_532),
    "cn_32_5": (128, 32, 2, S5, "inner", False, 4_326_244),
    "cn_64_5": (128, 64, 2, S5, "inner", False, 8_635_236),
}

# (preset, figure printed in the paper, decimals printed) - tables 2 and 3.
PAPER_FIGURES = [
    ("nano", 1.43, 2),
    ("lite", 2.61, 2),
    ("nano_g", 2.22, 2),
    ("lite_g", 3.40, 2),
    ("cn_32_3", 4.8, 1),
    ("cn_64_5", 8.6, 1),
]


def test_the_preset_table_is_exactly_this_set():
    assert set(CliffordNet.MODEL_VARIANTS) == set(PRESETS)


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_a_preset_has_the_published_architecture(name):
    channels, depth, patch, shifts, cli_mode, glob, _ = PRESETS[name]
    config = CliffordNet.MODEL_VARIANTS[name]
    assert config["channels"] == channels
    assert config["depth"] == depth
    assert config["patch_size"] == patch
    assert config["shifts"] == shifts
    assert config["cli_mode"] == cli_mode
    assert config["ctx_mode"] == "diff"
    assert config["use_global_context"] is glob


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_a_preset_has_its_pinned_parameter_count(name):
    model = CliffordNet.from_variant(name, num_classes=100)
    model.build((None, 32, 32, 3))
    assert model.count_params() == PRESETS[name][-1]


@pytest.mark.parametrize("name,figure,decimals", PAPER_FIGURES)
def test_a_paper_preset_truncates_to_the_printed_figure(name, figure, decimals):
    scale = 10 ** decimals
    truncated = math.floor(PRESETS[name][-1] / 1e6 * scale) / scale
    assert truncated == figure
