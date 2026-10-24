"""The TopoLM model README says what the code does (review pass 6, concerns 1, 10, 11, 13).

The README's tables carried claims no test read: a parameter table, a differences table,
an options table. This file reads them back and checks each against the code, so a
sentence that drifts from the model fails here. Numbers are built from the code (a
model is constructed and counted), never copied from the README into the test; the
sentences are matched by the facts they must carry.

Run on CPU (``CUDA_VISIBLE_DEVICES=``).
"""

import pathlib
import re
from typing import Dict, List

import keras
import numpy as np
import pytest

from dl_techniques.models.language.topolm.components import accepted_ffn_types

ROOT = pathlib.Path(__file__).resolve().parents[3]
README = ROOT / "src/dl_techniques/models/language/topolm/README.md"
MODEL_PY = ROOT / "src/dl_techniques/models/language/topolm/model.py"


def readme_text() -> str:
    return README.read_text(encoding="utf-8")


def table_rows(heading_fragment: str) -> List[List[str]]:
    """Cells of the first markdown table after the first line containing the fragment."""
    lines = readme_text().splitlines()
    start = next(i for i, line in enumerate(lines) if heading_fragment in line)
    rows: List[List[str]] = []
    for line in lines[start + 1:]:
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not set("".join(cells)) <= set("-: "):
                rows.append(cells)
        elif rows:
            break
    return rows


def options_table() -> Dict[str, List[str]]:
    rows = table_rows("### The block's options")
    return {name: row for row in rows[1:] for name in re.findall(r"`([a-z_]+)`", row[0])}


class TestTheOptionsTable:
    def test_the_ffn_type_row_names_exactly_the_accepted_types(self):
        row = options_table()["ffn_type"]
        assert row[1] == "`'mlp'`"
        assert accepted_ffn_types() == ["mlp"]
        text = row[2]
        assert "only accepted value" in text
        assert "output_kernel_initializer" in text
        assert "refused by name" in text

    @pytest.mark.parametrize("ffn_type", ["swiglu", "geglu"])
    def test_a_refused_type_is_never_described_as_accepted(self, ffn_type):
        text = readme_text()
        for line in text.splitlines():
            if f"'{ffn_type}'" in line:
                assert "refused" in line, line
        assert not re.search(rf"{ffn_type}.{{0,40}}(is|are) accepted", text)

    def test_the_options_are_those_of_the_block(self):
        from dl_techniques.models.language.topolm import TopoLMBlock
        import inspect

        parameters = set(inspect.signature(TopoLMBlock.__init__).parameters)
        documented = {
            "ffn_type", "ffn_args", "activation", "use_bias", "normalization_type",
            "norm_center", "attention_norm_args", "ffn_norm_args", "tap_placement",
        }
        assert documented <= parameters
        table = options_table()
        for name in documented:
            assert name in table, name


class TestNothingCallsTheBlockAGpt2Block:
    """GPT-2's attention projections carry biases; the block's never do (concern 11)."""

    def test_the_readme_does_not_claim_a_gpt2_block_with_biases(self):
        text = readme_text()
        assert "GPT-2 style model" not in text
        assert "still not a GPT-2 block" in text

    def test_the_model_docstring_does_not_either(self):
        source = MODEL_PY.read_text(encoding="utf-8")
        assert "A GPT-2-with-biases model" not in source
        assert "is not a GPT-2 block" in source

    def test_the_block_really_has_no_attention_bias(self):
        """The claim the sentence rests on, measured on a built block."""
        from .test_topolm_ffn import make_block

        block = make_block(use_bias=True)
        names = [w.path for w in block.attention.weights]
        assert names and not [n for n in names if n.endswith("/bias")], names


def parameter_table() -> Dict[str, List[int]]:
    """``{variant: [trainable, non-trainable, count_params]}`` as the README prints them."""
    rows = table_rows("### Parameter counts")
    assert [c.strip("`") for c in rows[0]][:2] == ["variant", "trainable parameters"]
    return {
        row[0].strip("`"): [int(cell.replace(",", "")) for cell in row[1:4]]
        for row in rows[1:]
    }


def built_counts(name: str) -> List[int]:
    """Trainable, non-trainable and ``count_params`` of ``create_topolm(name)``, built."""
    import numpy as np
    from dl_techniques.models.language.topolm import create_topolm

    model = create_topolm(name)
    model.build((None, 8))
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    non_trainable = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)
    assert model.count_params() == trainable + non_trainable
    return [trainable, non_trainable, model.count_params()]


class TestTheParameterTableIsWhatTheCodeBuilds:
    """Each row is rebuilt from ``create_topolm(name)`` with NO vocabulary argument.

    That is the README's own example, so a class default that disagrees with the table
    (the 100277 of review pass 6, concern 2) fails here as 167,950,048 against
    128,771,216.
    """

    @pytest.mark.parametrize("name", ["tiny", "small", "paper", "tiny3d", "paper3d"])
    def test_the_row_equals_the_built_model(self, name):
        keras.utils.set_random_seed(0)
        assert parameter_table()[name] == built_counts(name), name

    def test_every_variant_has_a_row(self):
        from dl_techniques.models.language.topolm import MODEL_VARIANTS

        assert sorted(parameter_table()) == sorted(MODEL_VARIANTS)

    def test_the_paper_headline_is_the_paper_rows_trainable_count(self):
        """"``paper`` is 128.8M trainable": the sentence and the row say one number."""
        match = re.search(r"`paper` is ([0-9.]+)M trainable", readme_text())
        assert match, "the headline sentence is gone"
        trainable = parameter_table()["paper"][0]
        assert round(trainable / 1e6, 1) == float(match.group(1))

    def test_the_closed_form_of_the_text_holds_for_the_paper_row(self):
        """``V*H + S*H + L*(4*H*H + 2*H*I + 2*H) + H`` with the paper variant's numbers."""
        from dl_techniques.models.language.topolm import MODEL_VARIANTS

        row = MODEL_VARIANTS["paper"]
        h, i, s, l = (row["embed_dim"], row["ffn_intermediate_size"],
                      row["max_seq_len"], row["depth"])
        v = 50304
        assert v * h + s * h + l * (4 * h * h + 2 * h * i + 2 * h) + h == (
            parameter_table()["paper"][0])


# ---------------------------------------------------------------------
# the differences table and the initialisation sentence
# ---------------------------------------------------------------------


def differences() -> Dict[str, List[str]]:
    rows = table_rows("### Against the paper and against the released code")
    assert rows[0][1:] == ["this repository", "released code", "paper text"]
    return {row[0]: row for row in rows[1:]}


def kernel_std_of(depth: int) -> Dict[str, float]:
    """Measured std of an expansion kernel and of the output projections of a built block."""
    from .test_topolm_ffn import make_block, weights_by_suffix

    block = make_block(hidden=256, inner=1024, heads=4, num_layers=depth)
    ffn = weights_by_suffix(block.ffn)
    attention = weights_by_suffix(block.attention)
    return {"fc1": float(ffn["fc1/kernel"].std()), "fc2": float(ffn["fc2/kernel"].std()),
            "proj": float(attention["proj/kernel"].std()),
            "qkv": float(attention["qkv/kernel"].std())}


def states(measured: float, stated: float) -> bool:
    """Whether a rounded README figure states a measured one: 2.5 per cent covers the
    rounding of a 2-digit figure and the sampling noise of the 262,144-draw kernels."""
    return abs(measured / stated - 1.0) < 0.025


class TestTheDifferencesTableMatchesTheCode:
    def test_the_rows_are_the_ones_the_text_discusses(self):
        assert list(differences()) == [
            "block", "initial weights", "residual-projection init",
            "bias, norm shift", "activation", "vocabulary", "tapped tensor",
            "neighbourhood", "neighbourhood draw",
        ]

    def test_the_initial_weights_row_says_truncated_here_and_normal_there(self):
        """Concern 10: the kernels start at 0.88x the released std, 0.0176 for 0.02."""
        row = differences()["initial weights"]
        assert "TruncatedNormal" in row[1] and "2 sigma" in row[1]
        assert "0.88" in row[1] and "0.0176 against 0.02" in row[1]
        assert "normal(0, std)" in row[2]
        measured = kernel_std_of(12)
        for key in ("fc1", "qkv"):
            assert states(measured[key], 0.0176), (key, measured[key])
        assert states(measured["fc1"] / 0.02, 0.88)

    def test_the_residual_projection_row_matches_the_measured_ratio(self):
        measured = kernel_std_of(12)
        assert "std / sqrt(2 * depth)" in differences()["residual-projection init"][1]
        assert abs(measured["fc2"] / measured["fc1"] * 24 ** 0.5 - 1.0) < 0.03
        assert abs(measured["proj"] / measured["qkv"] * 24 ** 0.5 - 1.0) < 0.03

    def test_the_depth_twelve_standard_deviations_of_the_text_are_the_measured_ones(self):
        """"0.0036 against 0.0124 at depth 12": the depth factor present, and missing."""
        match = re.search(
            r"standard deviation of ([0-9.]+) against ([0-9.]+) at depth 12", readme_text())
        assert match, "the sentence is gone"
        with_factor = kernel_std_of(12)["fc2"]
        without = kernel_std_of(1)["fc2"] * 1.0  # a one-layer block is the old std / sqrt(2)
        assert states(with_factor, float(match.group(1))), with_factor
        assert states(without, float(match.group(2))), without

    def test_the_defaults_row_claims_hold(self):
        from dl_techniques.models.language.topolm import TopoLMBlock

        block = TopoLMBlock(hidden_size=64, num_heads=4, intermediate_size=128, radius=1)
        assert block.use_bias is False and block.norm_center is False
        assert "none by default" in differences()["bias, norm shift"][1]
        assert block.tap_placement == "branch"
        assert "tap_placement='branch'" in differences()["tapped tensor"][1]

    def test_the_activation_row_says_exact_gelu_and_the_default_is_exact(self):
        from dl_techniques.layers.activations import resolve_activation
        import scipy.special

        assert differences()["activation"][1] == "exact GELU"
        value = float(resolve_activation("gelu")(np.float32(1.0)))
        assert abs(value - 0.5 * (1 + scipy.special.erf(1 / 2 ** 0.5))) < 1e-6
        assert abs(value - 0.5 * (1 + np.tanh(0.7978845608 * (1 + 0.044715)))) > 1e-5

    def test_the_vocabulary_row_is_the_class_default(self):
        from dl_techniques.models.language.topolm.model import DEFAULT_VOCAB_SIZE

        assert f"{DEFAULT_VOCAB_SIZE:,}" in differences()["vocabulary"][1]
        assert "50,304" in differences()["vocabulary"][2]

    def test_the_neighbourhood_row_counts_follow_from_the_radii(self):
        """11x11 = 121 units and 324 centres at radius 5, 9x9 = 81 and 400 at radius 4,
        on the 28 x 28 grid of width 784."""
        from dl_techniques.models.language.topolm.config import DEFAULT_RADIUS

        row = differences()["neighbourhood"]
        for radius, units, centres in ((5, 121, 324), (4, 81, 400)):
            assert (2 * radius + 1) ** 2 == units
            assert (28 - 2 * radius) ** 2 == centres
        assert DEFAULT_RADIUS == 5
        assert "radius 5" in row[1] and "11x11 = 121 units, 324 centres" in row[1]
        assert "9x9 = 81 units, 400 centres" in row[2]

    def test_both_geometry_settings_the_text_names_build(self):
        from dl_techniques.models.language.topolm import create_topolm

        for kwargs in ({"radius": 5, "distance": "linf"}, {"radius": 4, "distance": "l2"}):
            model = create_topolm("tiny", vocab_size=64, depth=1, max_seq_len=16, **kwargs)
            model.build((None, 8))
            assert model.blocks[0].attention_tap.distance == kwargs["distance"]
            assert model.blocks[0].attention_tap.radius == kwargs["radius"]
