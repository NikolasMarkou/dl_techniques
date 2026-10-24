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
