"""The sequence pooling GUIDE and README teach what the code does.

Guard for plan-2026-10-14T062616-81938e81 step 5.5 (D-204). Measured at
fb05466c3: ``sequence_pooling/GUIDE.md``'s Pattern B example masked with
``scores + (1.0 - ops.cast(mask, scores.dtype)) * (-1e9)``, the additive form
the package removed because it is NaN under ``mixed_float16``, and its masking
paragraph said ``max`` used "additive ``-1e9``" while the code selects
``mask_sentinel`` with ``keras.ops.where``. The README had no single place
saying which sub-layers one ``SequencePooling`` knob reaches, and neither the
README nor the NLP heads README said that the heads' ``'attention'`` pooling is
a different mechanism from ``SequencePooling('attention')``.

Checked here: no fenced code block in either document carries an additive mask
or a ``from keras import``; the GUIDE's Pattern B block parses; every row of the
README's shared-knob table names a real ``SequencePooling`` parameter and its
"Reaches" claim holds on a built layer; the NLP-head sentence is in both
READMEs and true.
"""

import ast
import inspect
import pathlib
import re

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.nlp.factory import create_nlp_head
from dl_techniques.layers.heads.nlp.task_types import NLPTaskConfig, NLPTaskType
from dl_techniques.layers.sequence_pooling import SequencePooling

ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "dl_techniques" / "layers"
GUIDE = ROOT / "sequence_pooling" / "GUIDE.md"
README = ROOT / "sequence_pooling" / "README.md"
NLP_README = ROOT / "heads" / "nlp" / "README.md"

ADDITIVE_MASK = re.compile(
    r"\(\s*1(\.0)?\s*-\s*[^()]*(\([^()]*\))?[^()]*\)\s*\*\s*\(?\s*-?\s*(1e\d+|MASK|NEG)"
)


def _code_blocks(path):
    return re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S)


@pytest.mark.parametrize("doc", [GUIDE, README], ids=["GUIDE", "README"])
def test_no_code_block_masks_by_addition(doc):
    blocks = _code_blocks(doc)
    assert blocks, f"{doc.name} has no python blocks; the scan would be vacuous"
    offenders = [line for b in blocks for line in b.splitlines() if ADDITIVE_MASK.search(line)]
    assert offenders == [], offenders
    assert "* (-1e9)" not in doc.read_text(encoding="utf-8")


def test_the_additive_pattern_is_recognised():
    """Positive control: the line the GUIDE used to carry is caught."""
    assert ADDITIVE_MASK.search(
        "scores = scores + (1.0 - ops.cast(mask, scores.dtype)) * (-1e9)"
    )
    assert not ADDITIVE_MASK.search("scores = keras.ops.where(keep, scores, sentinel)")


@pytest.mark.parametrize("doc", [GUIDE, README], ids=["GUIDE", "README"])
def test_no_code_block_imports_from_keras(doc):
    offenders = [b for b in _code_blocks(doc) if re.search(r"^\s*from keras import", b, re.M)]
    assert offenders == []


def test_the_guide_pattern_b_example_parses():
    blocks = [b for b in _code_blocks(GUIDE) if "class MyPooling" in b]
    assert len(blocks) == 1
    tree = ast.parse(blocks[0])
    source = blocks[0]
    assert "mask_sentinel(self.compute_dtype)" in source
    assert "keras.ops.where" in source
    assert any(isinstance(n, ast.ClassDef) and n.name == "MyPooling" for n in tree.body)


def _knob_table():
    text = README.read_text(encoding="utf-8")
    section = text.split("#### Shared knobs", 1)[1].split("\n\n####", 1)[0]
    rows = [r for r in section.splitlines() if r.startswith("| `")]
    knobs = {}
    for row in rows:
        first = row.split("|")[1]
        for name in re.findall(r"`([a-z_]+)`", first):
            knobs[name] = row
    return knobs


def _components(**kwargs):
    layer = SequencePooling(
        strategy=["attention", "multi_head_attention", "weighted"],
        aggregation_method="weighted_sum",
        weighted_max_seq_len=8,
        **kwargs,
    )
    layer(np.zeros((2, 5, 6), "float32"))
    c = layer.learnable_components
    return layer, (c["attention"], c["multi_head_attention"]), c["weighted"]


def _check_temperature():
    _, attention, weighted = _components(temperature=2.5)
    assert {a.temperature for a in attention} == {2.5} and weighted.temperature == 2.5


def _check_attention_dropout_rate():
    _, attention, weighted = _components(attention_dropout_rate=0.3)
    assert {a.dropout_rate for a in attention} == {0.3} and weighted.dropout_rate == 0.3


def _check_attention_hidden_dim():
    _, attention, _ = _components(attention_hidden_dim=7)
    assert {a.hidden_dim for a in attention} == {7}


def _check_attention_num_heads():
    _, (single, multi), _ = _components(attention_num_heads=3)
    assert single.num_heads == 1 and multi.num_heads == 3


def _check_bias():
    layer, attention, weighted = _components(
        use_bias=True,
        bias_initializer=keras.initializers.Constant(0.4),
        bias_regularizer=keras.regularizers.L2(1.0),
    )
    for a in attention:
        np.testing.assert_array_equal(np.asarray(a.attention_dense.bias), np.float32(0.4))
    assert not [w.path for w in weighted.weights if "bias" in w.path]
    assert not [w.path for w in layer.weights if "aggregation" in w.path and "bias" in w.path]


def _check_kernel_initializer():
    layer, attention, weighted = _components(
        kernel_initializer=keras.initializers.Constant(0.25)
    )
    for a in attention:
        np.testing.assert_array_equal(np.asarray(a.attention_dense.kernel), 0.25)
        np.testing.assert_array_equal(np.asarray(a.context_vector), 0.25)
    np.testing.assert_array_equal(np.asarray(weighted.position_weights), 1.0)
    np.testing.assert_array_equal(np.asarray(layer.aggregation_weights), 1.0)


def _check_kernel_regularizer():
    layer, _, _ = _components(kernel_regularizer=keras.regularizers.L2(1.0), use_bias=False)
    # 2 projection kernels + 2 context vectors + position weights + aggregation weights
    assert len(layer.losses) == 6


def _check_weighted_max_seq_len():
    _, _, weighted = _components()
    assert weighted.max_seq_len == 8


def _check_top_k():
    assert SequencePooling(strategy="top_k_mean", top_k=4).top_k == 4


CHECKS = {
    "temperature": _check_temperature,
    "attention_dropout_rate": _check_attention_dropout_rate,
    "attention_hidden_dim": _check_attention_hidden_dim,
    "attention_num_heads": _check_attention_num_heads,
    "use_bias": _check_bias,
    "bias_initializer": _check_bias,
    "bias_regularizer": _check_bias,
    "kernel_initializer": _check_kernel_initializer,
    "kernel_regularizer": _check_kernel_regularizer,
    "weighted_max_seq_len": _check_weighted_max_seq_len,
    "top_k": _check_top_k,
}


def test_every_table_knob_is_a_parameter_and_has_a_check():
    knobs = _knob_table()
    params = set(inspect.signature(SequencePooling.__init__).parameters)
    assert set(knobs) <= params, sorted(set(knobs) - params)
    assert set(knobs) == set(CHECKS), (sorted(set(knobs) ^ set(CHECKS)))


@pytest.mark.parametrize("knob", sorted(CHECKS))
def test_the_table_claim_holds(knob):
    CHECKS[knob]()


@pytest.mark.parametrize("doc", [README, NLP_README], ids=["pooling_README", "nlp_README"])
def test_both_readmes_say_the_nlp_attention_pooling_is_its_own_scorer(doc):
    text = " ".join(doc.read_text(encoding="utf-8").split())
    assert "inline `Dense(1, tanh)` scorer" in text


def test_the_nlp_attention_pooling_is_a_dense_tanh_scorer():
    config = NLPTaskConfig(name="t", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=3)
    head = create_nlp_head(config, input_dim=8, pooling_type="attention")
    assert head.sequence_pooler is None
    scorer = head.attention_pooling
    assert isinstance(scorer, keras.layers.Dense) and scorer.units == 1
    assert scorer.activation is keras.activations.tanh
