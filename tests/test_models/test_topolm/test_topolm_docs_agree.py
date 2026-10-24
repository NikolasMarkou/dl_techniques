"""The documents that quote TopoLM's measurements quote the CURRENT ones (review pass 6, concerns 7, 14).

Step 12.2 (D-031) re-measured the ``aggregate_backbone_losses`` claim on the corrected
model (17.67 aggregated against 7.68 not) and its message said every document was
updated; ``REPO_MAP.md``, the documented starting point for orientation, still quoted
the previous model's 15.31 against 5.36 and the old composition (``create_ffn_layer``),
and ``models/language/AGENTS.md`` had lost a comma that turned "not on identical
weights" into the opposite of "not aggregated". Nothing read those files.

The measurement is REPRODUCED here (the procedure of the README: the ``tiny`` variant,
alpha 2.5, vocabulary 2048, two batches of random tokens, SGD, one epoch, CPU) and the
documents are held to it, so a document that quotes another pair fails.

Run on CPU (``CUDA_VISIBLE_DEVICES=``).
"""

import pathlib
import re
from typing import List

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.common.masked_language_model import CausalLanguageModel
from dl_techniques.models.language.topolm import create_topolm

ROOT = pathlib.Path(__file__).resolve().parents[3]

#: Every file that states TopoLM's aggregated-loss measurement, or the composition of
#: the block. A file that quotes the measurement is listed with True.
DOCS = {
    "REPO_MAP.md": True,
    "src/dl_techniques/models/language/AGENTS.md": True,
    "src/dl_techniques/models/language/topolm/README.md": True,
    "src/train/topolm/README.md": True,
    "src/train/topolm/common.py": True,
    "src/train/topolm/__init__.py": True,
    "tests/test_callbacks/test_topographic_callbacks.py": True,
}

VOCAB = 2048


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _data() -> tf.data.Dataset:
    x = np.random.default_rng(7).integers(0, VOCAB, size=(4, 12)).astype("int32")
    y = np.random.default_rng(8).integers(0, VOCAB, size=(4, 12)).astype("int32")
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(2)


def measured_training_loss(aggregate: bool) -> float:
    """The reported training loss of one epoch, with or without the backbone's losses."""
    keras.utils.set_random_seed(0)
    backbone = create_topolm("tiny", vocab_size=VOCAB, max_seq_len=12)
    head = CausalLanguageModel(
        backbone=backbone, vocab_size=VOCAB, skip_head=True, output_key="logits",
        pre_shifted=True, aggregate_backbone_losses=aggregate, verify_causality=False)
    head.compile(optimizer=keras.optimizers.SGD(1e-3))
    head(np.zeros((1, 12), dtype="int32"), training=False)
    return float(head.fit(_data(), epochs=1, verbose=0).history["loss"][0])


@pytest.fixture(scope="module")
def measured() -> List[float]:
    return [measured_training_loss(True), measured_training_loss(False)]


class TestTheQuotedMeasurementIsTheReproducedOne:
    def test_the_two_numbers_reproduce(self, measured):
        aggregated, plain = measured
        assert abs(aggregated - 17.67) < 0.02, aggregated
        assert abs(plain - 7.68) < 0.02, plain

    def test_the_gap_is_the_derived_one(self, measured):
        """8 taps at alpha 2.5, each penalty about 0.5 at random weights: 10.0."""
        gap = measured[0] - measured[1]
        taps = len(create_topolm("tiny", vocab_size=VOCAB, max_seq_len=12).tap_layers)
        assert taps == 8
        assert abs(gap - taps * 2.5 * 0.5) < 0.05, gap

    @pytest.mark.parametrize("relative", sorted(DOCS))
    def test_each_document_quotes_the_current_pair(self, relative, measured):
        text = _read(relative)
        for number in (f"{measured[0]:.2f}", f"{measured[1]:.2f}"):
            assert number in text, (relative, number)

    @pytest.mark.parametrize("relative", sorted(DOCS))
    def test_no_document_quotes_the_previous_models_pair(self, relative):
        text = _read(relative)
        for stale in ("15.31", "5.36"):
            assert not re.search(rf"(?<![\d.]){re.escape(stale)}(?!\d)", text), (relative, stale)


class TestTheReadingOfTheMeasurement:
    def test_the_agents_row_says_not_aggregated_and_has_no_dash(self):
        row = next(line for line in _read(
            "src/dl_techniques/models/language/AGENTS.md").splitlines()
            if line.startswith("| `aggregate_backbone_losses=True`"))
        assert "aggregated against 7.68 not aggregated" in row
        assert "same seed" in row
        assert chr(0x2014) not in row and chr(0x2013) not in row  # no em dash, no en dash

    def test_the_agents_row_is_a_three_cell_table_row(self):
        row = next(line for line in _read(
            "src/dl_techniques/models/language/AGENTS.md").splitlines()
            if line.startswith("| `aggregate_backbone_losses=True`"))
        assert row.count("|") == 3


class TestRepoMapNamesTheCompositionThatIsBuilt:
    def test_it_names_the_feed_forward_path_the_block_uses(self):
        text = " ".join(_read("REPO_MAP.md").split())
        sentence = text.split("`TopoLMBlock` there composes", 1)[1].split("Precedents", 1)[0]
        assert "build_transformer_ffn_config" in sentence
        assert "create_ffn_from_config" in sentence
        assert "create_ffn_layer" not in sentence

    def test_the_block_really_composes_those_calls(self):
        source = _read("src/dl_techniques/models/language/topolm/components.py")
        for call in ("create_attention_layer(", "create_normalization_layer(",
                     "build_transformer_ffn_config(", "create_ffn_from_config("):
            assert call in source, call
        code = [line for line in source.splitlines() if not line.lstrip().startswith("#")]
        assert not [line for line in code if "create_ffn_layer(" in line]
