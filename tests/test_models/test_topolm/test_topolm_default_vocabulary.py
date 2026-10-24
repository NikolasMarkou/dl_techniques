"""The model's own default vocabulary is the padded GPT-2 one, like the trainer's.

plan-2026-10-23T190557-972a9ac5 step 13.2 (review pass 6, concern 2). Step 11.2
(D-031 e) moved the TRAINER's ``--vocab-size`` to the tokenizer's need padded to 64
(50304 for the default gpt2 encoding) but left the CLASS default at cl100k's 100277:
``create_topolm("paper")`` built 167,950,048 trainable parameters, 50k never-used rows
more than the 128,771,216 the README presents, and the README's training head example
paired that backbone with ``vocab_size=50304`` without an error.

The expected number is computed here from ``tiktoken`` and ``math.ceil``; nothing is
read back from the code under test but the built model.

Run on CPU (``CUDA_VISIBLE_DEVICES=``).
"""

import math

import numpy as np
import pytest
import tiktoken

from dl_techniques.models.language.topolm import TopoLM, create_topolm
from dl_techniques.models.language.topolm import model as topolm_model

#: The four special ids the trainer's config assigns after the base vocabulary
#: (``cls``, ``sep``, ``pad``, ``mask``: 50257 to 50260).
TRAINER_SPECIAL_IDS = (50257, 50258, 50259, 50260)


def padded_gpt2_vocabulary() -> int:
    base = tiktoken.get_encoding("gpt2").n_vocab
    required = max(base, max(TRAINER_SPECIAL_IDS) + 1)
    return int(math.ceil(required / 64) * 64)


SMALL = dict(embed_dim=64, depth=1, num_heads=4, ffn_intermediate_size=128,
             max_seq_len=16, radius=1)


class TestTheClassDefaultIsThePaddedGpt2Vocabulary:
    def test_the_expected_value_is_50304(self):
        assert padded_gpt2_vocabulary() == 50304

    def test_the_module_constant(self):
        assert topolm_model.DEFAULT_VOCAB_SIZE == padded_gpt2_vocabulary()

    def test_a_directly_built_model_uses_it(self):
        model = TopoLM(**SMALL)
        assert model.vocab_size == 50304
        assert model.get_config()["vocab_size"] == 50304

    @pytest.mark.parametrize("build", [TopoLM.from_variant, create_topolm],
                             ids=["from_variant", "create_topolm"])
    def test_the_variant_entry_points_use_it(self, build):
        model = build("tiny")
        assert model.vocab_size == 50304

    def test_the_embedding_table_and_the_logits_have_that_many_columns(self):
        model = TopoLM(**SMALL)
        out = model(np.zeros((1, 4), dtype="int32"), training=False)
        assert tuple(model.word_embeddings.embeddings.shape) == (50304, 64)
        assert int(out["logits"].shape[-1]) == 50304

    def test_an_explicit_vocabulary_is_used_as_typed(self):
        assert TopoLM(vocab_size=1000, **SMALL).vocab_size == 1000
        assert create_topolm("tiny", vocab_size=512).vocab_size == 512

    def test_the_docstring_does_not_name_the_old_default(self):
        doc = TopoLM.__doc__
        assert "100277" not in doc and "Default 50304" in doc
        assert "cl100k" not in doc.split(":param vocab_size:")[1].split(":type")[0]


class TestTheCausalHeadExampleIsConsistent:
    def test_the_backbone_of_the_readme_example_has_the_head_vocabulary(self):
        """The README builds ``create_topolm("paper")`` and a head over 50304."""
        import pathlib
        import re

        readme = pathlib.Path(topolm_model.__file__).with_name("README.md").read_text()
        block = readme.split("head = CausalLanguageModel(")[1].split(")\n```")[0]
        typed = int(re.search(r"vocab_size=(\d+)", block).group(1))
        assert 'create_topolm("paper")' in block
        assert create_topolm("paper").vocab_size == typed
