"""NanoVLM's factory is the house shape: `from_variant` owns the table and
`pretrained`, and `create_nanovlm` only delegates.

Before plan-2026-10-14T062616-81938e81 (D-006, D-037) there was no
`from_variant`, `create_nanovlm` built from the table itself and handed
`pretrained=True` to the constructor (Keras "Unrecognized keyword arguments"),
and a second factory, `create_modern_nanovlm`, built from hard-coded dicts
that bypassed `MODEL_VARIANTS`. Its configuration is now the row `'modern'`
(see `test_modern_factory.py` for its behaviour).

Everything here constructs only; nothing is built or called, so the
224px rows cost no forward pass.
"""

import copy

import pytest

import dl_techniques.models.vision_language.nano_vlm as package
import dl_techniques.models.vision_language.nano_vlm.model as module
from dl_techniques.models.vision_language.nano_vlm import NanoVLM, create_nanovlm

#: A tiny shape laid over any row: the one-level merge keeps every other key.
_SMALL = dict(
    vision_config={"img_size": 32, "embed_dim": 16, "depth": 1, "num_heads": 2},
    text_config={"embed_dim": 16, "depth": 1, "num_heads": 2, "max_seq_len": 16},
    fusion_config={"attention_config": {"num_heads": 2}, "num_fusion_layers": 1},
)


def test_the_table_holds_the_four_named_rows():
    assert sorted(NanoVLM.MODEL_VARIANTS) == ["base", "large", "mini", "modern"]


def test_the_modern_factory_is_gone():
    assert not hasattr(module, "create_modern_nanovlm")
    assert package.__all__ == ["NanoVLM", "create_nanovlm"]


def test_an_unknown_variant_is_refused_by_name():
    with pytest.raises(ValueError, match="Unknown variant 'huge'"):
        NanoVLM.from_variant("huge")


@pytest.mark.parametrize("pretrained", [True, "weights.keras"])
@pytest.mark.parametrize("entry", ["from_variant", "create_nanovlm"])
def test_pretrained_is_refused_before_anything_is_built(monkeypatch, entry, pretrained):
    """No NanoVLM weights ship and the model has no backbone/head split, so
    `True` and a path both raise `NotImplementedError` (D-033 ruling), before
    the constructor runs."""
    def no_construction(self, *args, **kwargs):
        raise AssertionError("NanoVLM was constructed before the refusal")

    monkeypatch.setattr(NanoVLM, "__init__", no_construction)
    build = NanoVLM.from_variant if entry == "from_variant" else create_nanovlm
    with pytest.raises(NotImplementedError, match="keras.models.load_model"):
        build("mini", pretrained=pretrained)


@pytest.mark.parametrize("entry", ["from_variant", "create_nanovlm", "class"])
def test_an_unknown_keyword_is_refused(entry):
    with pytest.raises(ValueError, match="bogus_kw"):
        if entry == "class":
            config = copy.deepcopy(NanoVLM.MODEL_VARIANTS["mini"])
            config["text_config"]["vocab_size"] = 32
            NanoVLM(**config, vocab_size=32, bogus_kw=1)
        elif entry == "from_variant":
            NanoVLM.from_variant("mini", vocab_size=32, bogus_kw=1)
        else:
            create_nanovlm("mini", vocab_size=32, bogus_kw=1)


def test_an_unknown_text_component_type_is_refused():
    """It used to fall through to the encoder branch silently."""
    with pytest.raises(ValueError, match="text_component_type"):
        NanoVLM.from_variant("mini", vocab_size=32, text_component_type="enc", **_SMALL)


def test_the_config_dicts_merge_one_level_and_nothing_shared_is_mutated():
    table_before = copy.deepcopy(NanoVLM.MODEL_VARIANTS)
    overrides = copy.deepcopy(_SMALL)
    overrides_before = copy.deepcopy(overrides)

    model = NanoVLM.from_variant("modern", vocab_size=64, **overrides)

    # The passed keys replace the row's ...
    assert model.vision_config["img_size"] == 32
    assert model.text_config["depth"] == 1
    assert model.fusion_config["num_fusion_layers"] == 1
    # ... and every other key of the row stays.
    assert model.vision_config["attention_type"] == "differential"
    assert model.text_config["embedding_type"] == "factorized"
    assert model.fusion_config["num_tensor_projections"] == 8
    assert model.vocab_size == 64 and model.text_config["vocab_size"] == 64
    # Neither the class table nor the caller's dicts were written into.
    assert NanoVLM.MODEL_VARIANTS == table_before
    assert overrides == overrides_before


@pytest.mark.parametrize("variant", ["mini", "base", "large", "modern"])
def test_every_row_defaults_to_cross_attention_and_the_caller_can_override(variant):
    default = NanoVLM.from_variant(variant, vocab_size=32, **_SMALL)
    assert default.fusion_config["fusion_strategy"] == "cross_attention"
    pooled = NanoVLM.from_variant(variant, vocab_size=32,
                                  fusion_strategy="attention_pooling", **_SMALL)
    assert pooled.fusion_config["fusion_strategy"] == "attention_pooling"


def test_create_nanovlm_builds_what_from_variant_builds():
    a = create_nanovlm("mini", vocab_size=32, **copy.deepcopy(_SMALL))
    b = NanoVLM.from_variant("mini", vocab_size=32, **copy.deepcopy(_SMALL))
    assert a.get_config() | {"name": None} == b.get_config() | {"name": None}


def test_a_factorized_decoder_refuses_to_tie():
    """'modern' uses factorized word embeddings, which have no full table to
    tie the output to. The row says `use_shared_embedding=False`; asking for
    the tie raises instead of being ignored (it was ignored before D-037)."""
    assert NanoVLM.MODEL_VARIANTS["modern"]["use_shared_embedding"] is False
    model = NanoVLM.from_variant("modern", vocab_size=32, **_SMALL)
    assert model.ties_output_to_embeddings is False
    assert model.output_projection is not None
    with pytest.raises(ValueError, match="factorized"):
        NanoVLM.from_variant("modern", vocab_size=32, use_shared_embedding=True, **_SMALL)
