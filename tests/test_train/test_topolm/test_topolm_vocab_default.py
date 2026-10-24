"""The trainer's ``--vocab-size`` default is the tokenizer's need, padded to 64 rows.

plan-2026-10-23T190557-972a9ac5 step 11.2 (D-031 e). The CLI used to default to
100277 rows (cl100k's size) while the trainer's gpt2 tokenizer emits ids up to
50260: 39M unused embedding rows (50k x 784 floats is 39M parameters) in the paper
variant. The released code pads GPT-2's 50257 to 50304, the next multiple of 64.

The default is now ``None`` and :func:`train.topolm.pretrain._config_from_args`
resolves it, like the radius, so ``config.json`` records the integer the model is
built with: the smallest multiple of 64 at or above what the tokenizer named by the
config's ``encoding_name`` can emit. An explicit value is kept as typed, and the
too-small refusal is the existing guard.

The expected numbers are computed here from ``tiktoken`` directly and from
``math.ceil``, not read back from the trainer.

CPU only (``CUDA_VISIBLE_DEVICES=``).
"""

import json
import math

import pytest

import tiktoken

from train.topolm import common as C
from train.topolm import pretrain as cli
from train.topolm import TopoLMTrainingConfig, build_backbone


def _parse(*argv):
    return cli._build_parser().parse_args(list(argv))


def _config(*argv):
    return cli._config_from_args(_parse(*argv))


def padded(n: int, multiple: int = 64) -> int:
    """The independent oracle for the rounding."""
    return int(math.ceil(n / multiple) * multiple)


class TestTheDefaultIsResolvedNotTyped:
    def test_the_parser_leaves_the_flag_unset(self):
        """None means "not typed": it is what lets an explicit value be told apart."""
        assert _parse().vocab_size is None

    def test_the_gpt2_default_resolves_to_50304(self):
        """tiktoken gpt2: 50257 base ids, the trainer's specials run 50257..50260."""
        base = tiktoken.get_encoding("gpt2").n_vocab
        specials = (50257, 50258, 50259, 50260)
        required = max(base, max(specials) + 1)
        assert (base, required) == (50257, 50261)
        assert _config().vocab_size == padded(required) == 50304

    def test_the_resolved_value_is_a_plain_int_multiple_of_64(self):
        value = _config().vocab_size
        assert type(value) is int and value % 64 == 0

    def test_the_value_follows_the_encoding_the_config_names(self):
        """Another encoding moves the requirement: cl100k_base has 100277 base ids."""
        base = tiktoken.get_encoding("cl100k_base").n_vocab
        assert base == 100277
        config = TopoLMTrainingConfig(encoding_name="cl100k_base")
        assert C.resolve_vocab_size(config) == padded(base) == 100288

    def test_a_special_id_above_the_base_vocabulary_counts(self):
        config = TopoLMTrainingConfig(
            encoding_name="gpt2", cls_token_id=50257, sep_token_id=50258,
            pad_token_id=50300, mask_token_id=50260,
        )
        assert C.resolve_vocab_size(config) == padded(50301) == 50304
        config = TopoLMTrainingConfig(
            encoding_name="gpt2", cls_token_id=50257, sep_token_id=50258,
            pad_token_id=50304, mask_token_id=50260,
        )
        assert C.resolve_vocab_size(config) == padded(50305) == 50368

    def test_resolving_builds_no_tokenizer(self, monkeypatch):
        """``_config_from_args`` runs before the variant/radius refusal, and a refused
        run does no run work (D-018): the resolver reads the encoding, builds no
        tokenizer."""
        def refuse(*args, **kwargs):
            raise AssertionError("create_tokenizer was reached while resolving")

        monkeypatch.setattr(C, "create_tokenizer", refuse)
        assert _config().vocab_size == 50304

    @pytest.mark.parametrize("required", [1, 63, 64, 65, 50257, 50261, 50303, 50304,
                                          50305, 100277])
    def test_the_rounding_is_the_next_multiple_at_or_above(self, required):
        assert C.round_up_vocab_size(required) == padded(required)
        assert C.round_up_vocab_size(required) >= required
        assert C.round_up_vocab_size(required) - required < 64


class TestAnExplicitValueIsKept:
    @pytest.mark.parametrize("typed", [32000, 50261, 50304, 70000, 100277])
    def test_the_typed_integer_reaches_the_config_unchanged(self, typed):
        assert _config("--vocab-size", str(typed)).vocab_size == typed

    @pytest.mark.parametrize("rows", [50257, 50260])
    def test_a_typed_table_that_is_too_small_is_still_refused_by_name(self, rows):
        config = _config("--vocab-size", str(rows))
        assert config.vocab_size == rows
        preprocessor = C.create_tokenizer(
            config.encoding_name, config.max_seq_length, config.cls_token_id,
            config.sep_token_id, config.pad_token_id, config.mask_token_id,
        )
        with pytest.raises(ValueError, match=r"--vocab-size") as excinfo:
            C._require_vocab_covers_tokenizer(config, preprocessor)
        assert "50261" in str(excinfo.value)
        with pytest.raises(ValueError, match=r"--vocab-size"):
            build_backbone(
                config, required_vocab_size=C._required_vocab_size(preprocessor)
            )

    def test_the_resolved_default_passes_the_same_guard(self):
        config = _config()
        preprocessor = C.create_tokenizer(
            config.encoding_name, config.max_seq_length, config.cls_token_id,
            config.sep_token_id, config.pad_token_id, config.mask_token_id,
        )
        C._require_vocab_covers_tokenizer(config, preprocessor)
        build_backbone(config, required_vocab_size=C._required_vocab_size(preprocessor))


class TestConfigJsonRecordsTheInteger:
    def test_the_written_config_holds_the_resolved_number(self, tmp_path):
        config = _config()
        C.prepare_run_dir(
            config, output_dir=str(tmp_path), config_filename="config.json"
        )
        written = json.loads((tmp_path / "config.json").read_text())
        assert written["vocab_size"] == 50304
        assert type(written["vocab_size"]) is int

    def test_a_typed_value_is_what_the_file_holds(self, tmp_path):
        config = _config("--vocab-size", "50261")
        C.prepare_run_dir(
            config, output_dir=str(tmp_path), config_filename="config.json"
        )
        written = json.loads((tmp_path / "config.json").read_text())
        assert written["vocab_size"] == 50261

    def test_the_help_text_states_the_rule(self):
        parser = cli._build_parser()
        action = next(a for a in parser._actions if "--vocab-size" in a.option_strings)
        assert action.default is None
        assert "multiple of 64" in action.help and "50304" in action.help
        assert "100277" not in action.help
