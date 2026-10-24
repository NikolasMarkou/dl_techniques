"""The stimulus framing needs the encoder API; there is no call-only fallback.

plan-2026-10-23T190557-972a9ac5 step 11.3 (D-031 f). Until now a preprocessor that
could only be CALLED (no ``.tokenizer.encode`` and ``.tokenizer.eot_token``), or
``context=None``, fell back to the call-only framing: whatever the preprocessor's
``__call__`` returned (``[CLS] text [SEP] [PAD]...``), with the special ids left out
of the pooling. That is the framing the model never sees in training and the one
that was found to manufacture significant units; D-028 made it loud instead of
removing it. It is removed: :func:`train.topolm.common._encode_stimuli` and
everything above it (``extract_tap_activations``, ``evaluate_topography``,
``stimulus_summary``, the controls) require the encoder API and raise
``TypeError``, naming what is missing, before any forward pass.

The fakes below implement only the encoder surface (and, for the refusals, less of
it). They are independent of the trainer test module's ``StubTokenizer``.

CPU only (``CUDA_VISIBLE_DEVICES=``).
"""

import numpy as np
import pytest

import keras

from train.topolm import (
    build_backbone,
    evaluate_topography,
    extract_tap_activations,
)
from train.topolm import common as C
from train.topolm import controls as controls_module
from train.topolm import TopoLMTrainingConfig

VOCAB = 300
EOT = 299


class Encoder:
    """The encoder surface the framing reads: ``encode`` and ``eot_token``."""

    eot_token = EOT

    def encode(self, text):
        return [(sum(map(ord, word)) * 7) % (VOCAB - 1) for word in str(text).split()]


class WithEncoder:
    """The smallest preprocessor the framing accepts: a ``.tokenizer`` and nothing else."""

    tokenizer = Encoder()


class CallOnly:
    """The old stub's shape: callable, CLS/SEP/PAD attributes, no raw encoder."""

    cls_token_id, sep_token_id, pad_token_id = 7, 8, 9

    def __call__(self, texts):
        texts = [texts] if isinstance(texts, str) else texts
        ids = np.full((len(texts), 12), 9, dtype="int32")
        for row, text in enumerate(texts):
            ids[row, :3] = [7, 10 + len(str(text)) % 50, 8]
        return {"input_ids": ids}


def _backbone():
    keras.utils.set_random_seed(0)
    config = TopoLMTrainingConfig(
        model_variant="tiny", vocab_size=VOCAB, max_seq_length=64, num_layers=1,
        spatial_radius=2, spatial_neighborhoods=3,
    )
    return build_backbone(config)


PROMPTS = ["alpha bravo charlie", "delta echo", "foxtrot golf hotel"]


class _NoForward:
    """A backbone-shaped object that fails if a forward pass is attempted."""

    built = True
    max_seq_len = 64
    tap_layers = ()
    grid_shape = (8, 8)
    grid_dim = 2

    def __call__(self, *args, **kwargs):
        raise AssertionError("a forward pass ran before the refusal")


class TestAnEncoderIsRequired:
    def test_the_minimal_encoder_preprocessor_works(self):
        """No ``max_length``, no ``pad_token_id``: pads fall back to the eot id."""
        backbone = _backbone()
        pooled = extract_tap_activations(backbone, PROMPTS, WithEncoder(), context="")
        assert len(pooled) == len(backbone.tap_layers)
        for matrix in pooled.values():
            assert matrix.shape == (len(PROMPTS), backbone.embed_dim)
            assert np.isfinite(matrix).all()

    def test_a_call_only_preprocessor_is_refused_by_name(self):
        with pytest.raises(TypeError) as excinfo:
            extract_tap_activations(_NoForward(), PROMPTS, CallOnly())
        message = str(excinfo.value)
        assert "CallOnly" in message
        assert "tokenizer.encode" in message and "tokenizer.eot_token" in message

    @pytest.mark.parametrize("context", ["", C.STIMULUS_CONTEXT])
    def test_the_refusal_does_not_depend_on_the_context(self, context):
        with pytest.raises(TypeError, match="tokenizer"):
            extract_tap_activations(_NoForward(), PROMPTS, CallOnly(), context=context)

    def test_a_missing_piece_is_named_alone(self):
        class NoEncode(WithEncoder):
            class tokenizer:  # noqa: N801
                eot_token = EOT

        class NoEot(WithEncoder):
            class tokenizer:  # noqa: N801
                @staticmethod
                def encode(text):
                    return [1]

        class NoneEot(WithEncoder):
            class tokenizer:  # noqa: N801
                eot_token = None

                @staticmethod
                def encode(text):
                    return [1]

        class EncodeNotCallable(WithEncoder):
            class tokenizer:  # noqa: N801
                eot_token = EOT
                encode = "not callable"

        for preprocessor, needle in (
            (NoEncode(), "tokenizer.encode"),
            (NoEot(), "tokenizer.eot_token"),
            (NoneEot(), "tokenizer.eot_token"),
            (EncodeNotCallable(), "tokenizer.encode"),
        ):
            with pytest.raises(TypeError) as excinfo:
                extract_tap_activations(_NoForward(), PROMPTS, preprocessor, context="")
            message = str(excinfo.value)
            assert needle in message, (type(preprocessor).__name__, message)
            other = "tokenizer.eot_token" if needle == "tokenizer.encode" else "tokenizer.encode"
            assert other not in message.split("missing:")[1], message

    def test_there_is_no_context_none_fallback(self):
        with pytest.raises(TypeError, match="context must be a string"):
            extract_tap_activations(_NoForward(), PROMPTS, WithEncoder(), context=None)

    def test_the_refusal_comes_before_the_empty_prompt_check_and_any_forward_pass(self):
        """A bad preprocessor is refused even when the prompts would also be refused."""
        with pytest.raises(TypeError):
            extract_tap_activations(_NoForward(), ["", "alpha"], CallOnly())

    def test_the_evaluation_refuses_a_call_only_preprocessor_before_any_forward_pass(self):
        with pytest.raises(TypeError, match="tokenizer.eot_token"):
            evaluate_topography(
                _NoForward(), {"a": PROMPTS, "b": PROMPTS}, ("a", "b"), CallOnly(),
                readout_fwhm=None, min_cluster_size=2,
            )

    def test_the_evaluation_refuses_context_none(self):
        with pytest.raises(TypeError, match="context must be a string"):
            evaluate_topography(
                _NoForward(), {"a": PROMPTS, "b": PROMPTS}, ("a", "b"),
                WithEncoder(), readout_fwhm=None, min_cluster_size=2,
                stimulus_context=None,
            )

    def test_the_summary_refuses_a_call_only_preprocessor(self):
        with pytest.raises(TypeError, match="tokenizer.encode"):
            C.stimulus_summary(CallOnly(), {"a": PROMPTS}, 64)

    def test_the_summary_names_only_the_two_framings_that_remain(self):
        for context, framing in ((C.STIMULUS_CONTEXT, "context+eot"), ("", "eot")):
            block = C.stimulus_summary(WithEncoder(), {"a": PROMPTS}, 400,
                                       context=context, contrast=("a", "a"))
            assert block["framing"] == framing
        assert not hasattr(C, "FRAMING_LEGACY")

    def test_no_warning_machinery_is_left_for_the_removed_path(self):
        for name in ("_warn_legacy_framing", "_LEGACY_FRAMING_WARNED",
                     "_NON_TEXT_ID_ATTRIBUTES"):
            assert not hasattr(C, name), name

    def test_the_framing_table_has_no_call_only_row(self):
        """The 'cls' row measured the removed framing; the table keeps three."""
        backbone = _backbone()
        table = controls_module.framing_table(
            backbone, WithEncoder(),
            stimuli={"a": ["alpha bravo", "charlie delta"],
                     "b": ["echo foxtrot", "golf hotel"]},
            min_cluster_size=2,
        )
        assert list(table) == ["context", "eot", "none"]
        assert [row["framing"] for row in table.values()] == [
            "context+eot", "eot", "none"]

    def test_the_framing_table_refuses_a_call_only_preprocessor(self):
        with pytest.raises(TypeError, match="tokenizer"):
            controls_module.framing_table(_NoForward(), CallOnly(),
                                          stimuli={"a": PROMPTS, "b": PROMPTS})
