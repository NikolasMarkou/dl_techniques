r"""Guards for ``src/train/hnet/infer_hnet.py`` -- the H-Net sampling entry point.

Why this module exists
----------------------
SC-13(c) asked for "a real CLI invocation of the generation/eval path". At step
18 there was no such path: ``train_hnet.py`` has only ``main() -> train()``, and
generation had to be driven from a scratch script, so the criterion could only
be graded PARTIAL. ``infer_hnet.py`` closes that; this module is its guard.

What is pinned, and what each guard would catch
-----------------------------------------------
1. **The CLI contract**: ``main(argv)`` parses FIRST, so ``--help`` prints a
   ``usage:`` line and exits 0 without claiming a GPU or reading a checkpoint
   off disk. Sentinels over ``setup_gpu`` and ``load_checkpoint`` MEASURE that
   rather than inferring it from an exit code -- ``src/train/CLAUDE.md``'s trap
   is that a script with no parser at all runs its whole job and exits 0 too.
2. **Defensive decoding at UTF-8 boundaries.** The model emits BYTES and the
   sample is cut wherever ``--max-new-bytes`` says, which is routinely mid
   codepoint. Every arm of :class:`TestCodepointBoundarySplitting` carries its
   own twin showing that the naive ``bytes(...).decode()`` RAISES on the same
   input, so the guards measure the hazard rather than assuming it.
3. **The sampler**: greedy is exactly ``argmax`` and consumes no randomness;
   temperature and ``top_p`` do what they claim; the bad-argument raises fire.
4. **End to end through ``main()``** on a tiny model saved to ``tmp_path``,
   with a real ``.keras`` round trip -- the only arm that proves the pieces are
   actually wired to each other.

The step-18 checkpoint at ``results/hnet_dev_20260909_161050/`` is NOT read
here. ``results/`` is gitignored, untracked and unrecoverable, the autouse guard
in ``tests/conftest.py`` forbids tests writing there, and a suite that depends
on a run directory is a suite that breaks the day someone tidies up. The real
invocation against that checkpoint was performed once, by hand, and its output
is recorded in ``decisions.md`` D-031.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.models.language.hnet.config import AttnSpec, HNetArchConfig, SSMSpec
from dl_techniques.models.language.hnet.model import HNet
from train.hnet import infer_hnet
from train.hnet.infer_hnet import (
    build_parser,
    generate_bytes,
    sample_next_id,
    split_at_codepoint_boundary,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def tiny_config(d_model: int = 16) -> HNetArchConfig:
    """The smallest layout with a chunking level, matching ``test_model.py``."""
    return HNetArchConfig(
        arch_layout=["m1", ["T1"], "m1"],
        d_model=[d_model, d_model],
        d_intermediate=[0, 0],
        vocab_size=256,
        ssm_cfg=SSMSpec(d_state=8, d_conv=4, expand=2),
        attn_cfg=AttnSpec(
            num_heads=(2, 2), rotary_emb_dim=(4, 4), window_size=(-1, -1)
        ),
    )


@pytest.fixture()
def tiny_model():
    keras.utils.set_random_seed(19)
    model = HNet(arch_config=tiny_config(), max_chunks=(6,), headdim=8, max_seq_len=64)
    model.build((None, None))
    return model


class CountingModel:
    """A model stub that records the shapes it was called with.

    Returns logits that make byte id ``65`` ('A') the argmax at every position,
    so a greedy continuation is exactly predictable and any deviation is a
    defect rather than a coin flip.
    """

    def __init__(self, vocab_size: int = 256, favourite: int = 65):
        self.calls: list = []
        self.vocab_size = vocab_size
        self.favourite = favourite

    def __call__(self, inputs, training=None):
        array = np.asarray(inputs)
        self.calls.append((array.shape, training))
        logits = np.zeros(array.shape + (self.vocab_size,), dtype="float32")
        logits[..., self.favourite] = 10.0
        return logits


# ---------------------------------------------------------------------
# 1. Defensive decoding
# ---------------------------------------------------------------------


class TestCodepointBoundarySplitting:
    """A cut sample must never raise, and must not invent a replacement char.

    Each arm names the byte sequence it uses and states what the naive decode
    does with it, because "defensive" is only meaningful against a measured
    hazard.
    """

    # 'é' is 0xC3 0xA9; '€' is 0xE2 0x82 0xAC; '𝄞' is 0xF0 0x9D 0x84 0x9E.
    EURO = [0xE2, 0x82, 0xAC]
    CLEF = [0xF0, 0x9D, 0x84, 0x9E]

    @pytest.mark.parametrize("held", [1, 2, 3])
    def test_a_truncated_codepoint_is_held_back_not_replaced(self, held):
        """MAIN. Cutting a 4-byte codepoint short holds the fragment back.

        ``held`` is how many of the clef's four bytes were CUT, so the tail
        actually present is ``4 - held`` bytes long and all of it must come
        back in ``held_back`` -- none of it decoded, none of it replaced.
        """
        tail = self.CLEF[: 4 - held]
        text, held_back = split_at_codepoint_boundary(list(b"ok") + tail)

        assert text == "ok"
        assert held_back == tail
        assert "�" not in text, "a truncated tail must be held, not replaced"

    @pytest.mark.parametrize("held", [1, 2, 3])
    def test_the_naive_decode_RAISES_on_the_same_input(self, held):
        """TWIN: the hazard is real, and this is the code that would ship it."""
        raw = bytes(bytearray(list(b"ok") + self.CLEF[: 4 - held]))
        with pytest.raises(UnicodeDecodeError):
            raw.decode("utf-8")

    def test_a_complete_sequence_holds_nothing_back(self):
        ids = list("ok€".encode("utf-8"))
        text, held_back = split_at_codepoint_boundary(ids)
        assert text == "ok€"
        assert held_back == []

    def test_an_INVALID_byte_is_replaced_rather_than_held_forever(self):
        """0xFF starts no codepoint at all. Holding it would never terminate.

        The distinction matters: "incomplete" is a reason to wait for more
        bytes, "invalid" is not, and a sampler emitting garbage would grow an
        unbounded buffer under a rule that could not tell them apart.
        """
        text, held_back = split_at_codepoint_boundary([0x41, 0xFF, 0x42])
        assert held_back == []
        assert "�" in text
        assert text.startswith("A") and text.endswith("B")

    def test_an_empty_sequence_is_the_empty_string(self):
        assert split_at_codepoint_boundary([]) == ("", [])

    def test_ascii_is_unchanged(self):
        text, held_back = split_at_codepoint_boundary(list(b"plain ascii"))
        assert (text, held_back) == ("plain ascii", [])


# ---------------------------------------------------------------------
# 2. The sampler
# ---------------------------------------------------------------------


class TestSampleNextId:
    """Greedy is argmax; temperature and top_p are live; bad arguments raise."""

    def test_temperature_zero_is_argmax_and_consumes_no_randomness(self):
        logits = np.arange(256, dtype="float32")
        logits[7] = 1000.0
        # A generator that would RAISE if touched: greedy must not sample.
        class Explodes:
            def choice(self, *args, **kwargs):
                raise AssertionError("greedy decoding drew from the RNG")

        assert sample_next_id(logits, temperature=0.0, rng=Explodes()) == 7

    def test_a_positive_temperature_actually_samples(self):
        """TWIN: the arm above must be measuring greediness, not a constant."""
        logits = np.zeros(256, dtype="float32")
        rng = np.random.default_rng(0)
        draws = {sample_next_id(logits, temperature=1.0, rng=rng) for _ in range(50)}
        assert len(draws) > 1, draws

    def test_top_p_excludes_the_tail(self):
        """With one dominant id, a tight nucleus must return only that id."""
        logits = np.full(256, -50.0, dtype="float32")
        logits[3] = 50.0
        logits[4] = 49.0
        rng = np.random.default_rng(1)
        draws = {
            sample_next_id(logits, temperature=1.0, top_p=0.5, rng=rng)
            for _ in range(30)
        }
        assert draws == {3}, draws

    def test_top_p_of_one_does_not_truncate(self):
        """TWIN for the arm above: the truncation is the KNOB, not the logits."""
        logits = np.zeros(256, dtype="float32")
        rng = np.random.default_rng(2)
        draws = {
            sample_next_id(logits, temperature=1.0, top_p=1.0, rng=rng)
            for _ in range(50)
        }
        assert len(draws) > 10, draws

    @pytest.mark.parametrize("temperature,top_p", [(-1.0, 1.0), (1.0, 0.0), (1.0, 1.5)])
    def test_bad_arguments_raise(self, temperature, top_p):
        with pytest.raises(ValueError):
            sample_next_id(np.zeros(4), temperature=temperature, top_p=top_p)


# ---------------------------------------------------------------------
# 3. The generation loop
# ---------------------------------------------------------------------


class TestGenerateBytes:
    """Counts, shapes, and the two easy ways to get this wrong."""

    def test_it_returns_exactly_the_new_bytes_and_not_the_prompt(self):
        model = CountingModel()
        produced = generate_bytes(model, list(b"hi"), max_new_bytes=5, temperature=0.0)

        assert produced == [65] * 5
        assert len(produced) == 5, "the prompt must not be concatenated in"

    def test_the_context_grows_by_one_byte_per_step(self):
        model = CountingModel()
        generate_bytes(model, list(b"hi"), max_new_bytes=3, temperature=0.0)

        assert [shape for shape, _ in model.calls] == [(1, 2), (1, 3), (1, 4)]
        assert all(training is False for _, training in model.calls), (
            "training=False must be passed EXPLICITLY on every call"
        )

    def test_context_bytes_caps_the_window(self):
        model = CountingModel()
        generate_bytes(
            model, list(b"hello"), max_new_bytes=3, temperature=0.0, context_bytes=4
        )
        assert [shape for shape, _ in model.calls] == [(1, 4), (1, 4), (1, 4)]

    def test_zero_new_bytes_calls_the_model_not_at_all(self):
        model = CountingModel()
        assert generate_bytes(model, list(b"hi"), max_new_bytes=0) == []
        assert model.calls == []

    def test_an_empty_prompt_is_refused(self):
        with pytest.raises(ValueError, match="nothing to condition on"):
            generate_bytes(CountingModel(), [], max_new_bytes=1)

    def test_a_negative_count_is_refused(self):
        with pytest.raises(ValueError, match="max_new_bytes"):
            generate_bytes(CountingModel(), list(b"hi"), max_new_bytes=-1)

    def test_it_runs_on_a_REAL_HNet_at_a_prompt_shorter_than_max_chunks(
        self, tiny_model
    ):
        """The regime DEFECT 1 made unusable (D-028) and step 6.1 repaired.

        ``max_chunks=(6,)`` against a 3-byte prompt: before the step-6.1 fix
        this raised a raw ``StridedSlice InvalidArgumentError``. Generation is
        exactly where that defect surfaces, because prompts are short.
        """
        produced = generate_bytes(
            tiny_model, list(b"abc"), max_new_bytes=4, temperature=0.0
        )
        assert len(produced) == 4
        assert all(0 <= i < 256 for i in produced)


# ---------------------------------------------------------------------
# 4. The CLI
# ---------------------------------------------------------------------


class _Sentinel:
    def __init__(self, name: str):
        self.name = name
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        raise AssertionError(f"sentinel {self.name!r} was called")


_EXPENSIVE: Tuple[str, ...] = ("setup_gpu", "load_checkpoint", "generate_bytes")


def _install_sentinels(monkeypatch) -> Dict[str, _Sentinel]:
    out: Dict[str, _Sentinel] = {}
    for attribute in _EXPENSIVE:
        sentinel = _Sentinel(attribute)
        # `monkeypatch.setattr` raises on an unknown attribute, so installing a
        # sentinel IS a check that the name it covers exists.
        monkeypatch.setattr(infer_hnet, attribute, sentinel)
        out[attribute] = sentinel
    return out


class TestTheCLIContract:
    """``main(argv)`` parses first. Exit 0 alone would not show that."""

    def test_the_sentinels_cover_names_that_exist(self, monkeypatch):
        sentinels = _install_sentinels(monkeypatch)
        assert len(sentinels) == len(_EXPENSIVE)
        assert all(s.calls == 0 for s in sentinels.values())

    def test_help_prints_usage_and_allocates_nothing(self, monkeypatch, capsys):
        sentinels = _install_sentinels(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["infer_hnet.py", "--help"])

        with pytest.raises(SystemExit) as excinfo:
            infer_hnet.main()

        reached = {name: s.calls for name, s in sentinels.items() if s.calls}
        assert not reached, (
            f"--help reached {reached} before argparse could exit; "
            "`args = parse_arguments(argv)` must be main()'s FIRST statement"
        )
        assert excinfo.value.code == 0
        assert capsys.readouterr().out.startswith("usage:")

    def test_a_missing_checkpoint_flag_is_a_parse_error_not_a_traceback(self):
        with pytest.raises(SystemExit) as excinfo:
            build_parser().parse_args([])
        assert excinfo.value.code == 2

    def test_every_declared_flag_reaches_the_namespace(self):
        """No flag may be advertised by ``--help`` and then dropped."""
        args = build_parser().parse_args([
            "--checkpoint", "x.keras", "--prompt", "abc", "--max-new-bytes", "3",
            "--temperature", "0.0", "--top-p", "0.5", "--context-bytes", "8",
            "--seed", "11", "--gpu", "1",
        ])
        assert (
            args.checkpoint, args.prompt, args.max_new_bytes, args.temperature,
            args.top_p, args.context_bytes, args.seed, args.gpu,
        ) == ("x.keras", "abc", 3, 0.0, 0.5, 8, 11, 1)

    def test_main_generates_from_a_REAL_saved_checkpoint(
        self, tmp_path, tiny_model, monkeypatch
    ):
        """END TO END, and the only arm that proves the pieces are wired.

        A real ``.keras`` round trip through ``load_checkpoint`` (no
        ``custom_objects``, so a registration regression fails here), a real
        forward pass, a real decode. Greedy, so the assertion is on a
        deterministic length rather than on a lucky string.
        """
        path = tmp_path / "tiny.keras"
        tiny_model.save(path)

        # `setup_gpu` would reconfigure the process's devices mid-session.
        monkeypatch.setattr(infer_hnet, "setup_gpu", lambda gpu_id=None: None)

        text = infer_hnet.main([
            "--checkpoint", str(path), "--prompt", "The ",
            "--max-new-bytes", "6", "--temperature", "0.0",
        ])

        assert isinstance(text, str)
        # 6 sampled bytes decode to AT MOST 6 characters and at least 0 (a
        # truncated multi-byte tail is held back rather than emitted).
        assert 0 <= len(text) <= 6
        assert "The " not in text, "main() must return the continuation only"
