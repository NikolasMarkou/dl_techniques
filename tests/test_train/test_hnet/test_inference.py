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
3b. **The three contracts that were unguarded until review pass 2** (D-032):
   the next byte is drawn from the LAST position's logits, ``--temperature``
   changes the distribution rather than merely being accepted, and
   ``--context-bytes`` keeps the TAIL of the context. All three survived a
   total-neutering mutation with 219 tests green, because the only model stub
   in this file is position-blind and the only window guard asserted shapes.
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
    """A model stub that records the shapes AND the contents it was called with.

    Returns logits that make byte id ``65`` ('A') the argmax at every position,
    so a greedy continuation is exactly predictable and any deviation is a
    defect rather than a coin flip.

    **This fixture is position-BLIND on purpose, and that has a cost.** Because
    its argmax is the same at every position, no assertion made against it can
    say anything about WHICH logit row ``generate_bytes`` reads --
    ``[0, -1, :]`` and ``[0, 0, :]`` produce identical output here. That blind
    spot was measured (review pass 2, S-6: the wrong row leaves 219 tests
    green), and :class:`PositionalModel` below exists to cover it. Keep the two
    separate: making THIS stub position-dependent would make every greedy
    assertion in the file depend on the window length as well.

    ``windows`` records a COPY of each input array, which is what lets a guard
    assert the sliding window's CONTENT rather than only its shape -- head
    slicing and tail slicing produce the same shapes forever.
    """

    def __init__(self, vocab_size: int = 256, favourite: int = 65):
        self.calls: list = []
        self.windows: list = []
        self.vocab_size = vocab_size
        self.favourite = favourite

    def __call__(self, inputs, training=None):
        array = np.asarray(inputs)
        self.calls.append((array.shape, training))
        self.windows.append(array.copy())
        logits = np.zeros(array.shape + (self.vocab_size,), dtype="float32")
        logits[..., self.favourite] = 10.0
        return logits


#: Base id for :class:`PositionalModel`. Chosen so ``BASE + position`` stays a
#: printable ASCII byte for every window this file uses and never collides with
#: :class:`CountingModel`'s 65.
POSITIONAL_BASE = 100


class PositionalModel:
    """A stub whose argmax DEPENDS on the position, so the row read is visible.

    Position ``t`` of the returned ``(1, L, vocab)`` logits puts its peak at
    ``POSITIONAL_BASE + t``. Reading the last row therefore yields
    ``POSITIONAL_BASE + L - 1`` and reading the first yields
    ``POSITIONAL_BASE`` -- two different bytes for every ``L > 1``, which is
    exactly the discrimination :class:`CountingModel` cannot provide.

    The autoregressive contract is that the next byte is drawn from the LAST
    position's logits, because ``pack_byte_windows`` trains position ``t`` to
    predict byte ``t + 1`` (``datasets/byte_lm.py``): the last row is the only
    one whose prediction is not already known.
    """

    def __init__(self, vocab_size: int = 256, base: int = POSITIONAL_BASE):
        self.calls: list = []
        self.vocab_size = vocab_size
        self.base = base

    def __call__(self, inputs, training=None):
        array = np.asarray(inputs)
        self.calls.append((array.shape, training))
        length = array.shape[1]
        logits = np.zeros(array.shape + (self.vocab_size,), dtype="float32")
        for position in range(length):
            logits[:, position, (self.base + position) % self.vocab_size] = 10.0
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
# 3b. The three sampler contracts that were unguarded until review pass 2
# ---------------------------------------------------------------------
#
# Every guard below was written against a MEASURED surviving mutation, not
# against a hypothesis. `decisions.md` D-032 carries the RED table.


class TestTheSamplerReadsTheLastPosition:
    """The autoregressive contract: the next byte comes from the LAST row.

    Reading ``[0, 0, :]`` instead of ``[0, -1, :]`` left the whole trainer
    suite at 219 passed (review pass 2, S-6). Two independent reasons the
    existing evidence could not see it, both defeated here:

    1. :class:`CountingModel` is position-blind by construction, so every
       greedy assertion in :class:`TestGenerateBytes` is satisfied by ANY row.
       :class:`PositionalModel` fixes that, and the first arm below proves the
       fixture really does discriminate before anything is asserted with it.
    2. The 170k-parameter step-18 checkpoint emits ``0x20`` from every
       position, so the two hand invocations D-031 records are byte-identical
       under the wrong row. The REAL-model arm below therefore asserts against
       a model whose per-position argmaxes are MEASURED to differ, and says so
       in its own assertion rather than assuming it.
    """

    def test_the_positional_fixture_can_actually_discriminate(self):
        """ANTI-VACUITY, and it comes first for a reason.

        A fixture that is uniform along the axis under test makes every
        assertion on that axis vacuous -- which is precisely how the defect
        this class exists for survived. So: measure the fixture.
        """
        model = PositionalModel()
        logits = np.asarray(model(np.zeros((1, 5), dtype="int32"), training=False))

        by_position = [int(np.argmax(logits[0, t])) for t in range(5)]
        assert by_position == [POSITIONAL_BASE + t for t in range(5)]
        assert len(set(by_position)) == 5, (
            "the fixture must differ ACROSS positions or the guards below "
            "cannot tell one row from another"
        )
        assert by_position[0] != by_position[-1]

    def test_greedy_generation_reads_the_LAST_row_not_the_first(self):
        """MAIN. RED against ``[0, -1, :]`` -> ``[0, 0, :]``.

        Prompt ``b"abc"`` is 3 bytes, so the first call sees ``L = 3`` and the
        last row's peak is ``POSITIONAL_BASE + 2``. Each sampled byte grows the
        context by one, so the window lengths are 3, 4, 5, 6 and the produced
        ids are ``BASE + 2, BASE + 3, BASE + 4, BASE + 5``. Reading row 0 would
        give ``BASE`` four times over -- a constant, and a different constant.
        """
        model = PositionalModel()
        produced = generate_bytes(
            model, list(b"abc"), max_new_bytes=4, temperature=0.0
        )

        assert produced == [POSITIONAL_BASE + k for k in (2, 3, 4, 5)], produced
        # Stated separately so the failure message names the defect: the
        # position-0 read yields this instead.
        assert produced != [POSITIONAL_BASE] * 4

    def test_the_windowed_read_is_the_last_row_of_the_WINDOW(self):
        """The row index is relative to the window, not to the full context.

        With ``context_bytes=3`` every call sees ``L = 3``, so the last row is
        always ``BASE + 2`` and the output is a constant -- but a DIFFERENT
        constant from the position-0 read's ``BASE``.
        """
        model = PositionalModel()
        produced = generate_bytes(
            model, list(b"hello"), max_new_bytes=3, temperature=0.0, context_bytes=3
        )
        assert produced == [POSITIONAL_BASE + 2] * 3, produced

    @pytest.mark.parametrize("prompt", [b"The ", b"hello world"])
    def test_on_a_REAL_HNet_the_first_byte_is_the_argmax_of_the_LAST_row(
        self, tiny_model, prompt
    ):
        """The same contract on a real forward pass, not a stub.

        MEASURED on this fixture (seed 19, CPU): ``b"The "`` gives per-position
        argmaxes ``[228, 69, 68, 69]`` and ``b"hello world"`` gives
        ``[107, ..., 15]``; the row-0 and row-(-1) logit vectors differ by
        0.30 and 0.44 respectively. The first assertion re-derives that
        difference at run time, so if the fixture ever became position-uniform
        this guard would FAIL rather than quietly stop testing anything.
        """
        ids = np.asarray([list(prompt)], dtype="int32")
        logits = np.asarray(tiny_model(ids, training=False))
        first_row = int(np.argmax(logits[0, 0]))
        last_row = int(np.argmax(logits[0, -1]))

        assert first_row != last_row, (
            f"vacuous fixture: rows 0 and -1 both peak at {last_row}; this "
            "guard can no longer distinguish the two reads"
        )

        produced = generate_bytes(
            tiny_model, list(prompt), max_new_bytes=1, temperature=0.0
        )
        assert produced == [last_row]


class TestTemperatureIsApplied:
    """``--temperature`` must change the DISTRIBUTION, not just be accepted.

    Neutering the division (``values / temperature`` -> ``values * 1.0``) left
    this module at 29 passed (review pass 2, S-7): every existing arm used
    either ``0.0`` (which short-circuits before the division) or ``1.0`` (for
    which the division is the identity). Both guards below therefore compare
    ACROSS temperatures on one fixed logit vector.
    """

    #: One id 4 nats above 255 flat rivals. Analytic argmax mass:
    #: 1.0 at T = 0.02, 0.17635 at T = 1.0, 0.00423 at T = 50.0.
    PEAK_ID = 7
    PEAK_LOGIT = 4.0

    def _logits(self) -> np.ndarray:
        logits = np.zeros(256, dtype="float32")
        logits[self.PEAK_ID] = self.PEAK_LOGIT
        return logits

    def _argmax_share(self, temperature: float, draws: int = 400) -> float:
        rng = np.random.default_rng(3)
        sampled = [
            sample_next_id(self._logits(), temperature=temperature, rng=rng)
            for _ in range(draws)
        ]
        return sampled.count(self.PEAK_ID) / draws

    def test_a_low_temperature_converges_to_greedy(self):
        """MAIN (low end). At T = 0.02 every draw is the argmax.

        MEASURED: 400/400 at T = 0.02. Under a temperature-ignoring sampler
        this reads 0.1575, because the effective temperature is 1.0.
        """
        assert self._argmax_share(0.02) == 1.0

    def test_a_moderate_temperature_does_NOT_converge(self):
        """TWIN. Without this, the arm above could be measuring a flat fixture.

        MEASURED: 0.1575 at T = 1.0 against an analytic 0.17635.
        """
        share = self._argmax_share(1.0)
        assert 0.05 < share < 0.35, share

    def test_a_high_temperature_flattens_the_distribution(self):
        """MAIN (high end). At T = 50 the peak is worth barely more than uniform.

        MEASURED: 0.0050 at T = 50.0 against a uniform 1/256 = 0.0039 and an
        analytic 0.00423. A temperature-ignoring sampler reads 0.1575 here.
        """
        assert self._argmax_share(50.0) < 0.02

    def test_the_argmax_share_falls_MONOTONICALLY_as_temperature_rises(self):
        """The three arms above as one ordering, so no single band can drift.

        This is the assertion a "spelling check" guard cannot make: accepting
        the argument proves nothing about the ordering of its effects.
        """
        shares = [self._argmax_share(t) for t in (0.02, 1.0, 50.0)]
        assert shares[0] > shares[1] > shares[2], shares


class TestContextBytesKeepsTheTail:
    """``--context-bytes`` must retain the MOST RECENT bytes.

    Keeping the head (``context[:context_bytes]``) also left this module at
    29 passed (review pass 2, S-8), because
    ``test_context_bytes_caps_the_window`` asserts only the shapes
    ``[(1,4), (1,4), (1,4)]`` -- which head slicing reproduces exactly. It is a
    live degeneracy, not a cosmetic one: once the cap is reached the model is
    re-fed the SAME frozen window forever and generation stops conditioning on
    what it just produced. These guards assert the window's CONTENT.
    """

    def test_the_window_is_the_TAIL_of_the_context(self):
        """MAIN. Hand-derived windows, byte for byte.

        Prompt ``b"hello"`` (``[104, 101, 108, 108, 111]``) at ``cap = 4`` with
        :class:`CountingModel`'s constant 65::

            step 1: "ello"          -> [101, 108, 108, 111]
            step 2: "llo" + 'A'     -> [108, 108, 111,  65]
            step 3: "lo" + 'AA'     -> [108, 111,  65,  65]

        Head slicing gives ``[104, 101, 108, 108]`` three times instead.
        """
        model = CountingModel()
        generate_bytes(
            model, list(b"hello"), max_new_bytes=3, temperature=0.0, context_bytes=4
        )

        windows = [w[0].tolist() for w in model.windows]
        assert windows == [
            [101, 108, 108, 111],
            [108, 108, 111, 65],
            [108, 111, 65, 65],
        ], windows

    def test_the_window_KEEPS_ADVANCING_past_the_cap(self):
        """The degeneracy stated as a property rather than as a literal.

        Generating more bytes than the cap must eventually flush the prompt out
        of the window entirely. A frozen window never does, whatever its
        length: it is the same array on every call.
        """
        model = CountingModel()
        generate_bytes(
            model, list(b"hello"), max_new_bytes=6, temperature=0.0, context_bytes=4
        )

        windows = [w[0].tolist() for w in model.windows]
        assert windows[-1] == [65, 65, 65, 65], windows[-1]
        assert len({tuple(w) for w in windows}) > 1, (
            "the window never changed: generation stopped conditioning on the "
            "bytes it produced"
        )
        # And each window after the first ends with the byte just sampled.
        for previous, current in zip(windows, windows[1:]):
            assert current[-1] == 65, (previous, current)

    def test_an_uncapped_context_keeps_everything(self):
        """TWIN: the tail slice is the CAP's doing, not the loop's."""
        model = CountingModel()
        generate_bytes(model, list(b"hello"), max_new_bytes=2, temperature=0.0)

        windows = [w[0].tolist() for w in model.windows]
        assert windows == [
            [104, 101, 108, 108, 111],
            [104, 101, 108, 108, 111, 65],
        ], windows


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
