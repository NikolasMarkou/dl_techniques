"""Tests for the compositional / QA benchmark generators under ``src/train/ntm/``.

Both generator modules shipped with zero test coverage, which is why three of
the five ``ScanSplit`` members could be degenerate or unimplemented without any
signal. These tests pin the corpus itself, so a grammar change can never again
move every split silently.

The SCAN group covers three defects that shared one symptom (an empty or
aliased test set) and had three independent causes:

* the conjunction enumerator never nested ``turn <dir>``, so the primitive
  ``add_prim_turn_left`` claims to hold out was unreachable in any compound and
  the split returned 254/0;
* ``_add_primitive_split`` matched the multiword primitive ``"turn left"`` on
  its first token alone, so every ``turn right`` command was routed as if it
  contained the held-out primitive;
* ``_length_split`` hardcoded the SCAN paper's absolute boundary (24 action
  tokens) onto a grammar whose sequences reach 8, putting the boundary above
  every sample.

Every count asserted below was MEASURED against the shipped code, not predicted.
"""

import collections
from typing import Dict, List

import pytest

from train.ntm.compositional_generators import (
    ScanGenerator,
    ScanSample,
    ScanSplit,
    _COMPOUND_POOL_SIZE,
    _SCAN_PAPER_CORPUS_MAX_ACTION_LENGTH,
    _SCAN_PAPER_TRAIN_MAX_ACTION_LENGTH,
)
from train.ntm.config import ScanTaskConfig


# ---------------------------------------------------------------------
# Measured constants (re-derive, never inherit)
# ---------------------------------------------------------------------

# Measured by executing ``ScanGenerator(ScanTaskConfig()).generate_all_samples()``
# against the shipped grammar.
MEASURED_CORPUS_SIZE = 446
MEASURED_ACTION_LENGTH_HISTOGRAM = {1: 6, 2: 84, 3: 204, 4: 136, 6: 8, 8: 8}
MEASURED_MAX_ACTION_LENGTH = 8

# Measured per-split (train, test) sizes.
MEASURED_SPLIT_SIZES = {
    "simple": (356, 90),
    "length": (294, 152),
    "add_prim_jump": (288, 158),
    "add_prim_turn_left": (392, 54),
}


def _generator(split_type: str = "simple") -> ScanGenerator:
    """Build a generator at the pinned seed.

    :param split_type: Split identifier forwarded to the config.
    :return: A freshly seeded generator.
    """
    return ScanGenerator(ScanTaskConfig(split_type=split_type))


def _histogram(samples: List[ScanSample]) -> Dict[int, int]:
    """Action-length histogram of a sample list.

    :param samples: Samples to summarise.
    :return: Mapping of action length to sample count, ascending.
    """
    counts = collections.Counter(len(s.action_tokens) for s in samples)
    return dict(sorted(counts.items()))


def _fake_samples(action_lengths: List[int]) -> List[ScanSample]:
    """Build throwaway samples carrying only the given action lengths.

    :param action_lengths: One action length per synthetic sample.
    :return: Samples whose only meaningful field is ``action_tokens``.
    """
    return [
        ScanSample(
            command="walk",
            actions=" ".join(["WALK"] * n),
            command_tokens=["walk"],
            action_tokens=["WALK"] * n,
        )
        for n in action_lengths
    ]


# ---------------------------------------------------------------------
# SCAN corpus
# ---------------------------------------------------------------------


class TestScanCorpus:
    """Pins the enumerated grammar so a change cannot move splits silently."""

    def test_corpus_size_and_action_length_histogram_are_the_measured_ones(self):
        samples = _generator().generate_all_samples()

        assert len(samples) == MEASURED_CORPUS_SIZE
        assert _histogram(samples) == MEASURED_ACTION_LENGTH_HISTOGRAM

    def test_corpus_fits_the_configured_sequence_bounds(self):
        config = ScanTaskConfig()
        samples = _generator().generate_all_samples()

        assert max(len(s.command_tokens) for s in samples) <= config.max_input_length
        assert max(len(s.action_tokens) for s in samples) <= config.max_output_length


class TestCompoundPoolComposesTurnCommands:
    """Guard for fix (a): ``turn <dir>`` must be reachable inside a compound."""

    def test_turn_left_appears_inside_a_compound_command(self):
        samples = _generator().generate_all_samples()

        compounds_with_turn_left = [
            s for s in samples
            if len(s.command_tokens) > 2
            and ScanGenerator._contains_phrase(s.command_tokens, "turn left")
        ]

        assert compounds_with_turn_left, (
            "no compound command nests 'turn left'; the add_prim_turn_left "
            "split can only be degenerate"
        )

    def test_pool_size_constant_bounds_the_whole_atomic_command_set(self):
        atomic_commands = (
            len(ScanGenerator.PRIMITIVES)
            + len(ScanGenerator.DIRECTIONS)
            + len(ScanGenerator.PRIMITIVES) * len(ScanGenerator.DIRECTIONS)
        )

        assert _COMPOUND_POOL_SIZE == atomic_commands


class TestMultiwordPrimitivePredicate:
    """Guard for fix (b): the held-out phrase is a token subsequence."""

    def test_contains_phrase_is_token_aware_not_substring(self):
        assert ScanGenerator._contains_phrase(["turn", "left"], "turn left")
        assert ScanGenerator._contains_phrase(
            ["walk", "and", "turn", "left"], "turn left"
        )
        assert not ScanGenerator._contains_phrase(["turn", "lefts"], "turn left")
        assert not ScanGenerator._contains_phrase(["turn", "right"], "turn left")
        assert not ScanGenerator._contains_phrase(["left", "turn"], "turn left")

    def test_every_held_out_sample_actually_contains_turn_left(self):
        _, test = _generator("add_prim_turn_left").generate_split()

        misrouted = [
            s.command for s in test
            if not ScanGenerator._contains_phrase(s.command_tokens, "turn left")
        ]

        assert misrouted == [], (
            f"{len(misrouted)} commands were held out without containing "
            f"'turn left', e.g. {misrouted[:3]}"
        )

    def test_turn_right_compounds_stay_in_train(self):
        train, _ = _generator("add_prim_turn_left").generate_split()

        turn_right_only = [
            s for s in train
            if ScanGenerator._contains_phrase(s.command_tokens, "turn right")
            and not ScanGenerator._contains_phrase(s.command_tokens, "turn left")
        ]

        assert len(turn_right_only) > 1, (
            "commands containing only 'turn right' were routed away from train"
        )


class TestLengthThresholdDerivation:
    """Guard for fix (c): the boundary tracks the corpus, not a constant."""

    def test_threshold_differs_from_the_old_hardcoded_constant(self):
        samples = _generator().generate_all_samples()

        threshold = ScanGenerator._derive_length_threshold(samples)

        assert threshold != 24

    def test_threshold_sits_strictly_inside_the_measured_distribution(self):
        samples = _generator().generate_all_samples()
        lengths = [len(s.action_tokens) for s in samples]

        threshold = ScanGenerator._derive_length_threshold(samples)

        assert min(lengths) <= threshold < max(lengths), (
            f"threshold {threshold} is outside [{min(lengths)}, "
            f"{max(lengths)}) so one side of the split must be empty"
        )

    def test_threshold_adapts_to_the_corpus_it_is_given(self):
        shallow = ScanGenerator._derive_length_threshold(
            _fake_samples([1, MEASURED_MAX_ACTION_LENGTH])
        )
        paper_scale = ScanGenerator._derive_length_threshold(
            _fake_samples([1, _SCAN_PAPER_CORPUS_MAX_ACTION_LENGTH])
        )

        assert shallow != paper_scale, "threshold ignored the corpus it was given"
        # At the paper's own scale the derivation reproduces the paper's boundary.
        assert paper_scale == _SCAN_PAPER_TRAIN_MAX_ACTION_LENGTH
        assert shallow == MEASURED_MAX_ACTION_LENGTH * paper_scale // (
            _SCAN_PAPER_CORPUS_MAX_ACTION_LENGTH
        )

    def test_length_split_holds_out_only_the_long_sequences(self):
        samples = _generator().generate_all_samples()
        threshold = ScanGenerator._derive_length_threshold(samples)

        train, test = _generator("length").generate_split()

        assert all(len(s.action_tokens) <= threshold for s in train)
        assert all(len(s.action_tokens) > threshold for s in test)


@pytest.mark.parametrize(
    "split_type,expected",
    sorted(MEASURED_SPLIT_SIZES.items()),
)
def test_measured_split_sizes(split_type: str, expected):
    """Pins each split's measured (train, test) sizes."""
    train, test = _generator(split_type).generate_split()

    assert (len(train), len(test)) == expected
