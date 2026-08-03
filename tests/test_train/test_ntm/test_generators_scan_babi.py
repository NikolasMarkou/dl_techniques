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
import re
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
from train.ntm.babi_generator import BabiGenerator
from train.ntm.config import BabiTaskConfig, BenchmarkSuiteConfig, ScanTaskConfig
from train.ntm.harness import BenchmarkHarness


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
    "template_around_right": (442, 4),
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


# ---------------------------------------------------------------------
# SCAN dispatch, template hold-out and the non-degeneracy seam
# ---------------------------------------------------------------------


class TestScanSplitDispatch:
    """Every enum member is handled; nothing else is silently accepted."""

    def test_every_enum_member_is_pinned_by_a_measured_size(self):
        assert {member.value for member in ScanSplit} == set(MEASURED_SPLIT_SIZES)

    @pytest.mark.parametrize("member", list(ScanSplit), ids=lambda m: m.value)
    def test_every_enum_member_partitions_non_degenerately(self, member: ScanSplit):
        train, test = _generator(member.value).generate_split()

        assert train, f"{member.value} produced an empty train set"
        assert test, f"{member.value} produced an empty test set"

    @pytest.mark.parametrize("member", list(ScanSplit), ids=lambda m: m.value)
    def test_every_enum_member_partitions_exhaustively_and_disjointly(
        self, member: ScanSplit
    ):
        generator = _generator(member.value)
        corpus = [s.command for s in generator.generate_all_samples()]

        train, test = generator.generate_split()
        train_commands = [s.command for s in train]
        test_commands = [s.command for s in test]

        assert len(train_commands) + len(test_commands) == len(corpus)
        assert sorted(train_commands + test_commands) == sorted(corpus)
        assert set(train_commands).isdisjoint(set(test_commands))

    def test_unknown_split_raises_instead_of_aliasing_to_simple(self):
        generator = _generator("not_a_real_split")

        with pytest.raises(ValueError) as excinfo:
            generator.generate_split()

        message = str(excinfo.value)
        assert "not_a_real_split" in message
        for member in ScanSplit:
            assert member.value in message

    def test_explicit_split_type_argument_overrides_the_config(self):
        generator = _generator("simple")

        train, test = generator.generate_split("template_around_right")

        assert (len(train), len(test)) == MEASURED_SPLIT_SIZES["template_around_right"]


class TestTemplateAroundRightHoldOut:
    """``template_around_right`` is a real hold-out, not an alias for simple."""

    def test_held_out_side_is_exactly_the_around_right_commands(self):
        generator = _generator("template_around_right")
        corpus = generator.generate_all_samples()
        expected = sorted(
            s.command for s in corpus
            if ScanGenerator._contains_phrase(s.command_tokens, "around right")
        )

        _, test = generator.generate_split()

        assert sorted(s.command for s in test) == expected

    def test_no_training_command_contains_the_held_out_template(self):
        train, _ = _generator("template_around_right").generate_split()

        leaked = [
            s.command for s in train
            if ScanGenerator._contains_phrase(s.command_tokens, "around right")
        ]

        assert leaked == []

    def test_it_no_longer_aliases_to_the_simple_split(self):
        simple = _generator("simple").generate_split()
        template = _generator("template_around_right").generate_split()

        assert [len(part) for part in simple] != [len(part) for part in template]


class TestNonDegeneracyValidator:
    """The seam refuses an empty side and says which split and why.

    Also the dead-component probe for the validator: forcing its predicate to
    accept everything makes both tests below go RED, so the validator is
    load-bearing rather than decorative.
    """

    @staticmethod
    def _uniform_length_generator(action_length: int = 4) -> ScanGenerator:
        """A generator whose corpus has a single action length.

        Every sample then sits strictly above the derived LENGTH threshold, so
        the ``length`` split has an empty train side.

        :param action_length: The single action length shared by all samples.
        :return: A generator with a monkeypatched corpus.
        """
        generator = _generator("length")
        corpus = _fake_samples([action_length] * 8)
        generator.generate_all_samples = lambda: corpus
        return generator

    def test_a_degenerate_partition_raises(self):
        generator = self._uniform_length_generator()

        with pytest.raises(ValueError) as excinfo:
            generator.generate_split()

        assert "degenerate" in str(excinfo.value)

    def test_the_refusal_names_the_split_both_sizes_and_the_histogram(self):
        generator = self._uniform_length_generator(action_length=4)

        with pytest.raises(ValueError) as excinfo:
            generator.generate_split()

        message = str(excinfo.value)
        assert "length" in message
        assert "0 train" in message
        assert "8 test" in message
        assert "{4: 8}" in message


# ---------------------------------------------------------------------
# bAbI coverage honesty
# ---------------------------------------------------------------------

# Measured against the shipped generators by calling generate() for each id
# 1..20 and recording which ones raise.
MEASURED_IMPLEMENTED_TASK_IDS = [1, 2, 3, 6, 7, 8, 11, 15, 17, 19]
MEASURED_UNIMPLEMENTED_TASK_IDS = [4, 5, 9, 10, 12, 13, 14, 16, 18, 20]


class TestBabiImplementedTaskIds:
    """One home for "which tasks exist", and a default that matches it."""

    def test_implemented_task_ids_are_exactly_the_ids_that_generate(self):
        generator = BabiGenerator(BabiTaskConfig(task_ids=[1]))

        generated = []
        for task_id in range(1, 21):
            try:
                generator.generate(task_id, num_samples=1)
            except ValueError:
                continue
            generated.append(task_id)

        assert generated == MEASURED_IMPLEMENTED_TASK_IDS
        assert set(generated) == BabiGenerator.IMPLEMENTED_TASK_IDS

    def test_default_task_ids_are_exactly_the_implemented_set(self):
        # The pin for the config.py restatement (D-010): the default list is
        # written out by hand because importing BabiGenerator into config.py
        # would be circular, so this assertion is what keeps the two in lockstep.
        assert set(BabiTaskConfig().task_ids) == BabiGenerator.IMPLEMENTED_TASK_IDS

    def test_the_default_config_constructs_and_generates_every_task_it_asks_for(self):
        config = BabiTaskConfig(num_samples_per_task=2)

        results = BabiGenerator(config).generate_all_tasks()

        assert sorted(results) == sorted(config.task_ids)
        assert all(len(samples) == 2 for samples in results.values())


class TestBabiRefusesUnimplementedTasks:
    """An unimplemented id fails at construction, naming all of them."""

    def test_construction_raises_and_names_every_unsupported_id(self):
        config = BabiTaskConfig(task_ids=list(range(1, 21)))

        with pytest.raises(ValueError) as excinfo:
            BabiGenerator(config)

        message = str(excinfo.value)
        for task_id in MEASURED_UNIMPLEMENTED_TASK_IDS:
            assert str(task_id) in message
        # Not just the first offender.
        assert str(MEASURED_UNIMPLEMENTED_TASK_IDS[-1]) in message

    def test_the_refusal_lists_the_implemented_ids(self):
        with pytest.raises(ValueError) as excinfo:
            BabiGenerator(BabiTaskConfig(task_ids=[4]))

        message = str(excinfo.value)
        assert str(sorted(BabiGenerator.IMPLEMENTED_TASK_IDS)) in message

    def test_requesting_an_unimplemented_task_raises_instead_of_returning_a_short_dict(self):
        # The old behaviour: generate_all_tasks() swallowed the ValueError and
        # returned a dict with ten fewer keys than requested, with no signal.
        config = BabiTaskConfig(task_ids=[1, 4], num_samples_per_task=1)

        with pytest.raises(ValueError):
            BabiGenerator(config).generate_all_tasks()


class TestBabiIsReachableFromTheFullSuite:
    """``run_full_suite`` must actually dispatch bAbI."""

    def test_babi_is_dispatched_by_the_default_full_suite(self):
        harness = BenchmarkHarness(BenchmarkSuiteConfig(verbose=False))
        called = []

        for name in (
            "run_copy_task_benchmark",
            "run_associative_recall_benchmark",
            "run_length_generalization_benchmark",
            "run_capacity_benchmark",
            "run_scan_benchmark",
            "run_babi_benchmark",
        ):
            setattr(
                harness,
                name,
                lambda model, _name=name: called.append(_name),
            )

        harness.run_full_suite(model=None, model_name="probe")

        assert "run_babi_benchmark" in called


# ---------------------------------------------------------------------
# bAbI task 19 (path finding)
# ---------------------------------------------------------------------

_RELATION = re.compile(r"^The (\w+) is (\w+) of the (\w+)$")
_QUESTION = re.compile(r"^How do you go from the (\w+) to the (\w+)\?$")


def _shortest_path_directions(story: List[str], question: str) -> List[str]:
    """Re-derive the travel directions from a task-19 sample, independently.

    Deliberately does NOT reuse the generator's index arithmetic: it rebuilds a
    direction graph from the story text and searches it, so a generator that
    randomises directions but derives the answer wrongly cannot agree with it by
    construction.

    :param story: The sample's story sentences.
    :param question: The sample's question.
    :return: The directions travelled from the question's origin to its target.
    :raises AssertionError: If a sentence or the question is not parseable, or
        no path exists.
    """
    edges: Dict[str, List] = {}
    for sentence in story:
        match = _RELATION.match(sentence)
        assert match is not None, f"unparseable story sentence: {sentence!r}"
        near, direction, far = match.groups()
        # "near is <direction> of far" => from far, travel <direction> to near.
        edges.setdefault(far, []).append((direction, near))
        edges.setdefault(near, []).append(
            (BabiGenerator.INVERSE_DIRECTIONS[direction], far)
        )

    q_match = _QUESTION.match(question)
    assert q_match is not None, f"unparseable question: {question!r}"
    origin, target = q_match.groups()

    queue = collections.deque([(origin, [])])
    seen = {origin}
    while queue:
        node, path = queue.popleft()
        if node == target:
            return path
        for direction, neighbour in edges.get(node, []):
            if neighbour in seen:
                continue
            seen.add(neighbour)
            queue.append((neighbour, path + [direction]))

    raise AssertionError(f"no path from {origin} to {target} in {story}")


class TestBabiTask19IsNotDegenerate:
    """Two independent properties: the answers vary, AND each one is correct.

    Either one alone is passable by a broken generator — a fixed answer passes
    nothing but a randomly-shuffled answer string would pass the variety test,
    and a generator that never randomised would pass the correctness test. Both
    mutations were proven RED separately.
    """

    @staticmethod
    def _samples(n: int = 100):
        """Generate task-19 samples at the pinned seed.

        :param n: Number of samples.
        :return: The generated samples.
        """
        config = BabiTaskConfig(task_ids=[19], num_samples_per_task=n)
        return BabiGenerator(config).generate(19)

    def test_task19_answers_are_not_all_the_same_string(self):
        answers = {sample.answer for sample in self._samples(100)}

        assert len(answers) > 1, (
            f"task 19 emitted a single answer for 100 samples: {answers}"
        )

    def test_task19_answer_is_the_derived_path_of_its_own_story(self):
        for sample in self._samples(100):
            expected = ", ".join(
                _shortest_path_directions(sample.story, sample.question)
            )
            assert sample.answer == expected, (
                f"answer {sample.answer!r} is not the path through "
                f"{sample.story} (expected {expected!r})"
            )
