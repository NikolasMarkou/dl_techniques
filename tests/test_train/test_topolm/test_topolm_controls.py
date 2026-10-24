"""The controls of the topographic contrast (D-029): framing, length, label shuffle.

``train.topolm.controls`` regenerates, from a finished run, the three controls the
README quotes next to its a-versus-b numbers, so that no figure rests on a script
in a gitignored plan directory:

* ``framing_table``: the same contrast under the shipped context, other contexts,
  ``<eot>`` only, no prefix and CLS;
* ``length_control``: word lists of 3 against 7 words (and 5 against 5) drawn from
  one vocabulary, seeded, ``n_draws`` a parameter;
* ``label_shuffle_null``: every split of the ten sentences into 5 and 5.

The statistics are the trainer's own chain (``score_contrast``, which
``evaluate_topography`` also calls). Three kinds of oracle:

* planted responses (the forward pass is replaced, so a failure is about the
  controls and never about a model): the effect is a known function of the prompt;
* an independent route for the same number (``evaluate_topography`` on
  re-labelled stimuli, a bare per-sentence forward pass);
* counting arguments (``C(10, 5)`` splits, a split and its mirror).

The tiny model is random: significance counts on it are mostly zero, so the
planted tests carry the discriminating power and the real-model tests carry the
shapes, the determinism and the agreement with ``evaluate_topography``.
"""

import ast
import json
import math
import os
import subprocess
import sys
import zlib

import numpy as np
import pytest

from train.topolm import SMOKE_STIMULI, evaluate_topography, train_topolm
from train.topolm import common as C
from train.topolm import controls as K

from .test_topolm_trainer import (  # noqa: F401  (the autouse fixture must be in scope)
    StubTokenizer,
    _block_effect,
    _FUNCTION_WORDS,
    _config,
    _gpt2_encoding,
    _model,
    _never_write_to_repo_results,
    _patch_data,
    _raw_taps,
    _stub_backbone,
    _tokenizer,
    _words,
)

#: The imported autouse fixture is what keeps run directories out of ``results/``.
_AUTOUSE_FIXTURES_IN_SCOPE = (_never_write_to_repo_results,)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

A_SENTENCES = list(SMOKE_STIMULI["a"])
B_SENTENCES = list(SMOKE_STIMULI["b"])
ALL_SENTENCES = A_SENTENCES + B_SENTENCES


def _noise(prompt, size):
    """Per-prompt noise, deterministic in the prompt text (a crc32 seed)."""
    return np.random.default_rng(zlib.crc32(prompt.encode())).standard_normal(size)


def _plant(monkeypatch, backbone, response):
    """Replace the forward pass: ``response(prompt, tap)`` is a unit-order vector.

    ``context`` and the other keywords of the real extraction are accepted and
    ignored, and every prompt list the controls extract is recorded in ``seen``.
    """
    seen = []

    def planted(backbone_, prompts, preprocessor, batch_size=32, **_kwargs):
        prompts = list(prompts)
        seen.append(prompts)
        return {
            tap.path: np.stack([response(prompt, tap) for prompt in prompts])
            for tap in backbone.tap_layers
        }

    monkeypatch.setattr(C, "extract_tap_activations", planted)
    return seen


def _stub():
    return _stub_backbone(num_units=256, grid_dim=2, seeds=(11, 12, 13, 14))


EFFECT_BOX = ((2, 2), (6, 6))  # a 4 x 4 block of cells = 16 units per tap


def _effect(tap):
    return _block_effect(tap.layout, *EFFECT_BOX, value=6.0)


# ---------------------------------------------------------------------
# the scorer the controls and evaluate_topography share
# ---------------------------------------------------------------------


class TestTheSharedScorer:
    """``score_contrast`` is the one place the t-map -> joint BH-FDR -> cluster
    chain lives (D-029: the controls must not keep a second copy of it)."""

    JITTER = 0.1 * np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    def _acts(self, backbone, box):
        values_a, values_b = {}, {}
        for tap in backbone.tap_layers:
            effect = _block_effect(tap.layout, *box, value=1.0)
            values_a[tap.path] = effect[None, :] + self.JITTER[:, None]
            values_b[tap.path] = np.zeros((5, tap.layout.num_units))
        return values_a, values_b

    def test_a_planted_block_is_one_cluster_on_its_own_cells(self):
        backbone = _stub()
        box = ((3, 4), (8, 9))  # 5 x 5 = 25 cells
        values_a, values_b = self._acts(backbone, box)
        scored = C.score_contrast(backbone, values_a, values_b, "a", "b", min_cluster_size=10)
        assert list(scored) == [tap.path for tap in backbone.tap_layers]
        expected = np.zeros((16, 16), dtype=bool)
        expected[3:8, 4:9] = True
        for tap in backbone.tap_layers:
            entry = scored[tap.path]
            np.testing.assert_array_equal(entry["sig_grid"], expected)
            assert entry["t_grid"].shape == (16, 16)
            assert len(entry["clusters"]["a"]) == 1 and len(entry["clusters"]["b"]) == 0
            cells = {tuple(int(c) for c in tap.layout.positions[u]) for u in entry["clusters"]["a"][0]}
            assert cells == {(r, c) for r in range(3, 8) for c in range(4, 9)}
            np.testing.assert_array_equal(entry["label_grid"] > 0, expected)
            assert not (entry["label_grid_b"] > 0).any()

    def test_a_cluster_below_the_minimum_size_is_not_grown(self):
        """A 3 x 3 block is significant in every tap but is a cluster only when the
        minimum size is 9 or less."""
        backbone = _stub()
        values_a, values_b = self._acts(backbone, ((5, 5), (8, 8)))
        for minimum, expected in ((10, 0), (9, 1)):
            scored = C.score_contrast(
                backbone, values_a, values_b, "a", "b", min_cluster_size=minimum
            )
            for tap in backbone.tap_layers:
                entry = scored[tap.path]
                assert int(entry["sig_grid"].sum()) == 9
                assert len(entry["clusters"]["a"]) == expected, (minimum, tap.path)

    def test_it_gives_evaluate_topographys_arrays_for_the_same_activations(
        self, monkeypatch
    ):
        backbone = _stub()
        values_a, values_b = self._acts(backbone, ((2, 2), (7, 7)))

        def planted(backbone_, prompts, preprocessor, batch_size=32, **_kwargs):
            return values_a if prompts[0].startswith("a") else values_b

        monkeypatch.setattr(C, "extract_tap_activations", planted)
        stimuli = {"a": [f"a{i}" for i in range(5)], "b": [f"b{i}" for i in range(5)]}
        report, arrays = evaluate_topography(
            backbone, stimuli, ("a", "b"), StubTokenizer(), readout_fwhm=None,
            min_cluster_size=10,
        )
        scored = C.score_contrast(backbone, values_a, values_b, "a", "b", min_cluster_size=10)
        raw = arrays["arms"]["raw"]
        for tap in backbone.tap_layers:
            np.testing.assert_array_equal(raw["t_maps"][tap.path], scored[tap.path]["t_grid"])
            np.testing.assert_array_equal(raw["sig_grids"][tap.path], scored[tap.path]["sig_grid"])
            np.testing.assert_array_equal(raw["label_grids"][tap.path], scored[tap.path]["label_grid"])
            assert report["arms"]["raw"]["per_tap"][tap.path]["cluster_sizes_a"] == [
                len(c) for c in scored[tap.path]["clusters"]["a"]
            ]


# ---------------------------------------------------------------------
# the alternative contexts
# ---------------------------------------------------------------------


class TestTheAlternativeContexts:
    """Three other contexts (about 30, 60 and 120 tokens, other topics), reported
    beside the shipped one whatever their numbers are."""

    #: function words and pronouns the smoke sentences also use
    ALLOWED = _FUNCTION_WORDS | {"before", "he", "over", "she", "they", "we"}

    def test_their_token_counts_are_31_59_and_123(self):
        encoding = _gpt2_encoding()
        counts = {name: len(encoding.encode(text)) for name, text in K.ALTERNATIVE_CONTEXTS.items()}
        assert counts == {"physics_31": 31, "sport_59": 59, "history_123": 123}, counts

    def test_none_shares_a_content_word_with_a_smoke_sentence(self):
        for name, text in K.ALTERNATIVE_CONTEXTS.items():
            shared = _words([text]) & _words(s for v in SMOKE_STIMULI.values() for s in v)
            assert shared <= self.ALLOWED, (name, shared - self.ALLOWED)

    def test_they_hold_no_special_id(self):
        encoding = _gpt2_encoding()
        for name, text in K.ALTERNATIVE_CONTEXTS.items():
            assert max(encoding.encode(text)) < 50256, name

    def test_the_shipped_context_is_not_one_of_them(self):
        assert C.STIMULUS_CONTEXT not in K.ALTERNATIVE_CONTEXTS.values()


# ---------------------------------------------------------------------
# the vocabulary helper
# ---------------------------------------------------------------------


class TestConditionVocabulary:
    def test_it_is_the_sorted_distinct_words_of_the_condition(self):
        vocabulary = K.condition_vocabulary(SMOKE_STIMULI, "a")
        expected = sorted({w for s in A_SENTENCES for w in s.split()})
        assert vocabulary == expected
        assert "the" in vocabulary and len(vocabulary) == len(set(vocabulary))

    def test_an_unknown_condition_is_refused_by_name(self):
        with pytest.raises(ValueError, match="'z'"):
            K.condition_vocabulary(SMOKE_STIMULI, "z")


# ---------------------------------------------------------------------
# the length control
# ---------------------------------------------------------------------


class TestTheLengthControl:
    """Word lists of ``short`` against ``long`` words from one vocabulary."""

    VOCAB_A = K.condition_vocabulary(SMOKE_STIMULI, "a")
    VOCAB_B = K.condition_vocabulary(SMOKE_STIMULI, "b")

    def _run(self, monkeypatch, **kwargs):
        backbone = _stub()

        def response(prompt, tap):
            # the effect scales with the number of words: 3 against 7 words is a
            # difference of 4 words, 5 against 5 is none
            return _effect(tap) * (len(prompt.split()) - 5) + _noise(
                prompt, tap.layout.num_units
            )

        seen = _plant(monkeypatch, backbone, response)
        result = K.length_control(
            backbone, None, self.VOCAB_A, self.VOCAB_B, **kwargs
        )
        return result, seen

    def test_the_keys_are_vocabulary_and_lengths(self, monkeypatch):
        result, _ = self._run(monkeypatch, n_draws=2, seed=0)
        assert sorted(result) == ["a_vocab_3v7", "a_vocab_5v5", "b_vocab_3v7", "b_vocab_5v5"]
        entry = result["a_vocab_3v7"]
        for key in ("vocabulary", "short_words", "long_words", "draws", "values",
                    "median", "min", "max", "clusters_median", "clusters_max"):
            assert key in entry, key
        assert entry["draws"] == 2 and len(entry["values"]) == 2
        assert (entry["short_words"], entry["long_words"]) == (3, 7)

    def test_a_planted_length_effect_is_found_at_3_against_7_and_not_at_5_against_5(
        self, monkeypatch
    ):
        result, _ = self._run(monkeypatch, n_draws=6, seed=0)
        for vocabulary in ("a", "b"):
            long_run = result[f"{vocabulary}_vocab_3v7"]
            equal_run = result[f"{vocabulary}_vocab_5v5"]
            # 16 planted units per tap, 4 taps: every one of them significant, and
            # next to no noise unit
            assert 16.0 <= long_run["median"] <= 20.0, long_run
            assert equal_run["median"] == 0.0, equal_run

    def test_each_draw_is_five_short_and_five_long_lists_of_distinct_vocabulary_words(
        self, monkeypatch
    ):
        _, seen = self._run(monkeypatch, n_draws=3, seed=0)
        # per (vocabulary, lengths, draw): the short side then the long side
        assert len(seen) == 2 * 2 * 3 * 2
        sides = iter(seen)
        for vocabulary, pool in (("a", self.VOCAB_A), ("b", self.VOCAB_B)):
            for short, long in ((3, 7), (5, 5)):
                for _ in range(3):
                    for words in (short, long):
                        lists = next(sides)
                        assert len(lists) == 5
                        for text in lists:
                            drawn = text.split()
                            assert len(drawn) == words
                            assert len(set(drawn)) == words, "a word repeated in a list"
                            assert set(drawn) <= set(pool), (vocabulary, text)

    def test_the_same_seed_draws_the_same_lists_and_another_seed_other_ones(
        self, monkeypatch
    ):
        _, first = self._run(monkeypatch, n_draws=2, seed=7)
        _, again = self._run(monkeypatch, n_draws=2, seed=7)
        _, other = self._run(monkeypatch, n_draws=2, seed=8)
        assert first == again
        assert first != other

    def test_the_number_of_draws_is_a_parameter(self, monkeypatch):
        for n_draws in (1, 4):
            result, seen = self._run(monkeypatch, n_draws=n_draws, seed=0)
            assert result["b_vocab_3v7"]["draws"] == n_draws
            assert len(seen) == 2 * 2 * n_draws * 2

    def test_lengths_are_a_parameter(self, monkeypatch):
        result, _ = self._run(monkeypatch, n_draws=2, seed=0, lengths=((4, 8),))
        assert sorted(result) == ["a_vocab_4v8", "b_vocab_4v8"]

    def test_a_vocabulary_smaller_than_the_longest_list_is_refused(self, monkeypatch):
        backbone = _stub()
        _plant(monkeypatch, backbone, lambda prompt, tap: np.zeros(tap.layout.num_units))
        with pytest.raises(ValueError, match="7"):
            K.length_control(backbone, None, ["one", "two", "three"], self.VOCAB_B)

    def test_on_a_real_tiny_model_the_draws_are_reproducible(self):
        backbone, preprocessor = _model("gpt2", 128), _tokenizer("gpt2", 100)
        first = K.length_control(
            backbone, preprocessor, self.VOCAB_A, self.VOCAB_B, n_draws=2, seed=3,
            lengths=((3, 7),), min_cluster_size=2,
        )
        again = K.length_control(
            backbone, preprocessor, self.VOCAB_A, self.VOCAB_B, n_draws=2, seed=3,
            lengths=((3, 7),), min_cluster_size=2,
        )
        assert first == again
        assert all(np.isfinite(v) for v in first["a_vocab_3v7"]["values"])
        assert first["a_vocab_3v7"]["mean_tokens"][0] < first["a_vocab_3v7"]["mean_tokens"][1]


# ---------------------------------------------------------------------
# the label-shuffle null
# ---------------------------------------------------------------------


class TestTheLabelShuffleNull:
    """Every split of the ten sentences into 5 and 5 is scored."""

    def _null(self, monkeypatch, effect_on, **kwargs):
        """``effect_on``: the set of sentences that carry the planted effect."""
        backbone = _stub()

        def response(prompt, tap):
            return _effect(tap) * (prompt in effect_on) + _noise(
                prompt, tap.layout.num_units
            )

        _plant(monkeypatch, backbone, response)
        return K.label_shuffle_null(
            backbone, None, SMOKE_STIMULI, ("a", "b"), **kwargs
        )

    def test_the_splits_add_up(self, monkeypatch):
        result = self._null(monkeypatch, set(A_SENTENCES))
        assert result["n_splits"] == math.comb(10, 5) == 252
        splits = [tuple(entry["a_index"]) for entry in result["per_split"]]
        assert len(splits) == len(set(splits)) == 252
        assert all(len(s) == 5 and s == tuple(sorted(s)) for s in splits)
        # every split meets its mirror (the complement)
        everything = set(range(10))
        assert {tuple(sorted(everything - set(s))) for s in splits} == set(splits)

    def test_the_real_labelling_is_the_first_five_sentences_against_the_rest(
        self, monkeypatch
    ):
        result = self._null(monkeypatch, set(A_SENTENCES))
        assert result["real"]["a_index"] == [0, 1, 2, 3, 4]

    def test_when_the_effect_follows_the_real_labelling_only_it_and_its_mirror_are_at_the_top(
        self, monkeypatch
    ):
        result = self._null(monkeypatch, set(A_SENTENCES))
        assert result["real"]["mean_significant_units_per_tap"] >= 16.0
        assert result["n_strictly_above"] == 0
        assert result["competitors"] == []
        assert result["rank_from_top"] == 1
        # the real split, its mirror and nothing else may tie it, so p is
        # (at or above) / 252, never below 2/252
        assert result["n_at_or_above"] >= 2
        assert result["p"] == pytest.approx(result["n_at_or_above"] / 252)
        assert result["p"] >= 2 / 252

    def test_a_split_that_beats_the_real_labelling_is_named_as_a_competitor(
        self, monkeypatch
    ):
        """Plant the effect on a0, a1, b0, b1, b2: that split (and its mirror)
        carries the contrast and the real labelling does not."""
        planted = {A_SENTENCES[0], A_SENTENCES[1], B_SENTENCES[0],
                   B_SENTENCES[1], B_SENTENCES[2]}
        result = self._null(monkeypatch, planted)
        found = {tuple(c["a_index"]) for c in result["competitors"]}
        assert (0, 1, 5, 6, 7) in found and (2, 3, 4, 8, 9) in found, found
        assert result["n_strictly_above"] == len(result["competitors"]) >= 2
        assert result["rank_from_top"] == result["n_strictly_above"] + 1
        real = result["real"]["mean_significant_units_per_tap"]
        assert all(
            c["mean_significant_units_per_tap"] > real for c in result["competitors"]
        )
        assert result["p"] == pytest.approx(result["n_at_or_above"] / 252)

    def test_the_counts_agree_with_evaluate_topography_on_relabelled_stimuli(
        self, monkeypatch
    ):
        """An independent route to the same number, for five splits."""
        backbone = _stub()

        def response(prompt, tap):
            return _effect(tap) * (prompt in set(A_SENTENCES)) + _noise(
                prompt, tap.layout.num_units
            )

        _plant(monkeypatch, backbone, response)
        result = K.label_shuffle_null(backbone, None, SMOKE_STIMULI, ("a", "b"))
        by_index = {tuple(e["a_index"]): e for e in result["per_split"]}
        for a_index in [(0, 1, 2, 3, 4), (0, 1, 2, 3, 5), (0, 2, 4, 6, 8),
                        (5, 6, 7, 8, 9), (1, 3, 5, 7, 9)]:
            stimuli = {
                "a": [ALL_SENTENCES[i] for i in a_index],
                "b": [ALL_SENTENCES[i] for i in range(10) if i not in a_index],
            }
            report, _ = _evaluate_with_planted(backbone, stimuli)
            arm = report["arms"]["raw"]
            entry = by_index[a_index]
            assert entry["mean_significant_units_per_tap"] == arm["mean_num_significant_units"]
            assert (entry["clusters_a"], entry["clusters_b"]) == (
                arm["total_clusters_a"], arm["total_clusters_b"],
            )
            assert entry["mean_morans_i"] == pytest.approx(arm["mean_morans_i"], abs=1e-12)

    def test_the_summary_statistics_describe_the_per_split_values(self, monkeypatch):
        result = self._null(monkeypatch, set(A_SENTENCES))
        values = np.array([e["mean_significant_units_per_tap"] for e in result["per_split"]])
        morans = np.array([e["mean_morans_i"] for e in result["per_split"]])
        assert result["sig"]["median"] == pytest.approx(float(np.median(values)))
        assert result["sig"]["max"] == pytest.approx(float(values.max()))
        assert result["morans_i"]["min"] == pytest.approx(float(morans.min()))
        assert result["morans_i"]["max"] == pytest.approx(float(morans.max()))
        real = result["real"]["mean_morans_i"]
        assert result["morans_i"]["real_rank_from_top"] == int((morans >= real).sum())
        assert result["n_splits_with_a_cluster"] == sum(
            (e["clusters_a"] + e["clusters_b"]) > 0 for e in result["per_split"]
        )

    def test_a_subsample_is_seeded_and_always_holds_the_real_labelling(self, monkeypatch):
        first = self._null(monkeypatch, set(A_SENTENCES), max_splits=20, seed=1)
        again = self._null(monkeypatch, set(A_SENTENCES), max_splits=20, seed=1)
        other = self._null(monkeypatch, set(A_SENTENCES), max_splits=20, seed=2)
        indices = lambda r: [tuple(e["a_index"]) for e in r["per_split"]]  # noqa: E731
        assert first["n_splits"] == 20 and len(indices(first)) == 20
        assert (0, 1, 2, 3, 4) in indices(first)
        assert indices(first) == indices(again)
        assert indices(first) != indices(other)

    def test_conditions_of_unequal_size_are_refused(self, monkeypatch):
        backbone = _stub()
        _plant(monkeypatch, backbone, lambda p, tap: np.zeros(tap.layout.num_units))
        with pytest.raises(ValueError, match="equal"):
            K.label_shuffle_null(
                backbone, None, {"a": A_SENTENCES[:3], "b": B_SENTENCES}, ("a", "b")
            )


def _evaluate_with_planted(backbone, stimuli):
    """``evaluate_topography`` over the planted extraction (already patched).

    The tokenizer only counts the report's ``stimuli`` block; the activations are
    planted.
    """
    return evaluate_topography(
        backbone, stimuli, ("a", "b"), StubTokenizer(), readout_fwhm=None,
        min_cluster_size=10,
    )


# ---------------------------------------------------------------------
# the framing table
# ---------------------------------------------------------------------


class TestTheFramingTable:
    def _setup(self):
        return _model("gpt2", 128), _tokenizer("gpt2", 100)

    def test_the_default_rows_and_their_fields(self):
        backbone, preprocessor = self._setup()
        table = K.framing_table(backbone, preprocessor, min_cluster_size=2)
        assert list(table) == ["context", "eot", "none", "cls"]
        assert [table[k]["framing"] for k in table] == [
            "context+eot", "eot", "none", "legacy-call-only",
        ]
        assert table["context"]["context_tokens"] == 60
        assert table["eot"]["context_tokens"] == table["none"]["context_tokens"] == 0
        for row in table.values():
            assert len(row["significant_units_per_tap"]) == len(backbone.tap_layers)
            assert row["mean_significant_units_per_tap"] == pytest.approx(
                np.mean(row["significant_units_per_tap"])
            )
            for key in ("clusters_a", "clusters_b", "mean_morans_i"):
                assert key in row

    def test_extra_contexts_become_rows_in_order(self):
        backbone, preprocessor = self._setup()
        table = K.framing_table(
            backbone, preprocessor, min_cluster_size=2,
            extra_contexts={"physics": "A field assigns a number to every point."},
        )
        assert list(table) == ["context", "eot", "none", "cls", "context:physics"]
        assert table["context:physics"]["framing"] == "context+eot"
        assert 0 < table["context:physics"]["context_tokens"] < 20

    def test_each_row_equals_evaluate_topography_under_the_same_framing(self):
        backbone, preprocessor = self._setup()
        table = K.framing_table(backbone, preprocessor, min_cluster_size=2)
        for name, context in (("context", C.STIMULUS_CONTEXT), ("eot", ""), ("cls", None)):
            report, _ = evaluate_topography(
                backbone, {"a": SMOKE_STIMULI["a"], "b": SMOKE_STIMULI["b"]},
                ("a", "b"), preprocessor, readout_fwhm=None, min_cluster_size=2,
                stimulus_context=context,
            )
            arm = report["arms"]["raw"]
            row = table[name]
            assert row["mean_significant_units_per_tap"] == arm["mean_num_significant_units"], name
            assert (row["clusters_a"], row["clusters_b"]) == (
                arm["total_clusters_a"], arm["total_clusters_b"],
            ), name
            assert row["mean_morans_i"] == pytest.approx(arm["mean_morans_i"], abs=1e-12), name

    def test_the_no_prefix_row_is_the_bare_sentence_alone(self, monkeypatch):
        """Oracle: one forward pass per sentence with no prefix at all, mean over
        every position of that sentence."""
        backbone, preprocessor = self._setup()
        backbone(np.zeros((1, 4), dtype="int32"), training=False)  # tap paths exist once built
        encoding = preprocessor.tokenizer
        acts = {}
        for condition in ("a", "b"):
            rows = {tap.path: [] for tap in backbone.tap_layers}
            for sentence in SMOKE_STIMULI[condition]:
                raw = _raw_taps(backbone, np.asarray([encoding.encode(sentence)], dtype="int32"))
                for path, values in raw.items():
                    rows[path].append(values[0].mean(axis=0))
            acts[condition] = {path: np.stack(v) for path, v in rows.items()}

        table = K.framing_table(backbone, preprocessor, min_cluster_size=2)

        def planted(backbone_, prompts, preprocessor_, batch_size=32, **_kwargs):
            condition = "a" if list(prompts) == SMOKE_STIMULI["a"] else "b"
            return acts[condition]

        monkeypatch.setattr(C, "extract_tap_activations", planted)
        report, _ = evaluate_topography(
            backbone, {"a": SMOKE_STIMULI["a"], "b": SMOKE_STIMULI["b"]},
            ("a", "b"), preprocessor, readout_fwhm=None, min_cluster_size=2,
        )
        arm = report["arms"]["raw"]
        row = table["none"]
        assert row["mean_significant_units_per_tap"] == arm["mean_num_significant_units"]
        assert row["mean_morans_i"] == pytest.approx(arm["mean_morans_i"], abs=1e-6)


# ---------------------------------------------------------------------
# everything for one run, the facts file and the command line
# ---------------------------------------------------------------------


class TestControlsForARunAndTheFactsFile:
    def _controls(self):
        # 160 positions: the longest alternative context is 123 tokens
        backbone, preprocessor = _model("gpt2", 160), _tokenizer("gpt2", 160)
        return K.controls_for_run(
            backbone, preprocessor, n_draws=2, seed=0, min_cluster_size=2,
            lengths=((3, 7),),
        )

    def test_the_block_holds_every_control(self):
        block = self._controls()
        assert sorted(block) == [
            "framings", "label_shuffle_null", "length_control", "n_draws",
            "observed_a_vs_b", "seed",
        ]
        assert block["n_draws"] == 2 and block["seed"] == 0
        assert block["label_shuffle_null"]["n_splits"] == 252
        assert "per_split" not in block["label_shuffle_null"], "the facts stay small"
        assert sorted(block["length_control"]) == ["a_vocab_3v7", "b_vocab_3v7"]
        observed = block["observed_a_vs_b"]
        assert observed["mean_significant_units_per_tap"] == block["framings"]["context"][
            "mean_significant_units_per_tap"
        ]
        json.dumps(block, allow_nan=False)  # strict JSON: no NaN, no numpy scalars

    def test_the_facts_file_is_written_from_a_run_directory_and_keeps_what_it_did_not_touch(
        self, monkeypatch, tmp_path
    ):
        _patch_data(monkeypatch)
        config = _config(
            vocab_size=50261, max_seq_length=160, save_dir="run_for_facts",
            readout_fwhm=None, min_cluster_size=2,
        )
        result = train_topolm(config)
        run_dir = result["results_dir"]

        existing = tmp_path / "existing.json"
        existing.write_text(json.dumps({
            "generated_from": {"seed": 42},
            "files": {"keep.png": {"sha256": "abc"}},
            "runs": {"2d": {"exit_code": 0, "controls": {"stale": True}}},
        }))
        out = tmp_path / "facts.json"
        facts = K.facts(
            {"2d": run_dir}, str(out), n_draws=2, seed=0, existing_path=str(existing),
            lengths=((3, 7),),
        )
        on_disk = json.loads(out.read_text())
        assert on_disk == facts
        assert on_disk["files"] == {"keep.png": {"sha256": "abc"}}
        run = on_disk["runs"]["2d"]
        assert run["exit_code"] == 0, "a key the writer does not own is kept"
        assert "stale" not in run["controls"], "the controls block is replaced"
        assert run["controls"]["label_shuffle_null"]["n_splits"] == 252
        assert run["stimuli"]["framing"] == "context+eot"
        assert on_disk["generated_from"]["run_dirs"] == {"2d": run_dir}
        assert on_disk["generated_from"]["n_draws"] == 2

    def test_a_facts_file_can_be_written_with_nothing_to_extend(self, monkeypatch, tmp_path):
        _patch_data(monkeypatch)
        config = _config(
            vocab_size=50261, max_seq_length=160, save_dir="run_for_facts2",
            readout_fwhm=None, min_cluster_size=2,
        )
        run_dir = train_topolm(config)["results_dir"]
        out = tmp_path / "fresh.json"
        facts = K.facts({"3d": run_dir}, str(out), n_draws=1, seed=0, lengths=((3, 7),))
        assert sorted(facts) == ["generated_from", "runs"]
        assert sorted(facts["runs"]) == ["3d"]


class TestTheFactsFileKeepsItsFormat:
    def test_the_tracked_file_is_reproduced_byte_for_byte(self, tmp_path):
        """``dump_facts`` is the format of ``figure_facts.json``: loading the tracked
        file and writing it back changes nothing, so extending it with new keys
        leaves the old ones as they are."""
        tracked = os.path.join(REPO, "src", "train", "topolm", "figures", "figure_facts.json")
        original = open(tracked, "rb").read()
        out = tmp_path / "again.json"
        K.dump_facts(str(out), json.loads(original))
        assert out.read_bytes() == original


class TestTheCommandLine:
    def test_help_exits_zero_with_a_usage_line(self):
        env = {**os.environ, "PYTHONPATH": os.path.join(REPO, "src"), "CUDA_VISIBLE_DEVICES": ""}
        done = subprocess.run(
            [sys.executable, "-m", "train.topolm.controls", "--help"],
            capture_output=True, text=True, env=env, cwd=REPO,
        )
        assert done.returncode == 0, done.stderr[-500:]
        assert "usage:" in done.stdout

    def test_arguments_are_parsed_before_any_run_work(self, monkeypatch, capsys):
        def boom(*args, **kwargs):
            raise AssertionError("the run was touched before the arguments were parsed")

        monkeypatch.setattr(K, "load_run", boom)
        monkeypatch.setattr(K, "facts", boom)
        with pytest.raises(SystemExit) as excinfo:
            K.main([])
        assert excinfo.value.code == 2
        assert "usage:" in capsys.readouterr().err
        with pytest.raises(SystemExit) as excinfo:
            K.main(["--run", "no-equals-sign", "--facts", "x.json"])
        assert excinfo.value.code == 2

    def test_main_passes_its_arguments_to_facts(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(K, "facts", lambda *a, **k: calls.append((a, k)) or {})
        K.main([
            "--run", "2d=/some/run", "--run", "3d=/other/run",
            "--facts", str(tmp_path / "f.json"), "--n-draws", "5", "--seed", "9",
        ])
        (args, kwargs), = calls
        assert args[0] == {"2d": "/some/run", "3d": "/other/run"}
        assert args[1] == str(tmp_path / "f.json")
        assert kwargs["n_draws"] == 5 and kwargs["seed"] == 9


class TestTheModuleStandsOnTheRepoAlone:
    def test_it_imports_nothing_from_a_plan_directory(self):
        path = os.path.join(REPO, "src", "train", "topolm", "controls.py")
        tree = ast.parse(open(path).read())
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
        assert modules, "no imports found: the check is vacuous"
        assert not [m for m in modules if m.split(".")[0] in ("plans", "scratch")], modules
        source = open(path).read()
        assert "sys.path" not in source, "a path hack could reach plans/"
        assert "plans/" not in source.replace("``plans/``", "")
