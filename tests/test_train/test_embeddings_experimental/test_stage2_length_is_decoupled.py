"""Stage 2 and the evaluation must run at ONE length, and it may differ from stage 1.

`contrastive_seq_length` exists to hold the contrastive objective fixed while the
pretraining context varies. In InfoNCE the batch size IS the number of in-batch
negatives, so a cell that halves `contrastive_batch_size` to fit a longer
sequence is solving an easier contrastive task -- easier in the same direction as
any improvement the longer context is meant to demonstrate. Measured 2026-08-30:
`ascii_bert` at 1024 peaks at **12.5 GB** at batch 16, so batch 64 at 1024 does
not fit on a 24 GB card, and the clifford arm already OOMed once at 512/64.

Two properties are pinned, and the second is the one a silent defect would break.

1. The default must be inert. Every run before this field existed used
   `max_seq_length` for stage 2 and for evaluation, so `None` must reproduce that
   exactly or the field re-interprets 100+ existing run directories.
2. Stage 2 and the evaluation must resolve through the SAME producer. An encoder
   contrastively trained at one length and evaluated at another would measure a
   length mismatch on top of whatever it is meant to measure -- and nothing about
   the run would look wrong: shapes match, losses are finite, `eval_ok` is True.
"""

import ast
import os

import pytest

from train.embeddings_experimental.config import (
    ExperimentConfig,
    resolve_contrastive_seq_length,
)
import train.embeddings_experimental.train_embeddings as trainer


class TestTheDefaultIsInert:
    """`None` must mean exactly what the code did before the field existed."""

    @pytest.mark.parametrize("length", [64, 512, 1024])
    def test_none_follows_max_seq_length(self, length: int) -> None:
        config = ExperimentConfig(max_seq_length=length)
        assert config.contrastive_seq_length is None
        assert resolve_contrastive_seq_length(config) == length

    def test_an_explicit_value_overrides(self) -> None:
        config = ExperimentConfig(max_seq_length=1024, contrastive_seq_length=512)
        assert resolve_contrastive_seq_length(config) == 512

    def test_zero_is_treated_as_unset_not_as_a_length(self) -> None:
        """A zero-length sequence is meaningless; falling back is the safe read."""
        config = ExperimentConfig(max_seq_length=512, contrastive_seq_length=0)
        assert resolve_contrastive_seq_length(config) == 512


class TestBothConsumersGoThroughTheProducer:
    """The stage-2 dataset and the evaluation must not name a length themselves.

    Asserted over the module's AST rather than by running training: the defect is
    a *stale reference* (`config.max_seq_length` left at one of the two sites),
    which no smoke run would surface because both values are valid lengths.
    """

    @staticmethod
    def _function(name: str) -> ast.FunctionDef:
        source = open(trainer.__file__).read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        raise AssertionError(f"{name} not found in {trainer.__file__}")

    @staticmethod
    def _calls_producer(node: ast.AST) -> bool:
        return any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "resolve_contrastive_seq_length"
            for n in ast.walk(node)
        )

    @staticmethod
    def _reads_max_seq_length(node: ast.AST) -> bool:
        return any(
            isinstance(n, ast.Attribute) and n.attr == "max_seq_length"
            for n in ast.walk(node)
        )

    def test_the_contrastive_stage_resolves_the_length(self) -> None:
        node = self._function("run_contrastive_stage")
        assert self._calls_producer(node), (
            "run_contrastive_stage does not call resolve_contrastive_seq_length; "
            "it is naming a sequence length itself, so --contrastive-seq-length "
            "is a knob that silently does nothing for stage 2"
        )
        assert not self._reads_max_seq_length(node), (
            "run_contrastive_stage still reads config.max_seq_length directly. "
            "Stage 2 must go through the producer or the two lengths can drift."
        )

    def test_the_evaluation_resolves_the_same_length(self) -> None:
        node = self._function("run_study_cell")
        source = ast.unparse(node)
        assert "max_length=resolve_contrastive_seq_length(config)" in source, (
            "the EvalConfig is not built from resolve_contrastive_seq_length, so "
            "the encoder could be contrastively trained at one length and "
            "evaluated at another -- a length mismatch measured as content"
        )
