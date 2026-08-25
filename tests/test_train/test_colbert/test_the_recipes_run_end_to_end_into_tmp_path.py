"""Each recipe runs end to end, into ``tmp_path``, with its OWN loss.

Three claims, all measured by executing the shipped training functions rather
than by reading them:

1. **The synthetic pipeline's element spec is what the model and the loss
   consume.** A dataset whose keys or row count disagree with
   ``ColBERT.call`` / the loss's ``(batch, nway)`` reshape fails at the first
   step; a finite ``loss`` in ``history`` is the evidence that it does not.
2. **v1 and v2 pair with different losses.** Each run's compiled loss object is
   asserted to be its own class AND asserted not to be the other's -- a single
   positive assertion passes if one recipe were a copy of the other.
3. **Every artefact lands under the caller's output root.** Repo-root
   ``results/`` is gitignored and untracked, so a stray run directory there is
   unrecoverable; ``tests/conftest.py``'s autouse fixture errors in teardown if
   one appears, and the assertions here pin the run directory positively.

Geometry
--------

The smallest configuration the pipeline accepts: one batch of ``batch_size *
nway`` rows, one epoch, ``dim=16``, ``query_maxlen=8``, ``doc_maxlen=24`` on the
``tiny`` backbone. ``vocab_size`` is deliberately left at its default -- the
ColBERT tokenizer emits Tiktoken ``cl100k_base`` ids, including its specials and
the two ``[Q]``/``[D]`` markers, from the TOP of a 100277-entry range, so a
smaller embedding table would make every marker an out-of-range lookup.

Both runs are module-scoped: each costs a real ``fit``, and the assertions below
only read the resulting objects.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

from train.common.callbacks import best_checkpoint_path
from dl_techniques.losses import ColBERTDistillationLoss, ColBERTPairwiseSoftmaxLoss
from train.language.colbert import train_colbert_v1 as v1
from train.language.colbert import train_colbert_v2 as v2
from train.language.colbert.common import TrainingConfig, build_datasets

#: v2 reads this; v1 ignores it. A non-default value, so its arrival is visible.
PROBE_DISTILLATION_ALPHA = 0.5


def _tiny_config(output_root: str, **overrides: Any) -> TrainingConfig:
    """The smallest runnable configuration, rooted at ``output_root``.

    Interface contract (3 callers: the two run fixtures and the element-spec
    test, which must all see the same geometry or they measure different
    pipelines):

    :param output_root: Directory the timestamped run directory is created in.
        Always a ``tmp_path``; never repo-root ``results``.
    :param overrides: Extra :class:`TrainingConfig` fields, e.g. ``nway``.
    :returns: A config with one training batch and one validation batch.
    """
    values: Dict[str, Any] = dict(
        colbert_variant="tiny",
        dim=16,
        query_maxlen=8,
        doc_maxlen=24,
        batch_size=2,
        epochs=1,
        warmup_epochs=0,
        num_train_groups=2,
        num_val_groups=2,
        patience=1,
        seed=42,
        output_root=output_root,
    )
    values.update(overrides)
    return TrainingConfig(**values)


@pytest.fixture(scope="module")
def v1_run(tmp_path_factory) -> Tuple[Any, Any, str, TrainingConfig]:
    """One real ``train_colbert_v1`` run. Returns ``(model, history, dir, config)``."""
    root = tmp_path_factory.mktemp("colbert_v1_run")
    config = _tiny_config(str(root))
    model, history, results_dir = v1.train_colbert_v1(config)
    return model, history, results_dir, config


@pytest.fixture(scope="module")
def v2_run(tmp_path_factory) -> Tuple[Any, Any, str, TrainingConfig]:
    """One real ``train_colbert_v2`` run. Returns ``(model, history, dir, config)``."""
    root = tmp_path_factory.mktemp("colbert_v2_run")
    config = _tiny_config(
        str(root), nway=3, distillation_alpha=PROBE_DISTILLATION_ALPHA
    )
    model, history, results_dir = v2.train_colbert_v2(config)
    return model, history, results_dir, config


# ---------------------------------------------------------------------
# The pipeline feeds the model and the loss
# ---------------------------------------------------------------------


@pytest.mark.parametrize("use_teacher_targets", [False, True], ids=["v1", "v2"])
def test_the_element_spec_matches_what_the_model_consumes(
    use_teacher_targets, tmp_path
) -> None:
    """Batch keys, row count and lengths, read off the real dataset.

    The row count is the load-bearing one: both losses recover a candidate group
    by reshaping a flat ``(batch * nway,)`` score vector, so a batch that is not
    a whole multiple of ``nway`` regroups across group boundaries.
    """
    config = _tiny_config(str(tmp_path), nway=3)
    train_dataset, val_dataset, _ = build_datasets(config, use_teacher_targets)

    for dataset in (train_dataset, val_dataset):
        inputs, targets = next(iter(dataset))
        assert set(inputs) == {
            "query_input_ids",
            "query_attention_mask",
            "doc_input_ids",
            "doc_attention_mask",
            "doc_skiplist_mask",
        }, f"unexpected input keys {sorted(inputs)}"
        assert set(targets) == {"score"}, (
            f"the label side must mirror the model's supervised OUTPUT KEY; got "
            f"{sorted(targets)}"
        )
        rows = config.batch_size * config.nway
        for name, tensor in inputs.items():
            assert tensor.shape[0] == rows, (
                f"{name} has {tensor.shape[0]} rows, expected batch_size * nway "
                f"= {rows}; the loss reshape would regroup across groups"
            )
        assert inputs["query_input_ids"].shape[1] == config.query_maxlen
        assert inputs["doc_input_ids"].shape[1] == config.doc_maxlen
        assert targets["score"].shape[0] == rows


@pytest.mark.parametrize("run", ["v1_run", "v2_run"])
def test_one_epoch_produces_a_finite_loss(run, request) -> None:
    """The whole wiring, executed: data -> model -> loss -> optimizer."""
    _, history, _, _ = request.getfixturevalue(run)
    for key in ("loss", "val_loss"):
        assert key in history.history, f"{run}: history has no {key!r}"
        values = history.history[key]
        assert values, f"{run}: {key} is empty -- fit ran zero epochs"
        for value in values:
            assert math.isfinite(value), f"{run}: {key} is not finite ({value})"


# ---------------------------------------------------------------------
# Each recipe pairs with its own loss
# ---------------------------------------------------------------------


def test_v1_compiles_the_pairwise_softmax_loss(v1_run) -> None:
    model = v1_run[0]
    loss = model.loss["score"]
    assert isinstance(loss, ColBERTPairwiseSoftmaxLoss), (
        f"v1 compiled {type(loss).__name__}; the v1 recipe is softmax CE over "
        f"the nway candidates"
    )
    assert not isinstance(loss, ColBERTDistillationLoss), (
        "v1 compiled the v2 distillation loss"
    )


def test_v2_compiles_the_distillation_loss(v2_run) -> None:
    model = v2_run[0]
    loss = model.loss["score"]
    assert isinstance(loss, ColBERTDistillationLoss), (
        f"v2 compiled {type(loss).__name__}; the v2 recipe is KL distillation "
        f"against the teacher scores"
    )
    assert not isinstance(loss, ColBERTPairwiseSoftmaxLoss), (
        "v2 compiled the v1 pairwise loss"
    )


def test_the_two_recipes_do_not_compile_the_same_loss(v1_run, v2_run) -> None:
    """The disagreement, asserted directly -- neither is silently the other."""
    assert type(v1_run[0].loss["score"]) is not type(v2_run[0].loss["score"])


def test_the_loss_reads_the_configured_nway_and_alpha(v1_run, v2_run) -> None:
    """Config -> loss constructor, for the two fields the recipes differ on."""
    v1_model, _, _, v1_config = v1_run
    v2_model, _, _, v2_config = v2_run
    assert v1_model.loss["score"].nway == v1_config.nway
    assert v2_model.loss["score"].nway == v2_config.nway
    assert v2_model.loss["score"].distillation_alpha == PROBE_DISTILLATION_ALPHA


# ---------------------------------------------------------------------
# Artefacts land where the caller asked
# ---------------------------------------------------------------------


@pytest.mark.parametrize("run", ["v1_run", "v2_run"])
def test_the_run_directory_is_under_the_configured_output_root(
    run, request
) -> None:
    """``--output-root`` is what makes a trainer test possible at all."""
    _, _, results_dir, config = request.getfixturevalue(run)
    root = Path(config.output_root).resolve()
    assert Path(results_dir).resolve().is_relative_to(root), (
        f"{run}: run directory {results_dir} is not under output_root {root}"
    )
    assert Path(results_dir).is_dir()
    assert config.results_dir_prefix in Path(results_dir).name


@pytest.mark.parametrize("run", ["v1_run", "v2_run"])
def test_the_best_checkpoint_is_written(run, request) -> None:
    """The ONE producer of that path is ``best_checkpoint_path``; use it."""
    _, _, results_dir, _ = request.getfixturevalue(run)
    assert best_checkpoint_path(results_dir), "no checkpoint path was produced"
    assert Path(best_checkpoint_path(results_dir)).exists(), (
        f"{run}: fit completed but wrote no best checkpoint"
    )
