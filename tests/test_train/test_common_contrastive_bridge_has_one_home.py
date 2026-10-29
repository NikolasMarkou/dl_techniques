"""The contrastive stock-``fit`` bridge has one home, ``train.common.contrastive``.

Guard for plan-2026-10-14T062616-81938e81 step 12 Part F item 9 (D-086). The
bridge (`StackedCLIPLogits`, its loss and metrics, `compile_for_contrastive_training`,
`as_contrastive_fit_batches`) lived in ``train.clip.train_clip`` and
``build_contrastive_loss`` in ``train.cliffordnet.train_clip`` (that trainer is gone), so the
CliffordCLIP trainer imported the CLIP trainer to train. The CLIP trainer now
imports the bridge from ``train.common.contrastive`` and defines none of it.
"""

import ast
from pathlib import Path

import pytest

import train.clip.train_clip as clip_trainer
import train.common.contrastive as contrastive

SRC = Path(__file__).resolve().parents[2] / "src"

BRIDGE = (
    "unstack_clip_logits",
    "StackedCLIPLogits",
    "StackedCLIPContrastiveLoss",
    "StackedCLIPAccuracy",
    "StackedCLIPRecallAtK",
    "compile_for_contrastive_training",
    "as_contrastive_fit_batches",
    "build_contrastive_loss",
)

TRAINERS = {
    "train/clip/train_clip.py": clip_trainer,
}

#: What each trainer actually uses from the bridge.
USES = {
    "train/clip/train_clip.py": ("compile_for_contrastive_training",
                                 "as_contrastive_fit_batches"),
}


def _tree(relative):
    return ast.parse((SRC / relative).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", BRIDGE)
def test_the_bridge_is_defined_in_the_common_module(name):
    assert hasattr(contrastive, name)
    defined = {node.name for node in _tree("train/common/contrastive.py").body
               if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    assert name in defined


@pytest.mark.parametrize("relative", sorted(TRAINERS))
def test_no_trainer_defines_any_of_the_bridge(relative):
    defined = {node.name for node in ast.walk(_tree(relative))
               if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    duplicated = sorted(defined & set(BRIDGE))
    assert not duplicated, f"{relative} defines bridge names again: {duplicated}"


@pytest.mark.parametrize("relative", sorted(TRAINERS))
def test_each_trainer_uses_the_common_objects(relative):
    module = TRAINERS[relative]
    for name in USES[relative]:
        assert getattr(module, name) is getattr(contrastive, name), (
            f"{relative}.{name} is not train.common.contrastive.{name}")
