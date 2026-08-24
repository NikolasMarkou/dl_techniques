"""R-038 root cause **RD-8**: does HRM's halting ``q_head`` ever train?

Plan ``plan-2026-08-22T035419-a11304c8``, ruling **D-056**.

The R-038 inventory (MG-13) classified this REAL DEFECT: the shipped factory
compiles ``loss={"logits": StableMaxCrossEntropy()}`` and nothing on
``q_halt_logits`` / ``q_continue_logits``, so the advertised adaptive-halting
head "receives exactly zero gradient under the model's own default compile".

**The first half of that is true and the conclusion is not.** Measured on CPU,
seeded, on a tiny HRM (vocab 16, seq 8, dim 32, ``halt_max_steps=2``), gradient
read AFTER a real optimizer step at ``lr=1.0`` -- never at init:

===================================================  ==================  =======================
arm                                                  ``q_head`` movement  live weights
===================================================  ==================  =======================
factory compile + stock ``fit()``, one step          ``0.0`` EXACTLY      20 of 22
``HRMTrainer.train_step`` (``HRMLoss`` + tape)       kernel ``0.99996``   **23 of 23**
                                                     bias ``0.99993``
===================================================  ==================  =======================

The Q head trains -- under ``src/train/hrm/train_hrm.py``, the loop that exists
for exactly that purpose. The factory's logits-only default is deliberate: the
Q term couples ``q_halt_logits`` with ``target_q_continue`` and BOTH are model
*outputs*, so Keras 3's ``CompileLoss``, which broadcasts a ``Loss`` across
output keys and pairs it with a same-keyed target, cannot express it. That is
already recorded verbatim in the factory's ``loss:`` docstring and in the D-032
anchor at the compile site.

So this file does NOT change behaviour. It pins BOTH readings so that neither
can drift silently: the deliberate deadness under ``fit()`` (which someone will
otherwise "fix" into a broken compile) and the real liveness under the trainer
(which is the claim the module docstring makes).
"""

import importlib.util
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import keras
from keras import ops

from dl_techniques.models.hierarchical_reasoning_model.model import (
    create_hierarchical_reasoning_model,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VOCAB, _SEQ, _BATCH = 16, 8, 4

_MODEL_KWARGS = dict(
    vocab_size=_VOCAB, seq_len=_SEQ, embed_dim=32, num_puzzle_identifiers=4,
    num_heads=2, h_layers=1, l_layers=1, h_cycles=1, l_cycles=1,
    halt_max_steps=2,
)


def _batch():
    return (
        np.random.RandomState(0).randint(0, _VOCAB, (_BATCH, _SEQ)).astype("int32"),
        np.zeros((_BATCH,), "int32"),
        np.random.RandomState(1).randint(0, _VOCAB, (_BATCH, _SEQ)).astype("int32"),
    )


def _snapshot(model) -> Dict[str, np.ndarray]:
    return {v.path: ops.convert_to_numpy(v).copy() for v in model.trainable_variables}


def _movement(model, before) -> Dict[str, float]:
    return {
        v.path: float(np.max(np.abs(ops.convert_to_numpy(v) - before[v.path])))
        for v in model.trainable_variables
    }


def _train_hrm_module():
    """Import the trainer by PATH; `src/train` is not an installed package here."""
    path = _REPO_ROOT / "src" / "train" / "hrm" / "train_hrm.py"
    spec = importlib.util.spec_from_file_location("_probe_train_hrm", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_probe_train_hrm"] = module
    spec.loader.exec_module(module)
    return module


def test_the_factory_compile_leaves_the_q_head_dead_and_that_is_documented():
    """The deliberate half. Pinned so nobody 'fixes' it into a broken compile.

    Do NOT make this pass by adding a Q term to the factory's default `loss=`:
    measured, Keras 3's `CompileLoss` raises
    `KeyError: The path: ('logits',) ... can't be found` for any single `Loss`
    object over this output dict, and the Q term needs two model OUTPUTS, not a
    label. See the D-032 anchor at the compile site.
    """
    keras.utils.set_random_seed(0)
    model = create_hierarchical_reasoning_model(
        optimizer="adam", learning_rate=1.0, **_MODEL_KWARGS
    )
    ids, puzzle_ids, labels = _batch()
    before = _snapshot(model)
    # Keras' own report of the deliberate deadness. Asserted here so this
    # test survives `-W error::UserWarning` and so the warning's disappearance
    # would be caught alongside the movement assertion below.
    with pytest.warns(UserWarning, match="Gradients do not exist"):
        model.fit(
            {"token_ids": ids, "puzzle_ids": puzzle_ids},
            {"logits": labels},
            epochs=1, batch_size=_BATCH, verbose=0,
        )
    moved = _movement(model, before)
    q_head = {k: v for k, v in moved.items() if "q_head" in k}

    assert len(q_head) == 2, f"expected a q_head kernel and bias, got {q_head}"
    assert all(v == 0.0 for v in q_head.values()), (
        "the q_head moved under the factory's logits-only compile. If ACT "
        "supervision has genuinely been wired into `compile()`, update D-056, "
        "the factory's `loss:` docstring and the D-032 anchor in the same "
        f"change. Movement: {q_head}"
    )
    live = sum(1 for v in moved.values() if v > 0.0)
    assert live == len(moved) - 2, (
        f"exactly the two q_head weights should be dead here; {len(moved) - live} "
        f"are: {[k for k, v in moved.items() if v == 0.0]}"
    )


def test_the_shipped_trainer_actually_moves_the_q_head():
    """The load-bearing half: the advertised adaptive halting IS trained.

    Per LESSONS, movement is read AFTER a real optimizer step, never as a
    non-None gradient at init.
    """
    train_hrm = _train_hrm_module()
    config = {
        "model": dict(
            vocab_size=_VOCAB, seq_len=_SEQ, embed_dim=32,
            num_puzzle_identifiers=4, puzzle_emb_dim=32, num_heads=2,
            h_layers=1, l_layers=1, h_cycles=1, l_cycles=1,
            halt_max_steps=2, batch_size=_BATCH,
        ),
        "lm_loss_type": "stable_max",
        "q_loss_weight": 0.5,
        "learning_rate": {"initial_lr": 1.0, "warmup_steps": 0, "min_lr_ratio": 1.0},
        "optimizer": {"type": "adam"},
    }
    keras.utils.set_random_seed(0)
    trainer = train_hrm.HRMTrainer(config, train_dataset=None)
    model = trainer.model
    ids, puzzle_ids, labels = _batch()

    before = _snapshot(model)
    trainer.train_step(
        {"inputs": ids, "puzzle_identifiers": puzzle_ids, "labels": labels}
    )
    moved = _movement(model, before)
    q_head = {k: v for k, v in moved.items() if "q_head" in k}

    assert len(q_head) == 2, f"expected a q_head kernel and bias, got {q_head}"
    assert all(v > 0.0 for v in q_head.values()), (
        "the q_head did NOT move under the trainer that exists to supervise it. "
        "The adaptive-halting head advertised by the module docstring is inert. "
        f"Movement: {q_head}"
    )
    dead = [k for k, v in moved.items() if v == 0.0]
    assert dead == [], (
        f"the ACT loop left {len(dead)} trainable weight(s) dead: {dead}"
    )


def test_the_q_loss_weight_is_what_makes_the_difference():
    """Anti-vacuity: the trainer's Q supervision is the CAUSE, not the tape.

    Driving `q_loss_weight` to 0.0 must take the q_head's movement back to zero.
    Without this arm, the test above would pass against a trainer whose Q term
    was disconnected but whose tape happened to touch the head some other way.
    """
    train_hrm = _train_hrm_module()
    movements = {}
    for weight in (0.5, 0.0):
        config = {
            "model": dict(
                vocab_size=_VOCAB, seq_len=_SEQ, embed_dim=32,
                num_puzzle_identifiers=4, puzzle_emb_dim=32, num_heads=2,
                h_layers=1, l_layers=1, h_cycles=1, l_cycles=1,
                halt_max_steps=2, batch_size=_BATCH,
            ),
            "lm_loss_type": "stable_max",
            "q_loss_weight": weight,
            "learning_rate": {
                "initial_lr": 1.0, "warmup_steps": 0, "min_lr_ratio": 1.0
            },
            "optimizer": {"type": "adam"},
        }
        keras.utils.set_random_seed(0)
        trainer = train_hrm.HRMTrainer(config, train_dataset=None)
        ids, puzzle_ids, labels = _batch()
        before = _snapshot(trainer.model)
        trainer.train_step(
            {"inputs": ids, "puzzle_identifiers": puzzle_ids, "labels": labels}
        )
        moved = _movement(trainer.model, before)
        movements[weight] = max(v for k, v in moved.items() if "q_head" in k)

    assert movements[0.5] > 0.0
    assert movements[0.0] == 0.0, (
        f"with q_loss_weight=0.0 the q_head still moved by {movements[0.0]:.6e}; "
        f"something other than the Q term is reaching it, so the arm above does "
        f"not prove what it claims"
    )
