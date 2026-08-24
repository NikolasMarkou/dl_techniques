"""`HRMTrainer.__init__` must complete.

Rationale
---------
`__init__`'s own closing log line calls `self.model.count_params()`, and the
model returned by `create_hierarchical_reasoning_model` is UNBUILT. MEASURED at
HEAD:

    ValueError: You tried to call `count_params` on layer
    'hierarchical_reasoning_model', but the layer isn't built.
      at src/train/hrm/train_hrm.py, line 61

so `python -m train.hrm.train_hrm` could not START -- the module was unreachable
before any dataset, device or config was involved. `_create_model` now calls
`model.build()`, which `HierarchicalReasoningModel` implements as an
input_shape-ignoring materialisation of its reasoning core.

Note on the earlier test coverage of this class: the existing suite constructs
`HRMTrainer` via `object.__new__` plus a lambda, so NEITHER the real `__init__`
NOR the real `train_step` ever ran, which is why 12 of 34 step-override sites
were measured as never executed. This test calls the REAL constructor.

See decisions.md D-031 (plan-2026-08-19T163559-499b6f0e).
"""

import pytest

from train.hrm.train_hrm import HRMTrainer

# ---------------------------------------------------------------------

TINY_CONFIG = {
    "model": {
        "vocab_size": 16,
        "seq_len": 8,
        "embed_dim": 32,
        "num_puzzle_identifiers": 4,
        "puzzle_emb_dim": 32,
        "batch_size": 2,
        "h_layers": 1,
        "l_layers": 1,
        "h_cycles": 1,
        "l_cycles": 1,
        "num_heads": 2,
        "halt_max_steps": 2,
    },
}


@pytest.fixture(scope="module")
def trainer() -> HRMTrainer:
    """The REAL constructor, with no dataset -- it must not need one."""
    return HRMTrainer(config=TINY_CONFIG, train_dataset=None, val_dataset=None)


def test_the_constructor_completes(trainer):
    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.loss_fn is not None


def test_the_model_is_built_and_has_parameters(trainer):
    """The isolating assertion: `count_params()` is what raised."""
    assert trainer.model.built, (
        "`HRMTrainer._create_model` must build the model; an unbuilt model makes "
        "`__init__`'s own logging line raise and the trainer unimportable"
    )
    count = trainer.model.count_params()
    assert count > 0, f"built model reports {count} parameters"
