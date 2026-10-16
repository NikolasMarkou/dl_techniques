"""The default learning rate depends on whether the encoder starts pretrained.

Guard for plan-2026-10-14T062616-81938e81 step 13.6 (f) (D-115). At e259d9977
the default was 5e-6 for every run: the paper's base learning rate for a
*pretrained* encoder (arXiv 2401.10891, section 4.1), while no pretrained
encoder ships, so a default run trained a random encoder at a rate under which
the Step 13 smoke's validation loss did not move. The from-scratch default is
the measured one (the Step 13.6 sweep, D-115); a start from ``--init-from`` or
``--pretrained-encoder-weights`` keeps 5e-6, and an explicit rate always wins.
"""

import sys

import pytest

from train.depth_anything import train_depth_anything as module

#: The sweep's pick, restated so a silent change of the constant is visible.
MEASURED_FROM_SCRATCH = 2e-4


def _config(**kwargs):
    return module.DepthAnythingTrainingConfig(experiment_name="run", **kwargs)


def test_the_constants():
    assert module.PRETRAINED_LEARNING_RATE == 5e-6
    assert module.FROM_SCRATCH_LEARNING_RATE == MEASURED_FROM_SCRATCH


def test_a_random_encoder_gets_the_measured_rate():
    assert _config().learning_rate == MEASURED_FROM_SCRATCH


@pytest.mark.parametrize("start", [
    {"init_from": "previous_run/model_inference.keras"},
    {"pretrained_encoder_weights": "vit_s_encoder.keras"},
])
def test_a_pretrained_start_gets_the_papers_rate(start):
    assert _config(**start).learning_rate == 5e-6


@pytest.mark.parametrize("start", [{}, {"init_from": "previous_run/model_inference.keras"}])
def test_an_explicit_rate_wins(start):
    assert _config(learning_rate=3e-4, **start).learning_rate == 3e-4


@pytest.mark.parametrize("argv, expected", [
    ([], MEASURED_FROM_SCRATCH),
    (["--init-from", "previous_run/model_inference.keras"], 5e-6),
    (["--learning-rate", "1e-3"], 1e-3),
])
def test_the_command_line_reaches_the_resolved_rate(monkeypatch, argv, expected):
    seen = {}
    monkeypatch.setattr(module, "setup_gpu", lambda gpu_id=None: None)
    monkeypatch.setattr(module, "train_depth_anything",
                        lambda config: seen.setdefault("config", config))
    monkeypatch.setattr(sys, "argv", ["train_depth_anything", *argv])
    module.main()
    assert seen["config"].learning_rate == expected
