"""CLI contract guard for ``src/train/mini_vec2vec/train_mini_vec2vec.py``.

``mini_vec2vec`` has no ``common.py``/``config_from_args`` split (D-007: the
algorithm has one build step and one entry point, no dataset pipeline or
optimizer to factor out) and no ``--gpu`` flag (the trainer's own module
docstring: every stage of ``align()`` runs on CPU through
scikit-learn/scipy). This file adapts
``tests/test_train/test_zamba2/test_cli_contract.py``'s two scoped checks to
that shape:

1. **Exit 0 is not a passing ``--help``.** ``--help`` must print a
   ``usage:`` line AND must not reach ``run_alignment`` -- a sentinel
   installed over it is asserted uncalled.
2. **``argv -> config`` must wire, not silently default.** A representative
   flag per field group is driven through the REAL parser and asserted to
   land on the REAL config object, including the defaults path.

Plus a small, genuinely fast synthetic alignment run (not the trainer's own
large default: ``embedding_dim=8``, ``n_samples=40``, single-digit
cluster/refinement counts) exercising ``run_alignment`` end to end, writing
into a ``tmp_path`` (never repo-root ``results/``) via
``--output-dir``.
"""

from __future__ import annotations

import json

import pytest

from train.mini_vec2vec.train_mini_vec2vec import (
    MiniVec2VecTrainingConfig,
    build_aligner,
    build_parser,
    build_synthetic_embedding_spaces,
    config_from_args,
)
import train.mini_vec2vec.train_mini_vec2vec as trainer


# ---------------------------------------------------------------------
# --help: prints usage, allocates nothing
# ---------------------------------------------------------------------


def test_help_exits_zero_and_prints_usage(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        trainer.main(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("usage:"), (
        f"--help did not print a usage: line first. stdout:\n{captured.out}"
    )


def test_help_never_runs_the_alignment(monkeypatch, capsys) -> None:
    calls = {"run_alignment": 0}

    def _spy_run_alignment(*_args, **_kwargs):
        calls["run_alignment"] += 1
        raise AssertionError("run_alignment must not run on --help")

    monkeypatch.setattr(trainer, "run_alignment", _spy_run_alignment)

    with pytest.raises(SystemExit) as excinfo:
        trainer.main(["--help"])
    assert excinfo.value.code == 0
    assert calls == {"run_alignment": 0}, f"--help reached run_alignment: {calls}"


# ---------------------------------------------------------------------
# argv -> config wiring
# ---------------------------------------------------------------------


def test_no_flags_produces_the_dataclass_defaults() -> None:
    args = build_parser().parse_args([])
    config = config_from_args(args)
    assert config == MiniVec2VecTrainingConfig()


@pytest.mark.parametrize(
    ("argv", "field", "expected"),
    [
        (["--embedding-dim", "16"], "embedding_dim", 16),
        (["--n-samples", "500"], "n_samples", 500),
        (["--n-eval", "100"], "n_eval", 100),
        (["--n-clusters", "4"], "n_clusters", 4),
        (["--approx-clusters", "4"], "approx_clusters", 4),
        (["--approx-runs", "2"], "approx_runs", 2),
        (["--refine1-iterations", "3"], "refine1_iterations", 3),
        (["--refine2-clusters", "8"], "refine2_clusters", 8),
        (["--seed", "123"], "seed", 123),
        (["--output-dir", "/tmp/mini_vec2vec_out"], "output_dir", "/tmp/mini_vec2vec_out"),
    ],
)
def test_one_flag_reaches_its_field(argv, field, expected) -> None:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    defaults = MiniVec2VecTrainingConfig()
    assert getattr(config, field) == expected
    # Trap: the probe value must differ from the default, or a wiring bug
    # that drops the flag entirely would still pass this assertion.
    assert expected != getattr(defaults, field), (
        f"probe value for {field!r} equals the default; it cannot prove "
        f"the flag was forwarded"
    )


def test_no_gpu_flag_exists() -> None:
    """The module docstring's own deliberate omission, as an executable guard."""
    parser = build_parser()
    dests = {action.dest for action in parser._actions}
    assert "gpu" not in dests


# ---------------------------------------------------------------------
# Cheap, fast build checks (not the trainer's own large default sizes)
# ---------------------------------------------------------------------


def test_build_synthetic_embedding_spaces_and_aligner_construct() -> None:
    config = MiniVec2VecTrainingConfig(embedding_dim=8, n_samples=40, n_eval=10)
    xa, xb, xa_eval, xb_eval, ground_truth_w = build_synthetic_embedding_spaces(config)
    assert xa.shape == (40, 8)
    assert xb.shape == (40, 8)
    assert xa_eval.shape == (10, 8)
    assert ground_truth_w.shape == (8, 8)

    aligner = build_aligner(config)
    assert aligner.embedding_dim == 8


def test_tiny_synthetic_alignment_run_writes_artifacts(tmp_path) -> None:
    """End-to-end ``run_alignment`` at a genuinely tiny size, not the
    trainer's own large default (``n_samples=25000`` etc.) -- fast enough
    for a unit-test budget, writing only under ``tmp_path``."""
    config = MiniVec2VecTrainingConfig(
        embedding_dim=8,
        n_samples=40,
        n_eval=10,
        n_clusters=2,
        approx_clusters=2,
        approx_runs=1,
        approx_neighbors=2,
        refine1_iterations=1,
        refine1_sample_size=10,
        refine1_neighbors=2,
        refine2_clusters=2,
        output_dir=str(tmp_path),
    )
    _aligner, metrics, results_dir = trainer.run_alignment(config)

    assert "top1_accuracy" in metrics
    assert 0.0 <= metrics["top1_accuracy"] <= 1.0

    config_path = f"{results_dir}/config.json"
    with open(config_path) as fh:
        saved = json.load(fh)
    assert saved["embedding_dim"] == 8
