"""The trainer's CLI contract and the argv -> config path.

The recurring defect this guards is a flag that parses fine and then never
reaches the config, so setting it silently does nothing. ``config_from_args``
maps fields by name for exactly that reason, and the test below drives the full
path rather than inspecting the parser.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from train.embeddings_experimental.config import (
    BASELINE_MODEL,
    MODEL_REGISTRY,
    POOLING_STRATEGIES,
    VARIANTS,
    ExperimentConfig,
    available_models,
    build_model,
)
from train.embeddings_experimental.train_embeddings import (
    config_from_args,
    encoder_path,
    parse_args,
)

REPO_SRC = Path(__file__).resolve().parents[3] / "src"


class TestHelpGate:
    """``--help`` must exit 0 WITH a usage line, and allocate nothing.

    Exit 0 alone is not a passing ``--help``: a script with no parser ignores
    the flag entirely, runs its whole job and exits 0.
    """

    def test_help_exits_zero_and_prints_usage(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "train.embeddings_experimental.train_embeddings",
                "--help",
            ],
            cwd=REPO_SRC,
            capture_output=True,
            text=True,
            timeout=300,
            env={"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "", "HOME": "/tmp"},
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "usage:" in result.stdout

    def test_help_lists_every_registered_model(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "train.embeddings_experimental.train_embeddings",
                "--help",
            ],
            cwd=REPO_SRC,
            capture_output=True,
            text=True,
            timeout=300,
            env={"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "", "HOME": "/tmp"},
        )
        for name in available_models():
            assert name in result.stdout, name


class TestArgvReachesTheConfig:
    """Every flag must land on the config; none may be a silent no-op."""

    def test_defaults_round_trip(self):
        config = config_from_args(parse_args([]))
        assert config == ExperimentConfig()

    def test_every_config_field_has_a_flag(self):
        """``config_from_args`` raises if a field has no CLI flag.

        Asserted directly so the mapping cannot silently start dropping a
        field when one is added.
        """
        args = parse_args([])
        for name in ExperimentConfig().to_dict():
            assert hasattr(args, name), f"{name} has no CLI flag"

    @pytest.mark.parametrize(
        "flag,value,field,expected",
        [
            ("--model", "ascii_clifford_bert", "model", "ascii_clifford_bert"),
            ("--variant", "base", "variant", "base"),
            ("--pooling-strategy", "attention", "pooling_strategy", "attention"),
            ("--seed", "7", "seed", 7),
            ("--max-seq-length", "128", "max_seq_length", 128),
            ("--mlm-batch-size", "8", "mlm_batch_size", 8),
            ("--mlm-learning-rate", "0.001", "mlm_learning_rate", 0.001),
            ("--mask-ratio", "0.3", "mask_ratio", 0.3),
            ("--steps-per-epoch", "11", "steps_per_epoch", 11),
            ("--projection-dim", "64", "projection_dim", 64),
            ("--contrastive-temperature", "0.1", "contrastive_temperature", 0.1),
            ("--hidden-dropout-rate", "0.25", "hidden_dropout_rate", 0.25),
            ("--stochastic-depth-rate", "0.15", "stochastic_depth_rate", 0.15),
            ("--output-dir", "somewhere", "output_dir", "somewhere"),
        ],
    )
    def test_a_flag_reaches_the_config(self, flag, value, field, expected):
        config = config_from_args(parse_args([flag, value]))
        assert getattr(config, field) == expected

    def test_no_contrastive_switches_the_stage_off(self):
        assert config_from_args(parse_args([])).run_contrastive is True
        assert (
            config_from_args(parse_args(["--no-contrastive"])).run_contrastive
            is False
        )

    def test_mixed_bfloat16_is_a_flag(self):
        assert config_from_args(parse_args(["--mixed-bfloat16"])).mixed_bfloat16

    def test_an_unknown_model_is_rejected_by_the_parser(self):
        with pytest.raises(SystemExit):
            parse_args(["--model", "not_a_model"])


class TestRegistry:
    """The study axes are registry-driven, not hard-coded."""

    def test_the_baseline_is_registered(self):
        assert BASELINE_MODEL in MODEL_REGISTRY

    def test_both_arms_are_registered(self):
        assert set(MODEL_REGISTRY) >= {"ascii_bert", "ascii_clifford_bert"}

    def test_build_model_rejects_an_unregistered_name(self):
        config = ExperimentConfig(model="nope")
        with pytest.raises(ValueError, match="Unknown model"):
            build_model(config)

    @pytest.mark.parametrize("model", sorted(MODEL_REGISTRY))
    @pytest.mark.parametrize("pooling", POOLING_STRATEGIES)
    def test_every_arm_builds_at_every_pooling_strategy(self, model, pooling):
        import keras

        config = ExperimentConfig(
            model=model, variant="tiny", pooling_strategy=pooling,
            max_seq_length=32,
        )
        encoder = build_model(config)
        encoder.build((None, config.max_seq_length))
        assert encoder.count_params() > 0
        keras.backend.clear_session()

    @pytest.mark.parametrize("model", sorted(MODEL_REGISTRY))
    def test_every_arm_exposes_every_variant(self, model):
        """The size axis must line up, or the ladder is not comparable."""
        factory = MODEL_REGISTRY[model]
        import keras

        for variant in VARIANTS:
            encoder = factory(variant, max_position_embeddings=32)
            assert encoder is not None, (model, variant)
            keras.backend.clear_session()


class TestEncoderPathHasOneProducer:
    """A checkpoint filename known in two places is a latent defect.

    This repo has already paid for one: a name with several readers and zero
    writers that failed every default run after training had finished.
    """

    def test_the_path_is_derived_not_typed(self, tmp_path):
        assert encoder_path(str(tmp_path)).startswith(str(tmp_path))
        assert encoder_path(str(tmp_path)).endswith(".keras")

    def test_the_trainer_saves_through_the_same_producer(self):
        """The save site must call the helper, not a literal."""
        source = (
            REPO_SRC / "train/embeddings_experimental/train_embeddings.py"
        ).read_text()
        assert "encoder.save(encoder_path(run_dir))" in source or (
            "path = encoder_path(run_dir)" in source
            and "encoder.save(path)" in source
        )


class TestCellIdentity:
    """A cell id addresses one point of the sweep grid."""

    def test_cell_id_names_all_four_axes(self):
        config = ExperimentConfig(
            model="ascii_clifford_bert",
            variant="small",
            pooling_strategy="cls",
            seed=3,
        )
        assert config.cell_id() == "ascii_clifford_bert/small/cls/seed_3"

    def test_distinct_cells_have_distinct_ids(self):
        base = ExperimentConfig()
        variants = [
            ExperimentConfig(model="ascii_clifford_bert"),
            ExperimentConfig(variant="base"),
            ExperimentConfig(pooling_strategy="cls"),
            ExperimentConfig(seed=1),
        ]
        ids = {base.cell_id()} | {c.cell_id() for c in variants}
        assert len(ids) == 5


class TestOutputDirResolvesToTheRepoRoot:
    """Training artifacts belong in the repo's `results/`, never `src/results/`.

    The documented invocation is ``python -m train.<pkg>.<script>`` run from
    ``src/``, so a bare relative ``results`` resolves against ``src/`` and
    creates a second, wrong results tree. That is exactly what this trainer's
    first smoke runs did, which is why the resolution is now explicit and
    pinned here.
    """

    def test_a_relative_dir_resolves_against_the_repo_root(self):
        from train.embeddings_experimental.train_embeddings import (
            REPO_ROOT,
            resolve_output_dir,
        )

        resolved = resolve_output_dir("results")
        assert resolved == str(REPO_ROOT / "results")
        assert not resolved.endswith("src/results")

    def test_an_absolute_dir_is_left_alone(self, tmp_path):
        from train.embeddings_experimental.train_embeddings import (
            resolve_output_dir,
        )

        assert resolve_output_dir(str(tmp_path)) == str(tmp_path)

    def test_the_repo_root_is_the_one_holding_src(self):
        from train.embeddings_experimental.train_embeddings import REPO_ROOT

        assert (REPO_ROOT / "src" / "train").is_dir()
        assert (REPO_ROOT / "pyproject.toml").exists()
