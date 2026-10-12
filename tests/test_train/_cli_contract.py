"""CLI flag contract test for the Neural HMM trainer.

This test verifies that every CLI flag (via argparse) maps 1:1 to a
NeuralHMMConfig dataclass field, and that no CLI flag is entirely
redundant / undocumented.

Run with::

    pytest tests/test_train/_cli_contract.py -xvs

or via make::

    pytest tests/test_train/_cli_contract.py -xvs
"""

import argparse

from train.neural_hmm.train_neural_hmm import create_argument_parser


def test_cli_flags_map_one_to_one_to_config():
    """Every CLI flag maps to a NeuralHMMConfig field, and vice versa."""
    parser = create_argument_parser()
    cli_dests = {a.dest for a in parser._actions}  # noqa: SLF001  intentional

    # Fields of the dataclass (excluding private/internal ones)
    cfg_fields = {
        "experiment_name", "output_dir", "seed", "generator",
        "n_samples", "n_states", "means", "stds",
        "transition_type", "switch_frac", "obs_noise_factor",
        "mode", "encoder", "emission_context", "topology",
        "band_width", "self_loop_init", "min_scale",
        "vocab_size", "epochs", "batch_size",
        "learning_rate", "gradient_clip_norm", "optimizer",
        "mixed_precision",
        "patience", "include_terminate_on_nan", "include_analyzer",
        "save_visualizations", "visualization_frequency",
    }

    # Every config field should appear as a CLI flag (or have a sensible
    # default that the trainer handles internally).
    missing_from_cli = cfg_fields - cli_dests
    assert not missing_from_cli, (
        f"Config fields without CLI flags: {missing_from_cli}. "
        "Every config field must have a corresponding CLI flag so the trainer "
        "can be driven entirely from the command line."
    )

    # Every CLI flag should correspond to a config field.
    # Some argparse internals (e.g. ``prog``, ``usage``) won't be in cfg_fields,
    # so we only check the ones that look like user flags (no leading underscore,
    # not ``help``, not ``from_file_prefix``, etc.).
    user_dests = {d for d in cli_dests if not d.startswith("_") and d not in {"help", "version"}}
    unexpected_in_cli = user_dests - cfg_fields
    assert not unexpected_in_cli, (
        f"CLI flags without matching config fields: {unexpected_in_cli}. "
        "Every user-facing CLI flag should map to a NeuralHMMConfig field."
    )