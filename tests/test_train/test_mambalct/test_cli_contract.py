"""CLI contract for the MambaLCT trainer: --help exits 0 with a usage line."""

import train.mambalct.train_mambalct as train_mambalct


def test_help_exits_zero_with_usage(capsys) -> None:
    try:
        train_mambalct.parse_arguments(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:  # pragma: no cover - parse_args("--help") always raises SystemExit
        raise AssertionError("--help did not exit")
    out = capsys.readouterr().out
    assert "usage:" in out


def test_config_from_args_forwards_flags() -> None:
    args = train_mambalct.parse_arguments(
        [
            "--clip-length", "3", "--l1-weight", "1.0",
            "--variant", "mambalct-384",
            "--template-size", "96", "--search-size", "192",
            "--batch-size", "8", "--epochs", "5",
            "--learning-rate", "1e-3", "--lr-schedule", "constant",
            "--allow-custom-geometry",
        ]
    )
    config = train_mambalct.config_from_args(args)
    assert config.clip_length == 3
    assert config.l1_weight == 1.0
    assert config.variant == "mambalct-384"
    assert config.template_size == 96
    assert config.search_size == 192
    assert config.batch_size == 8
    assert config.epochs == 5
    assert config.learning_rate == 1e-3
    assert config.lr_schedule_type == "constant"
