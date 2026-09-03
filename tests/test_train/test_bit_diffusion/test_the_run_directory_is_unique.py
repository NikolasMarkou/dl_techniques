r"""Guard: two default ``bit_diffusion`` runs do not write into the same directory.

The defect class this exists to catch
-------------------------------------
A TRAINER DEFAULT THAT SILENTLY DESTROYS THE PREVIOUS RUN. ``TrainingConfig.
experiment_name`` used to default to the CONSTANT ``"bit_diffusion"``, and
``train.common.run_io.prepare_run_dir`` appends no timestamp of its own -- the
run directory is exactly ``Path(output_dir) / experiment_name``. So every
default invocation resolved to ``results/bit_diffusion/`` and the second run
overwrote the first run's ``config.json``, ``best_model.keras``,
``final_model.keras``, ``training_history.json`` and ``training_log.csv`` in
place. Nothing errors on an existing run directory, so the loss was silent.

The fix is the shape 23 of this repo's 25 trainer configs already use:
``Optional[str] = None`` plus ``default_experiment_name(...)`` in
``__post_init__``. These arms pin the three properties that makes it a fix
rather than a rename:

1. the default is derived from the CLOCK, not a constant (so run N+1 gets its
   own directory);
2. an explicit ``--experiment-name`` still WINS, so a caller that wants a fixed
   directory -- every test in this suite that passes ``experiment_name=`` --
   keeps getting one;
3. the resulting RUN DIRECTORY, not merely the name, differs.

Trap designed out: ``default_experiment_name`` stamps at ONE-SECOND resolution
(``run_io.TIMESTAMP_FORMAT`` is ``%Y%m%d_%H%M%S``), so two configs constructed
inside the same second legitimately share a name and an "they differ" assertion
written without regard to that is FLAKY, not strict -- it would pass or fail on
scheduler luck. ``test_two_default_configs_taken_a_second_apart_differ``
therefore crosses a real second boundary with a real sleep instead of asserting
on two back-to-back constructions. The one-second granularity is the contract;
this test does not pretend it is finer.

No run directory is created here: nothing calls ``prepare_run_dir``, only
``resolved_run_dir``, which is pure path arithmetic.
"""
from __future__ import annotations

import re
import time

from train.common.args import resolved_run_dir
from train.common.run_io import TIMESTAMP_FORMAT
from train.bit_diffusion.train_bit_diffusion import (
    TrainingConfig,
    config_from_argv,
)

#: ``bit_diffusion_<bridge_preset>_<variant>_YYYYmmdd_HHMMSS``.
_DEFAULT_NAME = re.compile(r"^bit_diffusion_[a-z]+_[A-Za-z]+_\d{8}_\d{6}$")


def test_the_default_is_no_longer_the_bare_constant():
    """The pre-fix value, verbatim, must not come back."""
    name = TrainingConfig().experiment_name
    assert name != "bit_diffusion", (
        "experiment_name defaulted to the bare constant again: every default "
        "run would collide at results/bit_diffusion/"
    )
    assert _DEFAULT_NAME.match(name), (
        f"default experiment_name {name!r} does not carry the "
        f"{TIMESTAMP_FORMAT!r} stamp that makes it collision-free"
    )


def test_two_default_configs_taken_a_second_apart_differ():
    """The uniqueness property itself, at the granularity the stamp provides.

    The sleep is load-bearing, not padding: the stamp has one-second
    resolution, so two constructions in the same second SHOULD agree and an
    assertion over them would be flaky. Crossing the boundary makes the
    assertion deterministic in both directions -- it is green iff the name
    tracks the clock, and it was RED against the old constant default, which
    is invariant across any amount of elapsed time.
    """
    first = TrainingConfig()
    time.sleep(1.05)
    second = TrainingConfig()

    assert first.experiment_name != second.experiment_name
    assert resolved_run_dir(first) != resolved_run_dir(second)


def test_an_explicit_experiment_name_still_wins():
    """``--experiment-name`` pins the directory, through the real parser."""
    config = config_from_argv(["--experiment-name", "pinned-run"])
    assert config.experiment_name == "pinned-run"
    assert resolved_run_dir(config).name == "pinned-run"

    # ... and constructed directly, which is how every other test in this
    # directory gets a stable run directory under `tmp_path`.
    assert TrainingConfig(experiment_name="pinned-run").experiment_name == (
        "pinned-run"
    )


def test_the_name_carries_the_fields_that_change_what_the_artifacts_mean():
    """A checkpoint is not loadable under a different preset or variant.

    Both fragments are in the name so a directory listing distinguishes runs
    whose artifacts are mutually incompatible, rather than only by timestamp.
    """
    config = TrainingConfig(bridge_preset="tiny", variant="tiny")
    assert config.experiment_name.startswith("bit_diffusion_tiny_tiny_")

    other = TrainingConfig(bridge_preset="sd", variant="S")
    assert other.experiment_name.startswith("bit_diffusion_sd_S_")
