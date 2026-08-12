"""CLI contract guard for ``src/train/tabm/train_tabm.py``.

This package had ZERO tests of any kind, and its entry point had NO
``ArgumentParser``. A repo-wide sweep of all 115 ``src/train/`` entry points
found it as the only offender outside ``bert/wikipedia/``, and its failure mode
is worse than a crash: ``python -m train.tabm.train_tabm --help`` IGNORED the
flag, ran all five example pipelines to completion -- minutes of real training
-- and exited **0** with ``All examples completed`` in the log and no ``usage:``
line anywhere. An exit-code-only sweep reads that as healthy, which is why the
guard below asserts the usage line and the side effects, not the exit code
alone.

What each guard pins:

``test_help_exits_zero_without_running_anything``
    The invariant is not "a parser exists" but "``--help`` exits having run
    NOTHING". A sentinel replaces every example function and one replaces
    ``setup_gpu``; each is asserted to have been called ZERO times. Deleting the
    ``parse_arguments(argv)`` call from the top of ``main()`` fails this by the
    named ``--help ran ...`` message rather than by the exit code.

``test_help_prints_a_usage_line``
    Split out deliberately: the pre-fix script exited 0 too. Exit 0 is not a
    passing ``--help``.

``test_examples_selects_only_what_was_asked_for``
    The repo's documented silent-no-op bug class: a flag that parses and is then
    never forwarded. Asserts the SET of example functions ``main()`` actually
    invoked equals the one named on argv -- so a ``--examples`` that parses and
    is ignored (running all five) fails.

``test_setup_gpu_is_wired_and_imported_lazily``
    ``--gpu`` must reach ``setup_gpu``. The lazy-import half is the measured one:
    ``train.common``'s package ``__init__`` builds a ``tf.constant`` at module
    scope, which initializes TF's eager context and ALLOCATES A GPU, so a
    module-scope ``from train.common import setup_gpu`` makes ``--help`` expensive
    no matter where argv is parsed (plan D-006).

No training run: every example function is sentinelled off.
"""

import ast
from pathlib import Path
from typing import Any, Dict, List

import pytest

import train.common
import train.tabm.train_tabm as train_tabm

SRC = Path(__file__).resolve().parents[3] / "src"
SCRIPT = SRC / "train" / "tabm" / "train_tabm.py"


class _Sentinel:
    """Callable that records contact and returns ``None``.

    Contract: ``calls`` counts invocations; the call is a no-op otherwise, which
    is what lets ``main()`` run to its end over stand-ins instead of training.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0
        self.args: List[Any] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        self.args.append((args, kwargs))
        return None


def _install_sentinels(monkeypatch: pytest.MonkeyPatch) -> Dict[str, _Sentinel]:
    """Replace every example function and ``setup_gpu`` with a recorder.

    Contract:
      - Returns ``{name: sentinel}`` keyed by the ``EXAMPLES`` keys plus
        ``setup_gpu``. Every sentinel counts calls and does nothing else.
      - ``setup_gpu`` is patched on ``train.common`` (not on the script module)
        because the script imports it INSIDE ``main()``.
      - ``EXAMPLES`` is patched as a whole dict, so ``main()`` dispatching
        through it picks the stand-ins up.
      - ``monkeypatch`` restores everything at teardown.
    """
    sentinels = {name: _Sentinel(name) for name in train_tabm.EXAMPLES}
    monkeypatch.setattr(train_tabm, "EXAMPLES", dict(sentinels))
    gpu = _Sentinel("setup_gpu")
    monkeypatch.setattr(train.common, "setup_gpu", gpu)
    sentinels["setup_gpu"] = gpu
    return sentinels


def test_help_exits_zero_without_running_anything(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    sentinels = _install_sentinels(monkeypatch)
    with pytest.raises(SystemExit) as excinfo:
        train_tabm.main(["--help"])
    ran = [name for name, s in sentinels.items() if s.calls]
    assert not ran, (
        f"--help ran {ran} -- argparse must exit before main() does any work. "
        "Before the fix this script ignored --help entirely and trained to "
        "completion."
    )
    assert excinfo.value.code == 0


def test_help_prints_a_usage_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Exit 0 is NOT enough: the pre-fix script exited 0 after training."""
    _install_sentinels(monkeypatch)
    with pytest.raises(SystemExit):
        train_tabm.main(["--help"])
    assert capsys.readouterr().out.startswith("usage:")


def test_parse_arguments_defaults_to_every_example() -> None:
    """A bare run must still do what it did before the script had a CLI."""
    args = train_tabm.parse_arguments([])
    assert sorted(args.examples) == sorted(train_tabm.EXAMPLES)
    assert args.gpu is None


def test_examples_selects_only_what_was_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinels = _install_sentinels(monkeypatch)
    train_tabm.main(["--examples", "regression", "real-dataset"])
    called = {name for name in train_tabm.EXAMPLES if sentinels[name].calls}
    assert called == {"regression", "real-dataset"}, (
        f"--examples parsed but main() ran {sorted(called)}; a flag that parses "
        "and is then ignored is the repo's silent-no-op bug class."
    )


def test_setup_gpu_is_wired_and_imported_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinels = _install_sentinels(monkeypatch)
    train_tabm.main(["--gpu", "1", "--examples", "regression"])
    gpu = sentinels["setup_gpu"]
    assert gpu.calls == 1, f"main() called setup_gpu {gpu.calls} times, expected 1"
    assert gpu.args[0][0] == (1,), f"--gpu never reached setup_gpu: {gpu.args[0]}"

    # The lazy-import half: a module-scope `from train.common import ...`
    # allocates a GPU at import, so --help could never be free.
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    module_scope = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any(
        m == "train.common" or m.startswith("train.common.") for m in module_scope
    ), (
        "train.common is imported at MODULE scope; its package __init__ builds a "
        "tf.constant at import time, which allocates a GPU before argparse runs."
    )
