"""Mechanism shared by the bert/fnet trainer guards.

This module holds ONLY mechanism (how to drive the scripts). Every pinned
FACT -- parameter counts, layer names, config defaults -- lives literally in
the test modules, so a wrong value here can never make a guard agree with the
code by construction.

Why it exists: ``src/train/bert/*.py`` and ``src/train/fnet/*.py`` had ZERO
test coverage. There is no ``build_config(args)`` seam in either script --
``FinetuneConfig``/``TrainingConfig`` are plain classes mutated inline inside
``main()`` -- so the only way to observe the ``argv -> config`` mapping without
refactoring the scripts is to short-circuit ``main()`` at the training call.
"""

import re
import sys
from typing import Any, Dict, List

import pytest


class ShortCircuit(Exception):
    """Raised by the stand-in training function to stop ``main()`` early."""


def assert_layer_names(observed: List[str], expected: List[str]) -> None:
    """Compare a model's layer names against a pinned list, tolerating Keras's
    global name uniquifier.

    Contract:
      - Position-by-position: ``observed[i]`` must equal ``expected[i]`` or be
        ``expected[i]`` plus a ``_<digits>`` uniquifier suffix. Keras appends
        one when a name is already taken PROCESS-WIDE, so ``bert`` becomes
        ``bert_5`` purely because of how many models earlier tests built --
        a bare equality here is flaky by session ordering, not by code.
      - Length must match exactly.
      - Failure mode: this cannot distinguish ``encoder_layer_0`` from a
        uniquified ``encoder_layer`` -- acceptable, because both sides are
        matched positionally, so an ORDER change still fires.
    """
    assert len(observed) == len(expected), (
        f"layer count changed: {observed!r} vs pinned {expected!r}"
    )
    for index, (got, want) in enumerate(zip(observed, expected)):
        assert re.fullmatch(re.escape(want) + r"(_\d+)?", got), (
            f"layer {index}: got {got!r}, pinned {want!r} "
            f"(a trailing _<digits> uniquifier is allowed)"
        )


def effective_config(config: Any) -> Dict[str, Any]:
    """Return the FULL effective field map of a plain-class trainer config.

    Contract:
      - ``config`` is an instance of a plain class whose fields are declared as
        annotated CLASS attributes (the bert/fnet/tree_transformer idiom) and
        which ``main()`` mutates per-instance.
      - Returns ``{field_name: current_value}`` over every annotated field of
        the whole MRO, class defaults included.
      - Failure mode: a field declared without an annotation is invisible here.
        Every field in the four scripts under guard is annotated; the tests
        additionally pin the KEY SET, so a newly added un-annotated field shows
        up as a key-set mismatch rather than passing silently.

    ``config.__dict__`` alone is NOT enough: it holds only what ``main()``
    assigned, so untouched defaults such as ``max_seq_length`` (256 for bert,
    128 for fnet) would be invisible -- and pinning exactly those is the point.
    """
    fields: Dict[str, Any] = {}
    for klass in reversed(type(config).__mro__):
        fields.update(getattr(klass, "__annotations__", {}))
    return {name: getattr(config, name) for name in fields}


def capture_config_from_argv(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    train_fn_name: str,
    argv: List[str],
) -> Any:
    """Drive ``module.main()`` over ``argv`` and return the config it built.

    Contract:
      - ``module`` is one of the four bert/fnet trainer modules.
      - ``train_fn_name`` names the module-level training entry point that
        ``main()`` calls with the config as its first positional argument
        (``finetune_sentiment_model`` / ``train_bert_mlm`` / ``train_fnet_mlm``).
      - ``setup_gpu`` is neutralised, so no device is touched; the stand-in
        training function raises :class:`ShortCircuit` BEFORE any dataset load.
      - Returns the config object exactly as ``main()`` left it.
      - Failure mode: if ``main()`` ever stops calling ``train_fn_name`` with a
        config, ``pytest.raises`` below fails the test rather than silently
        returning a default config. That is deliberate -- a probe that cannot
        tell "reached the call" from "returned early" is vacuous.
    """
    captured: Dict[str, Any] = {}

    def _stop(config: Any, *args: Any, **kwargs: Any) -> Any:
        captured["config"] = config
        raise ShortCircuit(train_fn_name)

    monkeypatch.setattr(module, "setup_gpu", lambda *a, **k: None)
    monkeypatch.setattr(module, train_fn_name, _stop)
    monkeypatch.setattr(sys, "argv", ["prog", *argv])

    with pytest.raises(ShortCircuit):
        module.main()

    assert "config" in captured, (
        f"{module.__name__}.main() never reached {train_fn_name}; the probe "
        f"measured nothing"
    )
    return captured["config"]
