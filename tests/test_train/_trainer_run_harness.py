"""Stub harness for EXECUTING a trainer's `train_*` body without training.

Why this exists: `src/train/gpt2/*` and `src/train/wave_field/*` adopted
`save_config_json` / `save_training_history_json` as one-line calls inside their
`train_*` functions. A one-line adoption that silently no-ops is the exact
defect class this repo keeps hitting, so the guard must EXECUTE the real
function body and observe the files on disk -- not grep the source, and not
call the helpers directly (which would only prove the helpers work, a fact
their own tests already cover).

Interface contract
------------------
`run_trainer(monkeypatch, module, train_fn, config, results_dir, **kw)`

- `monkeypatch`: pytest's fixture. Every patch is applied to `module`'s OWN
  namespace, so nothing leaks to other importers.
- `module`: the imported trainer module (e.g. `train.gpt2.pretrain`).
- `train_fn`: the callable to invoke, taken off `module`.
- `config`: a real config instance from that module -- NOT a stub. The whole
  point is that the config the trainer serializes is the real one.
- `results_dir`: the directory `create_nlp_callbacks` is made to return; the
  adopted calls must write into exactly this directory.
- Keyword flags select the per-module extras (`compile_attr`, `dataset_loader`,
  `steps_attr`, `plot_callback`, `model_loader`).

A `phase_scheduler: bool` flag lived here too, stubbing out `PhaseScheduler` for
`train.wave_field.train_memory`. That trainer was DELETED on user instruction
(2026-08-13) and it was the flag's only caller, so the flag was removed with it
rather than left as a knob nothing sets. Re-add it (2 lines) if a memory-bank
trainer is ever written again.

Returns the trainer's own return value. Raises whatever the trainer raises --
failures are NOT swallowed, because a swallowed failure is how a vacuous guard
is born.

Every stub is deliberately dumb: no fit, no tokenizer, no dataset. The only
real code that runs is the trainer's own control flow.
"""

from typing import Any, Dict, List, Optional


class _StubHistory:
    """Minimal stand-in for `keras.callbacks.History`."""

    def __init__(self, history: Optional[Dict[str, List[float]]] = None):
        self.history = history if history is not None else {
            "loss": [3.5, 3.0],
            "val_loss": [3.4, 2.9],
        }


class _StubModel:
    """Model stand-in whose `fit` records its call and returns a history."""

    def __init__(self, history: Optional[_StubHistory] = None):
        self.history = history or _StubHistory()
        self.fit_calls = 0
        self.save_calls: List[str] = []

    def fit(self, *args, **kwargs):
        self.fit_calls += 1
        return self.history

    def save(self, path, *args, **kwargs):
        self.save_calls.append(str(path))

    def __call__(self, *args, **kwargs):  # generation-probe closures
        raise AssertionError("stub model must never be invoked for a forward pass")


class _StubDataset:
    """`tf.data`-shaped stand-in supporting only `.take()`."""

    def take(self, n):
        return self


def _noop(*args, **kwargs):
    return None


def _stub_callback_factory(*args, **kwargs):
    """Stand-in for a Keras callback class -- returns a bare recorder object."""

    class _Rec:
        pass

    return _Rec()


def run_trainer(
        monkeypatch,
        module,
        train_fn,
        config,
        results_dir,
        *,
        compile_attr: str = "compile_model",
        dataset_loader: str = "load_train_val_datasets",
        steps_attr: Optional[str] = "_make_steps_per_epoch",
        plot_callback: bool = False,
        model_loader: Optional[str] = None,
        model: Optional[_StubModel] = None,
):
    """Execute `train_fn(config)` with every heavy dependency stubbed out."""
    stub_model = model or _StubModel()

    monkeypatch.setattr(module, "set_seeds", _noop, raising=True)
    monkeypatch.setattr(module, "create_tokenizer", lambda *a, **k: object())
    monkeypatch.setattr(module, compile_attr, _noop)
    monkeypatch.setattr(
        module, "create_nlp_callbacks",
        lambda *a, **k: ([], str(results_dir)),
    )

    if steps_attr is not None:
        monkeypatch.setattr(module, steps_attr, lambda *a, **k: 7)
    else:
        monkeypatch.setattr(
            module, "estimate_clm_steps_per_epoch", lambda *a, **k: 7)

    if dataset_loader == "load_train_val_datasets":
        monkeypatch.setattr(
            module, dataset_loader,
            lambda *a, **k: (_StubDataset(), _StubDataset(), 11),
        )
    else:
        monkeypatch.setattr(
            module, dataset_loader,
            lambda *a, **k: (_StubDataset(), _StubDataset()),
        )

    for name in ("StepCheckpointCallback", "GenerationProbeCallback"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, _stub_callback_factory)
    if plot_callback:
        monkeypatch.setattr(module, "StepPlotCallback", _stub_callback_factory)
    if hasattr(module, "generate_training_curves"):
        monkeypatch.setattr(module, "generate_training_curves", _noop)

    if model_loader is not None:
        monkeypatch.setattr(module, model_loader, lambda *a, **k: stub_model)
        return train_fn(config), stub_model

    return train_fn(config, model_factory=lambda cfg: stub_model), stub_model


__all__ = [
    "run_trainer",
    "_StubModel",
    "_StubHistory",
    "_StubDataset",
]
