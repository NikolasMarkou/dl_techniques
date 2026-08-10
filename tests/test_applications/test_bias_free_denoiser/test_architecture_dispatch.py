"""Regression tests for `DenoiserPrior.from_pretrained`'s ARCHITECTURE DISPATCH.

What is under test here is the branch DECISION, not Keras' ability to read a file:
which loader a checkpoint reaches given (a) its sniffed architecture and (b) the
requested ``resolution``. Both loaders are therefore monkeypatched to record-only
stubs, so these tests are millisecond-fast, need no GPU and no trained artifact.

Why this file exists (plan-2026-08-10-3649c19e/D-027, correcting D-009)
-----------------------------------------------------------------------
Step 2b of that plan deleted the ``bfcliffordunet`` model module and, with it, the
non-ConvUNext ``dynamic`` loader (a factory rebuild that genuinely imported the
module). The refusal it added was placed OUTSIDE the ``resolution`` branch, on the
stated premise that such a checkpoint "can no longer be loaded at all". That premise
is false: ``_load_fixed`` is a plain ``keras.models.load_model`` on the SAVED GRAPH
and consults no model module, so ``resolution="fixed256"`` worked before the deletion
and still works after it (measured on
``results/20260722_cliffordunet_denoiser_hfb3/final_model.keras``: 873,114 params,
finite ``(1, 256, 256, 3)`` forward). The over-scoped refusal turned a working load of
a real on-disk artifact into a hard ``ValueError``.

The two assertions that pin the correction:

* a non-ConvUNext checkpoint REACHES ``_load_fixed`` under ``resolution="fixed256"``
  (the regression: this raised for one commit range);
* ``resolution="dynamic"`` on the same checkpoint raises, and the message NAMES the
  removed ``bfcliffordunet`` factory and POINTS AT ``fixed256`` — i.e. it stays a
  legible refusal and does not regress into a bare ``ModuleNotFoundError``.

Proven RED (2026-08-10): with the refusal moved back outside the ``resolution``
branch (the pre-fix shape), ``test_non_convunext_checkpoint_loads_via_fixed256`` and
``test_fixed256_never_consults_the_architecture_sniffer`` both FAIL with
``ValueError: unsupported denoiser architecture 'cliffordunet'``.
"""

import json
from pathlib import Path
from typing import Dict, List

import keras
import numpy as np
import pytest

from applications.bias_free_denoiser.denoiser_prior import DenoiserPrior

# The stamp the shared provenance gate (D-005) requires; unrelated to architecture,
# but every fixture below must carry it or it is refused before dispatch is reached.
_UNIT_DOMAIN = "[0,1]"


def _tiny_model() -> keras.Model:
    """A 1-layer functional stand-in for a loaded denoiser (`from_pretrained` only
    reads ``.outputs`` / ``.count_params()`` / ``.name`` before wrapping it)."""
    inp = keras.Input(shape=(4, 4, 3))
    return keras.Model(inp, keras.layers.Conv2D(3, 1)(inp), name="stub_denoiser")


def _write_checkpoint(tmp_path: Path, name: str, config: Dict) -> Path:
    """Write a checkpoint directory whose ``.keras`` file is never actually read.

    The loaders are stubbed, so the ``.keras`` payload is irrelevant — only its
    EXISTENCE matters (`_resolve_paths` raises FileNotFoundError otherwise).
    """
    d = tmp_path / name
    d.mkdir()
    (d / "best_model.keras").write_bytes(b"not-a-real-keras-file")
    (d / "config.json").write_text(json.dumps(config))
    return d


@pytest.fixture()
def cliffordunet_ckpt(tmp_path: Path) -> Path:
    """A [0,1]-stamped checkpoint with NO ``convnext_version`` key.

    That absence is exactly what `_detect_architecture` reads, and it is how the real
    `results/20260722_cliffordunet_denoiser_hfb3/config.json` looks.
    """
    return _write_checkpoint(
        tmp_path, "clifford_run", {"data_range": _UNIT_DOMAIN, "variant": "base"},
    )


@pytest.fixture()
def convunext_ckpt(tmp_path: Path) -> Path:
    """A [0,1]-stamped checkpoint carrying ``convnext_version`` (the ConvUNext mark)."""
    return _write_checkpoint(
        tmp_path,
        "convunext_run",
        {"data_range": _UNIT_DOMAIN, "convnext_version": "v1", "variant": "base"},
    )


@pytest.fixture()
def calls(monkeypatch) -> List[str]:
    """Replace both loaders with record-only stubs; return the call log."""
    log: List[str] = []

    def fake_load_fixed(keras_path):
        log.append("fixed")
        return _tiny_model()

    def fake_build_dynamic(cls, keras_path):
        log.append("dynamic")
        return _tiny_model()

    monkeypatch.setattr(DenoiserPrior, "_load_fixed", staticmethod(fake_load_fixed))
    monkeypatch.setattr(
        DenoiserPrior, "_build_dynamic_convunext", classmethod(fake_build_dynamic)
    )
    return log


class TestArchitectureDispatch:
    """`resolution` x sniffed-architecture -> which loader runs (or which refusal)."""

    def test_non_convunext_checkpoint_loads_via_fixed256(
        self, cliffordunet_ckpt: Path, calls: List[str]
    ) -> None:
        """THE REGRESSION: fixed256 must not be gated on architecture.

        `_load_fixed` loads the saved graph, which carries its own layer configs and
        needs no model module — this path never depended on `bfcliffordunet`.
        """
        prior = DenoiserPrior.from_pretrained(
            str(cliffordunet_ckpt), resolution="fixed256"
        )
        assert calls == ["fixed"], (
            f"a non-ConvUNext checkpoint must reach _load_fixed under "
            f"resolution='fixed256'; loaders called: {calls}"
        )
        assert prior.model is not None

    def test_fixed256_never_consults_the_architecture_sniffer(
        self, cliffordunet_ckpt: Path, calls: List[str], monkeypatch
    ) -> None:
        """Anti-vacuity twin: even a sniffer that returns garbage cannot block fixed256.

        Guards the stronger property — `fixed256` dispatch is architecture-INDEPENDENT
        — so a future refusal keyed on some *other* verdict string cannot creep back in.
        """
        monkeypatch.setattr(
            DenoiserPrior,
            "_detect_architecture",
            staticmethod(lambda config_path: "some_unknown_architecture"),
        )
        DenoiserPrior.from_pretrained(str(cliffordunet_ckpt), resolution="fixed256")
        assert calls == ["fixed"]

    def test_non_convunext_dynamic_raises_naming_the_removed_factory(
        self, cliffordunet_ckpt: Path, calls: List[str]
    ) -> None:
        """The dynamic path IS gone for non-ConvUNext, and says so legibly."""
        with pytest.raises(ValueError) as exc:
            DenoiserPrior.from_pretrained(
                str(cliffordunet_ckpt), resolution="dynamic"
            )
        msg = str(exc.value)
        assert "bfcliffordunet" in msg, (
            f"the refusal must name the REMOVED factory module, got: {msg}"
        )
        assert "fixed256" in msg, (
            f"the refusal must point at the load path that still works, got: {msg}"
        )
        assert "dynamic" in msg
        assert calls == [], f"no loader may run before the refusal; called: {calls}"

    def test_convunext_still_takes_the_dynamic_graph_relax_path(
        self, convunext_ckpt: Path, calls: List[str]
    ) -> None:
        """Control: the supported architecture's default path is unchanged."""
        DenoiserPrior.from_pretrained(str(convunext_ckpt))  # default = "dynamic"
        assert calls == ["dynamic"]

    def test_convunext_fixed256_still_works(
        self, convunext_ckpt: Path, calls: List[str]
    ) -> None:
        """Control: fixed256 is unchanged for the supported architecture too."""
        DenoiserPrior.from_pretrained(str(convunext_ckpt), resolution="fixed256")
        assert calls == ["fixed"]

    def test_legacy_domain_still_refused_before_architecture(
        self, tmp_path: Path, calls: List[str]
    ) -> None:
        """Ordering guard (D-009's one correct call): the provenance gate runs FIRST.

        A legacy-domain checkpoint must get the DOMAIN refusal — the more fundamental
        defect — not an architecture message, and not a load.
        """
        legacy = _write_checkpoint(tmp_path, "legacy_run", {"variant": "base"})
        with pytest.raises(ValueError) as exc:
            DenoiserPrior.from_pretrained(str(legacy), resolution="fixed256")
        assert "data_range" in str(exc.value)
        assert calls == []


# The real-artifact counterpart to the stubbed tests above: skipped where the
# checkpoint is absent, so it is portable, but on this machine it is the thing that
# actually proves the saved graph loads with no model module present.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_REAL_CLIFFORD_CKPT = (
    _REPO_ROOT / "results" / "20260722_cliffordunet_denoiser_hfb3" / "final_model.keras"
)


@pytest.mark.slow
def test_real_cliffordunet_checkpoint_loads_and_forwards() -> None:
    """`fixed256` on the user's actual CliffordUNet artifact: loads, forwards, finite."""
    if not _REAL_CLIFFORD_CKPT.is_file():
        pytest.skip(f"checkpoint not present: {_REAL_CLIFFORD_CKPT}")
    prior = DenoiserPrior.from_pretrained(
        str(_REAL_CLIFFORD_CKPT), resolution="fixed256"
    )
    x = np.random.default_rng(0).random((1, 256, 256, 3)).astype("float32")
    out = np.asarray(prior.denoise(x))
    assert out.shape == (1, 256, 256, 3)
    assert np.all(np.isfinite(out))
