"""The token-mean language-model losses couple their rows through ``N``.

``NanoVLMLoss``, ``MaskedCausalLMLoss`` (and its subclasses),
``FocalCausalLMLoss`` and the language-model term of ``HRMLoss`` return one value
per row, ``S_i / N * B``, with ``N`` the number of non-masked tokens in the WHOLE
batch. ``call()`` never sees ``sample_weight``, so a zero weight removes a row's
contribution but not its tokens from ``N``:

    loss(w with row d zeroed) == loss(batch without row d) * (N - n_d) / N

which is the ``(B - 1) / B`` of a plain row drop only when every row has the same
number of valid tokens. Their docstrings used to say the batch axis "lets
``sample_weight`` and ``reduction=`` select rows"; this file pins the coupling that
is true and fails if that sentence comes back
(plan-2026-11-11T131349-13b28b37 D-046). The ``NanoVLMLoss`` float64 reference test
lives in ``test_the_nano_vlm_loss_scores_the_text_positions.py``.
"""

import numpy as np
import pytest

from dl_techniques.losses.focal_causal_lm_loss import FocalCausalLMLoss
from dl_techniques.losses.hrm_loss import HRMLoss
from dl_techniques.losses.masked_causal_lm_loss import (
    MaskedCausalLMLoss,
    PrefixMaskedCausalLMLoss,
)
from dl_techniques.losses.nano_vlm_loss import NanoVLMLoss

B, T, V = 4, 7, 11


def _batch(ignore):
    """Rows with 6, 6, 6 and 2 valid tokens (unequal on purpose)."""
    rng = np.random.default_rng(3)
    y_pred = rng.normal(size=(B, T, V)).astype("float32")
    y_true = rng.integers(0, V, (B, T)).astype("int32")
    y_true[3, 2:] = ignore
    return y_true, y_pred


def _plain(cls, **kwargs):
    def run(y_true, y_pred, sample_weight=None):
        return float(cls(**kwargs)(y_true, y_pred, sample_weight=sample_weight))
    return run, -1  # ignore_index, valid mask = labels != -1


def _hrm():
    def run(y_true, y_pred, sample_weight=None):
        loss = HRMLoss(q_loss_weight=0.0)
        return float(loss(
            {"labels": y_true},
            {"logits": y_pred,
             "q_halt_logits": np.zeros(y_true.shape[0], "float32")},
            sample_weight=sample_weight,
        ))
    return run, -100


def _nano():
    def run(y_true, y_pred, sample_weight=None):
        return float(NanoVLMLoss(ignore_index=-1)(
            y_true, y_pred, sample_weight=sample_weight))
    return run, -1


_CASES = {
    "MaskedCausalLMLoss": _plain(MaskedCausalLMLoss),
    "PrefixMaskedCausalLMLoss": _plain(PrefixMaskedCausalLMLoss),
    "FocalCausalLMLoss": _plain(FocalCausalLMLoss),
    "HRMLoss": _hrm(),
    "NanoVLMLoss": _nano(),
}


@pytest.mark.parametrize("dead", [0, 3])
@pytest.mark.parametrize("name", sorted(_CASES))
def test_a_zero_sample_weight_keeps_the_row_s_tokens_in_the_denominator(name, dead):
    run, ignore = _CASES[name]
    y_true, y_pred = _batch(ignore)
    weights = np.ones(B, dtype="float32")
    weights[dead] = 0.0
    keep = [i for i in range(B) if i != dead]

    weighted = run(y_true, y_pred, weights)
    dropped = run(y_true[keep], y_pred[keep])

    targets = y_true[:, 1:] if name == "NanoVLMLoss" else y_true
    counts = (targets != ignore).sum(axis=-1)
    ratio = (counts.sum() - counts[dead]) / counts.sum()
    np.testing.assert_allclose(weighted, dropped * ratio, rtol=3e-6, atol=0)
    assert abs(weighted * B / (B - 1) - dropped) > 0.05 * abs(dropped)


@pytest.mark.parametrize("cls", [
    MaskedCausalLMLoss, FocalCausalLMLoss, HRMLoss, NanoVLMLoss,
], ids=lambda cls: cls.__name__)
def test_no_docstring_claims_that_sample_weight_selects_rows(cls):
    """The coupled form must not be documented as row selection."""
    doc = " ".join((cls.call.__doc__ or "").split())
    assert "select rows" not in doc
    assert "COUPLED" in doc
