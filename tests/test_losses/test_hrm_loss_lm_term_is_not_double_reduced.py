"""`HRMLoss`'s LM term must be averaged over tokens EXACTLY ONCE.

The defect this pins, found 2026-08-31: `HRMLoss.call` computes
``lm_loss = sum(lm_losses) / sum(valid_counts)``, which is correct only if the LM sub-loss returns
PER-TOKEN values. The ``sparse_categorical_crossentropy`` branch always built its sub-loss with
``reduction="none"`` and was fine. The ``stable_max`` branch -- **the default** -- did not, so
``Loss.__call__`` handed back an already-averaged scalar and that line divided it by the token
count a second time.

Measured under-weighting of the LM term, which is exactly ``sum(valid_counts)``:
``24x`` at 4x6, ``512x`` at 8x64, ``4096x`` at 8x512. At realistic sizes the language-modelling
objective all but vanishes beside the Q-learning terms it is summed with -- while the loss stays
finite and the curve looks healthy. Both shipped trainers (`src/train/hrm/train_hrm.py:45`,
`src/train/tiny_recursive_model/train_trm.py:145`) default to ``stable_max``, so both were affected.

Nothing else in the suite pins an `HRMLoss` value, which is why this went unseen.

Note on the fixture: `_batch` emits NO ignored labels, and that is load-bearing for the
branch-comparison test below. `HRMLoss.__init__` passes
``ignore_class=ignore_index if ignore_index >= 0 else None`` to
`SparseCategoricalCrossentropy`, so at the DEFAULT ``ignore_index=-100`` that branch receives the
raw negative labels and TensorFlow raises
``Received a label value of -100 which is outside the valid range of [0, 11)``. Measured
2026-08-31; a pre-existing defect in the non-default branch, recorded here rather than fixed, and
the reason this file cannot compare the two branches on a masked batch.
"""

import numpy as np
import keras
import pytest

from dl_techniques.losses.hrm_loss import HRMLoss, StableMaxCrossEntropy


def _batch(B, T, V, seed=0):
    rng = np.random.default_rng(seed)
    y_true = {"labels": rng.integers(0, V, size=(B, T)).astype("int32")}
    y_pred = {
        "logits": rng.normal(size=(B, T, V)).astype("float32"),
        "q_halt_logits": rng.normal(size=(B,)).astype("float32"),
    }
    return y_true, y_pred


def test_the_stable_max_sub_loss_is_built_unreduced():
    """The default branch's sub-loss must hand back per-token values.

    This is the root cause in one assertion. A failure means someone dropped
    ``reduction="none"`` and the LM term is being divided twice again.
    """
    loss = HRMLoss(lm_loss_type="stable_max")
    assert loss.lm_loss_fn.reduction == "none", (
        f"HRMLoss's stable_max sub-loss has reduction={loss.lm_loss_fn.reduction!r}, not 'none'. "
        f"call() does sum(lm_losses)/sum(valid_counts), so a reducing sub-loss makes that a SECOND "
        f"division by the token count and the LM term is under-weighted by sum(valid_counts)."
    )


def test_the_sub_loss_returns_per_token_values_not_a_scalar():
    """`StableMaxCrossEntropy` under HRM's construction must be token-shaped."""
    B, T, V = 4, 6, 11
    y_true, y_pred = _batch(B, T, V)
    sub = HRMLoss(lm_loss_type="stable_max").lm_loss_fn
    out = sub(y_true["labels"], y_pred["logits"])
    shape = tuple(np.shape(keras.ops.convert_to_numpy(out)))
    assert shape == (B, T), (
        f"the LM sub-loss returned shape {shape}, expected {(B, T)} (per token). A rank-0 return "
        f"is the double-reduction defect."
    )


@pytest.mark.parametrize("B,T,V", [(4, 6, 11), (8, 64, 101)])
def test_the_lm_term_scale_does_not_collapse_with_sequence_length(B, T, V):
    """The LM term must not shrink as the token count grows.

    This is the behavioural tell, and it is what makes the defect detectable without knowing the
    implementation: under the bug the LM term is divided by ``sum(valid_counts)``, so it collapses
    toward zero as B*T grows. Correctly reduced, it is an average and stays O(1).
    """
    y_true, y_pred = _batch(B, T, V)
    labels = y_true["labels"]
    valid = np.maximum((labels != -100).sum(-1), 1.0).sum()

    sub = HRMLoss(lm_loss_type="stable_max").lm_loss_fn
    per_token = keras.ops.convert_to_numpy(sub(labels, y_pred["logits"]))
    lm_term = float(per_token.sum() / valid)

    assert lm_term > 0.5, (
        f"LM term {lm_term!r} at B={B} T={T} is implausibly small for a cross-entropy over "
        f"{V} classes (chance is ~{np.log(V):.2f}). Under the double-reduction defect this value "
        f"is the true one divided by sum(valid_counts)={valid:.0f}."
    )


def test_both_lm_loss_type_branches_agree_in_scale():
    """`stable_max` and the SCC branch must land on the same order of magnitude.

    They are different functions and must NOT be asserted equal -- but a 10x-plus gap between them
    means one branch is reducing differently from the other, which is exactly how this defect
    presented.
    """
    B, T, V = 8, 64, 101
    y_true, y_pred = _batch(B, T, V)

    # `call()` returns (batch,) since plan-2026-08-31T045723-c0d5ffa9 step 9, so
    # the comparison is on the REDUCED value -- which is the same quantity this
    # test read before that step, when `call()` was already scalar. The shape is
    # asserted here rather than assumed, so a regression to a scalar `call()`
    # fails loudly instead of silently changing what is being compared.
    for lm_type in ("stable_max", "sparse_categorical_crossentropy"):
        shape = tuple(
            np.shape(
                keras.ops.convert_to_numpy(
                    HRMLoss(lm_loss_type=lm_type).call(y_true, y_pred)
                )
            )
        )
        assert shape == (B,), (
            f"HRMLoss(lm_loss_type={lm_type!r}).call() returned {shape}, expected {(B,)}. "
            f"A scalar return makes sample_weight charge every row the batch aggregate."
        )

    a = float(keras.ops.convert_to_numpy(HRMLoss(lm_loss_type="stable_max")(y_true, y_pred)))
    b = float(keras.ops.convert_to_numpy(
        HRMLoss(lm_loss_type="sparse_categorical_crossentropy")(y_true, y_pred)))
    ratio = max(a, b) / min(a, b)
    assert ratio < 3.0, (
        f"stable_max total {a!r} vs scc total {b!r} differ by {ratio:.1f}x. The two branches "
        f"should differ only by the choice of cross-entropy, not by a reduction convention."
    )


# ---------------------------------------------------------------------
# The OTHER branch's defect: it crashed at its own default ignore_index
# ---------------------------------------------------------------------


@pytest.mark.parametrize("lm_loss_type", ["stable_max", "sparse_categorical_crossentropy"])
def test_both_branches_survive_masked_labels_at_the_default_ignore_index(lm_loss_type):
    """Neither branch may crash on a masked batch at the DEFAULT ``ignore_index=-100``.

    The ``sparse_categorical_crossentropy`` branch used to. ``__init__`` passes
    ``ignore_class=ignore_index if ignore_index >= 0 else None``, so at the default the sub-loss
    was told to ignore NOTHING and TF raised
    ``Received a label value of -100 which is outside the valid range of [0, V)``.

    Padding a batch is the normal case, not an edge case, so this branch was unusable exactly when
    it mattered. A failure here means the ignored labels are reaching the cross-entropy kernel raw
    again.
    """
    B, T, V = 4, 6, 11
    y_true, y_pred = _batch(B, T, V)
    labels = y_true["labels"]
    labels[1, 3:] = -100          # ragged padding, the ordinary case
    labels[2, 4:] = -100

    out = HRMLoss(lm_loss_type=lm_loss_type).call(y_true, y_pred)
    arr = keras.ops.convert_to_numpy(out)
    assert arr.shape == (B,), f"{lm_loss_type}: expected per-sample {(B,)}, got {arr.shape}"
    assert np.all(np.isfinite(arr)), f"{lm_loss_type}: non-finite loss {arr!r}"


def test_masked_positions_contribute_nothing_to_the_scc_branch():
    """Clamping the ignored labels must not let them into the loss.

    The fix replaces ignored labels with class 0 so the kernel's range check passes. That is only
    sound because those positions are zeroed afterwards. If the zeroing is dropped, class 0's
    cross-entropy silently enters the token sum, and this test is what notices.
    """
    B, T, V = 4, 6, 11
    y_true, y_pred = _batch(B, T, V)
    labels = y_true["labels"]
    labels[1, 3:] = -100

    loss = HRMLoss(lm_loss_type="sparse_categorical_crossentropy")
    base = keras.ops.convert_to_numpy(loss.call(y_true, y_pred))

    # Change the LOGITS only at masked positions. A correct implementation cannot see it.
    #
    # Perturb ONE class, not all of them. Adding a constant to every logit at a position
    # is invisible to cross-entropy because softmax is SHIFT-INVARIANT -- a uniform bump
    # makes this test pass whether or not the masking works, i.e. a guard that cannot
    # fail. Measured: the uniform version stayed green under an injection that deleted
    # the zeroing outright.
    y_pred2 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in y_pred.items()}
    y_pred2["logits"][1, 3:, 0] += 25.0
    after = keras.ops.convert_to_numpy(loss.call(y_true, y_pred2))

    np.testing.assert_allclose(
        base, after, rtol=0, atol=1e-6,
        err_msg=("perturbing the logits at IGNORED positions changed the loss, so those positions "
                 "are contributing to the token sum. The clamp-to-class-0 is leaking."),
    )
