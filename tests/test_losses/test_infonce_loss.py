"""Tests for :mod:`dl_techniques.losses.infonce_loss`.

The load-bearing artifact in this file is :func:`_reference_simcse_loss`: a **verbatim,
op-for-op transcription** of the trainer-private ``SimCSELoss.call`` body at
``src/train/embeddings_experimental/train_embeddings.py:260-281`` (read 2026-08-30).

It is transcribed from *that* file and never from
:class:`dl_techniques.losses.infonce_loss.SymmetricInfoNCELoss`. That independence is the
whole point: an oracle copied from the implementation it grades passes forever and proves
nothing. The trainer module is deliberately **not imported** -- importing it pulls in the
tensorflow-heavy study pipeline, and a live training run is executing against that file.
"""

import keras
import numpy as np
import pytest

from dl_techniques.losses.infonce_loss import SymmetricInfoNCELoss

# ---------------------------------------------------------------------
# Reference oracle -- verbatim transcription, do not "simplify"
# ---------------------------------------------------------------------


def _reference_simcse_loss(y_pred, temperature):
    """The reference ``SimCSELoss.call`` body, transcribed op-for-op.

    Source: ``src/train/embeddings_experimental/train_embeddings.py:260-281``.

    :param y_pred: Stacked views, ``(batch, 2, embed_dim)``.
    :param temperature: Softmax temperature over the cosine similarities.
    :returns: The reference **scalar** loss, ``mean(forward + backward) / 2``.
    """
    view_a = y_pred[:, 0, :]
    view_b = y_pred[:, 1, :]
    logits = keras.ops.matmul(
        view_a, keras.ops.transpose(view_b)
    ) / temperature
    targets = keras.ops.arange(keras.ops.shape(logits)[0])
    forward = keras.losses.sparse_categorical_crossentropy(
        targets, logits, from_logits=True
    )
    backward = keras.losses.sparse_categorical_crossentropy(
        targets, keras.ops.transpose(logits), from_logits=True
    )
    return keras.ops.mean(forward + backward) / 2.0


# ---------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------


def _l2_normalized_views(batch: int, dim: int, seed: int) -> np.ndarray:
    """Build an L2-normalized ``(batch, 2, dim)`` float32 array.

    Unit-norm rows are what make the logits genuine cosine similarities, which is the
    regime the study's temperature (0.05) is calibrated for.
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((batch, 2, dim)).astype("float32")
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    return (raw / norms).astype("float32")


# ---------------------------------------------------------------------
# H-2: numerical equivalence with the reference implementation
# ---------------------------------------------------------------------


@pytest.mark.parametrize("batch", [2, 4, 8, 64])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_the_loss_reproduces_the_reference_simcse_loss(batch, seed):
    """The reduced loss equals the trainer's scalar ``SimCSELoss`` value.

    A failure here means the library loss is **not** a drop-in replacement for
    ``train_embeddings.py``'s local ``SimCSELoss``, and the deferred trainer swap would
    silently change the in-flight study's numbers. Do NOT loosen the tolerance to make
    this pass -- diagnose the divergence.
    """
    y_pred = _l2_normalized_views(batch=batch, dim=256, seed=seed)
    temperature = 0.05

    expected = float(keras.ops.convert_to_numpy(
        _reference_simcse_loss(keras.ops.convert_to_tensor(y_pred), temperature)
    ))

    loss_fn = SymmetricInfoNCELoss(temperature=temperature)
    actual = float(keras.ops.convert_to_numpy(
        loss_fn(keras.ops.zeros((batch,), dtype="float32"), y_pred)
    ))

    assert actual == pytest.approx(expected, abs=1e-6, rel=0.0), (
        f"batch={batch} seed={seed}: new loss {actual!r} != reference {expected!r}. "
        f"The library loss is not equivalent to the trainer's SimCSELoss."
    )


# ---------------------------------------------------------------------
# (a) SC3 -- the return shape contract
# ---------------------------------------------------------------------


@pytest.mark.parametrize("batch", [2, 3, 8, 32])
def test_call_returns_one_loss_per_sample_not_a_scalar(batch):
    """``call()`` returns a rank-1 ``(batch,)`` vector, never a scalar.

    A failure here means the loss has collapsed its own per-sample axis before the parent
    :class:`keras.losses.Loss` gets to reduce it. Keras passes a scalar through
    ``reduce_values`` unchanged (``keras/src/losses/loss.py:143-147``), so such a loss
    silently ignores ``sample_weight``, ignores masking, and skips
    ``scale_loss_for_distribution`` -- it does not raise, it just quietly stops honouring
    three parts of the loss contract.
    """
    y_pred = _l2_normalized_views(batch=batch, dim=32, seed=11)
    loss_fn = SymmetricInfoNCELoss(temperature=0.05)

    per_sample = keras.ops.convert_to_numpy(loss_fn.call(None, y_pred))

    assert per_sample.ndim == 1, (
        f"batch={batch}: call() returned rank {per_sample.ndim} (shape "
        f"{per_sample.shape}); it must be rank 1. A rank-0 return means the loss reduced "
        f"internally and will silently ignore sample_weight and masking."
    )
    assert per_sample.shape[0] == batch, (
        f"batch={batch}: call() returned {per_sample.shape[0]} values; it must return "
        f"exactly one loss per sample so the parent reduction and sample_weight align "
        f"element-wise with the batch."
    )


# ---------------------------------------------------------------------
# (b) SC3 -- the scalar-vs-vector reduction identity (pins F-7, does not restate it)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("batch", [2, 4, 8, 64])
def test_the_default_reduction_of_the_per_sample_vector_equals_the_reference_scalar(batch):
    """``loss_fn(...)`` == ``mean(fwd + bwd)/2`` == ``sum(call(...))/batch``.

    This is the claim that lets the loss return a ``(batch,)`` vector *and* stay a
    drop-in for the reference's scalar. A failure means those two goals are in genuine
    tension: either the parent's ``sum_over_batch_size`` is not dividing by the batch as
    assumed, or ``call()`` is no longer returning the per-sample decomposition of the
    reference's mean. Do not "fix" it by reducing inside ``call()`` -- that trades the
    ``sample_weight`` contract away to hide the symptom.
    """
    y_pred = _l2_normalized_views(batch=batch, dim=256, seed=5)
    loss_fn = SymmetricInfoNCELoss(temperature=0.05)

    reduced = float(keras.ops.convert_to_numpy(
        loss_fn(keras.ops.zeros((batch,), dtype="float32"), y_pred)
    ))
    reference_scalar = float(keras.ops.convert_to_numpy(
        _reference_simcse_loss(keras.ops.convert_to_tensor(y_pred), 0.05)
    ))
    per_sample = keras.ops.convert_to_numpy(loss_fn.call(None, y_pred))
    hand_reduced = float(per_sample.sum() / batch)

    assert reduced == pytest.approx(reference_scalar, abs=1e-6, rel=0.0), (
        f"batch={batch}: reduced {reduced!r} != reference mean(fwd+bwd)/2 "
        f"{reference_scalar!r}. The vector return has changed the reported number, so it "
        f"is NOT a value-preserving refactor of the reference."
    )
    assert reduced == pytest.approx(hand_reduced, abs=1e-6, rel=0.0), (
        f"batch={batch}: reduced {reduced!r} != sum(call(...))/batch {hand_reduced!r}. "
        f"The parent reduction is not 'sum_over_batch_size' over this vector, so the "
        f"per-sample values do not decompose the reported loss."
    )


# ---------------------------------------------------------------------
# (c) SC4 -- input-format dispatch
# ---------------------------------------------------------------------


def test_all_four_accepted_y_pred_forms_dispatch_to_the_same_value():
    """Stacked, 2-tuple, 2-list and dict inputs produce the identical loss.

    A failure means ``_split_views`` transposes, reorders or mis-slices one of the forms.
    Because every form is numerically valid on its own, the resulting bug is invisible to
    any single-form test: a caller merely switching from a dict to a stacked tensor would
    silently train against a different objective.
    """
    stacked = _l2_normalized_views(batch=8, dim=64, seed=21)
    view_a, view_b = stacked[:, 0, :], stacked[:, 1, :]
    loss_fn = SymmetricInfoNCELoss(temperature=0.05)
    dummy = keras.ops.zeros((8,), dtype="float32")

    values = {
        "stacked": float(keras.ops.convert_to_numpy(loss_fn(dummy, stacked))),
        "tuple": float(keras.ops.convert_to_numpy(loss_fn(dummy, (view_a, view_b)))),
        "list": float(keras.ops.convert_to_numpy(loss_fn(dummy, [view_a, view_b]))),
        "dict": float(keras.ops.convert_to_numpy(
            loss_fn(dummy, {"view_a": view_a, "view_b": view_b})
        )),
    }

    for form, value in values.items():
        assert value == pytest.approx(values["stacked"], abs=1e-6, rel=0.0), (
            f"the {form!r} input form gives {value!r} but the stacked form gives "
            f"{values['stacked']!r}. The accepted input forms are supposed to be pure "
            f"notation; a caller changing form must not change the objective."
        )


# ---------------------------------------------------------------------
# (d) SC5 -- temperature validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize("temperature", [0.0, -0.05])
def test_a_non_positive_temperature_raises_at_construction(temperature):
    """``temperature <= 0`` is rejected in ``__init__``, not at the first forward pass.

    A failure means a caller can build the loss with a temperature that either divides by
    zero (0.0) or flips the sign of every logit (negative), turning the objective into its
    own opposite -- training that *runs*, produces finite numbers, and maximizes the
    distance between positives.
    """
    with pytest.raises(ValueError, match=r"temperature must be > 0"):
        SymmetricInfoNCELoss(temperature=temperature)


@pytest.mark.parametrize(
    "temperature", [True, False, "0.05", None, [0.05]],
    ids=["bool_true", "bool_false", "str", "none", "list"],
)
def test_a_non_numeric_temperature_raises_at_construction(temperature):
    """A bool or non-number temperature is rejected before it can silently coerce.

    ``bool`` is called out explicitly because ``True`` is a perfectly valid ``int`` of
    value 1 in Python: without the ``isinstance(..., bool)`` check a
    ``temperature=True`` typo would build a *working* loss at temperature 1.0, 20x off the
    study's setting, with no error anywhere.
    """
    with pytest.raises(ValueError, match=r"temperature must be a positive real number"):
        SymmetricInfoNCELoss(temperature=temperature)


# ---------------------------------------------------------------------
# (e) SC5 -- malformed y_pred
# ---------------------------------------------------------------------


def test_a_rank_2_y_pred_raises_instead_of_being_reinterpreted():
    """A single ``(batch, dim)`` tensor is rejected, not silently treated as two views.

    A failure means a caller who forgot to stack their views gets a plausible number back
    from an objective computed over the wrong axis.
    """
    loss_fn = SymmetricInfoNCELoss(temperature=0.05)
    with pytest.raises(ValueError, match=r"must be the stacked rank-3 form"):
        loss_fn.call(None, _l2_normalized_views(batch=4, dim=16, seed=1)[:, 0, :])


def test_a_stacked_y_pred_with_three_views_raises():
    """``shape[1] == 3`` is rejected rather than having its third view dropped.

    A failure means a three-view augmentation pipeline would train on views 0 and 1 only,
    silently discarding a third of the signal it paid to compute.
    """
    loss_fn = SymmetricInfoNCELoss(temperature=0.05)
    bad = np.zeros((4, 3, 16), dtype="float32")
    with pytest.raises(ValueError, match=r"exactly 2 views on axis 1"):
        loss_fn.call(None, bad)


def test_a_three_element_y_pred_sequence_raises():
    """A 3-tuple is rejected rather than having its extra element ignored.

    A failure means the same silent view-dropping as the rank-3 case, reached through the
    tuple form instead.
    """
    loss_fn = SymmetricInfoNCELoss(temperature=0.05)
    view = _l2_normalized_views(batch=4, dim=16, seed=2)[:, 0, :]
    with pytest.raises(ValueError, match=r"exactly 2 views"):
        loss_fn.call(None, (view, view, view))


def test_a_y_pred_dict_missing_a_view_key_raises_naming_the_missing_key():
    """A dict without ``view_b`` raises and names what is missing.

    A failure means a ``KeyError`` (or worse, a silently defaulted view) instead of a
    message that tells the caller which key their pipeline dropped.
    """
    loss_fn = SymmetricInfoNCELoss(temperature=0.05)
    view = _l2_normalized_views(batch=4, dim=16, seed=3)[:, 0, :]
    with pytest.raises(ValueError, match=r"missing the key\(s\) \['view_b'\]"):
        loss_fn.call(None, {"view_a": view})


# ---------------------------------------------------------------------
# (f) SC5 -- the degenerate batch of one
# ---------------------------------------------------------------------


def test_a_statically_known_batch_of_one_raises_rather_than_returning_zero():
    """A ``(1, 2, dim)`` input raises instead of reporting a healthy-looking 0.0.

    With one sample the logit matrix is 1x1, its softmax is identically 1.0, so the loss
    is exactly 0.0 and the gradient exactly zero for *any* embeddings. A failure here
    means that no-op is reachable silently.

    **NOT COVERED BY THIS GUARD: the dynamic batch axis.** Under ``fit()`` the batch axis
    is ``None``, ``shape[0]`` carries no ``1`` to test, and this check cannot fire at all.
    The guard is partial by construction (decisions.md D-004); the mitigation for the
    dynamic case is ``drop_remainder=True`` on the dataset, and nothing in this file
    verifies it.
    """
    loss_fn = SymmetricInfoNCELoss(temperature=0.05)
    with pytest.raises(ValueError, match=r"batch size of 1"):
        loss_fn.call(None, _l2_normalized_views(batch=1, dim=16, seed=4))


# ---------------------------------------------------------------------
# (g) perfect alignment -- the closed-form VALUE, not merely finiteness
# ---------------------------------------------------------------------


@pytest.mark.parametrize("temperature", [1.0, 0.5, 0.2])
def test_perfect_alignment_of_orthonormal_views_equals_the_closed_form(temperature):
    """With ``view_a == view_b`` and orthonormal rows the loss is a hand-derivable number.

    Derivation. Let the ``N`` rows of ``A`` be mutually orthonormal and ``view_b == A``.
    Then ``logits = A @ A.T / t = I / t``: every diagonal entry is ``1/t`` and every
    off-diagonal entry is exactly ``0``. Each row's softmax cross-entropy against its own
    index is therefore::

        -log( exp(1/t) / (exp(1/t) + (N-1)*exp(0)) ) = log(1 + (N-1)*exp(-1/t))

    identical for every row, and identical for the transposed direction (``I/t`` is
    symmetric), so their average -- and hence the reduced loss -- is that same number.
    With ``N = 4`` this is ``log(1 + 3*exp(-1/t))``.

    A failure here means the loss is not computing the softmax cross-entropy it claims to:
    a wrong temperature placement, a missing diagonal target, or a transposed logit matrix
    all move this value while leaving the loss finite, positive and superficially sane.
    Asserting only finiteness or "it's small" would catch none of them.
    """
    batch, dim = 4, 256
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.standard_normal((dim, batch)))
    rows = q.T.astype("float32")                      # (batch, dim), orthonormal rows
    assert np.abs(rows @ rows.T - np.eye(batch)).max() < 1e-5, (
        "the test's own fixture is not orthonormal, so the closed form below does not "
        "apply -- this is a defect in the test, not in the loss."
    )
    y_pred = np.stack([rows, rows], axis=1)           # view_a == view_b

    expected = float(np.log1p((batch - 1) * np.exp(-1.0 / temperature)))
    actual = float(keras.ops.convert_to_numpy(
        SymmetricInfoNCELoss(temperature=temperature)(
            keras.ops.zeros((batch,), dtype="float32"), y_pred
        )
    ))

    assert actual == pytest.approx(expected, abs=1e-5, rel=0.0), (
        f"t={temperature}: loss {actual!r} != closed form log(1+{batch - 1}*exp(-1/t)) "
        f"= {expected!r}. On perfectly aligned orthonormal views the value is fully "
        f"determined; a mismatch means the objective is not the softmax cross-entropy "
        f"over I/t that this class documents."
    )


# ---------------------------------------------------------------------
# (h) y_true is ignored
# ---------------------------------------------------------------------


def test_y_true_is_ignored_entirely():
    """Zeros and arbitrary garbage as ``y_true`` give the identical loss.

    The positives are *positional* -- row ``i`` of view A pairs with row ``i`` of view B.
    A failure means ``y_true`` has become load-bearing, so every existing caller passing a
    dummy tensor (as the reference trainer does) would be silently training against
    labels it never intended to supply.
    """
    batch = 8
    y_pred = _l2_normalized_views(batch=batch, dim=64, seed=31)
    loss_fn = SymmetricInfoNCELoss(temperature=0.05)

    zeros = keras.ops.zeros((batch,), dtype="float32")
    garbage = keras.ops.convert_to_tensor(
        np.random.default_rng(99).standard_normal((batch,)).astype("float32") * 1e3
    )

    with_zeros = float(keras.ops.convert_to_numpy(loss_fn(zeros, y_pred)))
    with_garbage = float(keras.ops.convert_to_numpy(loss_fn(garbage, y_pred)))

    assert with_zeros == with_garbage, (
        f"y_true changed the loss: zeros gave {with_zeros!r}, garbage gave "
        f"{with_garbage!r}. y_true must be entirely unused; if it is not, the documented "
        f"'pass any placeholder' contract is false."
    )


# ---------------------------------------------------------------------
# (i) normalize_inputs is a LIVE knob (dead-knob guard)
# ---------------------------------------------------------------------


def test_normalize_inputs_is_a_live_knob_on_unnormalized_input():
    """``normalize_inputs`` visibly changes the loss when the input is not unit-norm.

    A failure means the flag is DEAD: it is accepted, serialized and documented, but the
    normalization it promises never runs. Callers who set it precisely because they cannot
    guarantee unit-norm embeddings would then get the plausible, wrong number the flag
    exists to prevent -- with no error and no warning.
    """
    rng = np.random.default_rng(3)
    unnormalized = (rng.standard_normal((4, 2, 16)) * 3.0).astype("float32")
    dummy = keras.ops.zeros((4,), dtype="float32")

    off = float(keras.ops.convert_to_numpy(
        SymmetricInfoNCELoss(temperature=0.05, normalize_inputs=False)(dummy, unnormalized)
    ))
    on = float(keras.ops.convert_to_numpy(
        SymmetricInfoNCELoss(temperature=0.05, normalize_inputs=True)(dummy, unnormalized)
    ))

    assert abs(off - on) > 1e-3, (
        f"normalize_inputs is a DEAD knob: False gave {off!r} and True gave {on!r} on "
        f"deliberately unnormalized input, a difference of {abs(off - on)!r}. The flag "
        f"is not reaching the computation."
    )


def test_normalize_inputs_is_a_no_op_on_already_normalized_input():
    """On unit-norm input the flag changes nothing -- it normalizes, it does not rescale.

    A failure means ``normalize_inputs=True`` is doing something *other* than L2
    normalization (a different axis, a mean-subtraction, a scale), which would make the
    defensive setting silently change the objective for every caller whose embeddings were
    already normalized -- including the study this loss was written for.
    """
    normalized = _l2_normalized_views(batch=4, dim=16, seed=3)
    dummy = keras.ops.zeros((4,), dtype="float32")

    off = float(keras.ops.convert_to_numpy(
        SymmetricInfoNCELoss(temperature=0.05, normalize_inputs=False)(dummy, normalized)
    ))
    on = float(keras.ops.convert_to_numpy(
        SymmetricInfoNCELoss(temperature=0.05, normalize_inputs=True)(dummy, normalized)
    ))

    assert on == pytest.approx(off, abs=1e-6, rel=0.0), (
        f"on already-normalized input normalize_inputs=True gave {on!r} but False gave "
        f"{off!r}. Re-normalizing a unit-norm vector must be an identity; a difference "
        f"means the 'defensive' path is not the same objective."
    )


# ---------------------------------------------------------------------
# (j) SC6 -- serialization round trips, both re-evaluated on fixed input
# ---------------------------------------------------------------------


def _fixed_evaluation(loss_fn) -> float:
    """Evaluate ``loss_fn`` on one fixed input so two instances can be compared by VALUE."""
    y_pred = _l2_normalized_views(batch=8, dim=64, seed=77)
    return float(keras.ops.convert_to_numpy(
        loss_fn(keras.ops.zeros((8,), dtype="float32"), y_pred)
    ))


@pytest.mark.parametrize("normalize_inputs", [False, True])
def test_get_config_from_config_round_trip_preserves_the_computed_value(normalize_inputs):
    """``from_config(get_config())`` rebuilds a loss that computes the identical number.

    Comparing config dicts alone would pass even if a key were never read back by
    ``__init__``. A failure here means a constructor argument does not survive the round
    trip, so a model saved with this loss silently reloads with different behaviour.
    """
    original = SymmetricInfoNCELoss(temperature=0.11, normalize_inputs=normalize_inputs)
    restored = SymmetricInfoNCELoss.from_config(original.get_config())

    assert restored.temperature == original.temperature, (
        f"temperature did not survive get_config/from_config: {restored.temperature!r} "
        f"!= {original.temperature!r}."
    )
    assert restored.normalize_inputs == original.normalize_inputs, (
        f"normalize_inputs did not survive get_config/from_config: "
        f"{restored.normalize_inputs!r} != {original.normalize_inputs!r}."
    )
    assert _fixed_evaluation(restored) == pytest.approx(
        _fixed_evaluation(original), abs=1e-6, rel=0.0
    ), (
        "the rebuilt loss computes a different value on identical input, so the config "
        "does not fully describe the object's behaviour."
    )


@pytest.mark.parametrize("normalize_inputs", [False, True])
def test_keras_registry_serialize_deserialize_round_trip_preserves_the_value(
    normalize_inputs,
):
    """The loss survives ``serialize_keras_object``/``deserialize_keras_object``.

    This is the path a ``.keras`` archive actually takes, and it needs the class to be
    findable in the Keras registry. A failure means either the registration decorator is
    missing/misnamed (deserialization raises), or the rebuilt object behaves differently
    -- both of which make any model compiled with this loss unloadable or silently wrong.
    """
    original = SymmetricInfoNCELoss(temperature=0.11, normalize_inputs=normalize_inputs)
    payload = keras.saving.serialize_keras_object(original)
    restored = keras.saving.deserialize_keras_object(payload)

    assert isinstance(restored, SymmetricInfoNCELoss), (
        f"deserialization returned {type(restored).__name__}, not SymmetricInfoNCELoss; "
        f"the class is not resolvable through the Keras registry."
    )
    assert _fixed_evaluation(restored) == pytest.approx(
        _fixed_evaluation(original), abs=1e-6, rel=0.0
    ), (
        "the registry round trip changed the computed value, so a saved model would "
        "reload with a different objective than it was trained with."
    )


# ---------------------------------------------------------------------
# (k) SC1 -- the package surface
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["SymmetricInfoNCELoss", "create_symmetric_infonce_loss"]
)
def test_the_public_names_are_exported_from_the_losses_package(name):
    """Both public names are importable from ``dl_techniques.losses`` and in ``__all__``.

    ``losses/__init__.py`` has no auto-discovery: a module nobody adds a line for is
    invisible to every documented import path, however correct its contents. A failure
    here means the loss ships but cannot be reached the way every peer loss is reached.
    """
    import dl_techniques.losses as losses_pkg

    assert hasattr(losses_pkg, name), (
        f"{name!r} is not importable from dl_techniques.losses; the module is not wired "
        f"into losses/__init__.py, so the documented import path does not work."
    )
    assert name in losses_pkg.__all__, (
        f"{name!r} is missing from dl_techniques.losses.__all__, so it is excluded from "
        f"the declared public API and from `from dl_techniques.losses import *`."
    )


def test_the_factory_builds_an_equivalent_loss_to_the_constructor():
    """``create_symmetric_infonce_loss`` is a faithful wrapper, not a second surface.

    A failure means the factory drops or renames an argument -- the failure mode where the
    documented convenience entry point silently ignores the setting a caller passed it.
    """
    from dl_techniques.losses.infonce_loss import create_symmetric_infonce_loss

    built = create_symmetric_infonce_loss(temperature=0.11, normalize_inputs=True)
    direct = SymmetricInfoNCELoss(temperature=0.11, normalize_inputs=True)

    # `name` is excluded deliberately: keras.losses.Loss auto-uniquifies it per instance
    # ("..._45" vs "..._46"), so it can never match and says nothing about the wrapper.
    built_config = {k: v for k, v in built.get_config().items() if k != "name"}
    direct_config = {k: v for k, v in direct.get_config().items() if k != "name"}

    assert built_config == direct_config, (
        f"the factory produced config {built_config!r} but the constructor "
        f"produced {direct_config!r}; the factory is not a faithful wrapper."
    )
    assert _fixed_evaluation(built) == pytest.approx(
        _fixed_evaluation(direct), abs=1e-6, rel=0.0
    ), (
        "the factory-built loss computes a different value than the directly-constructed "
        "one on identical input, so the factory is not a pass-through."
    )


# ---------------------------------------------------------------------
# D-002: what the (batch,) return actually buys — sample_weight correctness
# ---------------------------------------------------------------------


def _scalar_returning_variant(temperature: float):
    """The reference's premature-``mean`` shape, as a Loss subclass.

    This exists ONLY as the counterfactual for the test below: it is what
    :class:`SymmetricInfoNCELoss` would be if ``call`` reduced internally.
    """

    class _ScalarVariant(keras.losses.Loss):
        def call(self, y_true, y_pred):
            del y_true
            view_a, view_b = y_pred[:, 0, :], y_pred[:, 1, :]
            logits = keras.ops.matmul(
                view_a, keras.ops.transpose(view_b)
            ) / temperature
            targets = keras.ops.arange(keras.ops.shape(logits)[0])
            forward = keras.losses.sparse_categorical_crossentropy(
                targets, logits, from_logits=True
            )
            backward = keras.losses.sparse_categorical_crossentropy(
                targets, keras.ops.transpose(logits), from_logits=True
            )
            return keras.ops.mean(forward + backward) / 2.0

    return _ScalarVariant()


def test_sample_weight_selects_rows_which_a_scalar_return_cannot_do():
    """The per-sample return makes ``sample_weight`` mean what it says.

    This is the guard for decisions.md D-002 -- the ACTUAL justification for returning
    ``(batch,)``. Without it D-002 is an unguarded comment.

    Keras applies ``values * sample_weight`` *before* reducing
    (``keras/src/losses/loss.py``, ``reduce_weighted_values``). So a rank-0 return does
    not "ignore" the weights: it broadcasts against them, yielding
    ``whole_batch_loss * mean(sample_weight)`` -- a plausible number that has silently
    discarded WHICH rows were weighted. A failure here means that distinction has been
    lost and the loss can no longer be row-weighted or masked.
    """
    batch, temperature = 8, 0.05
    y_pred = _l2_normalized_views(batch=batch, dim=64, seed=3)
    y_true = keras.ops.zeros((batch,), dtype="float32")
    keep_row_0 = keras.ops.convert_to_tensor(
        np.array([1.0] + [0.0] * (batch - 1), dtype="float32")
    )

    loss_fn = SymmetricInfoNCELoss(temperature=temperature)
    per_sample = keras.ops.convert_to_numpy(
        loss_fn.call(y_true, keras.ops.convert_to_tensor(y_pred))
    )
    weighted = float(keras.ops.convert_to_numpy(
        loss_fn(y_true, y_pred, sample_weight=keep_row_0)
    ))

    # Only row 0 survives, and `sum_over_batch_size` still divides by the full batch.
    expected = float(per_sample[0]) / batch
    assert weighted == pytest.approx(expected, abs=1e-6, rel=0.0), (
        f"weighting to row 0 alone gave {weighted!r}, expected {expected!r} "
        f"(= per_sample[0]/{batch}). sample_weight is not selecting rows, so the "
        f"per-sample return is not doing the one job it exists for."
    )

    # And the counterfactual: the scalar shape cannot express this.
    scalar_weighted = float(keras.ops.convert_to_numpy(
        _scalar_returning_variant(temperature)(y_true, y_pred, sample_weight=keep_row_0)
    ))
    assert abs(scalar_weighted - weighted) > 1e-3, (
        f"the scalar-returning variant produced {scalar_weighted!r}, indistinguishable "
        f"from the per-sample form's {weighted!r}. If these agree, this test is not "
        f"pinning anything and D-002's justification is unsupported."
    )
