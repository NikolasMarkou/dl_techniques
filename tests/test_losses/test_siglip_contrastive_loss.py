"""Tests for ``dl_techniques.losses.siglip_contrastive_loss``.

The module exports three ``keras.losses.Loss`` subclasses
(``SigLIPContrastiveLoss``, ``AdaptiveSigLIPLoss``, ``HybridContrastiveLoss``)
and three factories, and had ZERO tests of any kind before this file.

**Three test groups here are deliberately RED against the current tree.** They
are written to the CORRECT contract, not to the shipped behaviour, and they are
the RED proof for the per-sample-return fix (plan
``plan-2026-08-30T203107-30455f66`` step 4). Every function in the
``PER-SAMPLE CONTRACT (RED)`` section below is expected to FAIL today:

*   ``test_call_returns_one_value_per_sample`` — ``call()`` ends with an
    axis-less ``ops.mean(...)`` in all three classes, so it returns a SCALAR.
    A ``keras.losses.Loss`` must return a value whose leading axis is the batch.
*   ``test_sample_weight_weights_the_named_rows`` — because ``call()`` returns a
    scalar, ``reduce_weighted_values`` multiplies that scalar by
    ``sample_weight`` and reduces, giving exactly
    ``unweighted * mean(sample_weight)``. Every row is charged the batch
    aggregate and WHICH rows were weighted is discarded. Measured today:
    ``loss(y, p, sample_weight=[1,1,1,0]) == loss(y, p) * 0.75`` exactly.
*   ``test_reduction_none_returns_one_value_per_sample`` — with a scalar
    ``call()``, ``reduction`` is a DEAD KNOB: ``'none'``, ``'sum'`` and
    ``'sum_over_batch_size'`` all return the same scalar.

Everything else in this file passes against the current tree and must keep
passing.

Two behaviours of this module shape the tests and are worth stating once:

*   ``AdaptiveSigLIPLoss`` MUTATES ``self.adaptive_temperature`` on every call,
    so a second call on the same instance returns a different number. Every
    comparison here therefore uses a FRESH instance per measurement.
*   ``HybridContrastiveLoss.call`` invokes its inner loss through
    ``Loss.__call__``, which converts ``y_true`` to a tensor — so
    ``hybrid.call(None, ...)`` raises ``ValueError``. It also draws
    ``keras.random.normal`` for its score-matching term, which is NOT
    reproducible via ``keras.utils.set_random_seed`` here (measured spread
    ~1.008-1.069 over 8 identical calls). Deterministic assertions therefore use
    a ``y_pred`` WITHOUT embeddings (the score term is then exactly ``0.0``);
    the two knobs that require the stochastic branch are averaged over repeats
    with effect sizes an order of magnitude above that spread.
"""

import numpy as np
import pytest

import keras

from dl_techniques.losses.siglip_contrastive_loss import (
    AdaptiveSigLIPLoss,
    HybridContrastiveLoss,
    SigLIPContrastiveLoss,
    create_adaptive_siglip_loss,
    create_hybrid_loss,
    create_siglip_loss,
)

BATCH = 4
ALL_CLASSES = [SigLIPContrastiveLoss, AdaptiveSigLIPLoss, HybridContrastiveLoss]
ALL_IDS = [c.__name__ for c in ALL_CLASSES]


# ---------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------


def _logits(batch: int = BATCH, seed: int = 7):
    """Deterministic square logit pair -- no embeddings, no randomness."""
    rng = np.random.default_rng(seed)
    return {
        "logits_per_image": rng.standard_normal((batch, batch)).astype(
            "float32"
        ),
        "logits_per_text": rng.standard_normal((batch, batch)).astype(
            "float32"
        ),
    }


def _logits_with_embeddings(batch: int = BATCH, seed: int = 7, dim: int = 3):
    """``_logits`` plus the embedding keys that switch on Hybrid's score term."""
    rng = np.random.default_rng(seed)
    y_pred = _logits(batch, seed)
    y_pred["image_embeddings"] = rng.standard_normal((batch, dim)).astype(
        "float32"
    )
    y_pred["text_embeddings"] = rng.standard_normal((batch, dim)).astype(
        "float32"
    )
    return y_pred


def _dummy_labels(batch: int = BATCH):
    """SigLIP is self-supervised: ``y_true`` is unused but must be a tensor."""
    return np.zeros((batch,), dtype="float32")


def _value(loss, y_pred=None, batch: int = BATCH, **call_kwargs) -> float:
    """One reduced float from a FRESH-state loss instance."""
    y_pred = _logits(batch) if y_pred is None else y_pred
    return float(np.array(loss(_dummy_labels(batch), y_pred, **call_kwargs)))


def _mean_value(factory, y_pred, repeats: int = 5) -> float:
    """Mean reduced value over ``repeats`` fresh instances.

    Only for Hybrid's stochastic score-matching branch; see the module
    docstring for why a seeded single call is not available.
    """
    return float(
        np.mean([_value(factory(), y_pred=y_pred) for _ in range(repeats)])
    )


def _instance(cls, **kwargs):
    """Construct any of the three classes with its own default arguments."""
    return cls(**kwargs)


# ---------------------------------------------------------------------
# serialization round trips -- both must RE-EVALUATE, not compare dicts
# ---------------------------------------------------------------------

_NON_DEFAULT_CONFIG = {
    SigLIPContrastiveLoss: dict(
        temperature=2.5, use_learnable_temperature=False, name="siglip_rt"
    ),
    AdaptiveSigLIPLoss: dict(
        initial_temperature=1.7,
        min_temperature=0.05,
        max_temperature=4.0,
        adaptation_rate=0.25,
        target_entropy=0.8,
        name="adaptive_rt",
    ),
    HybridContrastiveLoss: dict(
        siglip_weight=0.75,
        score_weight=0.3,
        temperature=1.9,
        noise_level=0.2,
        name="hybrid_rt",
    ),
}


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=ALL_IDS)
def test_get_config_from_config_round_trip_reevaluates_to_the_same_value(cls):
    """A config round trip must reproduce the VALUE, not just the dict."""
    kwargs = _NON_DEFAULT_CONFIG[cls]
    original = cls(**kwargs)
    config = original.get_config()

    for key, expected in kwargs.items():
        assert config[key] == expected, f"{cls.__name__}.get_config lost {key}"

    # Fresh instances on both sides: AdaptiveSigLIPLoss mutates its own state
    # on every call, so "original after being called" is not the same object.
    before = _value(cls(**kwargs))
    restored = cls.from_config(config)
    after = _value(restored)

    assert type(restored) is cls
    assert restored.name == kwargs["name"]
    assert after == pytest.approx(before, abs=1e-6, rel=0)


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=ALL_IDS)
def test_keras_saving_registry_round_trip_reevaluates_to_the_same_value(cls):
    """``serialize`` / ``deserialize`` through the registry, re-evaluated.

    A round trip that only checks the name resolves proves nothing about the
    restored object's behaviour, so this re-runs the loss on fixed input.
    """
    kwargs = _NON_DEFAULT_CONFIG[cls]
    serialized = keras.saving.serialize_keras_object(cls(**kwargs))

    assert serialized["class_name"] == cls.__name__
    assert serialized["registered_name"] == (
        f"dl_techniques.losses.siglip_contrastive_loss>{cls.__name__}"
    )

    before = _value(cls(**kwargs))
    restored = keras.saving.deserialize_keras_object(serialized)
    after = _value(restored)

    assert type(restored) is cls
    assert after == pytest.approx(before, abs=1e-6, rel=0)


# ---------------------------------------------------------------------
# factories
# ---------------------------------------------------------------------


def test_create_siglip_loss_forwards_its_arguments():
    loss = create_siglip_loss(temperature=3.0, use_learnable_temperature=False)
    assert isinstance(loss, SigLIPContrastiveLoss)
    assert loss.temperature == 3.0
    assert loss.use_learnable_temperature is False
    # The factory's own default differs from the class's -- pin it, because a
    # silent flip changes which temperature the loss reads at run time.
    assert create_siglip_loss().use_learnable_temperature is True
    assert SigLIPContrastiveLoss().use_learnable_temperature is False


def test_create_adaptive_siglip_loss_forwards_its_arguments():
    loss = create_adaptive_siglip_loss(
        initial_temperature=2.0, target_entropy=0.9, min_temperature=0.2
    )
    assert isinstance(loss, AdaptiveSigLIPLoss)
    assert loss.initial_temperature == 2.0
    assert loss.target_entropy == 0.9
    assert loss.min_temperature == 0.2


def test_create_hybrid_loss_forwards_its_arguments():
    loss = create_hybrid_loss(
        siglip_weight=0.5, score_weight=0.4, temperature=2.0
    )
    assert isinstance(loss, HybridContrastiveLoss)
    assert loss.siglip_weight == 0.5
    assert loss.score_weight == 0.4
    assert loss.temperature == 2.0


# ---------------------------------------------------------------------
# dead-knob assertions: EVERY constructor argument must change something
# ---------------------------------------------------------------------


def test_siglip_temperature_is_live():
    assert _value(SigLIPContrastiveLoss(temperature=0.5)) != pytest.approx(
        _value(SigLIPContrastiveLoss(temperature=4.0)), abs=1e-5, rel=0
    )


def test_siglip_use_learnable_temperature_is_live():
    """The flag decides whether ``y_pred['temperature']`` is read at all."""
    y_pred = dict(_logits())
    y_pred["temperature"] = np.float32(5.0)

    on = _value(SigLIPContrastiveLoss(use_learnable_temperature=True), y_pred)
    off = _value(SigLIPContrastiveLoss(use_learnable_temperature=False), y_pred)
    assert on != pytest.approx(off, abs=1e-5, rel=0)


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=ALL_IDS)
def test_name_is_live(cls):
    named = cls(name="a_deliberately_odd_name")
    assert named.name == "a_deliberately_odd_name"
    assert named.get_config()["name"] == "a_deliberately_odd_name"
    assert cls().name != "a_deliberately_odd_name"


def test_adaptive_initial_temperature_is_live():
    assert _value(AdaptiveSigLIPLoss(initial_temperature=3.0)) != pytest.approx(
        _value(AdaptiveSigLIPLoss(initial_temperature=1.0)), abs=1e-5, rel=0
    )


def test_adaptive_min_temperature_is_live():
    """A floor above the initial temperature must bind."""
    assert _value(AdaptiveSigLIPLoss(min_temperature=5.0)) != pytest.approx(
        _value(AdaptiveSigLIPLoss()), abs=1e-5, rel=0
    )


def test_adaptive_max_temperature_is_live():
    """A ceiling below the initial temperature must bind."""
    assert _value(AdaptiveSigLIPLoss(max_temperature=0.02)) != pytest.approx(
        _value(AdaptiveSigLIPLoss()), abs=1e-5, rel=0
    )


def test_adaptive_adaptation_rate_is_live():
    assert _value(AdaptiveSigLIPLoss(adaptation_rate=10.0)) != pytest.approx(
        _value(AdaptiveSigLIPLoss(adaptation_rate=0.1)), abs=1e-6, rel=0
    )


def test_adaptive_target_entropy_is_live():
    assert _value(AdaptiveSigLIPLoss(target_entropy=50.0)) != pytest.approx(
        _value(AdaptiveSigLIPLoss(target_entropy=0.5)), abs=1e-5, rel=0
    )


def test_adaptive_temperature_variable_actually_moves_and_stays_in_range():
    """The ``assign`` is the class's whole point; pin that it happens."""
    loss = AdaptiveSigLIPLoss(min_temperature=0.01, max_temperature=10.0)
    start = float(np.array(loss.adaptive_temperature))
    _value(loss)
    moved = float(np.array(loss.adaptive_temperature))

    assert moved != pytest.approx(start, abs=1e-7, rel=0)
    assert 0.01 - 1e-6 <= moved <= 10.0 + 1e-6


def test_hybrid_siglip_weight_is_live():
    y_pred = _logits()  # no embeddings -> score term is exactly 0.0
    assert _value(HybridContrastiveLoss(siglip_weight=2.0), y_pred) != (
        pytest.approx(
            _value(HybridContrastiveLoss(siglip_weight=1.0), y_pred),
            abs=1e-5,
            rel=0,
        )
    )


def test_hybrid_temperature_is_live():
    y_pred = _logits()
    assert _value(HybridContrastiveLoss(temperature=4.0), y_pred) != (
        pytest.approx(
            _value(HybridContrastiveLoss(temperature=1.0), y_pred),
            abs=1e-5,
            rel=0,
        )
    )


def test_hybrid_score_weight_is_live():
    """Needs the stochastic branch: averaged, with a >10x effect size."""
    y_pred = _logits_with_embeddings()
    high = _mean_value(lambda: HybridContrastiveLoss(score_weight=0.9), y_pred)
    low = _mean_value(lambda: HybridContrastiveLoss(score_weight=0.1), y_pred)
    assert high - low > 1.0, f"score_weight looks dead: {high} vs {low}"


def test_hybrid_noise_level_is_live():
    """Needs the stochastic branch: averaged, with a >10x effect size."""
    y_pred = _logits_with_embeddings()
    high = _mean_value(lambda: HybridContrastiveLoss(noise_level=5.0), y_pred)
    low = _mean_value(lambda: HybridContrastiveLoss(noise_level=0.1), y_pred)
    assert high - low > 1.0, f"noise_level looks dead: {high} vs {low}"


def test_hybrid_score_term_is_absent_without_embeddings():
    """Without the embedding keys, Hybrid must equal weighted SigLIP alone."""
    y_pred = _logits()
    hybrid = _value(HybridContrastiveLoss(siglip_weight=1.0), y_pred)
    plain = _value(SigLIPContrastiveLoss(temperature=1.0), y_pred)
    assert hybrid == pytest.approx(plain, abs=1e-6, rel=0)


# ---------------------------------------------------------------------
# edge inputs
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [SigLIPContrastiveLoss, HybridContrastiveLoss],
    ids=["SigLIPContrastiveLoss", "HybridContrastiveLoss"],
)
def test_batch_of_one_is_finite(cls):
    value = _value(cls(), y_pred=_logits(batch=1), batch=1)
    assert np.isfinite(value)
    assert value >= 0.0


def test_adaptive_batch_of_one_is_finite_but_warns():
    """Batch of 1 is finite for the adaptive variant -- and it WARNS.

    ``AdaptiveSigLIPLoss.call`` takes ``softmax(logits_per_image, axis=-1)`` to
    estimate the batch's entropy, so at ``batch == 1`` Keras itself warns that
    the softmax can only return 1 (and the entropy term is therefore ~0, making
    the temperature update a pure ``target_entropy`` offset). That is a real
    wart of the adaptive branch, not of this test: pinned here rather than
    suppressed, so it cannot disappear unnoticed. The other two classes do not
    warn, which is why they are parametrized separately above.
    """
    with pytest.warns(UserWarning, match="softmax over axis"):
        value = _value(
            AdaptiveSigLIPLoss(), y_pred=_logits(batch=1), batch=1
        )
    assert np.isfinite(value)
    assert value >= 0.0


@pytest.mark.parametrize("temperature", [1e-6, 1e4])
@pytest.mark.parametrize(
    "cls", [SigLIPContrastiveLoss, HybridContrastiveLoss], ids=["siglip", "hybrid"]
)
def test_extreme_temperature_stays_finite(cls, temperature):
    """``softplus`` must not overflow at either end of the range."""
    value = _value(cls(temperature=temperature))
    assert np.isfinite(value), f"{cls.__name__} at t={temperature}: {value}"
    assert value >= 0.0


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=ALL_IDS)
def test_non_dict_predictions_raise_value_error(cls):
    """The dict schema is the contract; a bare tensor must be rejected."""
    with pytest.raises(ValueError):
        cls()(_dummy_labels(), _logits()["logits_per_image"])


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=ALL_IDS)
def test_perfectly_separated_logits_beat_random_ones(cls):
    """A basic value-sanity arm: the loss must prefer the correct pairing."""
    eye = np.eye(BATCH, dtype="float32") * 10.0 - 5.0
    good = {"logits_per_image": eye, "logits_per_text": eye}
    bad = {"logits_per_image": -eye, "logits_per_text": -eye}
    assert _value(cls(), good) < _value(cls(), bad)


# =====================================================================
# PER-SAMPLE CONTRACT (RED)
#
# The three tests below are written to the CORRECT ``keras.losses.Loss``
# contract and FAIL against the current tree. They are the RED proof for the
# per-sample-return fix. Do NOT xfail them and do NOT relax them to match the
# shipped behaviour -- if they start passing, the fix landed.
# =====================================================================


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=ALL_IDS)
def test_call_returns_one_value_per_sample(cls):
    """RED TODAY. ``call()`` must return shape ``(batch,)``, not a scalar."""
    per_sample = np.array(cls().call(_dummy_labels(), _logits()))
    assert per_sample.shape == (BATCH,), (
        f"{cls.__name__}.call returned shape {per_sample.shape}; a "
        f"keras.losses.Loss must return one value per sample"
    )


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=ALL_IDS)
def test_sample_weight_weights_the_named_rows(cls):
    """RED TODAY. Zeroing row 3 must not equal scaling the batch by 0.75.

    With a scalar ``call()``, ``reduce_weighted_values`` broadcasts that scalar
    against ``sample_weight`` and reduces, so the result is EXACTLY
    ``unweighted * mean(sample_weight)`` -- WHICH rows were weighted is thrown
    away. Fresh instance per measurement (AdaptiveSigLIPLoss is stateful).
    """
    y_pred = _logits()
    unweighted = _value(cls(), y_pred)
    weighted = _value(
        cls(), y_pred, sample_weight=np.array([1.0, 1.0, 1.0, 0.0], "float32")
    )

    assert weighted != pytest.approx(unweighted * 0.75, abs=1e-6, rel=0), (
        f"{cls.__name__}: sample_weight=[1,1,1,0] gave {weighted}, which is "
        f"exactly unweighted*mean(w) = {unweighted * 0.75} -- the loss charged "
        f"every row the batch aggregate instead of weighting row 3"
    )


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=ALL_IDS)
def test_reduction_none_returns_one_value_per_sample(cls):
    """RED TODAY. ``reduction`` is a dead knob on all three classes."""
    y_pred = _logits()
    unreduced = np.array(cls(reduction="none")(_dummy_labels(), y_pred))

    assert unreduced.shape == (BATCH,), (
        f"{cls.__name__} with reduction='none' returned shape "
        f"{unreduced.shape}; reduction is dead while call() returns a scalar"
    )


# =====================================================================
# ADAPTIVE TEMPERATURE: a closed loop that carries its state
# (plan-2026-11-11T131349-13b28b37 step 1, D-003)
#
# Before this step the entropy that drove the update was computed on the
# UNSCALED logits, so the temperature never fed back into the quantity it was
# controlling (open loop), and the update sign was `-rate * (H - target)`.
# Scaling the entropy by the temperature alone would have been positive
# feedback with that sign (a larger H lowers T, which flattens the softmax and
# raises H again); only scaled entropy AND `+rate * (H - target)` converges.
# A pre-edit numpy simulation (progress.md, "Step 1 pre-edit simulation")
# measured: as-is sign -> T pinned at 0.01 for all six start/scale cases;
# scaled entropy + old sign -> a bound in all six; scaled entropy + flipped
# sign -> strictly inside (0.01, 10.0) in all six.
# =====================================================================

_MIN_T, _MAX_T = 0.01, 10.0


def _entropy_at(logits: np.ndarray, temperature: float) -> float:
    """Independent numpy re-derivation of the controlled quantity."""
    z = logits.astype("float64") * temperature
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    return float(-np.mean(np.sum(p * np.log(p + 1e-8), axis=-1)))


def _live_temperature(loss) -> float:
    return float(keras.ops.convert_to_numpy(loss.adaptive_temperature))


def _scaled_logits(scale: float, size: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    logits = (rng.standard_normal((size, size)) * scale).astype("float32")
    return {"logits_per_image": logits, "logits_per_text": logits.T.copy()}


@pytest.mark.parametrize("initial_temperature", [0.5, 1.0, 3.0])
def test_adaptive_temperature_settles_strictly_inside_the_bounds(
    initial_temperature,
):
    """RED at HEAD: the open loop drifts to ``min_temperature`` (0.01)."""
    y_pred = _scaled_logits(scale=2.0)
    loss = AdaptiveSigLIPLoss(
        initial_temperature=initial_temperature,
        min_temperature=_MIN_T,
        max_temperature=_MAX_T,
        target_entropy=0.5,
    )
    for _ in range(200):
        loss(_dummy_labels(8), y_pred)

    final = _live_temperature(loss)
    assert _MIN_T < final < _MAX_T, f"T pinned at a bound: {final}"
    achieved = _entropy_at(y_pred["logits_per_image"], final)
    assert abs(achieved - 0.5) < 0.05, (
        f"T={final} leaves the scaled entropy at {achieved}, not the 0.5 target"
    )


def test_adaptive_entropy_is_measured_at_the_live_temperature():
    """RED at HEAD: the entropy ignored the temperature, so both runs moved alike.

    Same logits, two starting temperatures. At T=0.05 the scaled softmax is
    nearly uniform (entropy above the 0.5 target); at T=8 it is nearly one-hot
    (entropy below it). A closed loop must therefore push them in OPPOSITE
    directions.
    """
    y_pred = _scaled_logits(scale=2.0)
    hot = AdaptiveSigLIPLoss(initial_temperature=8.0)
    cold = AdaptiveSigLIPLoss(initial_temperature=0.05)
    loss_high_t = _live_temperature(hot)
    loss_low_t = _live_temperature(cold)
    hot(_dummy_labels(8), y_pred)
    cold(_dummy_labels(8), y_pred)

    assert _live_temperature(hot) < loss_high_t, "entropy below target must lower T"
    assert _live_temperature(cold) > loss_low_t, "entropy above target must raise T"


def test_adaptive_update_raises_temperature_when_entropy_is_above_target():
    """Pin the sign on its own: entropy above target RAISES T (sharpens)."""
    y_pred = _scaled_logits(scale=0.1)  # near-uniform rows, H ~ ln(8) >> 0.5
    loss = AdaptiveSigLIPLoss(initial_temperature=1.0, adaptation_rate=0.1)
    loss(_dummy_labels(8), y_pred)
    expected = 1.0 + 0.1 * (_entropy_at(y_pred["logits_per_image"], 1.0) - 0.5)
    assert _live_temperature(loss) == pytest.approx(expected, abs=1e-5, rel=0)


def test_adaptive_temperature_state_survives_a_config_round_trip():
    """RED at HEAD: ``get_config`` had no live T, so a reload reset it to 1.0."""
    loss = AdaptiveSigLIPLoss(initial_temperature=1.0)
    loss.adaptive_temperature.assign(3.3)

    restored = AdaptiveSigLIPLoss.from_config(loss.get_config())

    # EXACT equality: the variable is float64, so 3.3 does not round through
    # float32 (3.299999952...), the way DINOLoss's teacher_temp learned to avoid.
    assert _live_temperature(restored) == 3.3
    assert restored.initial_temperature == 1.0, (
        "initial_temperature must keep meaning 'where a fresh run starts'"
    )


def test_adaptive_temperature_state_survives_the_keras_serialize_round_trip():
    loss = AdaptiveSigLIPLoss(min_temperature=0.05, max_temperature=4.0)
    loss.adaptive_temperature.assign(2.7)

    restored = keras.saving.deserialize_keras_object(
        keras.saving.serialize_keras_object(loss)
    )

    assert type(restored) is AdaptiveSigLIPLoss
    assert _live_temperature(restored) == 2.7
    assert restored.min_temperature == 0.05
    assert restored.max_temperature == 4.0


def test_adaptive_state_is_not_lost_after_updates_either():
    """The round trip carries the value AFTER the loop has moved it."""
    y_pred = _scaled_logits(scale=2.0)
    loss = AdaptiveSigLIPLoss()
    for _ in range(5):
        loss(_dummy_labels(8), y_pred)
    moved = _live_temperature(loss)
    assert moved != 1.0

    restored = AdaptiveSigLIPLoss.from_config(loss.get_config())
    assert _live_temperature(restored) == moved


# =====================================================================
# REACHABILITY CONDITION (plan-2026-11-11T131349-13b28b37, D-044)
#
# The controller steers the mean row entropy to a FIXED absolute target
# (``target_entropy`` nats). It settles strictly inside the temperature bounds
# only when that target is reachable in the interval:
#     H(max_temperature) <= target <= H(min_temperature)
# and otherwise pins at the nearer bound. These tests pin both regimes and the
# equivalence between the condition and the outcome, so the class docstring
# cannot drift back to an unconditional "settles strictly inside".
# =====================================================================


def _cosine_logits(batch: int, aligned: bool, dim: int = 128, seed: int = 0):
    """Cosine similarities of random unit embeddings (the documented input).

    ``aligned=False`` is the untrained case (independent towers); ``aligned=True``
    makes the second tower a noisy copy of the first (diagonal cosine about 0.96).
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((batch, dim))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    if aligned:
        b = a + 0.3 * rng.standard_normal((batch, dim)) / np.sqrt(dim)
    else:
        b = rng.standard_normal((batch, dim))
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    logits = (a @ b.T).astype("float32")
    return {"logits_per_image": logits, "logits_per_text": logits.T.copy()}


def _run_controller(y_pred, calls: int = 300, **kwargs) -> float:
    batch = y_pred["logits_per_image"].shape[0]
    loss = AdaptiveSigLIPLoss(**kwargs)
    for _ in range(calls):
        loss(_dummy_labels(batch), y_pred)
    return _live_temperature(loss)


def test_untrained_cosine_logits_pin_the_temperature_at_the_upper_bound():
    """The 0.5-nat target is unreachable: even at T=10 the softmax is near uniform."""
    y_pred = _cosine_logits(batch=64, aligned=False)
    logits = y_pred["logits_per_image"]
    assert _entropy_at(logits, _MAX_T) > 3.0, "premise: target 0.5 is out of reach"

    final = _run_controller(y_pred, max_temperature=_MAX_T)

    assert final == _MAX_T, f"expected a pin at max_temperature, got {final}"
    assert _entropy_at(logits, final) > 3.0, "the pinned T leaves H far above 0.5"


def test_aligned_cosine_logits_settle_inside_the_bounds_at_the_target():
    y_pred = _cosine_logits(batch=64, aligned=True)
    logits = y_pred["logits_per_image"]
    assert _entropy_at(logits, _MAX_T) < 0.5 < _entropy_at(logits, _MIN_T)

    final = _run_controller(y_pred, min_temperature=_MIN_T, max_temperature=_MAX_T)

    assert _MIN_T < final < _MAX_T, f"T pinned at a bound: {final}"
    assert abs(_entropy_at(logits, final) - 0.5) < 0.05


def test_the_same_untrained_logits_settle_once_the_target_is_reachable():
    """It is the target, not the data: lower the bar below H(max_T) and it settles."""
    y_pred = _cosine_logits(batch=64, aligned=False)
    logits = y_pred["logits_per_image"]
    target = 0.5 * (_entropy_at(logits, _MAX_T) + _entropy_at(logits, _MIN_T))

    # H(T) is nearly flat for these logits (0.36 nats across the whole interval),
    # so the loop approaches the target slowly: give it more calls.
    final = _run_controller(y_pred, calls=1500, target_entropy=target)

    assert _MIN_T < final < _MAX_T, f"T pinned at a bound: {final}"
    assert abs(_entropy_at(logits, final) - target) < 0.05


def test_a_target_above_the_flattest_softmax_pins_at_the_lower_bound():
    y_pred = _scaled_logits(scale=2.0)
    logits = y_pred["logits_per_image"]
    target = _entropy_at(logits, _MIN_T) + 1.0  # above even the flattest softmax

    final = _run_controller(y_pred, min_temperature=_MIN_T, target_entropy=target)

    assert final == _MIN_T, f"expected a pin at min_temperature, got {final}"


# DECISION plan-2026-11-11T131349-13b28b37/D-044
# The target is absolute, so settling is a property of the logits, not a promise
# of the loop. This test fails if the documented condition and the outcome part
# ways (for example a target silently rescaled by log B).
@pytest.mark.parametrize(
    "case",
    [
        ("gaussian_scale_2_b8", lambda: _scaled_logits(scale=2.0), 0.5),
        ("untrained_cosine_b32", lambda: _cosine_logits(32, False), 0.5),
        ("untrained_cosine_b256", lambda: _cosine_logits(256, False), 0.5),
        ("aligned_cosine_b32", lambda: _cosine_logits(32, True), 0.5),
        ("aligned_cosine_b256", lambda: _cosine_logits(256, True), 0.5),
        ("aligned_cosine_b1024", lambda: _cosine_logits(1024, True), 0.5),
        ("gaussian_scale_2_b8_high_target", lambda: _scaled_logits(2.0), 2.0),
    ],
    ids=lambda case: case[0] if isinstance(case, tuple) else str(case),
)
def test_settling_inside_the_bounds_is_exactly_the_reachability_condition(case):
    """Settles strictly inside iff H(max_T) <= target <= H(min_T); else pinned."""
    _, make, target = case
    y_pred = make()
    logits = y_pred["logits_per_image"]
    reachable = _entropy_at(logits, _MAX_T) <= target <= _entropy_at(logits, _MIN_T)

    final = _run_controller(
        y_pred, min_temperature=_MIN_T, max_temperature=_MAX_T, target_entropy=target
    )

    pinned = final in (_MIN_T, _MAX_T)
    assert reachable == (not pinned), (
        f"reachable={reachable} but final T={final} (pinned={pinned})"
    )
