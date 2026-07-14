"""C4 guard: ``--pretrained-encoder`` really moves the trunk, bit-for-bit.

**The failure this guards.** ``warm_start_encoder()`` carries pre-mortem #3: the warm start
runs, prints nothing alarming, transfers NOTHING, and the classifier trains from random init
while the command line says "pretrained". The (poor) accuracy is then blamed on the
pretraining objective, and the diagnosis costs a day. Nothing about that failure is visible in
a loss curve.

Until this file existed the function was executed by ZERO tests: the plan's Verification
Strategy named a ``test_pretrained_encoder_transfer`` that was never written, and the C4 proof
lived only in a throwaway scratchpad script. A proof that only ever ran in a scratchpad is not
a guard.

**What is asserted, and why in this shape:**

* the trunks provably start APART (a fresh classifier at a DIFFERENT seed) -- otherwise the
  equality assert below is vacuous and would pass with the transfer deleted;
* after the transfer the trunk is BIT-EXACT against the checkpoint
  (``assert_array_equal``, max|delta| == 0.0) -- not ``allclose``: a name-matched
  ``set_weights`` is an exact copy, and anything approximate means something else happened;
* ``BACKBONE_NAME in report.loaded`` -- NOT ``num_loaded > 0``. The nested backbone transfers
  as ONE named layer (9 weight arrays), so a COUNT is the wrong assert in both directions: a
  complete transfer scores 1, while some other layer could make a count positive with the
  trunk still at random init. (``TransferReport`` has no ``transferred`` field at all -- the
  plan's original assert would have raised ``AttributeError`` on the happy path.)
* the classification head is UNTOUCHED (still at init), since it does not exist in the source;
* the three ways a transfer can silently do nothing -- skip-everything, a garbage checkpoint,
  a config-mismatched checkpoint -- all RAISE.

Runs at the realistic geometry (224/16 -> N=196), variant ``tiny``.
"""

from typing import List, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.models.energy_transformer import (
    BACKBONE_NAME,
    create_energy_transformer_classifier,
    create_energy_transformer_mim,
)
from train.energy_transformer.train_classification import (
    TrainingConfig,
    warm_start_encoder,
)

# --- realistic geometry. DO NOT SHRINK. -------------------------------------
IMAGE_SIZE = 224
PATCH_SIZE = 16
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
VARIANT = "tiny"
NUM_CLASSES = 10

MIM_SEED = 7
CLS_SEED = 1234  # DIFFERENT on purpose: the two trunks must start apart.


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _build_mim(patch_size: int = PATCH_SIZE) -> keras.Model:
    keras.utils.set_random_seed(MIM_SEED)
    model = create_energy_transformer_mim(
        variant=VARIANT, input_shape=INPUT_SHAPE, patch_size=patch_size,
    )
    model.build((None,) + INPUT_SHAPE)
    return model


def _build_classifier() -> keras.Model:
    keras.utils.set_random_seed(CLS_SEED)
    model = create_energy_transformer_classifier(
        variant=VARIANT,
        input_shape=INPUT_SHAPE,
        patch_size=PATCH_SIZE,
        num_classes=NUM_CLASSES,
    )
    model.build((None,) + INPUT_SHAPE)
    return model


@pytest.fixture(scope="module")
def mim_checkpoint(tmp_path_factory) -> Tuple[str, List[np.ndarray]]:
    """A saved ``EnergyTransformerMIM`` ``.keras`` + its trunk weights, built once."""
    model = _build_mim()
    path = tmp_path_factory.mktemp("warm_start") / "mim.keras"
    model.save(path)
    trunk = [np.array(w) for w in model.get_layer(BACKBONE_NAME).get_weights()]
    return str(path), trunk


@pytest.fixture(scope="module")
def mismatched_checkpoint(tmp_path_factory) -> str:
    """An MIM checkpoint at patch 32 (N=49): same LAYER NAME, incompatible trunk SHAPES.

    This is the config-drift case -- someone pretrains at one geometry and fine-tunes at
    another. ``strict=False`` merely RECORDS a shape mismatch and skips it, which is
    pre-mortem #3 wearing a report, so the trainer promotes it to fatal.
    """
    model = _build_mim(patch_size=32)
    path = tmp_path_factory.mktemp("warm_start_bad") / "mim_patch32.keras"
    model.save(path)
    return str(path)


@pytest.fixture(scope="module")
def garbage_checkpoint(tmp_path_factory) -> str:
    """A valid ``.keras`` file that is simply not an Energy Transformer."""
    model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(3)])
    path = tmp_path_factory.mktemp("warm_start_garbage") / "not_an_et.keras"
    model.save(path)
    return str(path)


def _trunk(model: keras.Model) -> List[np.ndarray]:
    return [np.array(w) for w in model.get_layer(BACKBONE_NAME).get_weights()]


def _head(model: keras.Model) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for layer_name in ("head_norm", "head_dense"):
        out += [np.array(w) for w in model.get_layer(layer_name).get_weights()]
    return out


def _max_abs_delta(a: List[np.ndarray], b: List[np.ndarray]) -> float:
    assert len(a) == len(b), f"weight-list lengths differ: {len(a)} vs {len(b)}"
    return max(float(np.max(np.abs(x - y))) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# 1. The vacuity guard -- the two trunks must START apart
# ---------------------------------------------------------------------------

def test_trunks_start_apart(mim_checkpoint) -> None:
    """Without this, the bit-exactness assert below would pass with the transfer DELETED.

    Two random inits at different seeds must differ. If they did not (a fixed init, a leaked
    global seed), every other test in this file would be a tautology.
    """
    _path, mim_trunk = mim_checkpoint
    cold = _trunk(_build_classifier())

    delta = _max_abs_delta(cold, mim_trunk)
    assert delta > 1e-3, (
        f"The cold classifier trunk is already within {delta:.3e} of the MIM checkpoint. "
        f"The transfer tests below would then pass VACUOUSLY."
    )


# ---------------------------------------------------------------------------
# 2. C4 -- the transfer itself
# ---------------------------------------------------------------------------

def test_pretrained_encoder_transfer(mim_checkpoint) -> None:
    """THE C4 guard: bit-exact trunk, untouched head, ``et_backbone`` in ``report.loaded``."""
    path, mim_trunk = mim_checkpoint
    model = _build_classifier()

    head_before = _head(model)
    pre_delta = _max_abs_delta(_trunk(model), mim_trunk)

    report = warm_start_encoder(model, path)

    # -- the membership assert, NOT a count. See the module docstring.
    assert BACKBONE_NAME in report.loaded, (
        f"{BACKBONE_NAME!r} not in report.loaded={report.loaded}: the trunk did NOT transfer "
        f"and the run would train from RANDOM INIT while claiming to be pretrained."
    )
    assert not report.shape_mismatch, f"shape mismatches: {report.shape_mismatch}"

    # -- BIT-EXACT. A name-matched set_weights is a copy; anything approximate is a bug.
    after = _trunk(model)
    assert len(after) == len(mim_trunk)
    for i, (got, want) in enumerate(zip(after, mim_trunk)):
        np.testing.assert_array_equal(
            got, want, err_msg=f"trunk weight #{i} is not bit-identical to the checkpoint"
        )
    post_delta = _max_abs_delta(after, mim_trunk)
    assert post_delta == 0.0, f"expected max|delta| == 0.0, got {post_delta:.3e}"
    assert pre_delta > 0.0, "pre-transfer distance was 0 -- the assert above is vacuous"

    # -- the head does not exist in the source, so it must still be at init.
    head_after = _head(model)
    for i, (got, want) in enumerate(zip(head_after, head_before)):
        np.testing.assert_array_equal(
            got, want,
            err_msg=f"head weight #{i} MOVED during the warm start; only the trunk may transfer",
        )

    # -- the MIM decoder is the only thing skipped by prefix; nothing else is missing.
    assert sorted(report.skipped_by_prefix) == ["decoder_norm", "decoder_proj"], (
        f"unexpected skipped_by_prefix={report.skipped_by_prefix}"
    )


def test_warm_started_model_still_predicts(mim_checkpoint) -> None:
    """A transfer that leaves the model unusable is not a transfer. Forward pass at N=196."""
    path, _trunk_weights = mim_checkpoint
    model = _build_classifier()
    warm_start_encoder(model, path)

    logits = model(np.zeros((2,) + INPUT_SHAPE, dtype="float32"), training=False)
    assert tuple(logits.shape) == (2, NUM_CLASSES)
    assert np.all(np.isfinite(np.array(logits)))


# ---------------------------------------------------------------------------
# 3. GUARD-BITE -- every way the transfer can silently do nothing must RAISE
# ---------------------------------------------------------------------------

def test_skip_everything_transfer_raises(mim_checkpoint) -> None:
    """THE guard-bite. ``skip_prefixes=("",)`` skips every layer, i.e. transfers NOTHING --
    exactly the silent failure of pre-mortem #3. It must be LOUD, and the trunk must be
    provably still at its (different) random init afterwards.
    """
    path, mim_trunk = mim_checkpoint
    model = _build_classifier()

    with pytest.raises(RuntimeError, match="RANDOM INIT"):
        warm_start_encoder(model, path, skip_prefixes=("",))

    delta = _max_abs_delta(_trunk(model), mim_trunk)
    assert delta > 1e-3, (
        f"skip-everything still moved the trunk to within {delta:.3e} of the checkpoint"
    )


def test_config_mismatched_checkpoint_raises(mismatched_checkpoint) -> None:
    """A checkpoint pretrained at a DIFFERENT geometry. ``strict=False`` would merely record
    the shape mismatch and skip it, leaving the trunk at init -- so the trainer promotes it.
    """
    model = _build_classifier()
    with pytest.raises(RuntimeError, match="do not match"):
        warm_start_encoder(model, mismatched_checkpoint)


def test_garbage_checkpoint_raises(garbage_checkpoint) -> None:
    """A valid ``.keras`` that contains no ``et_backbone`` at all.

    This one is caught one level DOWN, by ``load_weights_from_checkpoint`` itself ("No
    overlapping layers"), before ``warm_start_encoder``'s own ``RuntimeError`` can fire. Both
    nets are real; the point of the test is that the run ABORTS rather than proceeding from
    random init, so the assert is on loudness, not on which layer shouts.
    """
    model = _build_classifier()
    with pytest.raises(ValueError, match="No overlapping layers"):
        warm_start_encoder(model, garbage_checkpoint)


def test_missing_checkpoint_path_is_rejected_at_config_time(tmp_path) -> None:
    """A typo'd path must fail BEFORE the data pipeline warms up -- and must fail, not degrade
    into a random-init run.
    """
    with pytest.raises(FileNotFoundError):
        TrainingConfig(pretrained_encoder=str(tmp_path / "nope.keras"))

    with pytest.raises(ValueError, match="must be a .keras checkpoint"):
        TrainingConfig(pretrained_encoder=str(tmp_path / "nope.h5"))
