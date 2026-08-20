"""Guard: `nam` and `superpoint`'s stability floors are not smaller than one ULP.

Rule ``R-091``, plan ``plan-2026-08-19T163559-499b6f0e``, step 17.1.

Both packages divided by ``<reduction> + <literal>`` where the literal is
**exactly 0.0 in float16**:

* ``models/nam/model.py`` (masked mean) and ``models/nam/cell.py`` (halt input):
  ``+ 1e-9``. An all-padding row makes the numerator and the denominator both
  zero, so the division is ``0/0``.
* ``models/superpoint/model.py`` (descriptor L2 normalisation): ``+ 1e-12``. A
  zero descriptor vector is the same ``0/0``.

MEASURED at HEAD on the REAL models (CPU, not a synthetic expression):

============================ =================== ===============
subject                      ``mixed_float16``   ``float32``
============================ =================== ===============
`NAM`, all-PAD row           **nan 8 / 8**       nan 0 / 8
`NAM`, valid rows (liveness) nan 0 / 8           nan 0 / 8
`SuperPoint`, zero desc.     **nan 32768/32768** nan 0 / 32768
`SuperPoint`, random desc.   nan 0 / 32768       nan 0 / 32768
============================ =================== ===============

Neither fires at random initialisation, which is exactly why no existing test
sees them: a random descriptor is never exactly zero and the fixtures never feed
an all-padding row.

The remedy is D-027's instrument, not a bigger constant: the floor becomes
``max(literal, finfo(compute_dtype).tiny)``, which is **inert in float32** by
construction (``max(1e-9, 1.18e-38) == 1e-9``).

See ``decisions.md`` D-050.
"""

import numpy as np
import pytest
import keras

from dl_techniques.models.nam.model import NAM
from dl_techniques.models.nam.config import NAMConfig
from dl_techniques.models.nam.tokenizer import ArithmeticTokenizer
from dl_techniques.models.superpoint.model import SuperPoint


NAM_OUTPUT_KEYS = ("result", "valid", "q_halt_logits", "q_continue_logits")


@pytest.fixture
def float16_policy():
    """Set `mixed_float16` for one test and restore the previous policy after.

    The global dtype policy is PROCESS-GLOBAL, so the restore must happen in a
    `finally` and must be asserted, not assumed.
    """
    previous = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        yield
    finally:
        keras.mixed_precision.set_global_policy(previous)
    assert keras.mixed_precision.global_policy().name == previous.name


def _nan_count(tensor):
    array = np.asarray(keras.ops.convert_to_numpy(tensor), dtype="float64")
    return int(np.isnan(array).sum()), array.size


def _nam():
    config = NAMConfig(
        hidden_size=32, num_heads=4, num_tree_layers=1, intermediate_size=64,
        memory_size=8, num_read_heads=2, max_expression_len=16,
        halt_max_steps=4, hidden_dropout_rate=0.0, attention_dropout_rate=0.0,
    )
    return NAM(config=config)


def _nam_nans(model, input_ids):
    batch = {"input_ids": input_ids}
    carry = model.initial_carry(batch)
    _, outputs = model(carry, batch, training=False)
    total = size = 0
    for key in NAM_OUTPUT_KEYS:
        n, s = _nan_count(outputs[key])
        total += n
        size += s
    return total, size


def _superpoint(descriptor_dim=8):
    return SuperPoint(
        input_shape=(64, 64, 1), descriptor_dim=descriptor_dim,
        depths=[1, 1, 1], dims=[8, 16, 16],
    )


def _superpoint_nans(zero_the_head):
    model = _superpoint()
    image = np.random.RandomState(0).rand(1, 64, 64, 1).astype("float32")
    model(image, training=False)
    if zero_the_head:
        # Force the descriptor head to emit EXACTLY zero. This is the regime the
        # `+ 1e-12` guard exists for, and the ONLY regime in which it is
        # observable — a randomly initialised head never produces a zero vector.
        for weight in model.descriptor_head.weights:
            weight.assign(np.zeros(weight.shape, dtype="float32"))
    return _nan_count(model(image, training=False)["descriptors"])


# ---------------------------------------------------------------------------
# nam — the all-padding row
# ---------------------------------------------------------------------------

def test_nam_all_padding_row_is_finite_under_mixed_float16(float16_policy):
    model = _nam()
    all_pad = np.zeros((2, 16), dtype="int32")  # token 0 is PAD
    n_nan, size = _nam_nans(model, all_pad)
    assert n_nan == 0, (
        f"NAM produced {n_nan} of {size} NaN on an all-padding row under "
        "mixed_float16 — the `+ 1e-9` masked-mean floor is exactly 0.0 in "
        "float16, so the division is 0/0"
    )


def test_nam_valid_rows_are_finite_under_mixed_float16(float16_policy):
    """LIVENESS. Without it, a NAM broken in every regime would read as a pass."""
    model = _nam()
    tokenizer = ArithmeticTokenizer(max_len=16)
    n_nan, size = _nam_nans(model, tokenizer.encode_batch(["1 + 2", "3 * 4"]))
    assert n_nan == 0, f"NAM produced {n_nan} of {size} NaN on VALID rows"


def test_nam_all_padding_row_is_finite_under_float32():
    """The float32 CONTROL: this arm was green at HEAD and must stay green."""
    assert keras.mixed_precision.global_policy().name == "float32"
    model = _nam()
    n_nan, size = _nam_nans(model, np.zeros((2, 16), dtype="int32"))
    assert n_nan == 0, f"NAM float32 control: {n_nan} of {size} NaN"


# ---------------------------------------------------------------------------
# superpoint — the zero descriptor
# ---------------------------------------------------------------------------

def test_superpoint_zero_descriptor_is_finite_under_mixed_float16(float16_policy):
    n_nan, size = _superpoint_nans(zero_the_head=True)
    assert n_nan == 0, (
        f"SuperPoint produced {n_nan} of {size} NaN for a zero descriptor under "
        "mixed_float16 — the `+ 1e-12` L2 floor is exactly 0.0 in float16"
    )


def test_superpoint_random_descriptors_are_finite_under_mixed_float16(float16_policy):
    """LIVENESS, and it is the arm that shows why this was invisible."""
    n_nan, size = _superpoint_nans(zero_the_head=False)
    assert n_nan == 0, f"SuperPoint random descriptors: {n_nan} of {size} NaN"


def test_superpoint_zero_descriptor_is_finite_under_float32():
    """The float32 CONTROL."""
    assert keras.mixed_precision.global_policy().name == "float32"
    n_nan, size = _superpoint_nans(zero_the_head=True)
    assert n_nan == 0, f"SuperPoint float32 control: {n_nan} of {size} NaN"


# ---------------------------------------------------------------------------
# The instrument itself, pinned by value.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("literal", [1e-9, 1e-12])
def test_the_literals_are_exactly_zero_in_float16(literal):
    """This is WHY the guards were void. Not 'small' — absent."""
    assert float(np.float16(literal)) == 0.0


@pytest.mark.parametrize("literal", [1e-9, 1e-12])
def test_the_dtype_aware_floor_is_inert_in_float32(literal):
    """The float32 numerics must not move: the clamp returns the literal."""
    assert max(literal, float(np.finfo("float32").tiny)) == literal


@pytest.mark.parametrize("literal", [1e-9, 1e-12])
def test_the_dtype_aware_floor_is_representable_in_float16(literal):
    floor = max(literal, float(np.finfo("float16").tiny))
    assert float(np.float16(floor)) > 0.0


def test_the_fftnet_pinned_float32_guard_is_the_safe_side_control():
    """`fftnet/model.py:252` pins its `1e-8` to float32 and is NOT affected.

    Recorded as the contrast case: `1e-8` is ALSO exactly 0.0 in float16, so
    that site is safe only because of the explicit `dtype="float32"`. Pinning it
    here keeps "the constant is fine" from being read as the lesson.
    """
    assert float(np.float16(1e-8)) == 0.0
    assert float(np.float32(1e-8)) > 0.0
