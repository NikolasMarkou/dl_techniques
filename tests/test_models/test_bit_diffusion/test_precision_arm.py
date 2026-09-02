"""Forward passes are finite under ``float32``, ``mixed_float16`` and ``float64``.

Three dtype policies, one claim each: the model runs and every number it emits
is finite. That is a weaker claim than accuracy and a much stronger one than it
looks, because the two failure modes it catches are silent:

* a sub-layer that hard-codes ``"float32"`` raises a dtype mismatch under a
  ``float64`` policy -- which is loud -- but a sub-layer that CASTS to float32
  silently narrows a float64 policy and nothing raises at all;
* an intermediate that overflows or divides by zero in ``float16`` produces
  ``nan``/``inf`` while every shape, config and round-trip assertion passes.

**The ``float16`` arm exercises the specific place D-025 bites, on purpose.**
``keras.ops.normalize(order=2)`` evaluates ``x * minimum(rsqrt(sum_sq), 1/eps)``.
Upstream's ``F.normalize`` epsilon is ``1e-12``; in ``float16`` that constant is
**itself zero**, so ``1/eps`` is ``inf`` and a padding row -- a token whose
embedding is the exact zero vector -- computes ``0 * inf = nan``. A real token
row is unaffected and the shape never changes, so the defect is invisible to
every float32 test in the tree. ``_normalize_epsilon_for`` floors the epsilon at
the compute dtype's smallest normal. Both halves are asserted here: the floored
path stays finite, and the raw ``1e-12`` path is shown to produce ``nan``, so
the guard is not merely passing -- it is passing against a demonstrated hazard.

The bridge closed forms are covered from the other side: under a
``mixed_float16`` global policy they must still evaluate in ``float32``, because
``C`` is ``O(1e-4)`` near either endpoint and is divided by.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.bit_diffusion.model import DiTXA
from dl_techniques.models.vision_language.bit_diffusion.sde import (
    CosineDecayingVolatilitySDE,
    bridge_math_dtype,
)
from dl_techniques.models.vision_language.bit_diffusion.token_decoder import (
    SharedTokenDecoder,
)

from ._ditxa_helpers import activate, batch, np_

POLICIES = ["float32", "mixed_float16", "float64"]


@pytest.fixture
def policy(request):
    """Set a global dtype policy for one test and restore it afterwards.

    Restoration is in a ``finally``: a policy leaking out of a failing test
    would silently re-dtype every test that runs after it in the same process,
    and pytest ordering makes that intermittent.
    """
    previous = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy(request.param)
    try:
        yield request.param
    finally:
        keras.mixed_precision.set_global_policy(previous)


def input_dtype_for(policy_name):
    """The dtype the caller feeds in. ``mixed_float16`` takes float32 inputs."""
    return "float64" if policy_name == "float64" else "float32"


class TestDiTXAUnderEveryPolicy:
    """The whole model, end to end."""

    @pytest.mark.parametrize("policy", POLICIES, indirect=True)
    def test_the_forward_pass_is_finite(self, policy):
        model = DiTXA.from_variant("tiny")
        inputs = batch(model, batch_size=2)
        dtype = input_dtype_for(policy)
        inputs = {
            k: (v.astype(dtype) if v.dtype.kind == "f" else v)
            for k, v in inputs.items()
        }
        model(inputs)
        activate(model, seed=5)

        out = np_(model(inputs, training=False))
        assert np.all(np.isfinite(out)), (
            f"policy {policy}: {int((~np.isfinite(out)).sum())} of {out.size} "
            "output entries are non-finite"
        )
        assert np.any(out != 0.0), (
            f"policy {policy}: the output is the exact zero tensor, so "
            "'every entry is finite' is a claim about zeros"
        )

    @pytest.mark.parametrize("policy", POLICIES, indirect=True)
    def test_the_compute_dtype_is_the_one_the_policy_asked_for(self, policy):
        """Anti-vacuity: the arm above must actually be running in that dtype.

        Without this, a model that cast everything to float32 internally would
        pass the finiteness arm under all three policies while honouring none
        of them.
        """
        model = DiTXA.from_variant("tiny")
        expected = {
            "float32": "float32",
            "mixed_float16": "float16",
            "float64": "float64",
        }[policy]
        assert model.compute_dtype == expected
        for block in model.blocks:
            assert block.compute_dtype == expected

    @pytest.mark.parametrize("policy", ["mixed_float16"], indirect=True)
    def test_mixed_float16_keeps_its_variables_in_float32(self, policy):
        """That is what "mixed" means; a float16 master copy is a defect."""
        model = DiTXA.from_variant("tiny")
        model.build(
            {
                "x_t": (2, 8, 8, 4),
                "t": (2,),
                "y": (2,),
                "x_cond": (2, 8, 8, 4),
                "direction": (2,),
            }
        )
        assert model.variable_dtype == "float32"
        offenders = [
            w.path for w in model.weights if str(w.dtype) not in ("float32",)
        ]
        assert not offenders, offenders


class TestTheBridgeMathNeverNarrows:
    """A ``mixed_float16`` policy must not reach the closed forms."""

    @pytest.mark.parametrize("policy", POLICIES, indirect=True)
    def test_the_closed_forms_run_at_float32_or_wider(self, policy):
        sde = CosineDecayingVolatilitySDE()
        t = keras.ops.convert_to_tensor(
            np.array([0.0, 0.25, 0.6, 1.0], dtype=input_dtype_for(policy))
        )
        sigma = sde.sigma(t)
        covariance = sde.C(0.0, t, t)
        for name, value in (("sigma", sigma), ("C", covariance)):
            dtype = getattr(value.dtype, "name", None) or str(value.dtype)
            assert dtype in ("float32", "float64"), f"{name} came back {dtype}"
            assert np.all(np.isfinite(np_(value))), name

    def test_the_floor_is_a_floor_and_not_a_hard_coded_float32(self):
        assert bridge_math_dtype("float16") == "float32"
        assert bridge_math_dtype("float32") == "float32"
        assert bridge_math_dtype("float64") == "float64"
        assert bridge_math_dtype("float32", "float64") == "float64"


class TestTheDecoderPaddingPathUnderFloat16:
    """D-025, from both sides."""

    @pytest.mark.parametrize("policy", POLICIES, indirect=True)
    def test_the_decoder_is_finite_including_an_all_zero_token_row(self, policy):
        """The padding path, exercised rather than assumed.

        Row 0 of the batch is the exact zero vector for its FIRST token, which
        is precisely the input that makes the unfloored epsilon produce ``nan``.
        """
        keras.utils.set_random_seed(0)
        decoder = SharedTokenDecoder(
            vocab_size=13, hidden_dim=16, token_seq_len=4, token_emb_dim=8
        )
        dtype = input_dtype_for(policy)
        x = np.random.default_rng(2).normal(size=(3, 32)).astype(dtype)
        x[0, :8] = 0.0  # one all-zero token row: the padding case

        normalized = np_(decoder.normalize_tokens(keras.ops.convert_to_tensor(x)))
        assert np.all(np.isfinite(normalized)), (
            f"policy {policy}: normalize_tokens produced "
            f"{int((~np.isfinite(normalized)).sum())} non-finite entries on a "
            "padding row -- this is the D-025 hazard"
        )
        np.testing.assert_allclose(normalized[0, 0], 0.0, atol=0, rtol=0)

        logits = np_(decoder(keras.ops.convert_to_tensor(x), training=False))
        assert np.all(np.isfinite(logits)), policy
        assert np.any(logits != 0.0), policy

    def test_the_unfloored_epsilon_really_does_nan_in_float16(self):
        """The hazard, demonstrated. Without it the arm above proves nothing.

        This calls ``keras.ops.normalize`` directly with upstream's ``1e-12``
        under ``float16`` -- the thing ``_normalize_epsilon_for`` exists to
        prevent -- and asserts the ``nan``. If a future Keras changes that op's
        algebra this arm goes red, which is the correct outcome: the floor's
        justification would have expired.
        """
        zeros = keras.ops.convert_to_tensor(np.zeros((1, 4), dtype="float16"))
        unfloored = np_(keras.ops.normalize(zeros, axis=-1, order=2, epsilon=1e-12))
        assert np.all(np.isnan(unfloored)), (
            "upstream's 1e-12 epsilon no longer NaNs a float16 zero row; "
            f"got {unfloored} -- re-derive D-025 before relying on the floor"
        )

        floored = np_(
            keras.ops.normalize(
                zeros, axis=-1, order=2, epsilon=float(np.finfo(np.float16).tiny)
            )
        )
        np.testing.assert_allclose(floored, 0.0, atol=0, rtol=0)
