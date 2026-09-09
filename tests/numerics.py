"""Shared numeric tolerances for the test suite.

One home for bounds that must be DERIVED from a noise source rather than pasted as a
literal. Everything here is imported by more than one test module; nothing here is
allowed to become a magic number.

Moved out of `tests/test_layers/test_transformers/test_energy_transformer.py` on
2026-08-17 when `tests/test_layers/test_attention/test_energy_attention.py` became the
second consumer — that module carried a comment reading "Do NOT 'fix' this by bumping
the literal: derive it from the noise source, as `reassociation_atol` does", which is a
copy-the-reasoning instruction, i.e. the shape a shared helper exists to remove.
"""

import numpy as np


# ---------------------------------------------------------------------
# Derived float32 reduction-reassociation tolerance
# ---------------------------------------------------------------------

_F32_EPS = float(np.finfo(np.float32).eps)      # 1.1920929e-07
_F32_U = _F32_EPS / 2.0                         # unit roundoff
_TAIL_FACTOR = 8.0                              # 8-sigma tail on the random-walk model


# DECISION plan-2026-07-30T140922-8af1028f/D-024: this helper exists because the three
# assertions below shipped with `atol=1e-6`, which is BELOW the float32 noise floor of the
# path they measure — the measured cross-evaluation difference is 0.97 to 2.11 `eps_f32` of
# the output magnitude at every one of the four measured cells (measured 2026-07-30
# CPU-only, see the calibration table in `reassociation_atol`). A bound
# below the output dtype's own resolution can NEVER pass, so it measures NOTHING: all three
# tests were RED at the exact commit that introduced them (`bd1956fc`, `1eebb815` — bisected
# with `git worktree`, bit-identical numbers), never once green, for their whole lifetime.
# Widening THAT is a repair, not a loosening.
#
# THIS IS NOT A PRECEDENT FOR LOOSENING AN ATTAINABLE BOUND. The distinction is whether the
# bound is reachable by a CORRECT implementation in the output dtype:
#   - unattainable (this case) -> the number is broken and must be re-derived;
#   - attainable but flaky (e.g. `TestPowerIteration::test_convergence_iterations`, whose
#     `rtol=0.1` is reachable and whose ~6% failure rate has a fixable cause) -> widening is
#     FORBIDDEN; fix the cause instead.
# Do NOT cite this helper when arguing to relax a tolerance in the second category.
#
# Do NOT replace the derivation with the measured numbers. A pasted constant stops tracking
# the shapes it came from: change `num_steps`, `hopfield_dim`, `num_heads` or the sequence
# length and a magic number silently becomes either vacuous or unattainable again.
def reassociation_atol(reduction_lengths, num_steps: int, scale: float) -> float:
    """Bound on the float32 difference between two REASSOCIATED evaluations of one formula.

    Both quantities compared by the callers are mathematically identical; they differ only
    in the order the backend accumulates its reductions. Permuting the token axis
    (`test_no_token_mixing`) changes which token lands in which vectorized lane; padding the
    sequence (`TestKerasMaskIsHonored`) changes the batch SHAPE, and therefore the kernel's
    blocking. Neither changes the exact answer; both change the rounding.

    Derivation (so this is a bound, not a pasted magic number):

    1. Each contraction of length ``L`` performs ``L`` rounded multiply-adds, each
       contributing an error of order ``u * |partial sum|`` with ``u = eps / 2``.
    2. The rounding errors of a reduction are not adversarially aligned; over a chain of
       ``M`` rounded operations they accumulate as a random walk, giving a relative error of
       order ``sqrt(M) * u``. Charge BOTH sides of the comparison, so ``M = 2 * num_steps *
       sum(reduction_lengths)``.
    3. Take an 8-sigma tail factor for the walk, and scale by the output magnitude because
       the error is relative::

           atol = 8 * sqrt(2 * num_steps * sum(L)) * u * max(1, max|output|)

    Calibration (measured 2026-07-30, CPU-only, `CUDA_VISIBLE_DEVICES=""`, TF32 disabled by
    this module's `pytestmark = pytest.mark.usefixtures("tf32_disabled")` opt-in near the
    top of the file — the numbers below were taken at step 7 under the import-time toggle
    that opt-in REPLACED at step 10; the toggle is gone, the regime it produced is not, and
    the assertions stayed green across the swap). DERIVED FIRST, then
    checked — the formula and the
    8-sigma factor were fixed before any of these numbers were read, and none was tuned
    afterwards:

    ===========================  ========  ==========  ==========  =====
    site                         scale     derived     measured    ratio
    ===========================  ========  ==========  ==========  =====
    no_token_mixing[relu]        1.64e+01  7.68e-05    1.91e-06    40x
    no_token_mixing[softmax]     1.42e+00  6.66e-06    3.58e-07    19x
    mask tests (both, N=6, T=3)  1.24e+02  3.26e-03    1.53e-05    214x
    ===========================  ========  ==========  ==========  =====

    The measured RELATIVE noise is 1.16e-07 / 2.51e-07 (Hopfield relu / softmax) and
    1.23e-07 (block), i.e. 0.97, 2.11 and 1.03 ``eps_f32`` — one mechanism operating at the
    same order at all three sites, not two (plan assumption A-6). The bound sits above
    that because a random walk's bound must; the slack is NOT a vacuity hole, and that is
    proven by execution rather than argued. Injecting the real defect each test exists to
    catch puts the difference four orders of magnitude ABOVE these bounds:

    - `no_token_mixing`: `update += 0.1 * cumsum(update, axis=1)` -> 3.96 (relu) / 0.265
      (softmax) vs bounds of 9.27e-05 / 7.12e-06.
    - both mask tests: `mask = None` at the top of `EnergyTransformer.call` (the F-02b
      defect verbatim) -> 1.26e+02 vs a bound of 3.26e-03.

    See the RED proofs recorded in
    `plans/plan-2026-07-30T140922-8af1028f/verification.md` row 9.

    **This bound assumes TRUE float32 matmuls, and it is the one regime dependence worth
    naming.** It is expressed in ``eps_f32 = 1.19e-07``; TF32 truncates the matmul mantissa
    to 10 bits, i.e. one TF32 ulp is ``2**-11 = 4.88e-04``, ~4100x larger, so under TF32 the
    reassociation noise would exceed this bound and these assertions would need to be
    restated in TF32 ulps (the `_TF32_ULP` pattern at
    `test_gated_linear_attention_block.py:1298`). That is exactly why this module opts into
    the shared `tf32_disabled` fixture via the module-level `pytestmark` above — do NOT
    remove that `pytestmark` without re-deriving here. (Until step 10 the mechanism was a
    bare `enable_tensor_float_32_execution(False)` at import time, which was PROCESS-GLOBAL
    and never restored: it silently changed the regime of every module collected after this
    one. The `pytestmark` opt-in is the same regime, scoped and restored, and
    `test_this_module_really_runs_with_tf32_disabled` is its anti-vacuity guard.) Verified
    2026-07-30 in three regimes, all CPU-only: file alone (168 passed), co-collected behind
    `test_linear_attention.py` (208 passed), and with TF32 force-ENABLED after collection
    (168 passed, all four measured diffs bit-identical). The third is a WEAK check and is
    labelled as such: with no CUDA device present TF32 is inert, so it proves the tests do
    not depend on the toggle's VALUE on CPU, not that the bound survives real TF32 hardware.
    Measuring that needs a GPU.

    Callers must pass ``rtol=0``: `assert_allclose`'s default ``rtol=1e-7`` on an output of
    magnitude 1.2e+02 silently contributes 1.2e-05 of tolerance, which would dominate the
    derived bound and make it decorative.

    :param reduction_lengths: Contraction lengths on the compared path, per descent step.
    :type reduction_lengths: Sequence[int]
    :param num_steps: Number of times that path is applied (the recurrent descent length).
    :type num_steps: int
    :param scale: Magnitude of the compared output, ``max|expected|``.
    :type scale: float
    :return: Absolute tolerance.
    :rtype: float
    """
    ops_count = 2.0 * num_steps * float(sum(reduction_lengths))
    return _TAIL_FACTOR * np.sqrt(ops_count) * _F32_U * max(1.0, float(scale))


# ---------------------------------------------------------------------
# The ARITHMETIC REGIME of the active device's float32 matmul
# ---------------------------------------------------------------------

#: Unit roundoff of NVIDIA TensorFloat-32. TF32 carries a 10-bit explicit mantissa, so
#: one ulp near 1.0 is ``2**-10`` and the unit roundoff is half of that. ~4100x the
#: float32 unit roundoff of ``5.96e-08``. The literal is not new here: it is the
#: ``_TF32_ULP = 2.0 ** -11`` of
#: ``tests/test_layers/test_transformers/test_gated_linear_attention_block.py:1495``,
#: promoted to the shared home so a third copy is not written.
TF32_UNIT_ROUNDOFF = 2.0 ** -11

#: How many unit roundoffs of the ACTIVE matmul precision a reassociated / retraced
#: float32 comparison is allowed. Also not a new number: ``4.0 * _TF32_ULP * scale`` is
#: the bound `test_gated_linear_attention_block.py` already uses in three places
#: (``:1690``, ``:1792``, ``:1861``). One convention, not two.
MATMUL_ULP_ALLOWANCE = 4.0

#: Detection threshold for :func:`matmul_unit_roundoff`. The probe below perturbs a
#: float32 matmul operand by ``2**-13``, which float32 represents exactly and TF32
#: cannot represent at all; half of it separates "the perturbation survived" from
#: "the perturbation was rounded away".
_MATMUL_PROBE_DELTA = 2.0 ** -13


# DECISION plan-2026-09-09T042752-6d66ac56/D-030: this probe MEASURES the matmul
# precision; it does NOT read `tf.config.experimental.tensor_float_32_execution_enabled()`.
# Do NOT "simplify" it to that flag. MEASURED on this box: the flag reads True in a stock
# process with NO CUDA device visible, because it is a config flag whose NUMERIC effect --
# not its value -- is device-dependent. A bound selected off the flag would therefore be
# 4100x too loose on every CPU-only run, i.e. the guards it feeds would stop failing.
# The probe's three measured regimes (2026-09-09, RTX 4090):
#   GPU, TF32 on  -> error 1.221e-04 (== 2**-13: the perturbation was rounded away)
#   GPU, TF32 off -> error 0.0
#   CPU           -> error 0.0        (flag True, effect absent -- the case the flag lies about)
# Rationale: decisions.md D-030.
def matmul_unit_roundoff(probe_size: int = 64) -> float:
    """Measure the unit roundoff of the ACTIVE device's float32 matmul.

    Interface contract: pure w.r.t. the caller's state, never raises, returns a strictly
    positive float that is either :data:`TF32_UNIT_ROUNDOFF` or the float32 unit roundoff
    ``5.96e-08``. It is deliberately NOT cached: a session may toggle TF32 mid-run (the
    ``tf32_disabled`` fixture does exactly that), so a cached answer would describe a
    regime that is no longer in force.

    The probe multiplies ``full((n, n), 1 + 2**-13)`` by the identity. ``1 + 2**-13`` is
    exact in float32 and is NOT representable in TF32's 10-bit mantissa, so a reduced
    precision matmul returns exactly ``1.0`` and the error reads ``2**-13``.

    :param probe_size: Side of the square probe matmul. 64 already dispatches to tensor
        cores on this hardware (verified at 64/128/256/512 -- identical reading).
    :type probe_size: int
    :return: The unit roundoff in force for float32 matmul on the active device.
    :rtype: float
    """
    import keras  # local: this module is otherwise pure NumPy and imported very widely.

    value = 1.0 + _MATMUL_PROBE_DELTA
    left = keras.ops.convert_to_tensor(
        np.full((probe_size, probe_size), value, dtype="float32")
    )
    right = keras.ops.convert_to_tensor(np.eye(probe_size, dtype="float32"))
    product = keras.ops.convert_to_numpy(keras.ops.matmul(left, right))
    lost = float(np.max(np.abs(np.asarray(product) - value)))
    return TF32_UNIT_ROUNDOFF if lost > _MATMUL_PROBE_DELTA / 2.0 else _F32_U


def matmul_precision_atol(scale: float) -> float:
    """Bound contributed by the ACTIVE device's matmul precision alone.

    A companion to :func:`reassociation_atol`, not a replacement: that helper bounds the
    error of REORDERING true float32 arithmetic, and this one bounds the error of doing
    the arithmetic in a NARROWER format than float32. On a true-float32 device the two
    are nine orders apart and this term is inert; on a tensor-core GPU with TF32 enabled
    it dominates by ~4100x. Callers take the MAXIMUM of the two, so one expression covers
    both regimes and neither regime loosens the other.

    :param scale: Magnitude of the compared output, ``max|expected|``.
    :type scale: float
    :return: Absolute tolerance.
    :rtype: float
    """
    return MATMUL_ULP_ALLOWANCE * matmul_unit_roundoff() * max(1.0, float(scale))
