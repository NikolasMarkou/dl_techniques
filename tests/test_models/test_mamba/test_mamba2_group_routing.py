"""`ngroups > 1` must ROUTE heads to groups, not sum over the group axis.

F-46 of the 2026-08-18 deep review, fixed under
``plan-2026-08-18T140459-7991552f/D-042``. Only reachable at all since F-48
(D-036) plumbed ``ngroups`` through ``Mamba2ResidualBlock`` and ``Mamba2``.

`Mamba2Layer._ssm_scan` used to compute ``einsum("bh,bgn->bhgn")`` ->
``einsum("bhgn,bhp->bhpn")`` for the state update and ``einsum("bhpn,bgn->bhpg")``
-> ``sum(axis=3)`` for the output, both of which CONTRACT the group axis. Every
head therefore received ``sum_g B_g`` and ``sum_g C_g``: the layer allocated
``2 * ngroups * d_state`` projection channels and then collapsed them
additively, which is exactly ``ngroups = 1`` with a rescaled ``B``/``C``. GQA --
which ``mamba_v2.py``'s module docstring explicitly invokes -- broadcasts WITHIN
a group; it does not sum ACROSS groups.

MEASURED (CPU; the GPU's TF32 floor is ~4e-04 relative here and is NOT tight
enough to see the difference between "equal" and "equal to float32"):

===================================================  ==========  ==========
instrument (nheads=4, ngroups=4, |y|max ~ 10.2)      pre-fix     post-fix
===================================================  ==========  ==========
max|y(B,C) - y(permuted groups)|                     9.54e-07    4.08e+00
max|y(ngroups=4) - y(ngroups=1, B/C summed)|         1.31e-06    9.85e+00
ngroups=1 scan, pre-fix vs post-fix                  ----------- 0.0 exact
ngroups=1 full layer forward, pre-fix vs post-fix    ----------- 0.0 exact
===================================================  ==========  ==========

The last two rows are the compatibility statement: the DEFAULT and every
shipped checkpoint (`ngroups=1`) is bit-identical across the fix, because
summing over a length-1 axis and broadcasting from it are the same operation.
"""

import numpy as np
import keras
from keras import ops
import pytest

from dl_techniques.models.mamba.components_v2 import Mamba2Layer


D_MODEL, D_STATE, HEADDIM, D_SSM = 16, 4, 8, 32
NHEADS = D_SSM // HEADDIM  # 4
BATCH, SEQ = 2, 5


def _layer(ngroups):
    keras.utils.set_random_seed(0)
    layer = Mamba2Layer(
        d_model=D_MODEL,
        d_state=D_STATE,
        d_conv=4,
        expand=2,
        headdim=HEADDIM,
        d_ssm=D_SSM,
        ngroups=ngroups,
    )
    layer.build((None, SEQ + 1, D_MODEL))
    return layer


def _scan_inputs(ngroups, seed=0):
    rng = np.random.RandomState(seed)
    return dict(
        x=ops.convert_to_tensor(
            rng.randn(BATCH, SEQ, NHEADS, HEADDIM).astype("float32")
        ),
        dt=ops.convert_to_tensor(rng.randn(BATCH, SEQ, NHEADS).astype("float32")),
        A=ops.convert_to_tensor(-np.abs(rng.randn(NHEADS)).astype("float32")),
        B=rng.randn(BATCH, SEQ, ngroups, D_STATE).astype("float32"),
        C=rng.randn(BATCH, SEQ, ngroups, D_STATE).astype("float32"),
    )


def _scan(layer, parts, B=None, C=None):
    return ops.convert_to_numpy(
        layer._ssm_scan(
            parts["x"],
            parts["dt"],
            parts["A"],
            ops.convert_to_tensor(parts["B"] if B is None else B),
            ops.convert_to_tensor(parts["C"] if C is None else C),
        )
    )


class TestGroupsAreNotSummed:
    def test_permuting_the_group_axis_changes_the_output(self):
        """The sharpest RED: summation is permutation-INVARIANT, routing is not."""
        layer = _layer(NHEADS)
        parts = _scan_inputs(NHEADS)
        base = _scan(layer, parts)
        perm = [3, 1, 0, 2]
        permuted = _scan(
            layer, parts, B=parts["B"][:, :, perm], C=parts["C"][:, :, perm]
        )
        delta = np.abs(base - permuted).max()
        scale = np.abs(base).max()
        assert delta > 1e-2 * scale, (
            f"the scan is invariant to permuting the group axis "
            f"(max|delta| = {delta:.3e} on |y|max = {scale:.3e}) -- it is still "
            f"contracting over `g` instead of routing heads to groups"
        )

    def test_ngroups_four_is_not_ngroups_one_with_summed_bc(self):
        """The equivalence the review claimed, constructed and then broken.

        `_ssm_scan` reads only `self.dt_bias` off the layer (`A` is passed in),
        so an `ngroups=1` layer with the SAME `dt_bias` is a faithful stand-in
        for "the same model with the group axis already summed away". Pre-fix
        the two agreed to 1.31e-06 on |y|max 10.2 -- the float32 floor.
        """
        four, one = _layer(NHEADS), _layer(1)
        one.dt_bias.assign(four.dt_bias)
        parts = _scan_inputs(NHEADS)

        grouped = _scan(four, parts)
        collapsed = _scan(
            one,
            parts,
            B=parts["B"].sum(axis=2, keepdims=True),
            C=parts["C"].sum(axis=2, keepdims=True),
        )
        delta = np.abs(grouped - collapsed).max()
        assert delta > 1e-2 * np.abs(grouped).max(), (
            f"ngroups={NHEADS} still reduces to ngroups=1 with summed B/C "
            f"(max|delta| = {delta:.3e})"
        )

    def test_each_head_reads_only_its_own_group(self):
        """Perturbing ONE group must move only that group's heads.

        With ``nheads == ngroups`` the mapping is the identity, so group `g`
        owns head `g` and nothing else.
        """
        layer = _layer(NHEADS)
        parts = _scan_inputs(NHEADS)
        base = _scan(layer, parts)

        for g in range(NHEADS):
            bumped = parts["B"].copy()
            bumped[:, :, g] += 1.0
            moved = _scan(layer, parts, B=bumped)
            per_head = np.abs(moved - base).max(axis=(0, 1, 3))  # (H,)
            assert per_head[g] > 1e-3, f"group {g} did not reach head {g}"
            others = np.delete(per_head, g)
            assert others.max() < 1e-6, (
                f"group {g} leaked into heads {np.where(np.delete(np.arange(NHEADS), g))}"
                f" (max|delta| = {others.max():.3e})"
            )

    def test_two_groups_serve_contiguous_head_blocks(self):
        """`nheads=4, ngroups=2` -> heads (0,1) read group 0, (2,3) group 1.

        Order matters: the reference is ``repeat(B, "b l g n -> b l (g h) n")``,
        i.e. contiguous blocks, NOT an interleave.
        """
        layer = _layer(2)
        parts = _scan_inputs(2)
        base = _scan(layer, parts)

        bumped = parts["B"].copy()
        bumped[:, :, 0] += 1.0
        moved = _scan(layer, parts, B=bumped)
        per_head = np.abs(moved - base).max(axis=(0, 1, 3))

        assert per_head[0] > 1e-3 and per_head[1] > 1e-3
        assert per_head[2] < 1e-6 and per_head[3] < 1e-6, (
            f"group 0 reached heads 2/3 -- the head->group mapping is not "
            f"contiguous blocks (per-head deltas {per_head})"
        )


class TestNgroupsOneIsUntouched:
    def test_default_stays_ngroups_one(self):
        # d_model=64 -> d_inner=d_ssm=128, divisible by the default headdim=64.
        assert Mamba2Layer(d_model=64, d_state=D_STATE).ngroups == 1

    def test_ngroups_one_forward_is_finite_and_shape_preserving(self):
        layer = _layer(1)
        x = ops.convert_to_tensor(
            np.random.RandomState(0).randn(BATCH, SEQ + 1, D_MODEL).astype("float32")
        )
        y = ops.convert_to_numpy(layer(x))
        assert y.shape == (BATCH, SEQ + 1, D_MODEL)
        assert np.isfinite(y).all()


class TestDivisibilityGuard:
    def test_nheads_must_be_divisible_by_ngroups(self):
        with pytest.raises(ValueError, match="divisible by ngroups"):
            Mamba2Layer(
                d_model=D_MODEL, d_state=D_STATE, headdim=HEADDIM,
                d_ssm=D_SSM, ngroups=3,   # nheads = 4
            )

    def test_ngroups_must_be_positive(self):
        with pytest.raises(ValueError, match="ngroups must be positive"):
            Mamba2Layer(
                d_model=D_MODEL, d_state=D_STATE, headdim=HEADDIM,
                d_ssm=D_SSM, ngroups=0,
            )
