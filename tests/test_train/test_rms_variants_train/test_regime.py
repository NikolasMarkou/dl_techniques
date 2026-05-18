"""SC-7: --regime CLI on each of E1/E3/E4/E5 (E2 stub).

Each experiment trainer carries a module-level ``_REGIME_MAP`` dict mapping
regime name → ``(lr, batch, mp, depth_override, wd_override)`` 5-tuple,
and an argparse ``--regime`` flag with the same key set. The 5th slot
was added at step 6 of plan_2026-05-18_6776f8ba (Phase 3 v3) to support
the ``wd_zero`` stress regime. This test verifies:

1. ``_REGIME_MAP`` exists with the documented key set per trainer.
2. The argparse parser accepts every declared regime name.
3. Every regime maps to a well-formed 4-tuple (None or a typed value).
4. The mapping values match the Phase 3 design contract (lr/batch numbers,
   mp bools, depth override ints).
"""
from __future__ import annotations

import sys

import pytest

from train.rms_variants_train.experiments import (
    e1_vit_cifar10,
    e2_resnet_cifar100,
    e3_tinytransformer_imdb,
    e4_deep_residual_reg,
    e5_norm_layer_microbench,
)


EXPECTED_REGIMES = {
    "e1": {"default", "lr_low", "lr_high", "mp_fp16",
           "lr_extreme", "wd_zero", "bs_4", "mp_fp16_lowloss"},
    "e2": {"default"},
    "e3": {"default", "mp_fp16",
           "lr_extreme", "wd_zero"},
    "e4": {"default", "depth_12", "depth_48",
           "wd_zero", "mp_fp16_lowloss"},
    "e5": {"default", "bs_32", "bs_256", "lr_low", "lr_high",
           "lr_extreme", "wd_zero"},
}


MODULES = {
    "e1": e1_vit_cifar10,
    "e2": e2_resnet_cifar100,
    "e3": e3_tinytransformer_imdb,
    "e4": e4_deep_residual_reg,
    "e5": e5_norm_layer_microbench,
}


@pytest.mark.parametrize("exp_id", list(EXPECTED_REGIMES.keys()))
def test_regime_map_keys(exp_id: str) -> None:
    module = MODULES[exp_id]
    rmap = getattr(module, "_REGIME_MAP", None)
    assert rmap is not None, f"{exp_id} missing _REGIME_MAP"
    assert set(rmap.keys()) == EXPECTED_REGIMES[exp_id], (
        f"{exp_id} regime keys mismatch: got {set(rmap.keys())}, "
        f"expected {EXPECTED_REGIMES[exp_id]}"
    )


@pytest.mark.parametrize("exp_id", list(EXPECTED_REGIMES.keys()))
def test_regime_map_value_shape(exp_id: str) -> None:
    """Every value is a 5-tuple ``(lr, batch, mp, depth_override, wd_override)``.

    E2 retains the legacy 4-tuple shape because it does not consume the
    Phase 3 stress regimes (E2 has only ``default``); we tolerate both
    shapes there to avoid a no-op trainer edit.
    """
    module = MODULES[exp_id]
    rmap = module._REGIME_MAP
    for name, val in rmap.items():
        assert isinstance(val, tuple), f"{exp_id}/{name} not a tuple: {val}"
        if exp_id == "e2":
            assert len(val) == 4, f"{exp_id}/{name} length != 4: {val}"
            lr, batch, mp, depth = val
            wd = None
        else:
            assert len(val) == 5, f"{exp_id}/{name} length != 5: {val}"
            lr, batch, mp, depth, wd = val
        assert lr is None or isinstance(lr, float)
        assert batch is None or isinstance(batch, int)
        assert mp is None or isinstance(mp, bool)
        assert depth is None or isinstance(depth, int)
        assert wd is None or isinstance(wd, float)


@pytest.mark.parametrize("exp_id", list(EXPECTED_REGIMES.keys()))
def test_argparse_accepts_every_declared_regime(exp_id: str) -> None:
    """``--regime <name>`` parses for every name in _REGIME_MAP."""
    module = MODULES[exp_id]
    saved_argv = sys.argv
    try:
        for regime in EXPECTED_REGIMES[exp_id]:
            sys.argv = [
                exp_id,
                "--norm-type", "rms_norm",
                "--out-dir", "/tmp/_test_unused",
                "--regime", regime,
            ]
            ns = module._parse_args()
            assert ns.regime == regime
    finally:
        sys.argv = saved_argv


@pytest.mark.parametrize("exp_id", list(EXPECTED_REGIMES.keys()))
def test_argparse_default_is_default(exp_id: str) -> None:
    """Missing --regime defaults to 'default' — keeps prior CLI invocations working."""
    module = MODULES[exp_id]
    saved_argv = sys.argv
    try:
        sys.argv = [
            exp_id, "--norm-type", "rms_norm", "--out-dir", "/tmp/_test_unused",
        ]
        ns = module._parse_args()
        assert ns.regime == "default"
    finally:
        sys.argv = saved_argv


def test_e1_lr_low_maps_to_specific_lr() -> None:
    """Phase 3 design contract: E1 lr_low → lr=1e-4."""
    lr, batch, mp, depth, wd = e1_vit_cifar10._REGIME_MAP["lr_low"]
    assert lr == 1e-4
    assert batch is None
    assert mp is None
    assert depth is None
    assert wd is None


def test_e1_mp_fp16_maps_to_mp_true() -> None:
    lr, batch, mp, depth, wd = e1_vit_cifar10._REGIME_MAP["mp_fp16"]
    assert lr is None
    assert mp is True
    assert wd is None


def test_e4_depth_overrides_to_int() -> None:
    """E4 depth_12 → depth_override=12; depth_48 → 48."""
    _, _, _, depth12, _ = e4_deep_residual_reg._REGIME_MAP["depth_12"]
    _, _, _, depth48, _ = e4_deep_residual_reg._REGIME_MAP["depth_48"]
    assert depth12 == 12
    assert depth48 == 48


def test_e5_bs_32_maps_to_batch_32() -> None:
    _, batch, _, _, _ = e5_norm_layer_microbench._REGIME_MAP["bs_32"]
    assert batch == 32


def test_e5_bs_256_maps_to_batch_256() -> None:
    _, batch, _, _, _ = e5_norm_layer_microbench._REGIME_MAP["bs_256"]
    assert batch == 256


# ---------------------------------------------------------------------
# Phase 3 v3 stress regimes (plan_2026-05-18_6776f8ba step 6)
# ---------------------------------------------------------------------


def test_e1_lr_extreme_is_3em3() -> None:
    """E1 lr_extreme is 10× the trainer default (3e-4 → 3e-3)."""
    lr, _, _, _, wd = e1_vit_cifar10._REGIME_MAP["lr_extreme"]
    assert lr == 3e-3
    assert wd is None


def test_e1_wd_zero_drops_weight_decay() -> None:
    """E1 wd_zero forces weight_decay=0.0."""
    lr, _, _, _, wd = e1_vit_cifar10._REGIME_MAP["wd_zero"]
    assert wd == 0.0
    assert lr is None


def test_e1_bs_4_maps_to_batch_4() -> None:
    """E1 bs_4: tiny-batch stress (4 examples per step)."""
    _, batch, _, _, _ = e1_vit_cifar10._REGIME_MAP["bs_4"]
    assert batch == 4


def test_e1_mp_fp16_lowloss_drops_lr_under_fp16() -> None:
    """E1 mp_fp16_lowloss runs fp16 at lr=1e-4 (edge-of-stability)."""
    lr, _, mp, _, _ = e1_vit_cifar10._REGIME_MAP["mp_fp16_lowloss"]
    assert lr == 1e-4
    assert mp is True


def test_e3_lr_extreme_is_3em3() -> None:
    lr, _, _, _, _ = e3_tinytransformer_imdb._REGIME_MAP["lr_extreme"]
    assert lr == 3e-3


def test_e3_wd_zero_drops_weight_decay() -> None:
    _, _, _, _, wd = e3_tinytransformer_imdb._REGIME_MAP["wd_zero"]
    assert wd == 0.0


def test_e4_wd_zero_drops_weight_decay() -> None:
    _, _, _, _, wd = e4_deep_residual_reg._REGIME_MAP["wd_zero"]
    assert wd == 0.0


def test_e4_mp_fp16_lowloss_drops_lr_under_fp16() -> None:
    lr, _, mp, _, _ = e4_deep_residual_reg._REGIME_MAP["mp_fp16_lowloss"]
    assert lr == 1e-4
    assert mp is True


def test_e5_lr_extreme_is_3em3() -> None:
    lr, _, _, _, _ = e5_norm_layer_microbench._REGIME_MAP["lr_extreme"]
    assert lr == 3e-3


def test_e5_wd_zero_drops_weight_decay() -> None:
    _, _, _, _, wd = e5_norm_layer_microbench._REGIME_MAP["wd_zero"]
    assert wd == 0.0


# ---------------------------------------------------------------------
# E2 retains legacy 4-tuple
# ---------------------------------------------------------------------


def test_e2_default_is_4tuple() -> None:
    """E2 has no stress regimes; legacy 4-tuple shape is fine."""
    val = e2_resnet_cifar100._REGIME_MAP["default"]
    assert len(val) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
