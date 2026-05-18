"""SC-7: --regime CLI on each of E1/E3/E4/E5 (E2 stub).

Each experiment trainer carries a module-level ``_REGIME_MAP`` dict mapping
regime name → ``(lr, batch, mp, depth_override)`` 4-tuple, and an argparse
``--regime`` flag with the same key set. This test verifies:

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
    "e1": {"default", "lr_low", "lr_high", "mp_fp16"},
    "e2": {"default"},
    "e3": {"default", "mp_fp16"},
    "e4": {"default", "depth_12", "depth_48"},
    "e5": {"default", "bs_32", "bs_256", "lr_low", "lr_high"},
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
    """Every value is a 4-tuple ``(lr, batch, mp, depth_override)``."""
    module = MODULES[exp_id]
    rmap = module._REGIME_MAP
    for name, val in rmap.items():
        assert isinstance(val, tuple), f"{exp_id}/{name} not a tuple: {val}"
        assert len(val) == 4, f"{exp_id}/{name} length != 4: {val}"
        lr, batch, mp, depth = val
        assert lr is None or isinstance(lr, float)
        assert batch is None or isinstance(batch, int)
        assert mp is None or isinstance(mp, bool)
        assert depth is None or isinstance(depth, int)


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
    lr, batch, mp, depth = e1_vit_cifar10._REGIME_MAP["lr_low"]
    assert lr == 1e-4
    assert batch is None
    assert mp is None
    assert depth is None


def test_e1_mp_fp16_maps_to_mp_true() -> None:
    lr, batch, mp, depth = e1_vit_cifar10._REGIME_MAP["mp_fp16"]
    assert lr is None
    assert mp is True


def test_e4_depth_overrides_to_int() -> None:
    """E4 depth_12 → depth_override=12; depth_48 → 48."""
    _, _, _, depth12 = e4_deep_residual_reg._REGIME_MAP["depth_12"]
    _, _, _, depth48 = e4_deep_residual_reg._REGIME_MAP["depth_48"]
    assert depth12 == 12
    assert depth48 == 48


def test_e5_bs_32_maps_to_batch_32() -> None:
    _, batch, _, _ = e5_norm_layer_microbench._REGIME_MAP["bs_32"]
    assert batch == 32


def test_e5_bs_256_maps_to_batch_256() -> None:
    _, batch, _, _ = e5_norm_layer_microbench._REGIME_MAP["bs_256"]
    assert batch == 256


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
