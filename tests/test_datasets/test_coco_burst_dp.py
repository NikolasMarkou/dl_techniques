"""Tests for ``dl_techniques.datasets.vision.coco_burst_dp``.

Focused on the F1 plan (plan_2026-05-19_b225c8df): the ``aux_spec`` kwarg
on :func:`build_coco_burst_dp_datasets` must propagate into both train and
val loader configs.

The underlying COCO multi-task loader is heavy (requires a real COCO root
on disk); we skip the propagation test when the default COCO root is
absent. The factory signature check is always exercised.
"""

from __future__ import annotations

import inspect
import os

import pytest

from dl_techniques.datasets.vision.coco_burst_dp import (
    COCO2017BurstDPLoader,
    COCOBurstDPConfig,
    DistortionSpec,
    build_coco_burst_dp_datasets,
    default_aux_spec,
)
from dl_techniques.datasets.vision.coco_multitask_local import COCO_DEFAULT_ROOT


class TestBuildCocoBurstDpFactory:
    def test_aux_spec_kwarg_present(self) -> None:
        params = inspect.signature(build_coco_burst_dp_datasets).parameters
        assert "aux_spec" in params
        # Defaults to None ⇒ "no override".
        assert params["aux_spec"].default is None

    @pytest.mark.skipif(
        not os.path.isdir(COCO_DEFAULT_ROOT),
        reason=f"COCO root not present at {COCO_DEFAULT_ROOT}",
    )
    def test_aux_spec_propagates_to_loaders(self) -> None:
        custom = DistortionSpec(noise_sigma_range=(0.5, 0.5))
        train, val = build_coco_burst_dp_datasets(
            coco_root=COCO_DEFAULT_ROOT,
            image_size=64,
            batch_size=2,
            n_max=2,
            n_min=1,
            max_train_images=2,
            max_val_images=2,
            workers=1,
            aux_spec=custom,
            seed=0,
        )
        assert isinstance(train, COCO2017BurstDPLoader)
        assert isinstance(val, COCO2017BurstDPLoader)
        assert train.cfg.aux_spec.noise_sigma_range == (0.5, 0.5)
        assert val.cfg.aux_spec.noise_sigma_range == (0.5, 0.5)

    @pytest.mark.skipif(
        not os.path.isdir(COCO_DEFAULT_ROOT),
        reason=f"COCO root not present at {COCO_DEFAULT_ROOT}",
    )
    def test_aux_spec_none_preserves_default(self) -> None:
        train, _ = build_coco_burst_dp_datasets(
            coco_root=COCO_DEFAULT_ROOT,
            image_size=64,
            batch_size=2,
            n_max=2,
            n_min=1,
            max_train_images=2,
            max_val_images=2,
            workers=1,
            aux_spec=None,
            seed=0,
        )
        assert train.cfg.aux_spec.noise_sigma_range == default_aux_spec().noise_sigma_range


class TestCocoBurstDpConfigShape:
    """Config-only test that doesn't touch the COCO loader."""

    def test_config_accepts_aux_spec(self) -> None:
        custom = DistortionSpec(noise_sigma_range=(0.5, 0.5))
        cfg = COCOBurstDPConfig(aux_spec=custom)
        assert cfg.aux_spec.noise_sigma_range == (0.5, 0.5)

    def test_config_default_aux_spec(self) -> None:
        cfg = COCOBurstDPConfig()
        assert cfg.aux_spec.noise_sigma_range == default_aux_spec().noise_sigma_range
