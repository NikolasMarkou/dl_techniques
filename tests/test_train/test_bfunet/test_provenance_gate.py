"""The three ``src/train/bfunet/`` checkpoint-load paths REFUSE a legacy checkpoint.

plan_2026-07-12_e56909cd/D-005. The unit-domain migration made these tools EMIT and
CONSUME ``[0,1]`` data, but they used to accept any ``.keras`` file. A bias-free net is
degree-1 homogeneous and cannot subtract a DC offset, so pointing ``eval_psnr_vs_noise``
at one of the legacy ``[-0.5,+0.5]`` checkpoints printed a complete, finite, plausible —
and WRONG — PSNR table. These are the tools that produce every dB number this project
reports, so the gate must cover them, not just the application.

The gate runs BEFORE any Keras load, so a dummy ``.keras`` file is enough: these tests
stay fast and CPU-only. The synthetic checkpoint dir means they do not depend on
``results/`` existing.

Run:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg .venv/bin/python -m pytest \\
        tests/test_train/test_bfunet/test_provenance_gate.py -q
"""

import json

import pytest

from train.bfunet.common import BFUnetTrainingConfig, train
from train.bfunet.eval_psnr_vs_noise import load_denoiser
from train.bfunet.eval_per_pixel_uncertainty import _load_denoiser


def _legacy_ckpt(tmp_path):
    """A pre-migration checkpoint: a real-looking dir whose config.json has no stamp."""
    (tmp_path / "best_model.keras").write_bytes(b"not-a-real-model")
    (tmp_path / "config.json").write_text(
        json.dumps({"variant": "base", "patch_size": 256, "convnext_version": "v1"})
    )
    return tmp_path / "best_model.keras"


def _unit_ckpt(tmp_path):
    """A post-migration checkpoint: config.json carries the [0,1] stamp."""
    (tmp_path / "best_model.keras").write_bytes(b"not-a-real-model")
    (tmp_path / "config.json").write_text(json.dumps({"data_range": "[0,1]"}))
    return tmp_path / "best_model.keras"


class TestEvalPsnrVsNoise:
    def test_load_denoiser_refuses_legacy_checkpoint(self, tmp_path):
        with pytest.raises(ValueError, match=r"REFUSING.*data_range"):
            load_denoiser(str(_legacy_ckpt(tmp_path)))

    def test_load_denoiser_gate_passes_on_stamped_checkpoint(self, tmp_path):
        # The gate must ACCEPT the stamp and fall through to the real loader, which
        # then fails on our dummy file. Any error here must NOT be the gate's.
        with pytest.raises(Exception) as exc:  # noqa: PT011 - keras raises its own type
            load_denoiser(str(_unit_ckpt(tmp_path)))
        assert "REFUSING" not in str(exc.value)


class TestEvalPerPixelUncertainty:
    def test_load_denoiser_refuses_legacy_checkpoint(self, tmp_path):
        with pytest.raises(ValueError, match=r"REFUSING.*data_range"):
            _load_denoiser(str(_legacy_ckpt(tmp_path)))

    def test_load_denoiser_gate_passes_on_stamped_checkpoint(self, tmp_path):
        with pytest.raises(Exception) as exc:  # noqa: PT011 - keras raises its own type
            _load_denoiser(str(_unit_ckpt(tmp_path)))
        assert "REFUSING" not in str(exc.value)


class TestInitFromWarmStart:
    """The ``--init-from`` path is the subtle one.

    ``load_weights_from_checkpoint``'s existing "0 layers loaded" guard does NOT trip
    on a legacy checkpoint: one of the SAME architecture loads 100% of its layers
    happily. The run then warm-starts from weights trained in the wrong pixel domain
    with no signal at all. Only the provenance gate catches it.
    """

    @staticmethod
    def _config(tmp_path, init_from) -> BFUnetTrainingConfig:
        return BFUnetTrainingConfig(
            experiment_name="provenance_gate_test",
            output_dir=str(tmp_path / "out"),
            init_from=str(init_from),
        )

    def test_train_refuses_legacy_init_from(self, tmp_path):
        ckpt = _legacy_ckpt(tmp_path)
        config = self._config(tmp_path, ckpt)

        def _never_called(*args, **kwargs):
            raise AssertionError("train() must refuse BEFORE building anything")

        with pytest.raises(ValueError, match=r"REFUSING.*data_range"):
            train(
                config,
                build_model_fn=_never_called,
                verify_fn=_never_called,
                model_label="test",
                results_dir_prefix="test",
            )

    def test_train_refusal_precedes_any_output_dir(self, tmp_path):
        # The gate is a fail-fast: nothing (not even the results dir) is created.
        ckpt = _legacy_ckpt(tmp_path)
        config = self._config(tmp_path, ckpt)
        with pytest.raises(ValueError, match=r"REFUSING"):
            train(
                config,
                build_model_fn=lambda c: None,
                verify_fn=lambda m: None,
                model_label="test",
                results_dir_prefix="test",
            )
        assert not (tmp_path / "out").exists()
