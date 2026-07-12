"""Tests for the shared denoiser checkpoint provenance gate.

``dl_techniques.utils.denoiser_provenance.require_unit_domain_checkpoint`` is the ONE
implementation behind all four denoiser checkpoint-load paths
(plan_2026-07-12_e56909cd/D-005):

* ``src/train/bfunet/eval_psnr_vs_noise.py``       (``load_denoiser``)
* ``src/train/bfunet/eval_per_pixel_uncertainty.py`` (``_load_denoiser``)
* ``src/train/bfunet/common.py``                   (``train``'s ``--init-from`` warm-start)
* ``src/applications/bias_free_denoiser/denoiser_prior.py`` (``from_pretrained``)

It must FAIL CLOSED on every degenerate case: a bias-free net is degree-1 homogeneous
and cannot subtract a DC offset, so a legacy ``[-0.5,+0.5]`` checkpoint fed ``[0,1]``
data emits a plausible WRONG result rather than an error. A refusal is strictly better
than a plausible wrong number.

Run:
    CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \\
        tests/test_utils/test_denoiser_provenance.py -q
"""

import json

import pytest

from dl_techniques.utils.denoiser_provenance import (
    UNIT_DOMAIN_STAMP,
    require_unit_domain_checkpoint,
    resolve_config_path,
)


def _ckpt_dir(tmp_path, config_text: str | None):
    """A checkpoint directory: a dummy best_model.keras + an optional config.json."""
    (tmp_path / "best_model.keras").write_bytes(b"not-a-real-model")
    if config_text is not None:
        (tmp_path / "config.json").write_text(config_text)
    return tmp_path


class TestFailsClosed:
    """Every degenerate config.json must REFUSE, never fall through."""

    def test_absent_data_range_key_is_refused(self, tmp_path):
        # EVERY pre-migration checkpoint looks exactly like this.
        d = _ckpt_dir(tmp_path, json.dumps({"variant": "base", "patch_size": 256}))
        with pytest.raises(ValueError, match=r"REFUSING"):
            require_unit_domain_checkpoint(d)

    def test_legacy_data_range_value_is_refused(self, tmp_path):
        d = _ckpt_dir(tmp_path, json.dumps({"data_range": "[-0.5,0.5]"}))
        with pytest.raises(ValueError, match=r"data_range='\[-0\.5,0\.5\]'"):
            require_unit_domain_checkpoint(d)

    def test_missing_config_json_is_refused(self, tmp_path):
        d = _ckpt_dir(tmp_path, None)
        with pytest.raises(ValueError, match=r"missing or unreadable"):
            require_unit_domain_checkpoint(d)

    def test_malformed_json_is_refused(self, tmp_path):
        # A parse error must NOT silently pass the gate.
        d = _ckpt_dir(tmp_path, "{not: valid json,,,")
        with pytest.raises(ValueError, match=r"REFUSING"):
            require_unit_domain_checkpoint(d)

    def test_non_object_json_is_refused(self, tmp_path):
        # Valid JSON that is not a dict: `raw.get(...)` would raise AttributeError,
        # not the intended loud ValueError.
        d = _ckpt_dir(tmp_path, "[1, 2, 3]")
        with pytest.raises(ValueError, match=r"REFUSING"):
            require_unit_domain_checkpoint(d)

    def test_null_data_range_is_refused(self, tmp_path):
        d = _ckpt_dir(tmp_path, json.dumps({"data_range": None}))
        with pytest.raises(ValueError, match=r"REFUSING"):
            require_unit_domain_checkpoint(d)

    def test_refusal_message_names_checkpoint_legacy_domain_and_retrain(self, tmp_path):
        d = _ckpt_dir(tmp_path, json.dumps({"variant": "base"}))
        with pytest.raises(ValueError) as exc:
            require_unit_domain_checkpoint(d)
        msg = str(exc.value)
        assert str(tmp_path) in msg            # names the offending checkpoint
        assert "[-0.5,+0.5]" in msg            # names the legacy domain
        assert "RETRAIN" in msg.upper()        # states the remedy
        assert "no compatibility shim" in msg  # states there is no escape hatch


class TestAccepts:
    def test_unit_domain_stamp_passes(self, tmp_path):
        d = _ckpt_dir(tmp_path, json.dumps({"data_range": UNIT_DOMAIN_STAMP}))
        require_unit_domain_checkpoint(d)  # must not raise

    def test_accepts_a_keras_file_path_not_just_a_dir(self, tmp_path):
        d = _ckpt_dir(tmp_path, json.dumps({"data_range": UNIT_DOMAIN_STAMP}))
        require_unit_domain_checkpoint(d / "best_model.keras")  # must not raise


class TestPathResolution:
    def test_dir_resolves_to_sibling_config(self, tmp_path):
        assert resolve_config_path(tmp_path) == tmp_path / "config.json"

    def test_keras_file_resolves_to_sibling_config(self, tmp_path):
        p = tmp_path / "final_model.keras"
        assert resolve_config_path(p) == tmp_path / "config.json"
