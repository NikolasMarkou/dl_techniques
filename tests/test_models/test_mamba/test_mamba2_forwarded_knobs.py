"""F-48: six `Mamba2Layer` knobs were unreachable from any assembled model.

``ngroups``, ``dt_min``, ``dt_max``, ``dt_init_floor``, ``bias`` and
``conv_bias`` are declared, stored and USED by ``Mamba2Layer`` -- but
``Mamba2ResidualBlock.__init__`` declared none of them, and ``Mamba2`` builds
its entire stack out of those blocks. This is the "unreachable from the
assembled model" flavour of the dead-knob family rather than the
"declared-and-dropped" one: nothing was silently ignored, there was simply no
way to reach ``ngroups != 1`` (which is why the multi-group state path had no
caller), to disable ``conv_bias``, or to widen the Δ init range from any real
Mamba-2 model. ``norm_before_gate`` was plumbed through this same chain in a
2026-08-15 fix; its six siblings were not.

Every test below has two halves: the value ARRIVES at the wrapped
``Mamba2Layer``, and it CHANGES something measurable there. The first half alone
would pass against a block that stored the value and forwarded a copy of its own
default. See decisions.md plan-2026-08-18T140459-7991552f/D-036.
"""

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import keras
import numpy as np
import pytest

from dl_techniques.models.mamba.components_v2 import (
    Mamba2Layer,
    Mamba2ResidualBlock,
)
from dl_techniques.models.mamba.mamba_v2 import Mamba2


_FORWARDED = (
    "ngroups", "dt_min", "dt_max", "dt_init_floor", "bias", "conv_bias",
)

_BLOCK_KWARGS = dict(
    d_model=16, d_state=8, d_conv=4, expand=2, headdim=8, d_ssm=32,
)

_MODEL_KWARGS = dict(vocab_size=32, d_model=16, num_layers=2, d_state=8,
                     headdim=8)


def _defaults(fn):
    return {
        name: p.default
        for name, p in inspect.signature(fn).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


class TestTheDefaultsAreTheLayersDefaults:
    """A pass-through whose defaults drift from the layer's is a silent
    behaviour change for every existing caller, so this is pinned rather than
    trusted."""

    @pytest.mark.parametrize("owner", [Mamba2ResidualBlock, Mamba2])
    def test_every_forwarded_default_matches_mamba2layer(self, owner):
        layer_defaults = _defaults(Mamba2Layer.__init__)
        owner_defaults = _defaults(owner.__init__)
        for name in _FORWARDED:
            assert name in owner_defaults, (
                f"{owner.__name__} does not expose `{name}`, so no assembled "
                f"model can reach it (F-48)"
            )
            assert owner_defaults[name] == layer_defaults[name], name


class TestTheBlockForwardsToItsLayer:

    def test_ngroups_arrives_and_widens_the_projections(self):
        a = Mamba2ResidualBlock(**_BLOCK_KWARGS, ngroups=1)
        b = Mamba2ResidualBlock(**_BLOCK_KWARGS, ngroups=2)
        assert b.mamba2.ngroups == 2
        assert b.mamba2.in_proj.units > a.mamba2.in_proj.units, (
            "ngroups reached the layer but did not change the input "
            "projection width; it is being stored and not used"
        )
        assert b.mamba2.conv1d.filters > a.mamba2.conv1d.filters

    def test_conv_bias_arrives_and_removes_the_conv_bias(self):
        on = Mamba2ResidualBlock(**_BLOCK_KWARGS, conv_bias=True)
        off = Mamba2ResidualBlock(**_BLOCK_KWARGS, conv_bias=False)
        assert off.mamba2.conv_bias is False
        assert on.mamba2.conv1d.use_bias is True
        assert off.mamba2.conv1d.use_bias is False

    def test_bias_arrives_and_reaches_both_dense_projections(self):
        off = Mamba2ResidualBlock(**_BLOCK_KWARGS, bias=False)
        on = Mamba2ResidualBlock(**_BLOCK_KWARGS, bias=True)
        assert on.mamba2.in_proj.use_bias is True
        assert on.mamba2.out_proj.use_bias is True
        assert off.mamba2.in_proj.use_bias is False
        assert off.mamba2.out_proj.use_bias is False

    @pytest.mark.parametrize("name,value", [
        ("dt_min", 0.01), ("dt_max", 0.5), ("dt_init_floor", 1e-2),
    ])
    def test_the_dt_knobs_arrive_and_move_the_dt_bias_init(self, name, value):
        keras.utils.set_random_seed(0)
        base = Mamba2ResidualBlock(**_BLOCK_KWARGS)
        base.build((None, 6, 16))
        keras.utils.set_random_seed(0)
        changed = Mamba2ResidualBlock(**_BLOCK_KWARGS, **{name: value})
        changed.build((None, 6, 16))

        assert getattr(changed.mamba2, name) == value
        delta = float(np.max(np.abs(
            keras.ops.convert_to_numpy(base.mamba2.dt_bias)
            - keras.ops.convert_to_numpy(changed.mamba2.dt_bias)
        )))
        assert delta > 0.0, (
            f"`{name}` reached the layer but the dt_bias initialisation is "
            f"identical (max |delta| = {delta}); the value is stored, not used"
        )

    def test_the_block_serializes_all_six(self):
        block = Mamba2ResidualBlock(
            **_BLOCK_KWARGS, ngroups=2, dt_min=0.01, dt_max=0.5,
            dt_init_floor=1e-2, bias=True, conv_bias=False,
        )
        config = block.get_config()
        for name in _FORWARDED:
            assert name in config, name
        restored = Mamba2ResidualBlock.from_config(config)
        for name in _FORWARDED:
            assert getattr(restored.mamba2, name) == getattr(
                block.mamba2, name
            ), name


class TestTheModelForwardsToEveryBlock:

    def test_every_block_in_the_stack_receives_them(self):
        model = Mamba2(
            **_MODEL_KWARGS, ngroups=2, dt_min=0.01, dt_max=0.5,
            dt_init_floor=1e-2, bias=True, conv_bias=False,
        )
        expected = dict(ngroups=2, dt_min=0.01, dt_max=0.5,
                        dt_init_floor=1e-2, bias=True, conv_bias=False)
        assert len(model.encoder_layers) == _MODEL_KWARGS["num_layers"]
        for block in model.encoder_layers:
            for name, value in expected.items():
                assert getattr(block.mamba2, name) == value, (block.name, name)

    def test_ngroups_changes_the_assembled_parameter_count(self):
        """The end-to-end arm: reaching the knob from the top-level model
        actually builds a different network."""
        keras.utils.set_random_seed(0)
        one = Mamba2(**_MODEL_KWARGS, ngroups=1)
        one(np.zeros((1, 6), dtype="int32"))
        keras.utils.set_random_seed(0)
        two = Mamba2(**_MODEL_KWARGS, ngroups=2)
        two(np.zeros((1, 6), dtype="int32"))
        assert two.count_params() > one.count_params()

    def test_the_model_serializes_all_six(self):
        model = Mamba2(**_MODEL_KWARGS, ngroups=2, conv_bias=False)
        config = model.get_config()
        for name in _FORWARDED:
            assert name in config, name
        restored = Mamba2.from_config(config)
        assert restored.encoder_layers[0].mamba2.ngroups == 2
        assert restored.encoder_layers[0].mamba2.conv_bias is False

    def test_a_pre_change_config_without_them_still_loads(self):
        """Back-compat: a `.keras` saved before this change carries none of the
        six keys, and must restore at the (unchanged) defaults."""
        config = Mamba2(**_MODEL_KWARGS).get_config()
        for name in _FORWARDED:
            config.pop(name)
        restored = Mamba2.from_config(config)
        layer_defaults = _defaults(Mamba2Layer.__init__)
        for name in _FORWARDED:
            assert getattr(restored.encoder_layers[0].mamba2, name) == (
                layer_defaults[name]
            ), name
