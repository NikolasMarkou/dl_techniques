"""Tests for ``SSPStrategyConfig``, ``SSPStrategy`` and ``ssp_builder``.

The strategy object exists to carry ONE set of defaults for all four operations, so
the guards here are mostly about the master switch and the config plumbing:

* ``enable=False`` must be a real no-op on every operation, not a partially-enabled
  one. It is the A/B control the principle is measured against, so a switch that
  merely softened the weighting would make every comparison meaningless.
* Unknown keys must RAISE. A silently dropped ``mgpo_lamda`` typo leaves the
  weighting at its default and reads as "SSP did not help".
* ``get_config`` must round-trip, and the returned dict must be exactly what
  ``from_config`` accepts.
"""

from __future__ import annotations

import numpy as np
import pytest

import keras

from dl_techniques.optimization import constants as opt_constants
from dl_techniques.optimization.ssp import (
    FUSION_MODES,
    SAMPLING_MODES,
    SSPStrategy,
    SSPStrategyConfig,
    SSPType,
    ssp_builder,
)
from dl_techniques.optimization.ssp.signal import (
    group_relative_advantages,
    max_entropy_weight,
    mgpo_advantages,
)


def _np(value) -> np.ndarray:
    return np.asarray(keras.ops.convert_to_numpy(value), dtype=np.float64)


# ---------------------------------------------------------------------
# the config object
# ---------------------------------------------------------------------


class TestSSPStrategyConfig:
    def test_defaults_are_off(self) -> None:
        """The repo's opt-in convention: a recipe can carry the config
        unconditionally and flip one flag."""
        config = SSPStrategyConfig()
        assert config.enable is False
        assert config.pass_at_k == opt_constants.DEFAULT_SSP_PASS_AT_K
        assert config.estimator == opt_constants.DEFAULT_SSP_ESTIMATOR

    def test_every_default_matches_the_constants_module(self) -> None:
        """``constants.py`` is the single source of truth for ``DEFAULT_*``; a
        builder that re-declares them is the drift this repo's conventions forbid."""
        config = SSPStrategyConfig()
        assert config.fusion_mode == opt_constants.DEFAULT_SSP_FUSION_MODE
        assert config.fusion_weight_scheme == opt_constants.DEFAULT_SSP_FUSION_SCHEME
        assert config.fusion_temperature == opt_constants.DEFAULT_SSP_FUSION_TEMPERATURE
        assert config.fusion_coefficient == opt_constants.DEFAULT_SSP_FUSION_COEFFICIENT
        assert config.sampling_mode == opt_constants.DEFAULT_SSP_SAMPLING_MODE
        assert config.mgpo_lambda == opt_constants.DEFAULT_MGPO_LAMBDA
        assert config.mgpo_p0 == opt_constants.DEFAULT_MGPO_P0
        assert config.mgpo_eps == opt_constants.DEFAULT_MGPO_EPS
        assert config.mgpo_ddof == opt_constants.DEFAULT_MGPO_DDOF
        assert config.mgpo_normalize_weights == \
            opt_constants.DEFAULT_MGPO_NORMALIZE_WEIGHTS
        assert config.clip_eps == opt_constants.DEFAULT_MGPO_CLIP_EPS
        assert config.token_reduction == opt_constants.DEFAULT_SSP_TOKEN_REDUCTION
        assert config.zero_variance == opt_constants.DEFAULT_SSP_ZERO_VARIANCE
        assert config.enable == opt_constants.DEFAULT_SSP_ENABLED

    def test_the_k_ladder_is_powers_of_two_up_to_pass_at_k(self) -> None:
        assert SSPStrategyConfig(pass_at_k=8).ks == [1, 2, 4, 8]
        assert SSPStrategyConfig(pass_at_k=5).ks == [1, 2, 4]
        assert SSPStrategyConfig(pass_at_k=1).ks == [1]

    def test_config_round_trips(self) -> None:
        original = SSPStrategyConfig(
            enable=True, pass_at_k=16, estimator="plug_in",
            fusion_mode="task_arithmetic", fusion_weight_scheme="score_softmax",
            fusion_temperature=2.5, fusion_coefficient=0.3,
            sampling_mode="uniform", mgpo_lambda=3.0, mgpo_p0=0.3,
            mgpo_eps=1e-5, mgpo_ddof=1, mgpo_normalize_weights=True,
            clip_eps=0.1, token_reduction="per_token_mean",
            zero_variance="raise",
        )
        rebuilt = SSPStrategyConfig.from_config(original.get_config())
        assert rebuilt.get_config() == original.get_config()

    def test_get_config_is_exactly_what_from_config_accepts(self) -> None:
        config = SSPStrategyConfig()
        assert set(config.get_config()) == set(
            SSPStrategyConfig.__init__.__code__.co_varnames[
                1: SSPStrategyConfig.__init__.__code__.co_argcount
            ]
        )

    @pytest.mark.parametrize("value", ["nonsense", 3, None])
    def test_an_unknown_key_is_refused_loudly(self, value) -> None:
        """A silently dropped ``mgpo_lamda`` typo leaves the weighting at its default
        and reads as "SSP did not help"."""
        with pytest.raises(TypeError):
            SSPStrategyConfig.from_config({"mgpo_lamda": 2.0})

    def test_from_config_refuses_a_non_dict(self) -> None:
        with pytest.raises(TypeError, match="must be a dict"):
            SSPStrategyConfig.from_config(["pass_at_k", 8])

    @pytest.mark.parametrize("mode", ["linear", "task_arithmetic"])
    def test_every_declared_fusion_mode_is_accepted(self, mode: str) -> None:
        assert mode in FUSION_MODES
        assert SSPStrategyConfig(fusion_mode=mode).fusion_mode == mode

    @pytest.mark.parametrize("mode", SAMPLING_MODES)
    def test_every_declared_sampling_mode_is_accepted(self, mode: str) -> None:
        assert SSPStrategyConfig(sampling_mode=mode).sampling_mode == mode

    @pytest.mark.parametrize("field,value", [
        ("fusion_mode", "slerp"),
        ("sampling_mode", "argmax"),
        ("token_reduction", "sum"),
        ("zero_variance", "maybe"),
        ("estimator", "biased"),
        ("fusion_weight_scheme", "inverse"),
    ])
    def test_an_unknown_enum_member_is_refused(self, field: str, value) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            SSPStrategyConfig(**{field: value})

    @pytest.mark.parametrize("field,value", [
        ("mgpo_lambda", -1.0),
        ("mgpo_eps", 0.0),
        ("clip_eps", 0.0),
        ("fusion_temperature", -1.0),
        ("mgpo_ddof", -1),
        ("pass_at_k", 0),
    ])
    def test_out_of_domain_numbers_are_refused_at_construction(
            self, field: str, value) -> None:
        """Validation happens when the recipe is PARSED, not three stages into a run."""
        with pytest.raises(ValueError):
            SSPStrategyConfig(**{field: value})

    @pytest.mark.parametrize("p0", [0.0, 1.0, -0.5, 1.5])
    def test_a_degenerate_p0_is_refused(self, p0: float) -> None:
        with pytest.raises(ValueError, match="strictly inside"):
            SSPStrategyConfig(mgpo_p0=p0)

    def test_a_bool_ddof_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be an integer"):
            SSPStrategyConfig(mgpo_ddof=True)

    def test_a_non_string_enum_member_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be a string"):
            SSPStrategyConfig(fusion_mode=1)


# ---------------------------------------------------------------------
# the builder
# ---------------------------------------------------------------------


class TestSSPBuilder:
    def test_builds_from_the_house_config_shape(self) -> None:
        strategy = ssp_builder({"type": "ssp_v1", "config": {"enable": True}})
        assert isinstance(strategy, SSPStrategy)
        assert strategy.config.enable is True

    def test_the_config_key_may_be_omitted(self) -> None:
        """The no-arg form is a strict no-op, not a partially-enabled one."""
        strategy = ssp_builder({"type": "ssp_v1"})
        assert strategy.config.enable is False

    def test_type_is_case_and_whitespace_tolerant(self) -> None:
        assert ssp_builder({"type": "  SSP_V1  "}).config.enable is False

    def test_a_non_dict_config_is_refused(self) -> None:
        with pytest.raises(TypeError, match="must be a dictionary"):
            ssp_builder(["ssp_v1"])

    def test_a_missing_type_is_refused(self) -> None:
        with pytest.raises(ValueError, match="type cannot be None"):
            ssp_builder({"config": {}})

    def test_a_non_string_type_is_refused(self) -> None:
        with pytest.raises(TypeError, match="must be a string"):
            ssp_builder({"type": 1})

    def test_an_unknown_type_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown SSP type"):
            ssp_builder({"type": "ssp_v2"})

    def test_a_non_dict_inner_config_is_refused(self) -> None:
        with pytest.raises(TypeError, match="must be a dictionary"):
            ssp_builder({"type": "ssp_v1", "config": ["enable"]})

    def test_the_declared_type_is_reachable(self) -> None:
        assert [member.value for member in SSPType] == ["ssp_v1"]

    def test_strategy_exposes_its_config(self) -> None:
        strategy = ssp_builder({"type": "ssp_v1", "config": {"pass_at_k": 4}})
        assert strategy.get_config()["pass_at_k"] == 4

    def test_the_strategy_refuses_a_raw_dict(self) -> None:
        with pytest.raises(TypeError, match="must be an SSPStrategyConfig"):
            SSPStrategy({"enable": True})


# ---------------------------------------------------------------------
# the master switch
# ---------------------------------------------------------------------


class TestTheMasterSwitchIsARealNoOp:
    REWARDS = np.array(
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype="float32"
    )
    COVERAGE = np.array([0.0, 0.5, 1.0])

    @property
    def _on(self) -> SSPStrategy:
        return ssp_builder({"type": "ssp_v1", "config": {
            "enable": True, "pass_at_k": 4, "mgpo_lambda": 3.0,
            "fusion_weight_scheme": "score_softmax",
        }})

    @property
    def _off(self) -> SSPStrategy:
        return ssp_builder({"type": "ssp_v1", "config": {"pass_at_k": 4}})

    def test_weights_are_all_ones_when_off(self) -> None:
        assert _np(self._off.weights(np.array([0.0, 0.5, 1.0]))).tolist() == [1.0, 1.0, 1.0]

    def test_weights_are_real_when_on(self) -> None:
        weights = _np(self._on.weights(np.array([0.0, 0.5, 1.0])))
        assert weights[1] == pytest.approx(1.0)
        assert weights[0] < 1.0 and weights[2] < 1.0

    def test_advantages_fall_back_to_plain_group_relative(self) -> None:
        """The A/B control: the disabled path IS the plain group-relative
        advantage."""
        plain = _np(group_relative_advantages(self.REWARDS))
        assert _np(self._off.advantages(self.REWARDS)) == pytest.approx(
            plain, abs=1e-6
        )

    def test_an_enabled_strategy_with_lambda_zero_also_equals_plain(self) -> None:
        """``lambda = 0`` must make the weighting vanish on the ENABLED path too --
        the paper's own degeneration claim."""
        zero_lambda = ssp_builder({"type": "ssp_v1", "config": {
            "enable": True, "mgpo_lambda": 0.0,
        }})
        assert _np(zero_lambda.advantages(self.REWARDS)) == pytest.approx(
            _np(group_relative_advantages(self.REWARDS)), abs=1e-6
        )

    def test_the_enabled_path_is_not_vacuous(self) -> None:
        """The control above would pass if the weighting did nothing at all. With
        ``lambda = 3`` and a group at ``p_c = 1/3`` the advantage must move."""
        assert not np.allclose(
            _np(self._on.advantages(self.REWARDS)),
            _np(group_relative_advantages(self.REWARDS)),
            atol=1e-6,
        )

    def test_sampling_weights_are_uniform_when_off(self) -> None:
        assert _np(self._off.sampling_weights(self.COVERAGE)) == pytest.approx(
            np.full(3, 1.0 / 3.0)
        )

    def test_fusion_weights_are_uniform_when_off(self) -> None:
        scores = [0.1, 0.5, 0.9]
        assert _np(self._off.fusion_weights(scores)) == pytest.approx(
            np.full(3, 1.0 / 3.0)
        )

    def test_fusion_weights_follow_their_own_scheme_when_on(self) -> None:
        weights = _np(self._on.fusion_weights([0.0, 1.0, 2.0]))
        assert weights.sum() == pytest.approx(1.0)
        assert weights[2] > weights[0]

    def test_the_off_path_ignores_the_score_softmax_scheme(self) -> None:
        """A switch that merely softened the weighting would make every A/B
        comparison meaningless, so the OFF path must not consult the scheme at all."""
        strategy = ssp_builder({"type": "ssp_v1", "config": {
            "enable": False, "fusion_weight_scheme": "score_softmax",
        }})
        assert _np(strategy.fusion_weights([0.0, 1.0, 2.0])) == pytest.approx(
            np.full(3, 1.0 / 3.0)
        )


# ---------------------------------------------------------------------
# the bound operations
# ---------------------------------------------------------------------


class TestTheBoundOperations:
    def test_profile_uses_the_configs_k_ladder(self) -> None:
        strategy = ssp_builder({"type": "ssp_v1", "config": {"pass_at_k": 4}})
        outcomes = np.array([[1, 0, 1, 0, 0, 0, 0, 0]], dtype=float)
        profile = strategy.profile(outcomes)
        assert max(profile["pass_at_k_curve"]) == 4

    def test_profile_clips_the_ladder_to_the_sample_count(self) -> None:
        """A ladder wider than the pool would raise; the strategy trims it."""
        strategy = ssp_builder({"type": "ssp_v1", "config": {"pass_at_k": 64}})
        outcomes = np.array([[1, 0, 1, 0]], dtype=float)
        profile = strategy.profile(outcomes)
        assert max(profile["pass_at_k_curve"]) == 4

    def test_profile_honours_an_explicit_ladder(self) -> None:
        strategy = ssp_builder({"type": "ssp_v1", "config": {"pass_at_k": 4}})
        outcomes = np.array([[1, 0, 1, 0]], dtype=float)
        assert max(strategy.profile(outcomes, ks=[1, 2])["pass_at_k_curve"]) == 2

    def test_select_delegates_to_the_module(self) -> None:
        strategy = ssp_builder({"type": "ssp_v1", "config": {"enable": True}})
        selection = strategy.select(np.array([[0.1, 0.9], [0.8, 0.2]]))
        assert selection["checkpoint_index"].tolist() == [1, 0]

    def test_fuse_delegates_and_returns_new_arrays(self) -> None:
        strategy = ssp_builder({"type": "ssp_v1", "config": {"enable": True}})
        first = [np.array([0.0], dtype="float32")]
        second = [np.array([4.0], dtype="float32")]
        fused = strategy.fuse([first, second])
        assert fused[0] == pytest.approx(np.array([2.0], dtype="float32"))
        assert first[0][0] == pytest.approx(0.0)

    def test_fuse_passes_the_task_arithmetic_mode_through(self) -> None:
        strategy = ssp_builder({"type": "ssp_v1", "config": {
            "enable": True, "fusion_mode": "task_arithmetic",
        }})
        with pytest.raises(ValueError, match="needs `base`"):
            strategy.fuse([[np.zeros(1, dtype="float32")]])

    def test_soup_delegates(self) -> None:
        strategy = ssp_builder({"type": "ssp_v1", "config": {"enable": True}})
        pool = [[np.array([0.0], dtype="float32")],
                [np.array([4.0], dtype="float32")]]
        indices, weights = strategy.soup(pool, score_fn=lambda w: -abs(float(w[0][0]) - 2.0))
        assert set(indices) == {0, 1}
        assert weights == pytest.approx([0.5, 0.5])

    def test_advantages_delegate(self) -> None:
        strategy = ssp_builder({"type": "ssp_v1", "config": {
            "enable": True, "mgpo_lambda": 2.0,
        }})
        rewards = np.array([[1.0, 0.0, 0.0]], dtype="float32")
        assert _np(strategy.advantages(rewards)) == pytest.approx(
            _np(mgpo_advantages(rewards, lam=2.0)), abs=1e-6
        )


# ---------------------------------------------------------------------
# RED proofs
# ---------------------------------------------------------------------


class TestTheGuardsActuallyGoRed:
    def test_the_off_switch_guard_sees_a_softened_weight(self) -> None:
        """If ``enable=False`` merely reduced ``lambda`` instead of forcing 1.0,
        every A/B comparison the principle is measured by would be meaningless."""
        off = ssp_builder({"type": "ssp_v1", "config": {"enable": False}})
        softened = _np(off.weights(np.array([0.0, 1.0])))
        real = _np(max_entropy_weight(np.array([0.0, 1.0]), lam=1.0))
        assert softened.tolist() == [1.0, 1.0]
        assert real[0] < 1.0, "the control must be a genuine down-weight"

    def test_the_unknown_key_guard_sees_a_dropped_typo(self) -> None:
        """Without the strict check, ``mgpo_lamda`` would be ignored and the run
        would look like "SSP did not help" rather than "the config was wrong"."""
        with pytest.raises(TypeError, match="lamda"):
            SSPStrategyConfig.from_config({"mgpo_lamda": 2.0})

    def test_the_constants_guard_sees_a_re_declared_default(self) -> None:
        """If a builder re-declared its own default, changing ``constants.py`` would
        silently stop reaching it."""
        assert SSPStrategyConfig().mgpo_lambda == \
            opt_constants.DEFAULT_MGPO_LAMBDA
        assert opt_constants.DEFAULT_MGPO_LAMBDA != 2.0, (
            "the fixture default must differ from the value this guard perturbs"
        )
