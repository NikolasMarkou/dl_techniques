"""Tests for ``dit/config.py`` -- the variant registry and ``DiffusionConfig``.

The variant table is pinned by an INDEPENDENT transcription of the reference
file, typed out here from ``reference/models.py``'s ``DiT Configs`` section
rather than imported from the module under test. Importing the module's own
table and comparing it to itself would pass against any typo; the point of this
oracle is that a typo in ``config.py`` has to disagree with a second reading of
the source.
"""

import pytest

from dl_techniques.models.vision_language.dit.config import (
    DIT_VARIANTS,
    VARIANT_FIELDS,
    DiffusionConfig,
    get_variant_config,
    normalize_variant_name,
)
from dl_techniques.utils.ddpm_schedule import DDPMSchedule

# ---------------------------------------------------------------------
# Independent oracle -- transcribed from reference/models.py, NOT imported
# ---------------------------------------------------------------------
#
#   def DiT_XL_2(**kwargs): return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, ...)
#   ... through ...
#   def DiT_S_8(**kwargs):  return DiT(depth=12, hidden_size=384,  patch_size=8, num_heads=6,  ...)
#
# Tuples are (depth, hidden_size, patch_size, num_heads).
REFERENCE_VARIANTS = {
    "DiT-XL/2": (28, 1152, 2, 16),
    "DiT-XL/4": (28, 1152, 4, 16),
    "DiT-XL/8": (28, 1152, 8, 16),
    "DiT-L/2": (24, 1024, 2, 16),
    "DiT-L/4": (24, 1024, 4, 16),
    "DiT-L/8": (24, 1024, 8, 16),
    "DiT-B/2": (12, 768, 2, 12),
    "DiT-B/4": (12, 768, 4, 12),
    "DiT-B/8": (12, 768, 8, 12),
    "DiT-S/2": (12, 384, 2, 6),
    "DiT-S/4": (12, 384, 4, 6),
    "DiT-S/8": (12, 384, 8, 6),
}


class TestVariantRegistry:
    """The twelve rows, against a second reading of the reference file."""

    def test_the_registry_has_exactly_the_twelve_reference_names(self):
        assert set(DIT_VARIANTS) == set(REFERENCE_VARIANTS)
        assert len(DIT_VARIANTS) == 12

    @pytest.mark.parametrize("name", sorted(REFERENCE_VARIANTS))
    def test_each_row_equals_the_reference_transcription(self, name):
        row = DIT_VARIANTS[name]
        assert tuple(row[f] for f in VARIANT_FIELDS) == REFERENCE_VARIANTS[name], (
            f"{name}: config.py says "
            f"{tuple(row[f] for f in VARIANT_FIELDS)}, reference/models.py says "
            f"{REFERENCE_VARIANTS[name]}"
        )

    def test_the_row_schema_is_exactly_the_declared_fields(self):
        for name, row in DIT_VARIANTS.items():
            assert set(row) == set(VARIANT_FIELDS), name

    @pytest.mark.parametrize("name", sorted(REFERENCE_VARIANTS))
    def test_hidden_size_is_divisible_by_num_heads(self, name):
        row = DIT_VARIANTS[name]
        assert row["hidden_size"] % row["num_heads"] == 0

    def test_get_variant_config_returns_a_copy_not_the_registry_row(self):
        row = get_variant_config("DiT-S/2")
        row["depth"] = 999
        assert DIT_VARIANTS["DiT-S/2"]["depth"] == 12


class TestVariantNameNormalization:
    """One canonical key form, reached from every accepted spelling."""

    @pytest.mark.parametrize(
        "spelling",
        ["DiT-XL/2", "XL/2", "xl_2", "DIT-XL-2", "dit_xl_2", "  DiT-XL/2  "],
    )
    def test_accepted_spellings_normalize_to_the_canonical_key(self, spelling):
        assert normalize_variant_name(spelling) == "DiT-XL/2"

    @pytest.mark.parametrize("name", sorted(REFERENCE_VARIANTS))
    def test_every_canonical_key_is_its_own_fixed_point(self, name):
        assert normalize_variant_name(name) == name

    def test_an_unknown_variant_raises_naming_the_input(self):
        with pytest.raises(ValueError, match="DiT-XXL/2"):
            normalize_variant_name("DiT-XXL/2")

    def test_an_unparseable_name_raises(self):
        with pytest.raises(ValueError, match="scale"):
            normalize_variant_name("XL")

    def test_an_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            normalize_variant_name("   ")


class TestDiffusionConfigDefaults:
    """Defaults reproduce upstream ``DiT.__init__`` and ``create_diffusion``."""

    def test_the_defaults_match_upstream(self):
        cfg = DiffusionConfig()
        assert cfg.input_size == 32
        assert cfg.in_channels == 4
        assert cfg.num_classes == 1000
        assert cfg.class_dropout_rate == pytest.approx(0.1)
        assert cfg.learn_sigma is True
        assert cfg.mlp_ratio == pytest.approx(4.0)
        assert cfg.num_timesteps == 1000
        assert cfg.schedule_name == "linear"

    def test_learn_sigma_doubles_the_output_channels(self):
        assert DiffusionConfig(in_channels=4, learn_sigma=True).out_channels == 8
        assert DiffusionConfig(in_channels=4, learn_sigma=False).out_channels == 4

    def test_num_patches_is_the_squared_token_grid(self):
        cfg = DiffusionConfig()
        assert cfg.num_patches(2) == 256
        assert cfg.num_patches(4) == 64
        assert cfg.num_patches(8) == 16

    def test_the_config_is_frozen(self):
        cfg = DiffusionConfig()
        with pytest.raises(Exception):
            cfg.input_size = 64  # type: ignore[misc]

    def test_build_schedule_returns_the_tables_of_the_named_schedule(self):
        cfg = DiffusionConfig(num_timesteps=100, schedule_name="squaredcos_cap_v2")
        schedule = cfg.build_schedule()
        assert isinstance(schedule, DDPMSchedule)
        assert schedule.num_timesteps == 100


class TestDiffusionConfigValidation:
    """One arm per ``__post_init__`` invariant; each asserts the field is named."""

    @pytest.mark.parametrize(
        "field",
        ["input_size", "in_channels", "num_classes", "num_timesteps"],
    )
    def test_a_non_positive_dimension_raises_naming_the_field(self, field):
        with pytest.raises(ValueError, match=field):
            DiffusionConfig(**{field: 0})

    @pytest.mark.parametrize(
        "field",
        ["input_size", "in_channels", "num_classes", "num_timesteps"],
    )
    def test_a_negative_dimension_raises_naming_the_field(self, field):
        with pytest.raises(ValueError, match=field):
            DiffusionConfig(**{field: -3})

    @pytest.mark.parametrize("rate", [-0.1, 1.0, 1.5])
    def test_an_out_of_range_dropout_rate_raises_naming_the_field(self, rate):
        with pytest.raises(ValueError, match="class_dropout_rate"):
            DiffusionConfig(class_dropout_rate=rate)

    @pytest.mark.parametrize("rate", [0.0, 0.1, 0.999])
    def test_an_in_range_dropout_rate_is_accepted(self, rate):
        assert DiffusionConfig(class_dropout_rate=rate).class_dropout_rate == rate

    def test_a_non_positive_mlp_ratio_raises_naming_the_field(self):
        with pytest.raises(ValueError, match="mlp_ratio"):
            DiffusionConfig(mlp_ratio=0.0)

    def test_an_unknown_schedule_name_raises_naming_the_field(self):
        with pytest.raises(ValueError, match="schedule_name"):
            DiffusionConfig(schedule_name="cosine")

    def test_a_non_positive_patch_size_raises_naming_the_field(self):
        with pytest.raises(ValueError, match="patch_size"):
            DiffusionConfig().validate_patch_size(0)

    def test_an_indivisible_patch_size_raises_naming_both_fields(self):
        with pytest.raises(ValueError, match="input_size"):
            DiffusionConfig(input_size=30).validate_patch_size(4)
        with pytest.raises(ValueError, match="patch_size"):
            DiffusionConfig(input_size=30).validate_patch_size(4)


class TestTheLinearScheduleFloorIsDelegatedNotHardcoded:
    """The ``'linear'`` short-chain rule, measured rather than assumed.

    ``beta_end = (1000 / T) * 0.02 = 20 / T``, so ``T < 20`` produces a
    ``beta_end`` above ``1.0`` and is rejected -- *except* at ``T == 1``, where
    ``np.linspace(start, stop, 1)`` returns ``[start]`` and drops the endpoint
    entirely, leaving a single legal ``beta = 0.1``. The measured accepted set is
    therefore ``{1} union [20, inf)``, which is not a floor at all.

    Two prose claims died here: the plan and ``ddpm_schedule.py``'s own docstring
    both said the boundary was ``50``, and "there is a floor" was itself wrong.
    That is why ``DiffusionConfig`` delegates to the schedule instead of encoding
    a threshold. See decisions.md D-010.
    """

    @pytest.mark.parametrize("num_timesteps", [2, 5, 10, 19])
    def test_a_rejected_linear_chain_raises_naming_both_fields(self, num_timesteps):
        with pytest.raises(ValueError, match="num_timesteps"):
            DiffusionConfig(num_timesteps=num_timesteps, schedule_name="linear")
        with pytest.raises(ValueError, match="schedule_name"):
            DiffusionConfig(num_timesteps=num_timesteps, schedule_name="linear")

    @pytest.mark.parametrize("num_timesteps", [20, 21, 50, 1000])
    def test_a_linear_chain_at_or_above_twenty_is_accepted(self, num_timesteps):
        cfg = DiffusionConfig(num_timesteps=num_timesteps, schedule_name="linear")
        assert cfg.num_timesteps == num_timesteps

    def test_the_measured_boundary_is_twenty_not_fifty(self):
        """Anti-vacuity: pin the boundary itself, on both sides."""
        with pytest.raises(ValueError):
            DiffusionConfig(num_timesteps=19, schedule_name="linear")
        assert (
            DiffusionConfig(num_timesteps=20, schedule_name="linear").num_timesteps
            == 20
        )

    def test_a_single_step_linear_chain_is_accepted_because_linspace_drops_the_endpoint(
        self,
    ):
        """The exception that makes 'the floor is N' unstatable.

        ``np.linspace(0.1, 20.0, 1)`` is ``[0.1]``: the illegal endpoint never
        appears, so ``T == 1`` is legal while ``T == 2`` is not.
        """
        cfg = DiffusionConfig(num_timesteps=1, schedule_name="linear")
        betas = cfg.build_schedule().betas
        assert betas.shape == (1,)
        assert betas[0] == pytest.approx(0.1)
        with pytest.raises(ValueError):
            DiffusionConfig(num_timesteps=2, schedule_name="linear")

    def test_the_measured_accepted_set_below_twenty_six_is_exactly_one_and_twenty_up(
        self,
    ):
        """An executable census, so a change to the schedule reddens here."""
        accepted = []
        for num_timesteps in range(1, 26):
            try:
                DiffusionConfig(num_timesteps=num_timesteps, schedule_name="linear")
            except ValueError:
                continue
            accepted.append(num_timesteps)
        assert accepted == [1] + list(range(20, 26))

    @pytest.mark.parametrize("num_timesteps", [1, 5, 10, 19, 20, 50])
    def test_the_cosine_schedule_accepts_every_one_of_those_short_chains(
        self, num_timesteps
    ):
        cfg = DiffusionConfig(
            num_timesteps=num_timesteps, schedule_name="squaredcos_cap_v2"
        )
        assert cfg.build_schedule().num_timesteps == num_timesteps


class TestTheRoundTripSurface:
    """There is no ``as_dict``/``from_dict`` on ``DiffusionConfig`` yet.

    The config is not a Keras object and is not serialized by anything at this
    step; ``get_config`` round-tripping arrives with ``dit/model.py`` (step 6),
    which will carry the config's fields as plain kwargs. Rather than invent a
    round-trip API with no consumer, this arm pins the ONE property a later
    round trip will rely on: the dataclass is reconstructible from its own
    fields, by value.
    """

    def test_the_config_reconstructs_by_value_from_its_own_fields(self):
        import dataclasses

        cfg = DiffusionConfig(
            input_size=16,
            in_channels=8,
            num_classes=10,
            class_dropout_rate=0.2,
            learn_sigma=False,
            mlp_ratio=2.0,
            num_timesteps=100,
            schedule_name="squaredcos_cap_v2",
        )
        rebuilt = DiffusionConfig(**dataclasses.asdict(cfg))
        assert rebuilt == cfg
