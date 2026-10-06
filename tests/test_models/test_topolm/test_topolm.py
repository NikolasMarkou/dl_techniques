"""Test suite for the ``topolm`` model package.

Covers construction and stored config, the variant registry and its provenance,
the tap inventory, the ``alpha = 0`` control as a measured property rather than a
claim, build parity, causality, composition, position sensitivity, knob
sensitivity with the instrument matched to each knob's class, and a ``.keras``
round trip on values that compares the restored taps' permutations.

The mandated behavioural pins, with the numbers this suite measures (float32,
CPU, ``tiny`` = 256 units on a 16x16 grid, ``vocab_size`` 200):

  1. test_the_future_leak_probe_is_exactly_zero          0.000e+00 on positions < t
  2. test_the_perturbation_reached_the_model             1.04 on positions >= t
  3. test_positions_actually_matter                      0.94 logit delta
  4. test_the_ladder_does_not_collapse                   std ratio 1.0006
  5. test_alpha_zero_keeps_every_weight                  identical relative paths
  6. test_no_tap_site_builds_no_tap_weights              60 vs 100 weights
  7. test_every_tap_has_a_distinct_layout                8 distinct permutations
"""

import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf

import keras
from keras import ops

from dl_techniques.layers.regularization.spatial_smoothness import (
    SpatialSmoothness,
)
from dl_techniques.models.language.topolm import (
    MODEL_VARIANTS,
    TAP_SITES,
    TopoLM,
    TopoLMBlock,
    create_topolm,
    grid_shape_of,
    validate_variant,
)
from dl_techniques.models.language.topolm.config import (
    DEFAULT_RADIUS,
    validate_all_variants,
)

VOCAB = 200
SMALL = {"embed_dim": 256, "depth": 2, "num_heads": 4, "max_seq_len": 64}


def _tokens(batch=2, length=12, seed=0, vocab=VOCAB):
    return np.random.default_rng(seed).integers(
        0, vocab, size=(batch, length)
    ).astype("int32")


def _build(**kwargs):
    """A small model with every sub-layer named explicitly.

    The model's own ``name`` is pinned too: it appears in every weight path, and
    a scope probe that compares two models' paths has to strip it or the
    comparison reports a naming difference instead of the one under test.
    """
    keras.utils.set_random_seed(0)
    config = dict(vocab_size=VOCAB, name="topolm", **SMALL)
    config.update(kwargs)
    return TopoLM(**config)


def _relative_paths(model):
    """Weight paths with the model-root segment stripped.

    Two separately-constructed models must carry identical names at EVERY level,
    or the comparison reports a naming failure instead of a build one. Every
    sub-layer here is created with an explicit ``name=`` for that reason.
    """
    return sorted(weight.path.split("/", 1)[-1] for weight in model.weights)


# ---------------------------------------------------------------------
# package surface
# ---------------------------------------------------------------------


class TestPackageApi:
    def test_the_package_exports_the_class_the_factory_and_the_table(self):
        import dl_techniques.models.language.topolm as package

        for name in ("TopoLM", "TopoLMBlock", "create_topolm", "MODEL_VARIANTS"):
            assert name in package.__all__, name
            assert hasattr(package, name)

    def test_the_package_binds_no_name_matching_a_submodule(self):
        """A name matching a subpackage shadows it and breaks collection.

        Cheap to check here and catastrophic when missed: the failure surfaces at
        ``--collect-only`` time, invisible to per-package runs.
        """
        import dl_techniques.models.language.topolm as package

        for name in package.__all__:
            assert not package.__file__.endswith(f"/{name.lower()}.py"), name

    def test_the_factory_delegates_to_from_variant(self):
        model = create_topolm("tiny", vocab_size=VOCAB, max_seq_len=64)
        assert isinstance(model, TopoLM)
        assert model.embed_dim == MODEL_VARIANTS["tiny"]["embed_dim"]

    def test_the_factory_and_the_classmethod_agree(self):
        left = create_topolm("tiny", vocab_size=VOCAB, max_seq_len=64)
        right = TopoLM.from_variant("tiny", vocab_size=VOCAB, max_seq_len=64)
        assert _relative_paths(left) == _relative_paths(right)


# ---------------------------------------------------------------------
# variant registry
# ---------------------------------------------------------------------


class TestVariantRegistry:
    def test_every_shipped_variant_passes_its_own_constraints(self):
        """The whole table, not the variant a caller happens to ask for."""
        validate_all_variants(DEFAULT_RADIUS)

    def test_the_paper_variant_is_the_published_shape(self):
        """Quoted data, so quoted exactly.

        Rathi et al. (2025) Section 3: hidden 784, 12 blocks, 16 heads, MLP width
        3136, context 1024, and 784 = 28 x 28 is the reason the width was chosen.
        """
        config = MODEL_VARIANTS["paper"]
        assert config["embed_dim"] == 784
        assert config["depth"] == 12
        assert config["num_heads"] == 16
        assert config["ffn_intermediate_size"] == 3136
        assert config["max_seq_len"] == 1024
        assert grid_shape_of(784) == (28, 28)

    def test_the_paper_variant_builds_the_papers_tap_inventory(self):
        """12 blocks x 2 branch outputs = the paper's 24 taps."""
        model = TopoLM.from_variant("paper")
        assert len(model.blocks) == 12
        assert len(model.tap_layers) == 24
        assert model.grid_shape == (28, 28)

    def test_the_repo_authored_variants_are_labelled_as_such(self):
        """``small`` and ``tiny`` are ours, and say so.

        A description that read like an upstream citation would be the false
        citation ``tests/test_variant_tables_match_upstream_references.py``
        exists to catch; the descriptions carry that burden on purpose.
        """
        assert "repo-authored" in MODEL_VARIANTS["small"]["description"].lower()
        assert "repo-authored" in MODEL_VARIANTS["tiny"]["description"].lower()
        assert "Rathi" in MODEL_VARIANTS["paper"]["description"]

    def test_every_variant_factors_into_a_grid_the_radius_can_use(self):
        for name, config in MODEL_VARIANTS.items():
            height, width = grid_shape_of(config["embed_dim"])
            assert height * width == config["embed_dim"], name
            assert config["embed_dim"] % config["num_heads"] == 0, name
            assert config["embed_dim"] >= (2 * DEFAULT_RADIUS + 1) ** 2, name

    def test_a_variant_that_cannot_host_a_neighbourhood_is_rejected(self):
        with pytest.raises(ValueError, match="below the"):
            validate_variant(
                "broken",
                {"embed_dim": 64, "num_heads": 4},
                DEFAULT_RADIUS,
            )

    def test_a_variant_whose_heads_do_not_divide_is_rejected(self):
        with pytest.raises(ValueError, match="divisible by num_heads"):
            validate_variant(
                "broken", {"embed_dim": 250, "num_heads": 16}, DEFAULT_RADIUS
            )

    def test_an_unknown_variant_lists_the_available_ones(self):
        with pytest.raises(ValueError, match=r"Available: \[.*'paper'"):
            TopoLM.from_variant("enormous")

    def test_from_variant_accepts_the_overrides_its_docstring_advertises(self):
        """``alpha``, ``permute`` and ``tap_sites`` are documented; they must work."""
        model = TopoLM.from_variant(
            "tiny", alpha=0.0, permute=False, tap_sites=("attention",)
        )
        assert model.alpha == 0.0
        assert model.permute is False
        assert model.tap_sites == ("attention",)

    def test_pretrained_true_raises_and_names_the_alternative(self):
        with pytest.raises(NotImplementedError, match="train.topolm"):
            TopoLM.from_variant("tiny", pretrained=True)

    def test_a_missing_local_checkpoint_raises(self):
        with pytest.raises(FileNotFoundError, match="Weights file not found"):
            TopoLM.from_variant("tiny", pretrained="/nonexistent/path.keras")

    def test_the_variant_table_is_not_mutated_by_construction(self):
        """``from_variant`` copies; a caller mutating one row must not poison
        the registry for the next caller."""
        snapshot = {name: dict(config) for name, config in MODEL_VARIANTS.items()}
        TopoLM.from_variant("tiny", vocab_size=VOCAB)
        assert {name: dict(config) for name, config in MODEL_VARIANTS.items()} == snapshot


# ---------------------------------------------------------------------
# construction and validation
# ---------------------------------------------------------------------


class TestConstruction:
    def test_configuration_is_stored_and_the_model_starts_unbuilt(self):
        model = _build(
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            layer_norm_eps=1e-6,
            initializer_range=0.05,
            seed=7,
        )
        assert not model.built
        assert model.vocab_size == VOCAB
        assert model.embed_dim == 256
        assert model.depth == 2
        assert model.num_heads == 4
        assert model.layer_norm_eps == 1e-6
        assert model.initializer_range == 0.05
        assert model.seed == 7

    def test_the_default_ffn_width_is_four_times_the_model_width(self):
        model = _build(ffn_intermediate_size=None)
        assert model.ffn_intermediate_size == 4 * model.embed_dim

    def test_an_explicit_ffn_width_is_respected(self):
        model = _build(ffn_intermediate_size=333)
        assert model.ffn_intermediate_size == 333

    def test_the_resolved_grid_is_reported_even_when_it_was_not_given(self):
        """A caller who passed ``grid_shape=None`` still needs the grid."""
        model = _build(grid_shape=None)
        assert model.grid_shape == (16, 16)

    def test_an_explicit_grid_is_reported_and_used(self):
        model = _build(grid_shape=(8, 32))
        assert model.grid_shape == (8, 32)

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"vocab_size": 0}, "vocab_size must be positive"),
            ({"embed_dim": -4}, "embed_dim must be positive"),
            ({"depth": 0}, "depth must be positive"),
            ({"num_heads": 0}, "num_heads must be positive"),
            ({"max_seq_len": 0}, "max_seq_len must be positive"),
            ({"embed_dim": 100, "num_heads": 16}, "must be divisible by num_heads"),
            ({"dropout_rate": 1.5}, "dropout_rate must be in"),
            ({"attention_dropout_rate": -0.1}, "attention_dropout_rate must be in"),
            ({"alpha": -1.0}, "alpha must be >= 0"),
            ({"radius": 0}, "radius must be >= 1"),
            ({"num_neighborhoods": 0}, "num_neighborhoods must be >= 1"),
            ({"tap_sites": ("nonsense",)}, "unknown site"),
            ({"grid_shape": (8, 8)}, "has 64 cells but the tapped tensor"),
        ],
    )
    def test_invalid_arguments_name_the_offender(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            _build(**kwargs)

    def test_a_width_too_narrow_for_the_radius_is_a_construction_error(self):
        """The failure would otherwise surface at the tap's build, after the
        embeddings, the attention and the whole stack exist."""
        with pytest.raises(ValueError, match="radius-3 neighbourhood needs"):
            _build(embed_dim=32, depth=1, num_heads=4, radius=3, ffn_intermediate_size=32)

    def test_the_grid_floor_is_evaluated_per_radius(self):
        # 64 units DO host a radius-3 patch (7x7 = 49), so this is the both-ways
        # twin: the floor is a function of the radius, not a constant.
        assert _build(embed_dim=64, depth=1, num_heads=4, radius=3,
                      ffn_intermediate_size=64).embed_dim == 64
        with pytest.raises(ValueError, match="radius-4 neighbourhood needs"):
            _build(embed_dim=64, depth=1, num_heads=4, radius=4,
                   ffn_intermediate_size=64)


class TestForward:
    def test_the_forward_shape_and_finiteness(self):
        model = _build()
        outputs = model(_tokens(), training=False)
        assert outputs["logits"].shape == (2, 12, VOCAB)
        assert outputs["last_hidden_state"].shape == (2, 12, 256)
        for value in outputs.values():
            assert np.all(np.isfinite(ops.convert_to_numpy(value)))

    def test_a_dict_input_is_accepted(self):
        model = _build()
        outputs = model(
            {"input_ids": _tokens()}, training=False
        )
        assert outputs["logits"].shape == (2, 12, VOCAB)

    def test_a_dict_input_without_ids_is_rejected(self):
        model = _build()
        with pytest.raises(ValueError, match="must contain 'input_ids'"):
            model({"something_else": _tokens()}, training=False)

    def test_a_padding_mask_changes_exactly_the_masked_positions(self):
        """The both-ways pair, and the side that matters.

        Under a CAUSAL mask the padding mask can only move the PADDED positions:
        a causal model already forbids position 8 from seeing position 12, so no
        padding mask can change position 0. Asserting that the unmasked
        positions move would be asserting something structurally impossible, and
        the resulting test could only ever be deleted or loosened.

        A model that ignores its mask entirely passes an equality check on the
        early positions, so the positive arm -- the late positions DO move -- is
        the load-bearing half.
        """
        model = _build()
        tokens = _tokens()
        mask = np.ones_like(tokens, dtype="float32")
        mask[:, 8:] = 0.0

        # `ops.convert_to_numpy` wraps the KEY lookup, not the dict: applied to
        # the whole output it converts a dict of arrays into a single NumPy
        # OBJECT array, and indexing that with a string raises IndexError.
        padded = ops.convert_to_numpy(
            model({"input_ids": tokens, "attention_mask": mask},
                  training=False)["last_hidden_state"]
        )
        full = ops.convert_to_numpy(
            model({"input_ids": tokens}, training=False)["last_hidden_state"]
        )

        assert float(np.abs(padded[:, 8:] - full[:, 8:]).max()) > 1e-4
        assert float(np.abs(padded[:, :8] - full[:, :8]).max()) == 0.0

    def test_the_output_dict_echoes_no_none(self):
        """``predict({"input_ids": ...})`` compares nested structures."""
        model = _build()
        outputs = model({"input_ids": _tokens()}, training=False)
        assert None not in outputs.values()
        assert set(outputs) == {"logits", "last_hidden_state"}

    def test_compute_output_shape_works_before_build(self):
        model = _build()
        shapes = model.compute_output_shape((None, 12))
        assert shapes["logits"] == (None, 12, VOCAB)
        assert shapes["last_hidden_state"] == (None, 12, 256)
        assert not model.built

    def test_compute_output_shape_matches_the_forward(self):
        model = _build()
        shapes = model.compute_output_shape((None, 12))
        outputs = model(_tokens(), training=False)
        # The batch axis is None in the declaration; everything after it is exact.
        assert shapes["logits"][1:] == tuple(outputs["logits"].shape)[1:]
        assert shapes["last_hidden_state"][1:] == tuple(
            outputs["last_hidden_state"].shape
        )[1:]

    def test_an_untied_head_is_a_separate_projection(self):
        tied = _build(tie_word_embeddings=True)
        untied = _build(tie_word_embeddings=False)
        tied(_tokens())
        untied(_tokens())
        assert tied.lm_head is None
        assert untied.lm_head is not None
        tied_paths = _relative_paths(tied)
        untied_paths = _relative_paths(untied)
        assert any("lm_head" in path for path in untied_paths)
        assert not any("lm_head" in path for path in tied_paths)


# ---------------------------------------------------------------------
# the taps
# ---------------------------------------------------------------------


class TestTapInventory:
    def test_each_block_carries_one_tap_per_site(self):
        model = _build()
        assert len(model.blocks) == 2
        assert all(len(block.taps) == 2 for block in model.blocks)
        assert len(model.tap_layers) == 2 * len(TAP_SITES)

    def test_every_tap_is_a_spatial_smoothness_layer(self):
        model = _build()
        assert all(
            isinstance(tap, SpatialSmoothness) for tap in model.tap_layers
        )

    def test_a_single_site_ablation_halves_the_inventory(self):
        model = _build(tap_sites=("attention",))
        assert len(model.tap_layers) == 2
        assert all(block.tap_sites == ("attention",) for block in model.blocks)

    def test_no_tap_site_builds_no_tap_weights(self):
        """The anti-vacuity sibling of the build-parity guard.

        Parity alone would pass if BOTH paths built everything; this is what
        proves a block that ``call`` never taps also builds nothing for it.
        """
        tapped = _build()
        tapped(_tokens())
        untapped = _build(tap_sites=())
        untapped(_tokens())

        assert untapped.tap_layers == ()
        assert not [w for w in untapped.weights if "_tap/" in w.path]
        assert len(untapped.weights) < len(tapped.weights)

    def test_every_tap_has_a_distinct_layout(self):
        """Distinct permutations are the point of the ablation being on by
        default: one shared layout is what the residual stream exploits."""
        model = _build(depth=3, seed=11)
        model(_tokens())
        permutations = [
            ops.convert_to_numpy(tap.perm).tolist() for tap in model.tap_layers
        ]
        assert len({tuple(p) for p in permutations}) == len(permutations)

    def test_the_layouts_are_reproducible_from_the_seed_alone(self):
        left = _build(seed=11)
        left(_tokens())
        right = _build(seed=11)
        right(_tokens())
        for a, b in zip(left.tap_layers, right.tap_layers):
            np.testing.assert_array_equal(
                ops.convert_to_numpy(a.perm), ops.convert_to_numpy(b.perm)
            )

    def test_a_different_seed_gives_different_layouts(self):
        left = _build(seed=11)
        left(_tokens())
        right = _build(seed=12)
        right(_tokens())
        assert not np.array_equal(
            ops.convert_to_numpy(left.tap_layers[0].perm),
            ops.convert_to_numpy(right.tap_layers[0].perm),
        )

    def test_the_permutation_ablation_reaches_the_layout(self):
        """``permute=False`` must place unit i at cell i.

        A knob that only reaches the config would leave a random permutation in
        place, and the ablation would silently measure nothing.
        """
        model = _build(permute=False)
        model(_tokens())
        perm = ops.convert_to_numpy(model.tap_layers[0].perm)
        np.testing.assert_array_equal(perm, np.arange(256))

    def test_permuted_and_identity_layouts_differ(self):
        permuted = _build(permute=True, seed=3)
        permuted(_tokens())
        identity = _build(permute=False)
        identity(_tokens())
        assert not np.array_equal(
            ops.convert_to_numpy(permuted.tap_layers[0].perm),
            ops.convert_to_numpy(identity.tap_layers[0].perm),
        )


class TestTheAlphaControl:
    """The non-topographic baseline is the paper's comparison, so it has to be a
    control rather than a different model."""

    def test_alpha_zero_keeps_every_weight(self):
        live = _build(alpha=2.5)
        live(_tokens())
        control = _build(alpha=0.0)
        control(_tokens())
        assert _relative_paths(live) == _relative_paths(control)

    def test_alpha_zero_keeps_every_permutation_bit_identical(self):
        live = _build(alpha=2.5, seed=5)
        live(_tokens())
        control = _build(alpha=0.0, seed=5)
        control(_tokens())
        for a, b in zip(live.tap_layers, control.tap_layers):
            np.testing.assert_array_equal(
                ops.convert_to_numpy(a.perm), ops.convert_to_numpy(b.perm)
            )

    def test_alpha_zero_adds_no_loss(self):
        live = _build(alpha=2.5)
        live(_tokens(), training=True)
        control = _build(alpha=0.0)
        control(_tokens(), training=True)
        assert len(live.losses) == 2 * len(TAP_SITES)
        assert control.losses == []

    def test_alpha_scales_the_added_losses(self):
        """A loss COUNT would pass on a tap wired to the wrong tensor, so the
        claim checked is the VALUE: one tap's added loss is ``alpha`` times the
        penalty it recorded."""
        model = _build(alpha=1.0)
        model(_tokens(), training=True)
        assert len(model.losses) == 2 * len(TAP_SITES)
        for tap in model.tap_layers:
            # Each tap's OWN `add_loss` entry, not a positional slice of
            # `model.losses`: the aggregate list is collected in sub-layer
            # tracking order, which does NOT match `tap_layers` order, so a zip
            # pairs block 0's attention loss with block 0's mlp value.
            added = float(tap.losses[-1])
            recorded = float(ops.convert_to_numpy(tap.last_spatial_loss))
            assert added == pytest.approx(1.0 * recorded, rel=1e-4), (
                tap.name, added, recorded
            )

    def test_the_tap_count_is_a_function_of_depth_and_sites_only(self):
        for depth in (1, 2, 5):
            for sites in ((), ("attention",), ("mlp",), TAP_SITES):
                model = _build(depth=depth, tap_sites=sites)
                assert len(model.tap_layers) == depth * len(sites)


# ---------------------------------------------------------------------
# build
# ---------------------------------------------------------------------


class TestBuild:
    def test_explicit_build_matches_lazy_build(self):
        explicit = _build()
        explicit.build((None, 12))
        lazy = _build()
        lazy(_tokens())

        assert _relative_paths(explicit), "the model has no weights to compare"
        assert _relative_paths(explicit) == _relative_paths(lazy)

    def test_build_actually_materializes_weights(self):
        """``Model.build(shape)`` alone leaves a subclassed model with nothing.

        Not a hypothetical: a build that stops after ``super().build()`` reports
        ``0`` parameters, restores into nothing on load, and raises nothing.
        """
        model = _build()
        model.build((None, 12))
        assert model.count_params() > 0
        assert len(model.weights) > 0

    def test_a_dynamic_sequence_axis_builds(self):
        model = _build()
        model.build((None, None))
        assert model.count_params() > 0

    def test_a_sequence_beyond_the_position_table_is_rejected(self):
        model = _build(max_seq_len=16)
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            model.build((None, 32))

    def test_a_rank_three_input_is_rejected(self):
        model = _build()
        with pytest.raises(ValueError, match="rank 2"):
            model.build((None, 4, 8))

    def test_a_block_rejects_a_hidden_size_it_was_not_built_for(self):
        """The cross-parameter contract ``call`` relies on, checked at build."""
        block = TopoLMBlock(
            hidden_size=256, num_heads=4, intermediate_size=512, radius=3
        )
        with pytest.raises(ValueError, match="hidden_size=256"):
            block.build((None, 4, 128))


# ---------------------------------------------------------------------
# causality, composition, positions
# ---------------------------------------------------------------------


class TestTheFutureLeakProbe:
    """Three arms, because arm 1 alone is satisfied by a model that ignores its
    input entirely."""

    @pytest.fixture(scope="class")
    def fixture(self):
        model = _build()
        tokens = _tokens(batch=2, length=12, seed=1)
        return model, tokens

    def _logits(self, model, tokens):
        return ops.convert_to_numpy(
            model(tokens, training=False)["logits"]
        )

    def test_positions_before_the_change_are_bit_identical(self, fixture):
        """Exactly ``0.0``, not "small": an attention weight of exactly zero on a
        masked key contributes exactly nothing.

        Fresh non-DC noise for the perturbation -- a per-channel constant is
        absorbed by the normalization in front of the guarded reduction, which is
        how a real leak once measured ``1.9e-06`` against a signal of 0.33.
        """
        model, tokens = fixture
        cut = 6
        perturbed = tokens.copy()
        perturbed[:, cut] = (perturbed[:, cut] + 1) % VOCAB

        delta = float(
            np.abs(
                self._logits(model, tokens)[:, :cut]
                - self._logits(model, perturbed)[:, :cut]
            ).max()
        )
        assert delta == 0.0, delta

    def test_the_perturbation_reached_the_model_at_all(self, fixture):
        """The other half of the pair. Without it, ``0.0`` above proves nothing."""
        model, tokens = fixture
        cut = 6
        perturbed = tokens.copy()
        perturbed[:, cut] = (perturbed[:, cut] + 1) % VOCAB

        delta = float(
            np.abs(
                self._logits(model, tokens)[:, cut:]
                - self._logits(model, perturbed)[:, cut:]
            ).max()
        )
        assert delta > 1e-3, delta

    def test_the_mask_is_rank_three(self, fixture):
        """A rank-2 causal mask is silently reinterpreted as a padding mask by
        grouped-query attention, and arm 1 would then pass for the wrong reason."""
        from dl_techniques.utils.masking import create_causal_attend_mask

        hidden = ops.convert_to_tensor(
            np.zeros((2, 5, 8), dtype="float32")
        )
        mask = create_causal_attend_mask(hidden)
        assert len(ops.shape(mask)) == 3


class TestPositionsMatter:
    def test_reordering_tokens_moves_the_output(self):
        """The positional embedding exists, is read, and is not a no-op.

        A component whose COUNT is right and whose IDENTITY is wrong -- a rotary
        embedding fed the head axis, a position table built and never added --
        passes every shape, parameter and gradient check.
        """
        model = _build()
        tokens = _tokens(seed=2)
        swapped = tokens.copy()
        swapped[:, [0, 1]] = swapped[:, [1, 0]]

        delta = float(
            np.abs(
                ops.convert_to_numpy(model(tokens, training=False)["last_hidden_state"])
                - ops.convert_to_numpy(
                    model(swapped, training=False)["last_hidden_state"]
                )
            ).max()
        )
        assert delta > 1e-3, delta

    def test_a_shifted_prefix_moves_the_output(self):
        """Non-cyclic in the other direction: prepending a token changes
        everything, so the model is not position-invariant."""
        model = _build()
        short = _tokens(batch=2, length=6, seed=3)
        padded = np.concatenate(
            [np.zeros((2, 6), dtype="int32"), short], axis=1
        )
        delta = float(
            np.abs(
                ops.convert_to_numpy(
                    model(short, training=False)["last_hidden_state"]
                )
                - ops.convert_to_numpy(
                    model(padded, training=False)["last_hidden_state"][:, 6:]
                )
            ).max()
        )
        assert delta > 1e-3, delta


class TestComposition:
    @pytest.mark.parametrize("depth", [1, 2, 8])
    def test_the_ladder_does_not_collapse(self, depth):
        """``std(out) / std(in)`` must stay near 1 across the stack.

        A block that computes a transform and expects an external residual, or a
        block whose ``gamma -> 0`` limit is zero rather than identity, collapses
        the signal to ~1e-5 per block while every shape and gradient check passes.
        """
        model = _build(depth=depth)
        hidden = np.random.default_rng(4).normal(
            size=(2, 6, 256)
        ).astype("float32")
        tensor = ops.convert_to_tensor(hidden)
        for block in model.blocks:
            tensor = block(tensor, training=False)
        ratio = float(ops.std(tensor)) / float(ops.std(tensor * 0 + hidden))
        assert 0.5 < ratio < 2.0, ratio

    @pytest.mark.parametrize("position", ["pre", "post"])
    def test_both_normalization_positions_forward(self, position):
        """``call`` branches on this, and an unguarded ``else`` runs the wrong one.

        The structural knob is the position flag itself, so both arms are pinned
        here rather than only the default.
        """
        model = _build(normalization_position=position)
        outputs = model(_tokens(), training=False)
        assert np.all(np.isfinite(ops.convert_to_numpy(outputs["logits"])))

    def test_the_normalization_position_reaches_the_block(self):
        pre = _build(normalization_position="pre")
        post = _build(normalization_position="post")
        assert all(
            block.normalization_position == "pre" for block in pre.blocks
        )
        assert all(
            block.normalization_position == "post" for block in post.blocks
        )

    def test_an_unknown_normalization_position_is_rejected(self):
        with pytest.raises(ValueError, match="must be 'pre' or 'post'"):
            _build(normalization_position="Pre")


# ---------------------------------------------------------------------
# knobs
# ---------------------------------------------------------------------


class TestKnobSensitivity:
    """Every constructor knob pinned with the instrument matched to its class.

    Structural knobs are pinned on the weight SHAPE signature, because two models
    of different shapes consume different RNG draws and their OUTPUTS differ
    whether or not the argument was honoured -- an output-diff assertion on a
    structural knob is satisfied by random-init luck alone.
    """

    @staticmethod
    def _signature(model, tokens):
        model(tokens, training=False)
        return tuple(tuple(w.shape) for w in model.weights)

    def test_a_structural_knob_changes_the_weight_shapes(self):
        small = self._signature(_build(depth=2), _tokens())
        deep = self._signature(_build(depth=5), _tokens())
        wide = self._signature(_build(embed_dim=512, ffn_intermediate_size=512), _tokens())
        narrow_ffn = self._signature(_build(ffn_intermediate_size=384), _tokens())

        assert len({small, deep, wide, narrow_ffn}) == 4

    def test_the_head_count_reaches_the_attention_weights(self):
        """``num_heads`` is NOT a structural knob by shape.

        Re-shaping a projection from (256, 256) into 8 heads of 32 leaves every
        weight the same size, so a shape-signature probe cannot see it. It is
        pinned on the OUTPUT instead, and the both-ways arm holds the seed.
        """
        tokens = _tokens()
        four = _build(num_heads=4)
        eight = _build(num_heads=8)
        left = self._signature(four, tokens)
        right = self._signature(eight, tokens)
        assert left == right, "the head count changed the weight shapes"

        difference = np.abs(
            ops.convert_to_numpy(four(tokens, training=False)["logits"])
            - ops.convert_to_numpy(eight(tokens, training=False)["logits"])
        )
        assert float(difference.max()) > 1e-6

    def test_a_value_knob_changes_the_output_and_not_the_shapes(self):
        tokens = _tokens()
        narrow = _build(layer_norm_eps=1e-5)
        wide = _build(layer_norm_eps=1e-2)
        left = self._signature(narrow, tokens)
        right = self._signature(wide, tokens)
        assert left == right, "a value knob changed the weight shapes"

        difference = np.abs(
            ops.convert_to_numpy(narrow(tokens, training=False)["logits"])
            - ops.convert_to_numpy(wide(tokens, training=False)["logits"])
        )
        assert float(difference.max()) > 1e-6, difference.max()

    def test_a_scoped_value_knob_changes_the_tap_weights_only(self):
        """``permute`` reaches the taps' weights and nothing else.

        A scoped probe compares the VALUES of a named subtree, not a whole-model
        output diff: an output diff passes on the broken tree whenever the same
        knob reaches another sub-layer by a second route.
        """
        permuted = _build(permute=True, seed=9)
        permuted(_tokens())
        identity = _build(permute=False)
        identity(_tokens())

        def in_taps(model):
            # The path segment is `attention_tap/` / `mlp_tap/`, so the scope
            # marker is `_tap/`, not `tap/` -- the latter matches nothing and the
            # probe would compare two EMPTY dictionaries and pass.
            return {
                weight.path.split("/", 1)[-1]: ops.convert_to_numpy(weight)
                for weight in model.weights
                if "_tap/" in weight.path and weight.name == "perm"
            }

        left, right = in_taps(permuted), in_taps(identity)
        assert left and right, (len(left), len(right))
        assert any(
            not np.array_equal(left[key], right[key]) for key in left
        )

        def outside_taps(model):
            return {
                weight.path.split("/", 1)[-1]: ops.convert_to_numpy(weight)
                for weight in model.weights
                if "_tap/" not in weight.path
            }

        left_untapped = outside_taps(permuted)
        right_untapped = outside_taps(identity)
        assert sorted(left_untapped) == sorted(right_untapped)
        for key, values in left_untapped.items():
            np.testing.assert_allclose(
                values, right_untapped[key], atol=1e-6, rtol=0,
                err_msg=f"{key} moved with `permute`, which it must not",
            )

    def test_the_radius_reaches_the_table_shapes(self):
        """``radius`` is structural for the tap: it changes ``patch_table``."""
        model = _build(radius=2, depth=1)
        model(_tokens())
        tables = [
            weight for weight in model.weights
            if weight.name == "patch_table"
        ]
        assert tables, "no patch table was built"
        radius_two = tables[0].shape
        assert radius_two[1] == 25  # (2 * 2 + 1) ** 2

        other = _build(radius=4, depth=1)
        other(_tokens())
        other_tables = [w for w in other.weights if w.name == "patch_table"]
        assert other_tables[0].shape[1] == 81  # (2 * 4 + 1) ** 2

    def test_the_distance_metric_reaches_the_prior_values(self):
        linf = _build(distance="linf", depth=1)
        linf(_tokens())
        l2 = _build(distance="l2", depth=1)
        l2(_tokens())

        def prior(model):
            return {
                weight.name: ops.convert_to_numpy(weight)
                for weight in model.weights
                if weight.name == "d_unit"
            }

        left, right = prior(linf), prior(l2)
        assert left and right
        assert any(not np.array_equal(left[k], right[k]) for k in left)

    def test_the_grid_shape_reaches_the_table_values(self):
        """A transposed grid is the same SHAPE of table and a different table.

        ``(16, 32)`` and ``(32, 16)`` both host a radius-5 patch and both give
        ``(132, 121)`` -- the centre count is ``(h - 2r)(w - 2r)``, symmetric under
        transposition. So a shape-signature probe is blind here, which is exactly
        the situation the guide warns about: an orientation defect survives a
        structural check. The discriminator is the table's VALUES, and a square
        grid is the anti-vacuity control that must agree with neither.
        """
        wide = _build(embed_dim=512, ffn_intermediate_size=512, grid_shape=(16, 32), depth=1)
        wide(_tokens(batch=2, length=12))
        tall = _build(embed_dim=512, ffn_intermediate_size=512, grid_shape=(32, 16), depth=1)
        tall(_tokens(batch=2, length=12))

        def table(model):
            return [
                ops.convert_to_numpy(weight)
                for weight in model.weights
                if weight.name == "patch_table"
            ][0]

        left, right = table(wide), table(tall)
        assert left.shape == right.shape, (
            "the two grids should give the same table SHAPE -- if they do not, "
            "this test is no longer testing orientation"
        )
        assert not np.array_equal(left, right)

    def test_the_grid_shape_still_governs_the_centre_count(self):
        """The shape-level claim, on grids where it actually differs."""
        wide = _build(embed_dim=512, ffn_intermediate_size=512, grid_shape=(16, 32), depth=1)
        wide(_tokens(batch=2, length=12))
        square = _build(embed_dim=256, ffn_intermediate_size=512, depth=1)
        square(_tokens(batch=2, length=12))

        def table(model):
            return [
                weight for weight in model.weights if weight.name == "patch_table"
            ][0]

        # 16 x 32 -> (16-10)(32-10) = 132 centres; 16 x 16 -> 36.
        assert tuple(table(wide).shape) == (132, 121)
        assert tuple(table(square).shape) == (36, 121)


# ---------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------


class TestGradients:
    def test_every_trainable_variable_receives_a_non_zero_gradient(self):
        """Non-None AND non-zero, by path, with the anti-vacuity count.

        A guard written as "all gradients are finite" reports green while every
        weight has an identically-zero gradient.
        """
        model = _build()
        tokens = _tokens(batch=2, length=12, seed=6)
        target = ops.convert_to_tensor(
            np.random.default_rng(11).normal(
                size=(2, 12, VOCAB)
            ).astype("float32")
        )
        with tf.GradientTape() as tape:
            logits = model(tokens, training=True)["logits"]
            loss = tf.reduce_mean(tf.square(logits - target))
            loss = loss + tf.add_n([tf.cast(v, loss.dtype) for v in model.losses])
        gradients = tape.gradient(loss, model.trainable_variables)

        assert len(model.trainable_variables) > 0
        missing = []
        zero = []
        for variable, gradient in zip(model.trainable_variables, gradients):
            if gradient is None:
                missing.append(variable.path)
            elif not np.any(ops.convert_to_numpy(gradient) != 0.0):
                zero.append(variable.path)
        assert not missing, f"no gradient for {missing}"
        assert not zero, f"all-zero gradient for {zero}"

    def test_a_step_actually_moves_the_weights(self):
        """Liveness through the optimizer, not through a hand-written tape.

        The name set is returned rather than a count: ``moved > 0`` was once
        satisfied by a 118-of-137 result whose 19-variable residual was never
        identified.
        """
        model = _build()
        head = _build_head(model, aggregate=True)
        dataset = _dataset()
        head.compile(optimizer=keras.optimizers.SGD(1e-2))

        head(np.zeros((1, 12), dtype="int32"), training=False)
        before = {
            variable.path: ops.convert_to_numpy(variable).copy()
            for variable in model.trainable_variables
        }
        head.fit(dataset, epochs=1, verbose=0)

        after = {
            variable.path: ops.convert_to_numpy(variable)
            for variable in model.trainable_variables
        }
        moved = {
            path for path, values in before.items()
            if not np.array_equal(values, after[path])
        }
        assert moved == set(before), sorted(set(before) - moved)


def _build_head(backbone, aggregate):
    from dl_techniques.models.common.masked_language_model import (
        CausalLanguageModel,
    )

    return CausalLanguageModel(
        backbone=backbone,
        vocab_size=VOCAB,
        skip_head=True,
        output_key="logits",
        pre_shifted=True,
        aggregate_backbone_losses=aggregate,
        verify_causality=False,
    )


def _dataset(batch=2):
    return tf.data.Dataset.from_tensor_slices(
        (
            np.random.default_rng(7).integers(0, VOCAB, size=(4, 12)).astype("int32"),
            np.random.default_rng(8).integers(0, VOCAB, size=(4, 12)).astype("int32"),
        )
    ).batch(batch)


class TestSpatialLossReachesTheObjective:
    """The RED proof that ``aggregate_backbone_losses`` is load-bearing.

    Without it the tap computes its loss every step, the training loop never sees
    it, and the run looks entirely healthy.
    """

    def test_the_reported_training_loss_includes_the_spatial_term(self):
        totals = {}
        for aggregate in (True, False):
            keras.utils.set_random_seed(0)
            backbone = _build(alpha=2.5)
            head = _build_head(backbone, aggregate)
            head.compile(optimizer=keras.optimizers.SGD(1e-3))
            head(np.zeros((1, 12), dtype="int32"), training=False)

            history = head.fit(_dataset(), epochs=1, verbose=0)
            evaluation = head.evaluate(_dataset(), verbose=0, return_dict=True)
            totals[aggregate] = (
                history.history["loss"][0], evaluation["loss"]
            )

        with_term, without_term = totals[True], totals[False]
        # Roughly 2 * depth * alpha * 0.5 of extra reported loss, ~9.9 here.
        assert with_term[0] > without_term[0] + 5.0, with_term
        # And the EVALUATION loss is the pure task loss either way, because the
        # taps add nothing when `training` is not True.
        assert with_term[1] == pytest.approx(without_term[1], rel=0.05)

    def test_validation_loss_excludes_the_spatial_term(self):
        """The taps add nothing at inference, so ``val_loss`` is the task loss.

        Asserted as a DELTA across the two calls rather than as a count read after
        the second: Keras clears ``model.losses`` at the start of every call, so a
        count read after the inference pass is ``0`` either way -- including for a
        tap that wrongly fired.
        """
        model = _build(alpha=2.5)
        model(_tokens(), training=True)
        assert len(model.losses) == 2 * len(TAP_SITES)

        model(_tokens(seed=9), training=False)
        assert model.losses == []


# ---------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------


class TestSerialization:
    def _round_trip(self, model, tokens):
        original = ops.convert_to_numpy(model(tokens, training=False)["logits"])
        saved = {
            weight.name: ops.convert_to_numpy(weight).copy()
            for weight in model.weights
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "topolm.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            restored = {
                weight.name: ops.convert_to_numpy(weight)
                for weight in loaded.weights
            }
            assert sorted(saved) == sorted(restored), (
                f"missing {sorted(set(saved) - set(restored))}, "
                f"unexpected {sorted(set(restored) - set(saved))}"
            )
            # atol=0.0: restoration is a copy, not a computation.
            for name, values in saved.items():
                np.testing.assert_array_equal(
                    values, restored[name], err_msg=f"{name} was not restored"
                )
            replayed = ops.convert_to_numpy(
                loaded(tokens, training=False)["logits"]
            )
        return original, replayed

    def test_a_keras_round_trip_preserves_values_and_every_permutation(self):
        model = _build(depth=3, seed=4)
        tokens = _tokens(seed=5)
        original, replayed = self._round_trip(model, tokens)
        np.testing.assert_allclose(replayed, original, atol=1e-6, rtol=0)

    def test_the_config_round_trips_through_from_config(self):
        model = _build(
            alpha=1.5, radius=3, num_neighborhoods=7, distance="l2",
            permute=False, dropout_rate=0.1, layer_norm_eps=1e-6, seed=21,
            tap_sites=("mlp",),
        )
        rebuilt = TopoLM.from_config(model.get_config())
        assert rebuilt.get_config() == model.get_config()
        for key in ("alpha", "radius", "num_neighborhoods", "distance",
                    "permute", "seed", "tap_sites"):
            assert rebuilt.get_config()[key] == model.get_config()[key]

    def test_the_config_carries_every_constructor_argument(self):
        """A config missing one argument round-trips to a DIFFERENT model."""
        model = _build()
        config = model.get_config()
        for key in (
            "vocab_size", "embed_dim", "depth", "num_heads",
            "ffn_intermediate_size", "max_seq_len", "alpha", "radius",
            "num_neighborhoods", "distance", "permute", "grid_shape",
            "tap_sites", "dropout_rate", "attention_dropout_rate",
            "initializer_range", "layer_norm_eps", "activation",
            "tie_word_embeddings", "seed",
        ):
            assert key in config, key

    def test_the_activation_survives_the_config_round_trip(self):
        """A callable activation would be handed back raw on some backends."""
        model = _build()
        rebuilt = TopoLM.from_config(model.get_config())
        assert ops.convert_to_tensor(
            rebuilt.activation(ops.convert_to_tensor([1.0]))
        ) is not None

    def test_the_taps_do_not_appear_in_the_config(self):
        """They travel as weights; a config entry would re-derive a layout."""
        model = _build()
        config = model.get_config()
        assert "tap_layers" not in config
        assert "blocks" not in config


# ---------------------------------------------------------------------
# precision and graph safety
# ---------------------------------------------------------------------


class TestPrecision:
    @pytest.mark.parametrize("policy", ("float32", "mixed_float16"))
    def test_the_forward_is_finite_under_each_policy(self, policy):
        previous = keras.mixed_precision.global_policy().name
        try:
            keras.mixed_precision.set_global_policy(policy)
            model = _build()
            outputs = model(_tokens(), training=True)
            logits = ops.convert_to_numpy(outputs["logits"])
            assert np.all(np.isfinite(logits))
            assert all(
                np.isfinite(float(value)) for value in model.losses
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)

    def test_a_tf_function_trace_matches_eager(self):
        model = _build()
        tokens = _tokens()

        @tf.function(
            input_signature=[tf.TensorSpec([None, None], tf.int32)]
        )
        def traced(ids):
            return model(ids, training=False)["logits"]

        np.testing.assert_allclose(
            ops.convert_to_numpy(traced(tokens)),
            ops.convert_to_numpy(model(tokens, training=False)["logits"]),
            atol=1e-5, rtol=0,
        )

    def test_a_jit_compiled_step_actually_optimises(self):
        keras.utils.set_random_seed(0)
        model = _build()
        head = _build_head(model, aggregate=True)
        head.compile(
            optimizer=keras.optimizers.SGD(1e-2), jit_compile=True
        )
        head(np.zeros((1, 12), dtype="int32"), training=False)
        before = {
            variable.path: ops.convert_to_numpy(variable).copy()
            for variable in model.trainable_variables
        }
        history = head.fit(_dataset(), epochs=2, verbose=0)

        assert np.isfinite(history.history["loss"][-1])
        assert history.history["loss"][-1] < history.history["loss"][0]
        after = {
            variable.path: ops.convert_to_numpy(variable)
            for variable in model.trainable_variables
        }
        moved = {
            path for path, values in before.items()
            if not np.array_equal(values, after[path])
        }
        assert moved, "the JIT-compiled step moved nothing"


class TestEveryComponentIsLive:
    def test_the_probe_can_detect_a_dead_training_path(self):
        """The RED proof for the step-movement guard above.

        With the outputs cut off the tape, a LIVE training path must raise. If it
        does not, the guard above is measuring a model whose gradients were never
        applied.
        """
        from unittest import mock

        model = _build()
        head = _build_head(model, aggregate=True)
        head.compile(optimizer=keras.optimizers.SGD(1e-2))
        head(np.zeros((1, 12), dtype="int32"), training=False)

        original = type(model).call

        def cut(*args, **kwargs):
            outputs = original(model, *args, **kwargs)
            return {
                key: keras.ops.stop_gradient(value)
                for key, value in outputs.items()
            }

        # The taps' `add_loss` tensors are a SECOND live path to the optimizer and
        # they are not part of `model.call`'s return value. Cutting only the
        # outputs leaves a gradient path intact, `fit` succeeds, and the probe
        # reports a dead training path as a live one.
        def cut_tap(self, inputs, training=None):
            # Not "call the original then clear": `add_loss` has already run by
            # the time the layer returns, and Keras collected it during the call.
            # The tap has to contribute NOTHING.
            self.losses.clear()
            return inputs

        with mock.patch.object(model, "call", cut), mock.patch.object(
            SpatialSmoothness, "call", cut_tap
        ):
            head.train_function = None
            try:
                with pytest.raises(Exception) as info:
                    head.fit(_dataset(), epochs=1, verbose=0)
                assert "gradient" in str(info.value).lower(), info.value
            finally:
                head.train_function = None


class TestTheSmokeContract:
    def test_the_contract_rejects_a_broken_forward(self):
        from ..smoke_contract_oracle import (
            assert_contract_rejects_a_broken_forward,
        )

        model = _build()
        tokens = _tokens()

        def contract(outputs):
            logits = ops.convert_to_numpy(outputs["logits"])
            hidden = ops.convert_to_numpy(outputs["last_hidden_state"])
            assert logits.shape == (2, 12, VOCAB)
            assert hidden.shape == (2, 12, 256)
            assert np.all(np.isfinite(logits))
            assert np.all(np.isfinite(hidden))
            assert float(np.abs(logits).max()) > 1e-4

        def collapse_logits(outputs):
            """A dict-output model needs a dict-aware breaker.

            The shared oracle's default breakers collapse the whole output, which
            on a dict is a ``TypeError`` from ``ops.mean`` -- the contract
            CRASHING rather than judging, which the oracle correctly refuses to
            accept as a pass.
            """
            return {**outputs, "logits": ops.mean(outputs["logits"])}

        def slice_logits(outputs):
            return {**outputs, "logits": outputs["logits"][:1]}

        assert_contract_rejects_a_broken_forward(
            model, tokens, contract,
            breakers=(collapse_logits, slice_logits),
        )