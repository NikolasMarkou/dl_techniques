"""``DiT``: the model's shape contract, its frozen table, and the unpatchify orientation.

This file proves the model is correct and buildable. It deliberately stops short
of the comprehensive v2 §16.3 suite (step 9) and of the six upstream-quirk guards
(step 10); what is here is what a reader needs to believe ``model.py`` works at
all, plus the three properties that are invisible to every conventional test:

1. **The unpatchify interleave.** Upstream's einsum is channels-FIRST and the
   channels-last derivation has a plausible-looking transposed alternative that
   produces an IDENTICALLY SHAPED tensor on a square grid. The guard is a
   delta-impulse at an asymmetric patch coordinate on a NON-SQUARE token grid,
   with the destination index computed by hand from ``row = i*p + pi`` /
   ``col = j*p + pj`` and never by re-invoking the function's own arithmetic.
   A sibling arm asserts the transposed permutation is shape-identical, so the
   blindness the guard exists to cover is itself pinned.

2. **The positional table is a frozen WEIGHT.** Not a Python attribute (does not
   survive a round trip) and not ``add_weight(zeros).assign()`` in ``build()``
   (``StatelessScope`` discards the assign and leaves it all zeros, silently).
   Pinned by a value comparison against an independently computed NumPy table
   AND by a build-through-a-parent's-``call()`` probe, which is the code path
   where the ``.assign()`` failure mode actually bites.

3. **The label table's initializer.** The house default for
   ``ClassLabelEmbedding`` is ``"uniform"``; upstream is ``normal(std=0.02)``.
   The two differ in no shape, no count and no config, so the arm here reads the
   table's own dispersion and its tail beyond the uniform default's hard bound.

The premise everything rests on: the model emits **exactly 0.0** at
initialisation, because every block's modulation ``Dense`` and the whole final
layer are zero-initialised. Assertions here are therefore about shapes, weights
and structure, never about "the output changed".
"""

from typing import Any, Dict

import keras
import numpy as np
import pytest

from dl_techniques.layers.embedding.sincos_pos_embed_2d import get_2d_sincos_pos_embed
from dl_techniques.models.vision_language.dit.config import DIT_VARIANTS
from dl_techniques.models.vision_language.dit.model import (
    CFG_GUIDED_CHANNELS,
    DiT,
    create_dit,
    flattened_linear_xavier,
    unpatchify_tokens,
)

from ._dit_helpers import BATCH, TINY, tiny_inputs, tiny_model

# ---------------------------------------------------------------------
# A tiny, fully-specified configuration every arm shares
# ---------------------------------------------------------------------
#
# `TINY`, `tiny_model` and `tiny_inputs` MOVED to `_dit_helpers.py` at step 9,
# when the comprehensive suite became a second consumer. One home for the
# geometry, rather than two dicts kept equal by hand.


def expected_parameter_count(cfg: Dict[str, Any]) -> int:
    """Total parameter count derived from the config, term by term.

    Computed here from the ARCHITECTURE, not from the model: every term names
    the sub-layer it counts, so a count mismatch localizes the discrepancy
    instead of just reporting a wrong total.
    """
    d = cfg["hidden_size"]
    p = cfg["patch_size"]
    c_in = cfg["in_channels"]
    c_out = c_in * 2 if cfg["learn_sigma"] else c_in
    grid = cfg["input_size"] // p
    tokens = grid * grid
    freq = cfg["frequency_embedding_size"]
    mlp_hidden = int(cfg["mlp_ratio"] * d)
    label_rows = cfg["num_classes"] + (1 if cfg["class_dropout_rate"] > 0 else 0)

    # PatchEmbedding2D's Conv2D: kernel (p, p, C_in, D) + bias (D,)
    total = p * p * c_in * d + d
    # The frozen sin-cos table: (1, T, D)
    total += tokens * d
    # TimestepEmbedding: the frozen frequency ladder (freq // 2), then
    # Dense(freq -> D) and Dense(D -> D).
    total += freq // 2 + (freq * d + d) + (d * d + d)
    # ClassLabelEmbedding table
    total += label_rows * d
    # Each block: 6-way adaLN Dense, 4 attention projections, 2 MLP Denses.
    per_block = (d * 6 * d + 6 * d) + 4 * (d * d + d)
    per_block += (d * mlp_hidden + mlp_hidden) + (mlp_hidden * d + d)
    total += cfg["depth"] * per_block
    # Final layer: 2-way adaLN Dense, then the zero-init read-out projection.
    total += (d * 2 * d + 2 * d) + (d * (p * p * c_out) + p * p * c_out)
    return total


# ---------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------


class TestTheForwardPass:
    """Shape, finiteness and the static shape contract."""

    def test_output_shape_is_the_input_grid_with_out_channels(self) -> None:
        model = tiny_model()
        x, t, y = tiny_inputs()
        out = np.asarray(model([x, t, y], training=False))
        assert out.shape == (
            BATCH,
            TINY["input_size"],
            TINY["input_size"],
            2 * TINY["in_channels"],
        )

    def test_the_output_is_finite(self) -> None:
        model = tiny_model()
        x, t, y = tiny_inputs(seed=1)
        out = np.asarray(model([x, t, y], training=False))
        assert np.all(np.isfinite(out))

    def test_learn_sigma_false_halves_the_output_channels(self) -> None:
        model = tiny_model(learn_sigma=False)
        x, t, y = tiny_inputs(seed=2)
        out = np.asarray(model([x, t, y], training=False))
        assert out.shape[-1] == TINY["in_channels"]

    def test_compute_output_shape_matches_the_real_forward(self) -> None:
        model = tiny_model()
        x, t, y = tiny_inputs(seed=3)
        declared = model.compute_output_shape(
            [x.shape, t.shape, y.shape]
        )
        actual = np.asarray(model([x, t, y], training=False)).shape
        assert tuple(declared) == actual

    def test_a_non_dividing_patch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="divisible by patch_size"):
            tiny_model(patch_size=3)

    def test_a_rank_3_input_shape_raises_naming_the_layout(self) -> None:
        model = tiny_model()
        with pytest.raises(ValueError, match="channels-LAST"):
            model.build([(None, 8, 8), (None,), (None,)])

    def test_forward_with_cfg_returns_the_full_batch(self) -> None:
        model = tiny_model()
        x, t, y = tiny_inputs(seed=4)
        # Second half is the null row, exactly as a CFG sampler stacks it.
        y = np.concatenate(
            [y[: BATCH // 2], np.full((BATCH // 2,), TINY["num_classes"], "int32")]
        )
        out = np.asarray(model.forward_with_cfg(x, t, y, cfg_scale=1.5, training=False))
        assert out.shape == (
            BATCH,
            TINY["input_size"],
            TINY["input_size"],
            2 * TINY["in_channels"],
        )
        assert np.all(np.isfinite(out))
        # The guided channels are identical across the two halves by
        # construction (`concat([half_eps, half_eps])`).
        guided = out[..., :CFG_GUIDED_CHANNELS]
        np.testing.assert_array_equal(
            guided[: BATCH // 2], guided[BATCH // 2:]
        )


# ---------------------------------------------------------------------
# The unpatchify orientation
# ---------------------------------------------------------------------


class TestTheUnpatchifyOrientation:
    """A delta impulse at an asymmetric coordinate on a NON-SQUARE grid."""

    # A grid that is not square in EITHER the token layout or the impulse
    # coordinate, so a transposed interleave cannot hide behind a symmetry.
    GRID_H, GRID_W, PATCH, CHANNELS = 2, 3, 2, 1

    def _impulse(self, patch_row: int, patch_col: int, pixel_row: int, pixel_col: int):
        """One-hot token payload tensor, built by index arithmetic only."""
        h, w, p, c = self.GRID_H, self.GRID_W, self.PATCH, self.CHANNELS
        tokens = np.zeros((1, h * w, p * p * c), dtype="float32")
        # Token axis is row-major over the grid; payload axis is row-major over
        # (pixel_row, pixel_col, channel). Both spelled out, not derived from
        # the function under test.
        token_index = patch_row * w + patch_col
        payload_index = pixel_row * p * c + pixel_col * c + 0
        tokens[0, token_index, payload_index] = 1.0
        return tokens

    def test_the_impulse_lands_at_the_independently_computed_pixel(self) -> None:
        h, w, p, c = self.GRID_H, self.GRID_W, self.PATCH, self.CHANNELS
        patch_row, patch_col, pixel_row, pixel_col = 0, 2, 1, 0
        tokens = self._impulse(patch_row, patch_col, pixel_row, pixel_col)

        image = np.asarray(unpatchify_tokens(tokens, h, w, p, c))

        # Destination computed from the contract, NOT from the implementation.
        expected_row = patch_row * p + pixel_row
        expected_col = patch_col * p + pixel_col
        assert (expected_row, expected_col) == (1, 4)

        assert image[0, expected_row, expected_col, 0] == 1.0
        # And it is the ONLY non-zero pixel.
        assert int(np.count_nonzero(image)) == 1

    def test_every_pixel_of_a_full_ramp_lands_where_the_contract_says(self) -> None:
        """An exhaustive census, so the single-impulse arm is not a lucky index."""
        h, w, p, c = self.GRID_H, self.GRID_W, self.PATCH, self.CHANNELS
        for patch_row in range(h):
            for patch_col in range(w):
                for pixel_row in range(p):
                    for pixel_col in range(p):
                        tokens = self._impulse(
                            patch_row, patch_col, pixel_row, pixel_col
                        )
                        image = np.asarray(unpatchify_tokens(tokens, h, w, p, c))
                        row = patch_row * p + pixel_row
                        col = patch_col * p + pixel_col
                        assert image[0, row, col, 0] == 1.0, (
                            f"patch=({patch_row},{patch_col}) "
                            f"pixel=({pixel_row},{pixel_col}) -> ({row},{col})"
                        )
                        assert int(np.count_nonzero(image)) == 1

    def test_the_transposed_permutation_is_shape_identical_on_a_square_grid(
        self,
    ) -> None:
        """Why the orientation needs a VALUE guard: shapes cannot see it."""
        h = w = 3
        p, c = 2, 1
        tokens = np.arange(h * w * p * p * c, dtype="float32").reshape(
            1, h * w, p * p * c
        )
        correct = np.asarray(unpatchify_tokens(tokens, h, w, p, c))

        x = keras.ops.reshape(tokens, (1, h, w, p, p, c))
        transposed = np.asarray(
            keras.ops.reshape(
                keras.ops.transpose(x, (0, 2, 4, 1, 3, 5)), (1, h * p, w * p, c)
            )
        )
        assert transposed.shape == correct.shape
        assert not np.array_equal(transposed, correct)

    def test_the_model_method_agrees_with_the_free_function(self) -> None:
        model = tiny_model()
        grid = TINY["input_size"] // TINY["patch_size"]
        out_channels = 2 * TINY["in_channels"]
        payload = TINY["patch_size"] ** 2 * out_channels
        tokens = np.random.default_rng(7).standard_normal(
            (2, grid * grid, payload)
        ).astype("float32")
        np.testing.assert_array_equal(
            np.asarray(model.unpatchify(tokens)),
            np.asarray(
                unpatchify_tokens(
                    tokens, grid, grid, TINY["patch_size"], out_channels
                )
            ),
        )

    def test_a_zero_dimension_raises(self) -> None:
        with pytest.raises(ValueError, match="grid_width must be a positive int"):
            unpatchify_tokens(np.zeros((1, 1, 4), "float32"), 1, 0, 2, 1)


# ---------------------------------------------------------------------
# The frozen positional table
# ---------------------------------------------------------------------


class TestThePosEmbedTableIsAFrozenWeight:
    """Non-trainable, materialized, and equal to the independent NumPy table."""

    @staticmethod
    def _find(model: DiT, collection: str):
        return [w for w in getattr(model, collection) if w.path.endswith("pos_embed")]

    def test_it_is_a_non_trainable_weight(self) -> None:
        model = tiny_model()
        x, t, y = tiny_inputs(seed=5)
        model([x, t, y], training=False)

        assert len(self._find(model, "non_trainable_weights")) == 1
        assert self._find(model, "trainable_weights") == []

    def test_its_value_equals_an_independently_computed_table(self) -> None:
        model = tiny_model()
        x, t, y = tiny_inputs(seed=6)
        model([x, t, y], training=False)

        grid = TINY["input_size"] // TINY["patch_size"]
        expected = np.asarray(
            get_2d_sincos_pos_embed(TINY["hidden_size"], grid), dtype="float32"
        ).reshape(1, grid * grid, TINY["hidden_size"])

        actual = np.asarray(self._find(model, "non_trainable_weights")[0])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0.0)

    def test_it_is_not_all_zeros(self) -> None:
        """The exact symptom of the ``add_weight().assign()`` failure mode."""
        model = tiny_model()
        x, t, y = tiny_inputs(seed=7)
        model([x, t, y], training=False)
        table = np.asarray(self._find(model, "non_trainable_weights")[0])
        assert float(np.max(np.abs(table))) > 0.0

    def test_it_survives_a_build_through_a_parents_call(self) -> None:
        """``StatelessScope`` is entered when a PARENT builds the child."""

        class Parent(keras.Model):
            def __init__(self, child: DiT, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.child = child

            def call(self, inputs, training=None):
                return self.child(inputs, training=training)

        child = tiny_model()
        parent = Parent(child)
        x, t, y = tiny_inputs(seed=8)
        parent([x, t, y], training=False)

        grid = TINY["input_size"] // TINY["patch_size"]
        expected = np.asarray(
            get_2d_sincos_pos_embed(TINY["hidden_size"], grid), dtype="float32"
        ).reshape(1, grid * grid, TINY["hidden_size"])
        actual = np.asarray(self._find(child, "non_trainable_weights")[0])
        assert float(np.max(np.abs(actual))) > 0.0
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0.0)


# ---------------------------------------------------------------------
# The label table's initializer
# ---------------------------------------------------------------------


class TestTheLabelTableUsesTheUpstreamInitializer:
    """``normal(std=0.02)``, not the house ``"uniform"`` default."""

    #: Enough rows that the dispersion estimate's standard error is ~1e-4.
    CONFIG: Dict[str, Any] = dict(TINY, hidden_size=64, num_heads=4, num_classes=500)

    def _table(self) -> np.ndarray:
        model = DiT(**self.CONFIG)
        x, t, y = tiny_inputs(seed=9, batch=2, config=self.CONFIG)
        model([x, t, y], training=False)
        weights = [
            w for w in model.weights if "y_embedder" in w.path
        ]
        assert len(weights) == 1, [w.path for w in weights]
        return np.asarray(weights[0])

    def test_the_table_dispersion_is_the_upstream_stddev(self) -> None:
        table = self._table()
        assert table.shape == (self.CONFIG["num_classes"] + 1, 64)
        # normal(0.02) -> 0.020; the house "uniform" default is
        # RandomUniform(-0.05, 0.05) -> 0.05/sqrt(3) = 0.0289.
        assert 0.019 < float(np.std(table)) < 0.021

    def test_the_table_has_a_tail_past_the_uniform_defaults_hard_bound(self) -> None:
        """A bounded distribution cannot produce a single sample past 0.05."""
        table = self._table()
        assert int(np.count_nonzero(np.abs(table) > 0.05)) > 0


# ---------------------------------------------------------------------
# The patch-embed initializer
# ---------------------------------------------------------------------


class TestThePatchEmbedInitializer:
    """The flattened-``Linear`` Xavier bound, not the Keras conv default."""

    def test_the_limit_is_computed_over_the_flattened_fan_out(self) -> None:
        init = flattened_linear_xavier(fan_in=2 * 2 * 4, fan_out=32)
        draws = np.asarray(init(shape=(4096,)))
        limit = float(np.sqrt(6.0 / (16 + 32)))
        assert float(np.max(np.abs(draws))) <= limit
        # And it is measurably wider than the Keras conv fan_out would give
        # (fan_out = p*p*D = 128 instead of D = 32).
        keras_limit = float(np.sqrt(6.0 / (16 + 128)))
        assert limit > keras_limit

    def test_the_kernel_respects_that_bound(self) -> None:
        model = tiny_model()
        x, t, y = tiny_inputs(seed=10)
        model([x, t, y], training=False)
        kernel = [
            w for w in model.weights
            if "x_embedder" in w.path and "kernel" in w.path
        ]
        assert len(kernel) == 1
        limit = float(
            np.sqrt(
                6.0
                / (
                    TINY["patch_size"] ** 2 * TINY["in_channels"]
                    + TINY["hidden_size"]
                )
            )
        )
        assert float(np.max(np.abs(np.asarray(kernel[0])))) <= limit

    def test_a_non_positive_fan_raises(self) -> None:
        with pytest.raises(ValueError, match="both fans must be positive"):
            flattened_linear_xavier(fan_in=0, fan_out=8)


# ---------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------


class TestFromVariant:
    """Construction from the named table, and the pretrained refusal."""

    # The two cheapest variants: the S scale at the two largest patch sizes.
    SMALLEST = ("DiT-S/8", "DiT-S/4")

    @pytest.mark.parametrize("variant", SMALLEST)
    def test_it_takes_its_four_numbers_from_the_registry(self, variant: str) -> None:
        row = DIT_VARIANTS[variant]
        model = DiT.from_variant(
            variant,
            input_size=row["patch_size"] * 2,
            in_channels=4,
            num_classes=8,
            frequency_embedding_size=16,
        )
        assert model.depth == row["depth"]
        assert model.hidden_size == row["hidden_size"]
        assert model.patch_size == row["patch_size"]
        assert model.num_heads == row["num_heads"]

    def test_a_variant_model_runs(self) -> None:
        model = DiT.from_variant(
            "DiT-S/8",
            input_size=8,
            in_channels=4,
            num_classes=8,
            depth=1,
            hidden_size=32,
            num_heads=4,
            frequency_embedding_size=16,
        )
        x = np.zeros((2, 8, 8, 4), "float32")
        out = np.asarray(
            model([x, np.zeros((2,), "float32"), np.zeros((2,), "int32")],
                  training=False)
        )
        assert out.shape == (2, 8, 8, 8)

    @pytest.mark.parametrize("variant", sorted(DIT_VARIANTS))
    def test_pretrained_true_raises_naming_the_variant(self, variant: str) -> None:
        with pytest.raises(NotImplementedError) as excinfo:
            DiT.from_variant(variant, pretrained=True)
        assert variant in str(excinfo.value)

    def test_an_unknown_variant_raises_listing_the_available_keys(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            DiT.from_variant("DiT-XXL/2")
        message = str(excinfo.value)
        for key in DIT_VARIANTS:
            assert key in message

    def test_create_dit_delegates_to_from_variant(self) -> None:
        model = create_dit(
            "dit_s_8",
            input_size=8,
            in_channels=4,
            num_classes=8,
            depth=1,
            hidden_size=32,
            num_heads=4,
            frequency_embedding_size=16,
        )
        assert isinstance(model, DiT)
        assert model.patch_size == DIT_VARIANTS["DiT-S/8"]["patch_size"]

    def test_create_dit_forwards_the_pretrained_refusal(self) -> None:
        with pytest.raises(NotImplementedError, match="DiT-S/2"):
            create_dit("DiT-S/2", pretrained=True)


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------


class TestSerialization:
    """``get_config`` completeness and a ``.keras`` round trip on VALUES."""

    def test_get_config_round_trips_every_constructor_argument(self) -> None:
        model = tiny_model()
        config = model.get_config()
        for key, value in TINY.items():
            assert config[key] == value
        rebuilt = DiT.from_config(config)
        assert rebuilt.get_config() == config

    def test_the_keras_round_trip_reproduces_the_output_exactly(self, tmp_path) -> None:
        model = tiny_model()
        x, t, y = tiny_inputs(seed=11)
        before = np.asarray(model([x, t, y], training=False))

        path = str(tmp_path / "dit.keras")
        model.save(path)
        loaded = keras.models.load_model(path)

        # Weight VALUES first, before the loaded model has ever been called.
        assert len(loaded.weights) == len(model.weights)
        for original, restored in zip(model.weights, loaded.weights):
            np.testing.assert_allclose(
                np.asarray(restored), np.asarray(original), rtol=0, atol=0.0
            )

        after = np.asarray(loaded([x, t, y], training=False))
        np.testing.assert_allclose(after, before, rtol=0, atol=1e-6)


# ---------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------


class TestTheParameterCount:
    """Measured on a built model, against a total derived from the config."""

    def test_the_tiny_configuration_matches_the_derived_total(self) -> None:
        model = tiny_model()
        x, t, y = tiny_inputs(seed=12)
        model([x, t, y], training=False)
        measured = sum(int(np.prod(w.shape)) for w in model.weights)
        assert measured == expected_parameter_count(TINY)

    def test_the_derivation_is_sensitive_to_depth(self) -> None:
        """Anti-vacuity: the formula must not be a constant in disguise."""
        model = tiny_model(depth=3)
        x, t, y = tiny_inputs(seed=13)
        model([x, t, y], training=False)
        measured = sum(int(np.prod(w.shape)) for w in model.weights)
        assert measured == expected_parameter_count(dict(TINY, depth=3))
        assert expected_parameter_count(dict(TINY, depth=3)) != (
            expected_parameter_count(TINY)
        )

    def test_a_real_variant_count_is_measured_not_pasted(self) -> None:
        """``DiT-S/8`` at the published latent geometry, both sides derived."""
        config = dict(
            input_size=32,
            patch_size=8,
            in_channels=4,
            hidden_size=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.0,
            class_dropout_rate=0.1,
            num_classes=1000,
            learn_sigma=True,
            frequency_embedding_size=256,
        )
        model = DiT(**config)
        x, t, y = tiny_inputs(seed=14, batch=1, config=config)
        model([x, t, y], training=False)
        measured = sum(int(np.prod(w.shape)) for w in model.weights)
        assert measured == expected_parameter_count(config)
