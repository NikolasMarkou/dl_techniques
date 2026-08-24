"""FastVLM stage 3 must be able to tell WHERE something is.

F-86 of the 2026-08-18 deep review, fixed under
``plan-2026-08-18T140459-7991552f/D-043`` (max_seq_len plumbing) and
``D-044`` (the default flip).

`AttentionBlockVLM` flattens ``(B, H, W, C) -> (B, H*W, C)`` and runs a
`TransformerLayer`. With ``attention_type='multi_head'`` that layer builds
`MultiHeadAttention` with NO RoPE and NO relative bias, and nothing in the block
or in `FastVLM` adds a positional embedding, so every stage-3 block was exactly
permutation-equivariant over the spatial grid: ``f(Px) == P f(x)`` for any
permutation `P` of the ``H*W`` positions.

MEASURED on CPU, 14x14 grid, dim=64, 4 heads, random permutation,
``max|f(Px) - P f(x)|``:

=============================  ==================  ==================
                               use_layer_scale=F   use_layer_scale=T
=============================  ==================  ==================
``'multi_head'``               5.36e-07            3.73e-09
``'group_query'`` (the fix)    5.48e-01            4.88e-05
``|y|max``                     4.03 - 4.45         4.03 - 4.45
=============================  ==================  ==================

TWO instrument traps are pinned as tests below, because either one makes the
fix look like a no-op:

1. The shipped classification head is `GlobalAveragePooling2D`, which is itself
   permutation-INVARIANT. A probe on the model's LOGITS cannot see this defect
   at all; it must read `include_top=False` features (or the block directly).
2. At the shipped ``layer_scale_init=1e-4`` the block is ~identity at init, so
   the SAME positional signal is attenuated by 1e-4 to 4.88e-05 -- present, but
   easy to mistake for float noise. Probe with ``use_layer_scale=False``.

THE FIX IS A WEIGHT-PATH CHANGE. `'multi_head'` builds
``attention/cross_attention/{qkv,proj}/{kernel,bias}`` (4 tensors, fused QKV);
`'group_query'` builds ``attention/{w_q,w_k,w_v,w_o}/{kernel,bias}`` plus RoPE
tables. `FastVLM.get_config` serializes `attention_type`, so a `.keras` file
saved before 2026-08-19 carries ``'multi_head'`` and still loads correctly; a
bare `load_weights` into a default-constructed model does NOT.
"""

import numpy as np
import keras
from keras import ops
import pytest

from dl_techniques.models.fastvlm.components import AttentionBlockVLM
from dl_techniques.models.fastvlm.model import FastVLM


DIM, HEADS, GRID, BATCH = 64, 4, 14, 2


def _block(attention_type, use_layer_scale=False, **kwargs):
    keras.utils.set_random_seed(0)
    return AttentionBlockVLM(
        dim=DIM,
        num_heads=HEADS,
        attention_type=attention_type,
        use_layer_scale=use_layer_scale,
        **kwargs,
    )


def _equivariance_delta(layer, seed=0):
    """``max|f(Px) - P f(x)|`` over a random spatial permutation, and ``|y|max``."""
    rng = np.random.RandomState(seed)
    x_np = rng.randn(BATCH, GRID, GRID, DIM).astype("float32")
    y = ops.convert_to_numpy(layer(ops.convert_to_tensor(x_np)))

    perm = rng.permutation(GRID * GRID)
    x_perm = x_np.reshape(BATCH, GRID * GRID, DIM)[:, perm]
    y_perm = ops.convert_to_numpy(
        layer(ops.convert_to_tensor(x_perm.reshape(BATCH, GRID, GRID, DIM)))
    ).reshape(BATCH, GRID * GRID, DIM)

    reference = y.reshape(BATCH, GRID * GRID, DIM)[:, perm]
    return float(np.abs(y_perm - reference).max()), float(np.abs(y).max())


class TestStage3BlockIsPositionAware:
    def test_default_block_is_not_permutation_equivariant(self):
        delta, scale = _equivariance_delta(_block("group_query"))
        assert delta > 1e-2 * scale, (
            f"stage-3 block is permutation-equivariant (max|f(Px) - Pf(x)| = "
            f"{delta:.3e} on |y|max = {scale:.3e}); it carries no positional "
            f"information"
        )

    def test_multi_head_really_was_exactly_equivariant(self):
        """The defect, still reachable explicitly. This is the RED proof."""
        delta, scale = _equivariance_delta(_block("multi_head"))
        # 1e-3 relative, NOT tighter: on the GPU this measures 4.12e-04 on
        # |y|max 5.03 (TF32's ~1e-4 relative floor) against 5.36e-07 on CPU.
        # `'group_query'` sits at 1.1e-01 relative, a 100x margin either way.
        assert delta < 1e-3 * scale, (
            f"'multi_head' is no longer exactly permutation-equivariant "
            f"({delta:.3e} on {scale:.3e}) -- something now adds position to it, "
            f"and this file's premise needs re-measuring"
        )


class TestTheTwoInstrumentTraps:
    def test_layer_scale_attenuates_the_signal_by_four_orders(self):
        """Trap 2: at `layer_scale_init=1e-4` the block is ~identity at init."""
        loud, loud_scale = _equivariance_delta(
            _block("group_query", use_layer_scale=False)
        )
        quiet, quiet_scale = _equivariance_delta(
            _block("group_query", use_layer_scale=True, layer_scale_init=1e-4)
        )
        assert loud > 1e-2 * loud_scale
        assert quiet > 0.0
        assert quiet < loud / 100.0, (
            f"expected LayerScale to attenuate the positional delta by orders "
            f"of magnitude (loud={loud:.3e}, quiet={quiet:.3e})"
        )

    def test_global_average_pooling_head_cannot_see_position(self):
        """Trap 1: the shipped head is permutation-INVARIANT by construction."""
        rng = np.random.RandomState(0)
        feats = rng.randn(BATCH, GRID, GRID, DIM).astype("float32")
        pool = keras.layers.GlobalAveragePooling2D()

        a = ops.convert_to_numpy(pool(ops.convert_to_tensor(feats)))
        perm = rng.permutation(GRID * GRID)
        shuffled = feats.reshape(BATCH, GRID * GRID, DIM)[:, perm]
        b = ops.convert_to_numpy(
            pool(ops.convert_to_tensor(shuffled.reshape(BATCH, GRID, GRID, DIM)))
        )
        np.testing.assert_allclose(a, b, atol=1e-5)


class TestAssembledModelDefaults:
    @staticmethod
    def _tiny(**kwargs):
        keras.utils.set_random_seed(0)
        return FastVLM(
            num_classes=0,
            include_top=False,
            embed_dims=[16, 32, 64],
            depths=[1, 1, 1],
            num_heads=[2, 2, 4],
            input_shape=(64, 64, 3),
            **kwargs,
        )

    def test_default_attention_type_is_group_query(self):
        assert FastVLM.__init__.__defaults__ is not None
        model = self._tiny()
        assert model.attention_type == "group_query"

    def test_num_kv_heads_equals_num_heads(self):
        """`'group_query'` here is MHA arithmetic plus RoPE, not a capacity cut."""
        model = self._tiny(use_layer_scale=False)
        model(ops.convert_to_tensor(np.zeros((1, 64, 64, 3), dtype="float32")))
        block = model.stages[2].layers[0]
        attention = block.transformer.attention
        assert attention.num_kv_heads == attention.num_heads == 4

    def test_stage3_of_the_assembled_model_is_position_aware(self):
        """Read the STAGE, not the logits -- the head would hide this."""
        model = self._tiny(use_layer_scale=False)
        model(ops.convert_to_tensor(np.zeros((1, 64, 64, 3), dtype="float32")))
        stage3 = model.stages[2]

        rng = np.random.RandomState(0)
        side = 4  # 64 / 16
        x = rng.randn(BATCH, side, side, 64).astype("float32")
        y = ops.convert_to_numpy(stage3(ops.convert_to_tensor(x)))

        perm = rng.permutation(side * side)
        xp = x.reshape(BATCH, side * side, 64)[:, perm]
        yp = ops.convert_to_numpy(
            stage3(ops.convert_to_tensor(xp.reshape(BATCH, side, side, 64)))
        ).reshape(BATCH, side * side, 64)
        reference = y.reshape(BATCH, side * side, 64)[:, perm]

        delta = np.abs(yp - reference).max()
        assert delta > 1e-2 * np.abs(y).max(), (
            f"assembled stage 3 is still permutation-equivariant ({delta:.3e})"
        )

    def test_weight_paths_changed_and_this_is_deliberate(self):
        """The default flip renames the attention subtree. Pin both shapes."""
        default = self._tiny()
        legacy = self._tiny(attention_type="multi_head")
        probe = ops.convert_to_tensor(np.zeros((1, 64, 64, 3), dtype="float32"))
        default(probe)
        legacy(probe)

        def attention_paths(model):
            return {
                w.path.split("vision_transformer/")[-1]
                for w in model.weights
                if "stage3" in w.path and "/attention/" in w.path
            }

        legacy_paths = attention_paths(legacy)
        default_paths = attention_paths(default)

        assert "attention/cross_attention/qkv/kernel" in legacy_paths
        assert "attention/w_q/kernel" in default_paths
        assert not (legacy_paths & default_paths), (
            "the two attention types now share weight paths -- the "
            "'weight-path change' warning in this module's docstring, in "
            "FastVLM's docstring and in decisions D-044 is stale"
        )


class TestRopeTableCeiling:
    def test_max_seq_len_is_forwarded_to_the_attention(self):
        block = _block("group_query", max_seq_len=256)
        block.build((None, 8, 8, DIM))
        assert block.transformer.attention.max_seq_len == 256

    def test_max_seq_len_is_not_forwarded_to_multi_head(self):
        """`MultiHeadAttention` does not declare the key and the factory RAISES
        on undeclared keys since 2026-08-17, so it must not be passed."""
        block = _block("multi_head", max_seq_len=256)
        block.build((None, 8, 8, DIM))  # must not raise
        assert block.transformer.attention_args == {}

    def test_grid_larger_than_the_table_fails_loudly(self):
        """The ceiling is enforced at CALL time by `RotaryPositionEmbedding`,
        not at build; either way it is a loud ValueError naming the knob."""
        block = _block("group_query", max_seq_len=16)
        x = ops.convert_to_tensor(
            np.zeros((1, GRID, GRID, DIM), dtype="float32")
        )
        with pytest.raises(ValueError, match="max_seq_len"):
            block(x)  # 196 > 16

    def test_max_seq_len_survives_serialization(self):
        block = _block("group_query", max_seq_len=512)
        assert block.get_config()["max_seq_len"] == 512
        assert AttentionBlockVLM.from_config(block.get_config()).max_seq_len == 512

    def test_model_level_knob_is_serialized(self):
        model = FastVLM(
            num_classes=0, include_top=False, embed_dims=[16, 32, 64],
            depths=[1, 1, 1], num_heads=[2, 2, 4], input_shape=(64, 64, 3),
            attention_max_seq_len=512,
        )
        assert model.get_config()["attention_max_seq_len"] == 512
        assert model.stages[2].layers[0].max_seq_len == 512
