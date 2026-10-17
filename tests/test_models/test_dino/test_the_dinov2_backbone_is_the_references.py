"""The DINOv2 backbone keeps the reference's LayerScale, position table and layer hook.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 1.7. Each test
fails on the axis its defect moves; none needs the downloaded weights.

* D-134: the LayerScale gains are free parameters, as the reference's
  ``gamma``. About half of every official checkpoint's gains are negative; a
  ``non_neg`` constraint (``LayerScale``'s default, passed explicitly until
  this step) clips them all to 0 on the first optimizer step.
* D-135: with ``pos_embed_grid_size=M`` the stored square table is resampled
  to the input grid exactly as torch's ``upsample_bicubic2d`` at
  ``scale_factor = (n + offset) / M`` (or at the output size for offset 0),
  checked against the per-pixel loop of ``dinov2_numpy_reference``; the table
  is used as is when the grid equals ``M x M``; the gradient reaches the
  stored table.
* ``get_intermediate_layers(x, n)`` returns the last ``n`` block outputs of
  the backbone's own functional graph, each through the final norm, [CLS]
  and registers dropped.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.vision.dino.dino_v2 import (
    DINOv2PositionEmbedding,
    DINOv2VisionTransformer,
    torch_bicubic_resize_matrix,
)

from .dinov2_numpy_reference import torch_bicubic_2d


def _tiny(**kwargs):
    kwargs = dict(dict(image_size=56, patch_size=14, embed_dim=16, depth=6, num_heads=2,
                       init_values=1.0, pos_embed_grid_size=5), **kwargs)
    return DINOv2VisionTransformer(**kwargs)


def _images(n=2, size=56, seed=0):
    return np.random.default_rng(seed).normal(size=(n, size, size, 3)).astype("float32")


# --- D-134 -------------------------------------------------------------------


def test_negative_layerscale_gains_survive_an_optimizer_step():
    model = _tiny()
    gains = [b.ls1.gamma for b in model.transformer_blocks] + [b.ls2.gamma for b in model.transformer_blocks]
    for g in gains:
        g.assign(-np.linspace(0.5, 1.5, g.shape[0]).astype("float32"))
    optimizer = keras.optimizers.SGD(1e-4)
    x, masks = _images(), np.zeros((2, model.num_patches), bool)
    with tf.GradientTape() as tape:
        loss = keras.ops.mean(model([x, masks])["x_norm_patchtokens"] ** 2)
    optimizer.apply_gradients(zip(tape.gradient(loss, gains), gains))
    for g in gains:
        value = keras.ops.convert_to_numpy(g)
        assert (value < -0.4).all(), f"{g.path} was clipped: min {value.min()}, max {value.max()}"


# --- D-135 -------------------------------------------------------------------


@pytest.mark.parametrize("m,grid,offset", [
    (37, (18, 18), 0.1),     # 252 px, the gate's size
    (37, (16, 16), 0.0),     # 224 px, output-size convention
    (5, (3, 7), 0.1),        # non-square, up and down
    (5, (5, 9), 0.1),        # one axis equal to M is still resampled
])
def test_the_table_is_resampled_as_torch_does(m, grid, offset):
    layer = DINOv2PositionEmbedding(grid_size=m, output_grid=grid, interpolate_offset=offset)
    tokens = np.zeros((1, 1 + grid[0] * grid[1], 4), "float32")
    layer.build(tokens.shape)
    table = np.random.default_rng(0).normal(size=(1, 1 + m * m, 4)).astype("float32")
    layer.pos_embed.assign(table)
    got = keras.ops.convert_to_numpy(layer(tokens))[0]
    patch = table[0, 1:].reshape(m, m, 4).transpose(2, 0, 1).astype(np.float64)
    scales = ((grid[0] + offset) / m, (grid[1] + offset) / m) if offset else (None, None)
    want = torch_bicubic_2d(patch, grid[0], grid[1], *scales).transpose(1, 2, 0).reshape(-1, 4)
    np.testing.assert_array_equal(got[0], table[0, 0])
    np.testing.assert_allclose(got[1:], want, rtol=0, atol=1e-5)


def test_the_stored_grid_is_used_as_is():
    layer = DINOv2PositionEmbedding(grid_size=5, output_grid=(5, 5), interpolate_offset=0.1)
    tokens = np.random.default_rng(1).normal(size=(2, 26, 4)).astype("float32")
    layer.build(tokens.shape)
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(layer(tokens)),
        tokens + keras.ops.convert_to_numpy(layer.pos_embed))


def test_the_matrix_rows_are_torchs_taps():
    # Downsampling 8 -> 4 at the output-size scale: t = 0.5 everywhere,
    # torch's A = -0.75 taps (-0.09375, 0.59375, 0.59375, -0.09375), clamped at 0.
    matrix = torch_bicubic_resize_matrix(8, 4)
    np.testing.assert_allclose(matrix[0], [0.59375 - 0.09375, 0.59375, -0.09375, 0, 0, 0, 0, 0],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(matrix[1], [0, -0.09375, 0.59375, 0.59375, -0.09375, 0, 0, 0],
                               rtol=0, atol=1e-12)


def test_the_gradient_reaches_the_stored_table():
    layer = DINOv2PositionEmbedding(grid_size=5, output_grid=(3, 4))
    tokens = tf.zeros((1, 13, 4))
    layer.build(tokens.shape)
    with tf.GradientTape() as tape:
        out = keras.ops.sum(layer(tokens) * tf.random.normal((1, 13, 4), seed=0))
    grad = keras.ops.convert_to_numpy(tape.gradient(out, layer.pos_embed))[0]
    reached = np.abs(grad).sum(-1) > 0
    # The [CLS] row and more stored rows than there are outputs (each output
    # reads a 4 x 4 neighbourhood of the 5 x 5 grid).
    assert reached[0] and reached[1:].sum() > 12, reached


def test_the_backbone_round_trips_its_grid_and_table(tmp_path):
    model = _tiny()
    x, masks = _images(), np.zeros((2, 16), bool)
    before = keras.ops.convert_to_numpy(model([x, masks])["x_norm_patchtokens"])
    path = str(tmp_path / "dinov2.keras")
    model.save(path)
    loaded = keras.models.load_model(path, compile=False)
    assert loaded.pos_embed_grid_size == 5
    assert tuple(loaded.pos_embed.pos_embed.shape) == (1, 26, 16)
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(loaded([x, masks])["x_norm_patchtokens"]), before)


def test_antialiasing_is_refused_rather_than_ignored():
    with pytest.raises(ValueError, match="interpolate_antialias"):
        _tiny(interpolate_antialias=True)


# --- get_intermediate_layers -------------------------------------------------


@pytest.mark.parametrize("registers", [0, 2])
def test_intermediate_layers_are_the_graphs_last_blocks_through_the_norm(registers):
    model = _tiny(num_register_tokens=registers)
    x, masks = _images(), np.zeros((2, 16), bool)
    graph = keras.Model(model.inputs, [model.get_layer(f"block_{i}").output for i in range(6)])
    blocks = graph([x, masks])
    read = model.get_intermediate_layers(x, 4)
    assert len(read) == 4
    for got, block in zip(read, blocks[2:]):
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(got),
            keras.ops.convert_to_numpy(model.norm(block))[:, 1 + registers:])
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(read[-1]),
        keras.ops.convert_to_numpy(model([x, masks])["x_norm_patchtokens"]))
    with pytest.raises(ValueError, match="n must be in"):
        model.get_intermediate_layers(x, 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
