"""Mirrored tests for the SD3 MMDiT block + final layer.

Covers four configurations:
  (a) plain MMDiTBlock (context_pre_only=False, use_dual_attention=False)
  (b) dual-attention MMDiTBlock (use_dual_attention=True)
  (c) context_pre_only MMDiTBlock (single-tensor return)
  (d) MMDiTFinalLayer

For each: instantiation, forward shapes, compute_output_shape pre/post build,
get_config/from_config, and a .keras Functional round-trip @ atol 1e-6 under
variable batch + variable N_img/N_txt.
"""

import os
import tempfile
import zlib

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.sd3_mmdit.blocks import MMDiTBlock, MMDiTFinalLayer


# ---------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------

DIM = 64
NUM_HEADS = 4
N_IMG = 16
N_TXT = 7
BATCH = 2


@pytest.fixture
def img():
    return keras.random.normal((BATCH, N_IMG, DIM))


@pytest.fixture
def txt():
    return keras.random.normal((BATCH, N_TXT, DIM))


@pytest.fixture
def cond():
    return keras.random.normal((BATCH, DIM))


def _build_block_functional(block):
    """Wrap a (non-context_pre_only) block in a 3-input Functional model."""
    img_in = keras.Input(shape=(None, DIM), name="img")
    txt_in = keras.Input(shape=(None, DIM), name="txt")
    cond_in = keras.Input(shape=(DIM,), name="cond")
    h_out, e_out = block([img_in, txt_in, cond_in])
    return keras.Model([img_in, txt_in, cond_in], [h_out, e_out])


def _build_context_pre_only_functional(block):
    img_in = keras.Input(shape=(None, DIM), name="img")
    txt_in = keras.Input(shape=(None, DIM), name="txt")
    cond_in = keras.Input(shape=(DIM,), name="cond")
    h_out = block([img_in, txt_in, cond_in])
    return keras.Model([img_in, txt_in, cond_in], h_out)


def _build_final_functional(layer):
    h_in = keras.Input(shape=(None, DIM), name="img")
    cond_in = keras.Input(shape=(DIM,), name="cond")
    out = layer([h_in, cond_in])
    return keras.Model([h_in, cond_in], out)


# =====================================================================
# (a) plain block
# =====================================================================


class TestMMDiTBlockPlain:

    def test_instantiation(self):
        blk = MMDiTBlock(dim=DIM, num_heads=NUM_HEADS)
        assert blk.dim == DIM
        assert blk.attn2 is None
        assert blk.ff_context is not None
        assert blk.norm2_context is not None

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            MMDiTBlock(dim=63, num_heads=NUM_HEADS)  # not divisible
        with pytest.raises(ValueError):
            MMDiTBlock(dim=-1, num_heads=NUM_HEADS)
        with pytest.raises(ValueError):
            MMDiTBlock(dim=DIM, num_heads=NUM_HEADS, eps=0.0)
        with pytest.raises(ValueError):
            MMDiTBlock(dim=DIM, num_heads=NUM_HEADS, mlp_ratio=0.0)

    def test_forward_shapes(self, img, txt, cond):
        blk = MMDiTBlock(dim=DIM, num_heads=NUM_HEADS)
        h_out, e_out = blk([img, txt, cond])
        assert h_out.shape == (BATCH, N_IMG, DIM)
        assert e_out.shape == (BATCH, N_TXT, DIM)

    def test_compute_output_shape_pre_post_build(self, img, txt, cond):
        blk = MMDiTBlock(dim=DIM, num_heads=NUM_HEADS)
        shapes_in = [
            (None, N_IMG, DIM),
            (None, N_TXT, DIM),
            (None, DIM),
        ]
        pre = blk.compute_output_shape(shapes_in)
        assert pre == ((None, N_IMG, DIM), (None, N_TXT, DIM))
        blk([img, txt, cond])  # builds
        post = blk.compute_output_shape(shapes_in)
        assert post == pre

    def test_config_round_trip(self):
        blk = MMDiTBlock(
            dim=DIM, num_heads=NUM_HEADS, mlp_ratio=2.0, qk_norm=False
        )
        cfg = blk.get_config()
        blk2 = MMDiTBlock.from_config(cfg)
        assert blk2.dim == DIM
        assert blk2.mlp_ratio == 2.0
        assert blk2.qk_norm is False

    def test_variable_batch_and_seq(self, cond):
        blk = MMDiTBlock(dim=DIM, num_heads=NUM_HEADS)
        for b, n_img, n_txt in [(1, 5, 3), (3, 20, 11)]:
            h = keras.random.normal((b, n_img, DIM))
            e = keras.random.normal((b, n_txt, DIM))
            c = keras.random.normal((b, DIM))
            h_out, e_out = blk([h, e, c])
            assert h_out.shape == (b, n_img, DIM)
            assert e_out.shape == (b, n_txt, DIM)


# =====================================================================
# (b) dual-attention block
# =====================================================================


class TestMMDiTBlockDualAttention:

    def test_instantiation(self):
        blk = MMDiTBlock(
            dim=DIM, num_heads=NUM_HEADS, use_dual_attention=True
        )
        assert blk.attn2 is not None
        assert isinstance(blk.attn2, keras.layers.MultiHeadAttention)

    def test_forward_shapes(self, img, txt, cond):
        blk = MMDiTBlock(
            dim=DIM, num_heads=NUM_HEADS, use_dual_attention=True
        )
        h_out, e_out = blk([img, txt, cond])
        assert h_out.shape == (BATCH, N_IMG, DIM)
        assert e_out.shape == (BATCH, N_TXT, DIM)

    def test_config_round_trip(self):
        blk = MMDiTBlock(
            dim=DIM, num_heads=NUM_HEADS, use_dual_attention=True
        )
        blk2 = MMDiTBlock.from_config(blk.get_config())
        assert blk2.use_dual_attention is True
        assert blk2.attn2 is not None


# =====================================================================
# (c) context_pre_only block (single return)
# =====================================================================


class TestMMDiTBlockContextPreOnly:

    def test_instantiation(self):
        blk = MMDiTBlock(
            dim=DIM, num_heads=NUM_HEADS, context_pre_only=True
        )
        assert blk.ff_context is None
        assert blk.norm2_context is None
        assert blk.attn.context_pre_only is True

    def test_forward_single_return(self, img, txt, cond):
        blk = MMDiTBlock(
            dim=DIM, num_heads=NUM_HEADS, context_pre_only=True
        )
        out = blk([img, txt, cond])
        # Single tensor, not a pair.
        assert not isinstance(out, (list, tuple))
        assert out.shape == (BATCH, N_IMG, DIM)

    def test_compute_output_shape_single(self):
        blk = MMDiTBlock(
            dim=DIM, num_heads=NUM_HEADS, context_pre_only=True
        )
        out_shape = blk.compute_output_shape(
            [(None, N_IMG, DIM), (None, N_TXT, DIM), (None, DIM)]
        )
        assert out_shape == (None, N_IMG, DIM)

    def test_config_round_trip(self):
        blk = MMDiTBlock(
            dim=DIM, num_heads=NUM_HEADS, context_pre_only=True
        )
        blk2 = MMDiTBlock.from_config(blk.get_config())
        assert blk2.context_pre_only is True
        assert blk2.ff_context is None

    def test_the_final_block_builds_no_text_query(self, img, txt, cond):
        """The text stream gives keys and values only (D-043, GF-04)."""
        blk = MMDiTBlock(
            dim=DIM, num_heads=NUM_HEADS, context_pre_only=True
        )
        blk([img, txt, cond])
        assert blk.attn.add_q_proj is None
        assert blk.attn.norm_added_q is None
        paths = [w.path for w in blk.weights]
        assert not [p for p in paths if "add_q_proj" in p or "norm_added_q" in p]
        # The text keys/values are still built: the image queries read them.
        assert any("add_k_proj" in p for p in paths)
        assert any("add_v_proj" in p for p in paths)
        assert any("norm_added_k" in p for p in paths)


# ---------------------------------------------------------------------
# Final-block value pin (plan-2026-10-14-81938e81 step 10.1, D-043)
# ---------------------------------------------------------------------
#
# Until 2044e6987 the final (context_pre_only) block also built and applied a
# text query (`attn/add_q_proj`, `attn/norm_added_q`), attended with it and then
# discarded the text rows. Step 10.1 stopped building it. Each softmax row
# depends only on its own query, so the image output must not move.
#
# PINNED_FINAL_BLOCK_OUTPUT was computed by the code at 2044e6987 (the OLD
# layout, text query included) with exactly the fixture below, on CPU. Every
# weight is drawn from a RandomState seeded by the CRC32 of its path below the
# block, so the weights the two layouts share get identical values in both, and
# the old layout's three extra tensors got values of their own.

_PIN_DIM = 8
_PIN_HEADS = 2
_PIN_N_IMG = 3
_PIN_N_TXT = 2

PINNED_FINAL_BLOCK_OUTPUT = np.array(
    [
        2.016202449798584, 0.6109495162963867, -0.678856611251831,
        1.888751745223999, 2.173214912414551, -2.4482152462005615,
        1.0240740776062012, -0.35184246301651, -0.2795543670654297,
        0.5899345278739929, -1.1868754625320435, 1.037825107574463,
        0.8017204999923706, -0.5679530501365662, 0.4800978899002075,
        0.1414850652217865, 2.098407030105591, 0.17965167760849,
        -0.877736508846283, -1.4193127155303955, -2.893477201461792,
        -0.463049054145813, 1.9851731061935425, -0.8964107036590576,
    ],
    dtype="float32",
).reshape(1, _PIN_N_IMG, _PIN_DIM)


def _pin_inputs():
    rs = np.random.RandomState(0)
    img_ = rs.standard_normal((1, _PIN_N_IMG, _PIN_DIM)).astype("float32")
    txt_ = rs.standard_normal((1, _PIN_N_TXT, _PIN_DIM)).astype("float32")
    cond_ = rs.standard_normal((1, _PIN_DIM)).astype("float32")
    return img_, txt_, cond_


def _seed_weights_by_path(block):
    """Give every weight values drawn from a seed derived from its path."""
    for w in block.weights:
        rel = w.path.split("/", 1)[1]
        rng = np.random.RandomState(zlib.crc32(rel.encode()))
        w.assign((0.3 * rng.standard_normal(tuple(w.shape))).astype("float32"))


def _pinned_fixture_block_output(img_, txt_, cond_):
    blk = MMDiTBlock(
        dim=_PIN_DIM, num_heads=_PIN_HEADS, context_pre_only=True,
        name="final_block",
    )
    blk([img_, txt_, cond_])
    _seed_weights_by_path(blk)
    return keras.ops.convert_to_numpy(blk([img_, txt_, cond_]))


def test_the_final_block_image_output_matches_the_old_layout(
    golden_reference_device,
):
    """Dropping the text query leaves the image output where it was.

    Tolerance: the old layout ran the score matmul over N_img + N_txt query
    rows and the new one over N_img, so CPU float32 blocking may differ in
    the last ulps (measured at most 5.7e-6 at output scale 17 on a larger
    probe). 1e-5 absolute is below any real change: perturbing one text
    token moves this output by far more (asserted below), and a final block
    whose image queries stop reading the text keys and values misses the pin.
    """
    img_, txt_, cond_ = _pin_inputs()
    with keras.device(golden_reference_device):
        out = _pinned_fixture_block_output(img_, txt_, cond_)
        moved_txt = txt_.copy()
        # Not a constant shift: the text AdaLN's LayerNorm removes the
        # per-token mean, so `+= 1.0` leaves the block output unchanged.
        moved_txt[0, 0, :] += np.linspace(-2.0, 2.0, _PIN_DIM, dtype="float32")
        moved = _pinned_fixture_block_output(img_, moved_txt, cond_)

    np.testing.assert_allclose(out, PINNED_FINAL_BLOCK_OUTPUT, atol=1e-5, rtol=0)
    # The pin is not blind to the text path: text keys/values reach the image.
    assert np.abs(moved - PINNED_FINAL_BLOCK_OUTPUT).max() > 1e-2


# =====================================================================
# (d) MMDiTFinalLayer
# =====================================================================


class TestMMDiTFinalLayer:

    def test_instantiation(self):
        head = MMDiTFinalLayer(dim=DIM, out_channels=16)
        assert head.out_channels == 16

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            MMDiTFinalLayer(dim=DIM, out_channels=0)
        with pytest.raises(ValueError):
            MMDiTFinalLayer(dim=-1, out_channels=16)

    def test_forward_shape(self, img, cond):
        head = MMDiTFinalLayer(dim=DIM, out_channels=16)
        out = head([img, cond])
        assert out.shape == (BATCH, N_IMG, 16)

    def test_compute_output_shape(self):
        head = MMDiTFinalLayer(dim=DIM, out_channels=16)
        out_shape = head.compute_output_shape(
            [(None, N_IMG, DIM), (None, DIM)]
        )
        assert out_shape == (None, N_IMG, 16)

    def test_config_round_trip(self):
        head = MMDiTFinalLayer(dim=DIM, out_channels=16, eps=1e-5)
        head2 = MMDiTFinalLayer.from_config(head.get_config())
        assert head2.out_channels == 16
        assert head2.eps == 1e-5

    def test_variable_batch_and_seq(self):
        head = MMDiTFinalLayer(dim=DIM, out_channels=16)
        for b, n in [(1, 5), (3, 20)]:
            h = keras.random.normal((b, n, DIM))
            c = keras.random.normal((b, DIM))
            assert head([h, c]).shape == (b, n, 16)


# =====================================================================
# .keras round-trip (atol 1e-6) — exercises both dual + plain + final
# =====================================================================


class TestKerasRoundTrip:

    def _round_trip(self, model, inputs):
        outputs_before = model(inputs)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            outputs_after = reloaded(inputs)
        return outputs_before, outputs_after

    def _assert_close(self, before, after):
        if not isinstance(before, (list, tuple)):
            before, after = [before], [after]
        for b, a in zip(before, after):
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(b),
                keras.ops.convert_to_numpy(a),
                atol=1e-6,
            )

    def test_plain_block_round_trip(self, img, txt, cond):
        blk = MMDiTBlock(dim=DIM, num_heads=NUM_HEADS)
        model = _build_block_functional(blk)
        before, after = self._round_trip(model, [img, txt, cond])
        self._assert_close(before, after)

    def test_dual_attention_block_round_trip(self, img, txt, cond):
        blk = MMDiTBlock(
            dim=DIM, num_heads=NUM_HEADS, use_dual_attention=True
        )
        model = _build_block_functional(blk)
        before, after = self._round_trip(model, [img, txt, cond])
        self._assert_close(before, after)

    def test_context_pre_only_block_round_trip(self, img, txt, cond):
        blk = MMDiTBlock(
            dim=DIM, num_heads=NUM_HEADS, context_pre_only=True
        )
        model = _build_context_pre_only_functional(blk)
        before, after = self._round_trip(model, [img, txt, cond])
        self._assert_close(before, after)

    def test_final_layer_round_trip(self, img, cond):
        head = MMDiTFinalLayer(dim=DIM, out_channels=16)
        model = _build_final_functional(head)
        before, after = self._round_trip(model, [img, cond])
        self._assert_close(before, after)
