"""Tests for the RouterLayer (dynamic conditional computation)."""

import os
import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.layers.transformers.transformer import TransformerLayer
from dl_techniques.layers.router import RouterLayer

B, SEQ, HID = 2, 12, 16


def _transformer():
    return TransformerLayer(hidden_size=HID, num_heads=2, intermediate_size=32)


@pytest.fixture
def sample():
    return np.random.default_rng(0).standard_normal((B, SEQ, HID)).astype("float32")


@pytest.fixture
def sample3():
    """Batch of 3 for per-item teacher-forcing selection tests."""
    return np.random.default_rng(1).standard_normal((3, SEQ, HID)).astype("float32")


class TestRouterLayer:

    def test_construction(self):
        layer = RouterLayer(transformer_layer=_transformer(), num_windows=4)
        assert layer.num_windows == 4

    def test_invalid_transformer_type(self):
        with pytest.raises(TypeError):
            RouterLayer(transformer_layer="not-a-transformer")

    @pytest.mark.parametrize("bad", [
        {"router_bottleneck_dim": 0},
        {"num_windows": 0},
    ])
    def test_invalid_args_raise(self, bad):
        with pytest.raises(ValueError):
            RouterLayer(transformer_layer=_transformer(), **bad)

    def test_forward_pass(self, sample):
        layer = RouterLayer(transformer_layer=_transformer(), router_bottleneck_dim=8, num_windows=4)
        out, logits = layer(sample)
        assert tuple(out.shape) == (B, SEQ, HID)
        assert tuple(logits.shape) == (B, 3)

    def test_compute_output_shape(self):
        layer = RouterLayer(transformer_layer=_transformer(), num_windows=4)
        out_shape, logits_shape = layer.compute_output_shape((B, SEQ, HID))
        assert out_shape == (B, SEQ, HID)
        assert logits_shape == (B, 3)

    def test_serialization_round_trip(self, sample, tmp_path):
        inp = keras.Input(shape=(SEQ, HID))
        out, logits = RouterLayer(
            transformer_layer=_transformer(), router_bottleneck_dim=8,
            num_windows=4, name="router",
        )(inp)
        model = keras.Model(inp, [out, logits])
        y0, l0 = model(sample)
        path = os.path.join(tmp_path, "router.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        y1, l1 = loaded(sample)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(l0), keras.ops.convert_to_numpy(l1),
            rtol=1e-5, atol=1e-5,
        )

    # ------------------------------------------------------------------
    # 1. Teacher-forcing selection correctness
    # ------------------------------------------------------------------
    def test_teacher_forcing_selects_per_item(self, sample3):
        """layer_decision=[0,1,2] must select SKIP / EXECUTE / REPEAT per item.

        RED proof: references are computed from the SAME wrapped
        transformer_layer. If the one-hot select picked the wrong slice for
        any item (or always the same path), the per-item asserts below would
        fail — e.g. item 0 (SKIP) would equal EXECUTE(inputs)[0] != inputs[0].
        """
        layer = RouterLayer(
            transformer_layer=_transformer(), router_bottleneck_dim=8, num_windows=4
        )
        decision = np.array([0, 1, 2], dtype="int32")
        final_output, _ = layer(sample3, layer_decision=decision, training=False)
        final_output = ops.convert_to_numpy(final_output)

        # Independent references from the wrapped transformer (same weights).
        execute = ops.convert_to_numpy(
            layer.transformer_layer(sample3, training=False)
        )
        repeat = ops.convert_to_numpy(
            layer.transformer_layer(
                layer.transformer_layer(sample3, training=False), training=False
            )
        )

        # item 0 -> SKIP == inputs
        np.testing.assert_allclose(final_output[0], sample3[0], atol=1e-5)
        # item 1 -> EXECUTE == transformer(inputs)
        np.testing.assert_allclose(final_output[1], execute[1], atol=1e-5)
        # item 2 -> REPEAT == transformer(transformer(inputs))
        np.testing.assert_allclose(final_output[2], repeat[2], atol=1e-5)

    # ------------------------------------------------------------------
    # 2. Mask semantics
    # ------------------------------------------------------------------
    def test_mask_semantics(self, sample):
        """Mask must change the routing summary; all-ones == None; rank-3 works.

        RED proof (a): a mask-blind pooling would produce IDENTICAL logits for
        the half-masked and all-ones masks (probed: real diff ~0.29), so the
        not-allclose assertion fails against blind code. (b) is a byte-identity
        invariant that guards the None path from regressing. (c) pins the
        rank-reduction: a rank-3 mask broadcast over queries must reduce to the
        same per-key keep vector as the 2D mask (logits equal).
        """
        layer = RouterLayer(
            transformer_layer=_transformer(), router_bottleneck_dim=8, num_windows=4
        )
        _ = layer(sample, training=False)  # build

        ones = np.ones((B, SEQ), dtype="float32")
        half = np.ones((B, SEQ), dtype="float32")
        half[:, SEQ // 2:] = 0.0

        _, l_none = layer(sample, attention_mask=None, training=False)
        _, l_ones = layer(sample, attention_mask=ones, training=False)
        _, l_half = layer(sample, attention_mask=half, training=False)
        o_none, _ = layer(sample, attention_mask=None, training=False)
        o_ones, _ = layer(sample, attention_mask=ones, training=False)

        l_none = ops.convert_to_numpy(l_none)
        l_ones = ops.convert_to_numpy(l_ones)
        l_half = ops.convert_to_numpy(l_half)

        # (a) masking real tokens changes the routing summary/logits.
        assert not np.allclose(l_half, l_ones, atol=1e-4), (
            "half-masked logits equal all-ones logits: pooling ignores the mask"
        )

        # (b) all-ones mask is byte-identical to attention_mask=None.
        np.testing.assert_allclose(l_ones, l_none, atol=1e-6)
        np.testing.assert_allclose(
            ops.convert_to_numpy(o_ones), ops.convert_to_numpy(o_none), atol=1e-6
        )

        # (c) rank-3 (B, S, S) mask broadcast over queries == 2D keep pattern.
        mask3 = np.broadcast_to(
            half[:, None, :], (B, SEQ, SEQ)
        ).astype("float32").copy()
        _, l_3d = layer(sample, attention_mask=mask3, training=False)
        np.testing.assert_allclose(
            ops.convert_to_numpy(l_3d), l_half, atol=1e-6
        )

    # ------------------------------------------------------------------
    # 3. Non-divisible forward (seq_len % num_windows != 0)
    # ------------------------------------------------------------------
    def test_non_divisible_forward(self, sample):
        """seq_len=12, num_windows=8 (window_len=1, 4 trailing tokens dropped
        from the summary): forward succeeds with correct shapes."""
        layer = RouterLayer(transformer_layer=_transformer(), num_windows=8)
        out, logits = layer(sample, training=False)
        assert tuple(out.shape) == (B, SEQ, HID)
        assert tuple(logits.shape) == (B, 3)

    # ------------------------------------------------------------------
    # 4. Static-shape / jit forward
    # ------------------------------------------------------------------
    def test_static_shape_functional_and_jit(self, sample):
        """Functional model with a static seq dim runs; conditional jit smoke;
        symbolic (None, HID) input exercises the dynamic fallback branch."""
        inp = keras.Input(shape=(SEQ, HID))
        out, logits = RouterLayer(
            transformer_layer=_transformer(), router_bottleneck_dim=8, num_windows=4
        )(inp)
        model = keras.Model(inp, [out, logits])

        y, l = model.predict(sample, verbose=0)
        assert tuple(y.shape) == (B, SEQ, HID)
        assert tuple(l.shape) == (B, 3)

        # Conditional jit smoke: skip if the backend/hardware rejects XLA.
        try:
            model.compile(jit_compile=True)
            model.predict(sample, verbose=0)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"jit_compile unsupported on this backend/hardware: {exc}")

    def test_symbolic_dynamic_seq(self, sample):
        """keras.Input(shape=(None, HID)) builds and runs (dynamic path)."""
        inp = keras.Input(shape=(None, HID))
        out, logits = RouterLayer(
            transformer_layer=_transformer(), router_bottleneck_dim=8, num_windows=4
        )(inp)
        model = keras.Model(inp, [out, logits])
        y, l = model.predict(sample, verbose=0)
        assert tuple(y.shape) == (B, SEQ, HID)
        assert tuple(l.shape) == (B, 3)

    # ------------------------------------------------------------------
    # 5. Gradient flow
    # ------------------------------------------------------------------
    def test_gradient_flow_transformer_via_path_router_via_logits(self, sample3):
        """Task loss reaches the transformer via the selected path but NOT the
        router MLP (non-differentiable argmax select); the router MLP trains
        ONLY through a loss on the returned logits.

        RED proof: kernel_regularizer=None removes the only non-task path to
        the router weights, so in (a) a router gradient that is anything other
        than None-or-exactly-zero would signal a leak from task loss into the
        router. In (b) the same weights DO get non-zero gradient once the loss
        is placed on logits — proving the tape/vars are wired and the (a) zero
        is real, not a dead tape.
        """
        layer = RouterLayer(
            transformer_layer=_transformer(), router_bottleneck_dim=8,
            num_windows=4, kernel_regularizer=None,
        )
        _ = layer(sample3, training=False)  # build to populate variables

        router_vars = (
            layer.router_bottleneck.trainable_variables
            + layer.router_output.trainable_variables
        )
        tf_vars = layer.transformer_layer.trainable_variables
        assert len(router_vars) > 0 and len(tf_vars) > 0

        decision = np.ones((3,), dtype="int32")  # force EXECUTE for every item

        # (a) loss on final_output.
        with tf.GradientTape(persistent=True) as tape:
            out, _ = layer(sample3, layer_decision=decision, training=True)
            loss_out = ops.mean(ops.square(out))
        tf_grads = tape.gradient(loss_out, tf_vars)
        router_grads_from_out = tape.gradient(loss_out, router_vars)
        del tape

        # Transformer weights receive real gradient via the EXECUTE path.
        assert any(
            g is not None and float(ops.max(ops.abs(g))) > 1e-8 for g in tf_grads
        ), "transformer received no gradient from the selected EXECUTE path"

        # Router MLP weights receive NO task-loss gradient (None or exactly 0).
        for g in router_grads_from_out:
            assert g is None or float(ops.max(ops.abs(g))) == 0.0, (
                "task loss leaked a non-zero gradient into the router MLP"
            )

        # (b) loss on logits -> router MLP gets real gradient.
        with tf.GradientTape() as tape:
            _, logits = layer(sample3, training=True)
            loss_logits = ops.mean(ops.square(logits))
        router_grads_from_logits = tape.gradient(loss_logits, router_vars)
        assert any(
            g is not None and float(ops.max(ops.abs(g))) > 1e-8
            for g in router_grads_from_logits
        ), "router MLP received no gradient from a loss on logits"
