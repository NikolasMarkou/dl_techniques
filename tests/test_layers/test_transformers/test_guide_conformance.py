"""
Keras guide conformance tests for transformer layers fixed in plan_2026-06-13_250487cb.
Covers: EomtTransformer, BinaryMapper, FreeTransformerLayer, PFTBlock.
"""
import numpy as np
import pytest
import keras
from keras import ops
import tempfile
import os


class TestEomtTransformerConformance:
    def test_current_step_is_keras_variable(self):
        from dl_techniques.layers.transformers.eomt_transformer import EomtTransformer
        layer = EomtTransformer(hidden_size=64, num_heads=4)
        x = ops.zeros((1, 16, 64))
        layer(x, training=True)  # trigger build
        assert isinstance(layer.current_step, keras.Variable), (
            f"current_step must be keras.Variable, got {type(layer.current_step)}"
        )

    def test_current_step_increments(self):
        from dl_techniques.layers.transformers.eomt_transformer import EomtTransformer
        # mask_annealing_steps > 0 is required for the increment guard to fire
        layer = EomtTransformer(hidden_size=64, num_heads=4, mask_annealing_steps=100)
        x = ops.zeros((1, 16, 64))
        layer(x, training=True)
        val_before = float(layer.current_step)
        layer(x, training=True)
        val_after = float(layer.current_step)
        assert val_after == val_before + 1.0

    def test_compute_output_shape_matches_forward(self):
        from dl_techniques.layers.transformers.eomt_transformer import EomtTransformer
        layer = EomtTransformer(hidden_size=64, num_heads=4)
        x = np.zeros((2, 16, 64), dtype="float32")
        out = layer(x, training=False)
        cos = layer.compute_output_shape((2, 16, 64))
        assert tuple(out.shape) == tuple(cos), f"shape mismatch: {out.shape} vs {cos}"

    def test_round_trip(self):
        from dl_techniques.layers.transformers.eomt_transformer import EomtTransformer
        layer = EomtTransformer(hidden_size=64, num_heads=4)
        inp = keras.Input((16, 64))
        model = keras.Model(inp, layer(inp))
        x = np.zeros((1, 16, 64), dtype="float32")
        out_a = model(x, training=False)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "eomt.keras")
            model.save(path)
            m2 = keras.models.load_model(path)
        out_b = m2(x, training=False)
        assert np.allclose(out_a, out_b, atol=1e-5), "EomtTransformer round-trip failed"


class TestBinaryMapperConformance:
    def test_no_tf_import(self):
        import ast, pathlib
        src = pathlib.Path("src/dl_techniques/layers/transformers/free_transformer.py").read_text()
        assert "import tensorflow" not in src, "tf import still present"

    def test_forward_inference(self):
        from dl_techniques.layers.transformers.free_transformer import BinaryMapper
        # Determine param name from signature
        import inspect
        sig = inspect.signature(BinaryMapper.__init__)
        param = "num_bits" if "num_bits" in sig.parameters else "num_latent_bits"
        layer = BinaryMapper(**{param: 4})
        x = ops.zeros((2, 8, 4))
        out = layer(x, training=False)
        assert out is not None

    def test_compute_output_shape_matches_forward(self):
        from dl_techniques.layers.transformers.free_transformer import BinaryMapper
        import inspect
        sig = inspect.signature(BinaryMapper.__init__)
        param = "num_bits" if "num_bits" in sig.parameters else "num_latent_bits"
        layer = BinaryMapper(**{param: 4})
        x = np.zeros((2, 8, 4), dtype="float32")
        out = layer(x, training=False)
        cos = layer.compute_output_shape((2, 8, 4))
        assert tuple(out.shape) == tuple(cos), f"shape mismatch: {out.shape} vs {cos}"


class TestFreeTransformerLayerConformance:
    def test_compute_output_shape_matches_forward(self):
        from dl_techniques.layers.transformers.free_transformer import FreeTransformerLayer
        layer = FreeTransformerLayer(hidden_size=64, num_heads=4, intermediate_size=256)
        x = np.zeros((1, 16, 64), dtype="float32")
        out = layer(x, training=False)
        cos = layer.compute_output_shape((1, 16, 64))
        assert tuple(out.shape) == tuple(cos), f"shape mismatch: {out.shape} vs {cos}"

    def test_round_trip(self):
        from dl_techniques.layers.transformers.free_transformer import FreeTransformerLayer
        layer = FreeTransformerLayer(hidden_size=64, num_heads=4, intermediate_size=256)
        inp = keras.Input((16, 64))
        model = keras.Model(inp, layer(inp, training=False))
        x = np.zeros((1, 16, 64), dtype="float32")
        out_a = model(x, training=False)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "free_transformer.keras")
            model.save(path)
            m2 = keras.models.load_model(path)
        out_b = m2(x, training=False)
        assert np.allclose(out_a, out_b, atol=1e-5), "FreeTransformerLayer round-trip failed"


class TestPFTBlockConformance:
    def test_compute_output_shape_matches_forward(self):
        from dl_techniques.layers.transformers.progressive_focused_transformer import PFTBlock
        layer = PFTBlock(dim=64, num_heads=4, window_size=4)
        x = np.zeros((1, 8, 8, 64), dtype="float32")
        out = layer(x, training=False)
        # PFTBlock may return a tuple; take first element for shape check
        out_tensor = out[0] if isinstance(out, (list, tuple)) else out
        cos = layer.compute_output_shape((1, 8, 8, 64))
        cos_primary = cos[0] if isinstance(cos, (list, tuple)) else cos
        assert tuple(out_tensor.shape) == tuple(cos_primary), (
            f"shape mismatch: {out_tensor.shape} vs {cos_primary}"
        )

    def test_forward_training(self):
        from dl_techniques.layers.transformers.progressive_focused_transformer import PFTBlock
        layer = PFTBlock(dim=64, num_heads=4, window_size=4)
        x = np.zeros((1, 8, 8, 64), dtype="float32")
        out = layer(x, training=True)
        assert out is not None
