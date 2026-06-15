import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os


from dl_techniques.layers.embedding.class_token import ClassTokenPrepend


class TestClassTokenPrepend:
    """Behavioral test suite for the ClassTokenPrepend layer.

    Mirrors the class-based structure of ``test_positional_embedding.py``.
    The five load-bearing assertions are:
        (a) output shape ``(B, N+1, dim)`` from input ``(B, N, dim)``;
        (b) ``output[:, 0, :]`` equals the broadcast of ``cls_token[0, 0, :]``;
        (c) ``output[:, 1:, :]`` equals the input exactly (existing tokens unaltered);
        (d) ``get_config`` / ``from_config`` round-trip preserves ``initializer`` and
            produces numerically identical output after copying weights;
        (e) ``.keras`` save/load round-trip of a wrapping ``keras.Model`` yields
            bit-identical predictions.
    """

    @pytest.fixture
    def input_tensor(self) -> tf.Tensor:
        """Create a test input tensor of shape (batch, seq_len, dim)."""
        return tf.random.normal([4, 16, 32])  # batch=4, seq=16, dim=32

    @pytest.fixture
    def layer_instance(self) -> ClassTokenPrepend:
        """Create a default layer instance for testing."""
        return ClassTokenPrepend()

    def test_initialization_defaults(self):
        """Default initializer matches the DINO/ViT convention; weight is lazy."""
        layer = ClassTokenPrepend()
        assert layer.initializer == "truncated_normal"
        assert layer.cls_token is None

    def test_initialization_custom(self):
        """Custom initializer is retained verbatim."""
        layer = ClassTokenPrepend(initializer="zeros")
        assert layer.initializer == "zeros"

    def test_build_process(self, input_tensor: tf.Tensor):
        """The layer builds and creates the (1, 1, dim) class-token weight."""
        layer = ClassTokenPrepend()
        _ = layer(input_tensor)
        assert layer.built is True
        assert layer.cls_token is not None
        assert tuple(layer.cls_token.shape) == (1, 1, 32)

    def test_build_invalid_rank(self):
        """A non-3D input shape is rejected at build."""
        layer = ClassTokenPrepend()
        with pytest.raises(ValueError, match="expects a 3D input"):
            layer.build((4, 32))  # 2D

    def test_build_requires_static_dim(self):
        """A dynamic (None) feature dimension is rejected at build."""
        layer = ClassTokenPrepend()
        with pytest.raises(ValueError, match="static feature dimension"):
            layer.build((4, 16, None))

    # ----- (a) output shape -----
    def test_output_shape(self, input_tensor: tf.Tensor):
        """Output is (B, N+1, dim) from input (B, N, dim)."""
        layer = ClassTokenPrepend()
        out = layer(input_tensor)
        b, n, dim = input_tensor.shape
        assert tuple(out.shape) == (b, n + 1, dim)

        # compute_output_shape agrees with the forward pass
        cos = layer.compute_output_shape((b, n, dim))
        assert tuple(cos) == (b, n + 1, dim)

    # ----- (b) token@0 is the broadcast cls_token -----
    def test_prepended_token_is_cls_token(self, input_tensor: tf.Tensor):
        """output[:, 0, :] equals the broadcast of cls_token[0, 0, :] everywhere."""
        layer = ClassTokenPrepend()
        out = layer(input_tensor).numpy()
        cls_vec = layer.cls_token.numpy()[0, 0, :]  # (dim,)

        batch = input_tensor.shape[0]
        expected = np.broadcast_to(cls_vec, (batch, cls_vec.shape[0]))
        # position 0 of every batch element is exactly the (single) cls vector
        assert np.allclose(out[:, 0, :], expected, atol=1e-6)
        # and the vector is identical across the batch (true broadcast, not per-row)
        assert np.allclose(out[:, 0, :], out[0:1, 0, :], atol=1e-7)

    # ----- (c) existing tokens unaltered -----
    def test_existing_tokens_unchanged(self, input_tensor: tf.Tensor):
        """output[:, 1:, :] equals the input exactly (prepend doesn't alter tokens)."""
        layer = ClassTokenPrepend()
        out = layer(input_tensor).numpy()
        assert np.allclose(out[:, 1:, :], input_tensor.numpy(), atol=1e-7)

    # ----- (d) get_config / from_config round-trip -----
    def test_get_config_round_trip(self):
        """from_config preserves initializer and yields numerically identical output."""
        original = ClassTokenPrepend(initializer="glorot_uniform")
        input_shape = (None, 16, 32)
        original.build(input_shape)

        config = original.get_config()
        assert config["initializer"] == "glorot_uniform"

        recreated = ClassTokenPrepend.from_config(config)
        assert recreated.initializer == "glorot_uniform"

        recreated.build(input_shape)
        # Copy the single weight so outputs are bit-comparable.
        recreated.cls_token.assign(original.cls_token)

        test_input = tf.random.normal([2, 16, 32])
        orig_out = original(test_input, training=False).numpy()
        recr_out = recreated(test_input, training=False).numpy()
        assert np.allclose(orig_out, recr_out, atol=1e-7)

    # ----- (e) .keras model save/load round-trip -----
    def test_model_save_load(self, input_tensor: tf.Tensor):
        """A keras.Model wrapping the layer round-trips bit-identically."""
        inputs = keras.Input(shape=(16, 32))
        x = ClassTokenPrepend(name="cls_prepend")(inputs)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(5)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        original_prediction = model.predict(input_tensor, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "test_model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"ClassTokenPrepend": ClassTokenPrepend},
            )
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

        assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)
        assert isinstance(loaded_model.get_layer("cls_prepend"), ClassTokenPrepend)

    def test_batch_size_independence(self):
        """The prepended token is identical regardless of batch size."""
        layer = ClassTokenPrepend()
        _ = layer(tf.random.normal([1, 16, 32]))  # build
        cls_vec = layer.cls_token.numpy()[0, 0, :]

        for b in (1, 3, 8):
            out = layer(tf.random.normal([b, 16, 32])).numpy()
            assert out.shape == (b, 17, 32)
            assert np.allclose(out[:, 0, :], cls_vec, atol=1e-6)
