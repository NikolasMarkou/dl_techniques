import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os


from dl_techniques.layers.embedding.mask_token import MaskTokenApply


class TestMaskTokenApply:
    """Behavioral test suite for the MaskTokenApply layer.

    Mirrors ``test_class_token.py``. MaskTokenApply replaces masked positions of
    a ``(B, L, D)`` patch sequence with a single learnable ``(1, 1, D)`` mask
    token (iBOT / BEiT convention: ``True`` = masked, replace). The five
    load-bearing assertions are:
        (a) output shape ``(B, L, D)`` == input patch shape (unchanged);
        (b) where mask is True, output equals the broadcast learned mask token;
            where False, output equals the input patch exactly;
        (c) the ``mask_token`` weight is ``(1, 1, D)``, trainable, nonzero
            (truncated-normal init);
        (d) ``get_config`` / ``from_config`` round-trip preserves ``initializer``
            and yields numerically identical output after copying weights;
        (e) ``.keras`` save/load round-trip of a wrapping ``keras.Model`` yields
            bit-identical predictions.

    Call signature (confirmed from ``mask_token.py``): the layer takes a
    list/tuple of two tensors ``(patch_embeddings (B,L,D), mask (B,L) bool)``.
    """

    @pytest.fixture
    def patch_tensor(self) -> tf.Tensor:
        """Create a patch-embedding tensor of shape (batch, seq_len, dim)."""
        return tf.random.normal([4, 16, 32])  # batch=4, seq=16, dim=32

    @pytest.fixture
    def mask_tensor(self) -> tf.Tensor:
        """A mixed boolean mask of shape (batch, seq_len)."""
        rng = np.random.default_rng(0)
        return tf.constant(rng.random((4, 16)) > 0.5)

    @pytest.fixture
    def layer_instance(self) -> MaskTokenApply:
        """Create a default layer instance for testing."""
        return MaskTokenApply()

    def test_initialization_defaults(self):
        """Default initializer is the DINO/iBOT truncated-normal; weight is lazy."""
        layer = MaskTokenApply()
        # The bare-string default is resolved to a TruncatedNormal object.
        assert isinstance(layer.initializer, keras.initializers.TruncatedNormal)
        assert layer.mask_token is None

    def test_build_process(self, patch_tensor, mask_tensor):
        """The layer builds and creates the (1, 1, dim) mask-token weight."""
        layer = MaskTokenApply()
        _ = layer([patch_tensor, mask_tensor])
        assert layer.built is True
        assert layer.mask_token is not None
        assert tuple(layer.mask_token.shape) == (1, 1, 32)

    def test_build_invalid_arity(self):
        """A single (non-pair) input shape is rejected at build."""
        layer = MaskTokenApply()
        with pytest.raises(ValueError, match="expects two inputs"):
            layer.build((4, 16, 32))  # not a pair

    def test_build_invalid_rank(self):
        """Non-3D patch embeddings are rejected at build."""
        layer = MaskTokenApply()
        with pytest.raises(ValueError, match="expects 3D patch embeddings"):
            layer.build([(4, 32), (4, 16)])  # 2D embeddings

    def test_build_requires_static_dim(self):
        """A dynamic (None) feature dimension is rejected at build."""
        layer = MaskTokenApply()
        with pytest.raises(ValueError, match="static feature dimension"):
            layer.build([(4, 16, None), (4, 16)])

    # ----- (a) output shape == input patch shape -----
    def test_output_shape(self, patch_tensor, mask_tensor):
        """Output is (B, L, dim) == input patch shape (unchanged by masking)."""
        layer = MaskTokenApply()
        out = layer([patch_tensor, mask_tensor])
        b, n, dim = patch_tensor.shape
        assert tuple(out.shape) == (b, n, dim)

        # compute_output_shape agrees with the forward pass
        cos = layer.compute_output_shape([(b, n, dim), (b, n)])
        assert tuple(cos) == (b, n, dim)

    # ----- (b) masked -> mask_token; unmasked -> input exactly -----
    def test_per_position_selection(self, patch_tensor):
        """Masked positions equal the broadcast mask token; unmasked equal input.

        Uses a deterministic mixed mask so BOTH branches are exercised within the
        same forward, pinning per-position selection (not all-or-nothing).
        """
        layer = MaskTokenApply()
        # Build first so the mask_token weight exists.
        rng = np.random.default_rng(1)
        mask_np = rng.random((4, 16)) > 0.5
        # Guarantee at least one True and one False per row for a real mixed test.
        mask_np[:, 0] = True
        mask_np[:, 1] = False
        mask = tf.constant(mask_np)

        out = layer([patch_tensor, mask]).numpy()
        patches = patch_tensor.numpy()
        mask_vec = layer.mask_token.numpy()[0, 0, :]  # (dim,)

        masked = mask_np  # (B, L) bool
        # Where True -> the (broadcast) learned mask token.
        assert np.allclose(out[masked], mask_vec, atol=1e-6)
        # Where False -> the input patch, bit-for-bit.
        assert np.allclose(out[~masked], patches[~masked], atol=1e-7)

    # ----- (c) mask_token weight is (1,1,D), trainable, nonzero -----
    def test_mask_token_weight_properties(self, patch_tensor, mask_tensor):
        """mask_token is (1,1,D), trainable, and nonzero (truncated-normal init)."""
        layer = MaskTokenApply()
        _ = layer([patch_tensor, mask_tensor])
        assert tuple(layer.mask_token.shape) == (1, 1, 32)
        assert layer.mask_token.trainable is True
        # It is in the layer's trainable weights.
        assert any(w is layer.mask_token for w in layer.trainable_weights)
        # Truncated-normal init -> not the degenerate constant-zero vector.
        assert np.any(np.abs(layer.mask_token.numpy()) > 1e-8)

    # ----- (d) get_config / from_config round-trip -----
    def test_get_config_round_trip(self):
        """from_config preserves initializer and yields identical output."""
        original = MaskTokenApply(initializer="glorot_uniform")
        input_shape = [(None, 16, 32), (None, 16)]
        original.build(input_shape)

        config = original.get_config()
        recreated = MaskTokenApply.from_config(config)

        recreated.build(input_shape)
        # Copy the single weight so outputs are bit-comparable.
        recreated.mask_token.assign(original.mask_token)

        patches = tf.random.normal([2, 16, 32])
        mask = tf.constant(np.random.default_rng(2).random((2, 16)) > 0.5)
        orig_out = original([patches, mask], training=False).numpy()
        recr_out = recreated([patches, mask], training=False).numpy()
        assert np.allclose(orig_out, recr_out, atol=1e-7)

    # ----- (e) .keras model save/load round-trip -----
    def test_model_save_load(self, patch_tensor, mask_tensor):
        """A keras.Model wrapping the layer round-trips bit-identically."""
        patch_in = keras.Input(shape=(16, 32))
        mask_in = keras.Input(shape=(16,), dtype="bool")
        x = MaskTokenApply(name="mask_apply")([patch_in, mask_in])
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(5)(x)
        model = keras.Model(inputs=[patch_in, mask_in], outputs=outputs)

        original_prediction = model.predict([patch_tensor, mask_tensor], verbose=0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "test_model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"MaskTokenApply": MaskTokenApply},
            )
            loaded_prediction = loaded_model.predict(
                [patch_tensor, mask_tensor], verbose=0
            )

        assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)
        assert isinstance(loaded_model.get_layer("mask_apply"), MaskTokenApply)
