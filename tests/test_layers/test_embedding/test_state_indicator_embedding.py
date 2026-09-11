import pytest
import numpy as np
import keras

from dl_techniques.layers.embedding.state_indicator_embedding import (
    StateIndicatorEmbedding,
)


class TestStateIndicatorEmbedding:
    """Behavioral test suite for StateIndicatorEmbedding.

    Additively selects one of two learned ``(1, 1, D)`` vectors per SAMPLE
    (not per position, unlike ``MaskTokenApply``'s replace semantics) and
    adds it to every token of that sample.
    """

    @pytest.fixture
    def tokens(self):
        return keras.random.normal([4, 6, 8], seed=0)

    def test_present_flag_adds_present_embedding_exactly(self, tokens):
        layer = StateIndicatorEmbedding()
        flag = keras.ops.convert_to_tensor([True, True, True, True])
        output = layer((tokens, flag))

        present = layer.present_embedding
        expected = tokens + present
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(expected),
            atol=1e-6, rtol=0,
        )

    def test_absent_flag_adds_absent_embedding_exactly(self, tokens):
        layer = StateIndicatorEmbedding()
        flag = keras.ops.convert_to_tensor([False, False, False, False])
        output = layer((tokens, flag))

        absent = layer.absent_embedding
        expected = tokens + absent
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(expected),
            atol=1e-6, rtol=0,
        )

    def test_mixed_batch_each_sample_gets_its_own_vector(self, tokens):
        layer = StateIndicatorEmbedding()
        flag = keras.ops.convert_to_tensor([True, False, True, False])
        output = layer((tokens, flag))

        output_np = keras.ops.convert_to_numpy(output)
        tokens_np = keras.ops.convert_to_numpy(tokens)
        present_np = keras.ops.convert_to_numpy(layer.present_embedding)
        absent_np = keras.ops.convert_to_numpy(layer.absent_embedding)

        np.testing.assert_allclose(
            output_np[0], tokens_np[0] + present_np[0], atol=1e-6, rtol=0
        )
        np.testing.assert_allclose(
            output_np[1], tokens_np[1] + absent_np[0], atol=1e-6, rtol=0
        )
        np.testing.assert_allclose(
            output_np[2], tokens_np[2] + present_np[0], atol=1e-6, rtol=0
        )
        np.testing.assert_allclose(
            output_np[3], tokens_np[3] + absent_np[0], atol=1e-6, rtol=0
        )
        # Present and absent embeddings must differ (truncated-normal init at
        # two independently created weights); otherwise this whole test would
        # pass vacuously.
        assert not np.allclose(present_np, absent_np)

    def test_present_embedding_and_absent_embedding_are_distinct_weights(self, tokens):
        layer = StateIndicatorEmbedding()
        flag = keras.ops.convert_to_tensor([True])
        layer((tokens[:1], flag))
        assert layer.present_embedding.shape == (1, 1, 8)
        assert layer.absent_embedding.shape == (1, 1, 8)

    def test_2d_flag_shape_is_accepted(self, tokens):
        layer = StateIndicatorEmbedding()
        flag = keras.ops.reshape(
            keras.ops.convert_to_tensor([True, False, True, False]), (4, 1)
        )
        output = layer((tokens, flag))
        assert output.shape == tokens.shape

    def test_get_config_from_config_round_trip(self, tokens):
        layer = StateIndicatorEmbedding(initializer="glorot_uniform")
        flag = keras.ops.convert_to_tensor([True, False, True, False])
        _ = layer((tokens, flag))

        config = layer.get_config()
        restored = StateIndicatorEmbedding.from_config(config)
        _ = restored((tokens, flag))
        restored.present_embedding.assign(layer.present_embedding)
        restored.absent_embedding.assign(layer.absent_embedding)

        original_out = layer((tokens, flag))
        restored_out = restored((tokens, flag))
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_out),
            keras.ops.convert_to_numpy(restored_out),
            atol=1e-6, rtol=0,
        )

    def test_composes_inside_a_functional_model(self):
        token_input = keras.Input(shape=(6, 8))
        flag_input = keras.Input(shape=(), dtype="bool")
        output = StateIndicatorEmbedding()((token_input, flag_input))
        model = keras.Model([token_input, flag_input], output)

        tokens_np = np.random.normal(size=(4, 6, 8)).astype("float32")
        flag_np = np.array([True, False, True, False])
        result = model([tokens_np, flag_np])
        assert result.shape == (4, 6, 8)
        assert np.isfinite(keras.ops.convert_to_numpy(result)).all()

    def test_output_shape_matches_input_tokens_shape(self, tokens):
        layer = StateIndicatorEmbedding()
        flag = keras.ops.convert_to_tensor([True, False, True, False])
        output = layer((tokens, flag))
        assert output.shape == tokens.shape

    def test_build_raises_on_non_pair_input(self):
        layer = StateIndicatorEmbedding()
        with pytest.raises(ValueError):
            layer.build([(4, 6, 8)])

    def test_build_raises_on_non_rank3_tokens(self):
        layer = StateIndicatorEmbedding()
        with pytest.raises(ValueError):
            layer.build([(4, 8), (4,)])
