"""Tests for the `reglu` and `bilinear` GLU factory aliases.

Both keys are pure factory aliases of :class:`GLUFFN` (Shazeer 2020):
- ReGLU    = GLUFFN with activation='relu'
- Bilinear = GLUFFN with activation='linear' (identity gate, no nonlinearity)

These tests verify that the factory actually produces a ``GLUFFN`` whose gate
activation resolves correctly (PM2 falsification), that the alias output matches
a directly-constructed ``GLUFFN``, that the layer round-trips through ``.keras``,
and that the alias default activation remains user-overridable.
"""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.layers.ffn.glu_ffn import GLUFFN
from dl_techniques.layers.ffn.factory import create_ffn_layer, get_ffn_info


class TestReGLUBilinearAlias:
    """Alias-correctness tests for the `reglu` and `bilinear` factory keys."""

    def test_reglu_is_gluffn_relu(self):
        layer = create_ffn_layer('reglu', hidden_dim=64, output_dim=64)
        assert isinstance(layer, GLUFFN)
        assert keras.activations.serialize(layer.activation) == 'relu'

    def test_bilinear_is_gluffn_linear(self):
        layer = create_ffn_layer('bilinear', hidden_dim=64, output_dim=64)
        assert isinstance(layer, GLUFFN)
        assert keras.activations.serialize(layer.activation) == 'linear'

    def test_reglu_matches_direct_gluffn(self):
        x = keras.random.normal((4, 64))

        alias = create_ffn_layer('reglu', hidden_dim=64, output_dim=64)
        direct = GLUFFN(hidden_dim=64, output_dim=64, activation='relu')

        # Build both, then copy weights so the comparison isolates the
        # activation wiring (ReLU-gated GLU) rather than RNG init.
        alias(x)
        direct(x)
        direct.set_weights(alias.get_weights())

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(alias(x)),
            keras.ops.convert_to_numpy(direct(x)),
            rtol=1e-6, atol=1e-6,
        )

    def test_bilinear_matches_direct_gluffn(self):
        x = keras.random.normal((4, 64))

        alias = create_ffn_layer('bilinear', hidden_dim=64, output_dim=64)
        direct = GLUFFN(hidden_dim=64, output_dim=64, activation='linear')

        alias(x)
        direct(x)
        direct.set_weights(alias.get_weights())

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(alias(x)),
            keras.ops.convert_to_numpy(direct(x)),
            rtol=1e-6, atol=1e-6,
        )

    def test_serialization_cycle_reglu(self):
        x = keras.random.normal((2, 32))

        inputs = keras.Input(shape=(32,))
        outputs = create_ffn_layer('reglu', hidden_dim=32, output_dim=32)(inputs)
        model = keras.Model(inputs, outputs)

        original = model(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'reglu_alias.keras')
            model.save(path)
            loaded = keras.models.load_model(path)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original),
            keras.ops.convert_to_numpy(loaded(x)),
            rtol=1e-6, atol=1e-6,
        )

    def test_user_can_override_activation(self):
        layer = create_ffn_layer('reglu', hidden_dim=32, output_dim=32, activation='gelu')
        assert isinstance(layer, GLUFFN)
        assert keras.activations.serialize(layer.activation) == 'gelu'

    def test_factory_info(self):
        info = get_ffn_info()
        for key in ('reglu', 'bilinear'):
            assert key in info
            entry = info[key]
            for field in ('class', 'description', 'required_params', 'optional_params', 'use_case'):
                assert field in entry, f"{key} missing {field}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
