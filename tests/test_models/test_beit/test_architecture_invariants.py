"""What the BEiT composition ACTUALLY wired, asserted by inspecting the objects.

Moved verbatim from ``test_model.py`` (class ``TestBeitArchitectureValidation``,
section 6) during the step-8 decomposition of plan-2026-08-24T074054-247151fd.

This file owns the structural pins: which layer classes were instantiated, which
weights exist and which do not, which knobs reached the sub-layer that honours them.
Every assertion reads the constructed object graph, never the config dict that was
passed in -- a kwarg that never arrived and a kwarg that arrived and was honoured are
indistinguishable from the caller's side. It is deliberately independent of forward
and serialization mechanics, which live in ``test_model.py``.
"""

import numpy as np
import pytest
from keras import ops

from dl_techniques.layers.attention.beit_attention import BeitAttention
from dl_techniques.layers.embedding.class_token import ClassTokenPrepend
from dl_techniques.layers.embedding.mask_token import MaskTokenApply
from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.models.vision.beit import (
    BACKBONE_NAME,
    BeitModel,
    create_beit_backbone,
)
from tests.test_models.test_beit.beit_test_geometry import (
    IMG,
    PATCH,
    GRID,
    SEQ_LEN,
    EPS,
    _tiny,
    _images,
)

class TestBeitArchitectureValidation:
    """What the composition ACTUALLY wired, asserted by inspection."""

    def test_every_block_is_a_transformer_layer_with_beit_attention(self):
        model = _tiny()
        assert len(model.encoder_layers) == model.num_layers == 12
        for i, layer in enumerate(model.encoder_layers):
            assert isinstance(layer, TransformerLayer), i
            assert layer.name == f"encoder_layer_{i}"
            assert isinstance(layer.attention, BeitAttention), i
            assert layer.attention.window_size == GRID
            assert layer.attention.num_tokens == SEQ_LEN
            assert layer.attention.use_relative_position_bias is True
            # BEiT's asymmetric bias survived the whole factory chain.
            assert layer.attention.q_dense.use_bias is True
            assert layer.attention.k_dense.use_bias is False
            assert layer.attention.v_dense.use_bias is True

    def test_the_encoder_layer_list_is_stored_flat(self):
        """`List[List[Layer]]` loses weights on a `.keras` round trip while every
        structural check still passes."""
        model = _tiny()
        assert isinstance(model.encoder_layers, list)
        assert all(
            isinstance(l, TransformerLayer) for l in model.encoder_layers
        ), "encoder_layers must be a FLAT list of layers"

    def test_every_norm_epsilon_is_1e_12(self):
        model = _tiny()
        model.build((None,) + IMG)
        for i, layer in enumerate(model.encoder_layers):
            assert layer.attention_norm.epsilon == EPS, i
            assert layer.output_norm.epsilon == EPS, i

    def test_the_epsilon_assertion_is_falsifiable(self):
        """Control: a model asked for a different epsilon must report it.

        Why this can fail if the implementation is wrong: if the epsilon were being
        ignored, every norm would carry the factory default and the assertion above
        would be pinning that default rather than the config. These two models must
        disagree.
        """
        other = _tiny(layer_norm_eps=1e-7)
        other.build((None,) + IMG)
        assert other.encoder_layers[0].attention_norm.epsilon == 1e-7
        assert other.encoder_layers[0].output_norm.epsilon == 1e-7

    def test_normalization_is_pre_norm(self):
        for layer in _tiny().encoder_layers:
            assert layer.normalization_position == 'pre'
            assert layer.normalization_type == 'layer_norm'
            assert layer.ffn_type == 'mlp'

    def test_layer_scale_is_on_at_the_scale_value_and_signed(self):
        model = _tiny(scale='large', num_layers=2)
        model.build((None,) + IMG)
        for layer in model.encoder_layers:
            assert layer.use_layer_scale is True
            assert layer.layer_scale_init_value == 1e-5
            # BEiT's gamma is a SIGNED scale; LearnableMultiplier's own default is a
            # non_neg constraint, which TransformerLayer overrides with None.
            assert layer.attention_layer_scale.constraint is None

    def test_stochastic_depth_is_the_exact_linear_ramp(self):
        """Assert the WHOLE list, not just that the rates differ.

        Why this can fail if the implementation is wrong: a reversed ramp, a constant
        rate, or an off-by-one denominator all produce a list of distinct-looking
        floats that trains fine and is wrong. The oracle below is the schedule
        transcribed independently (linear 0 -> drop_path_rate over num_layers).
        """
        rate = 0.1
        model = _tiny(drop_path_rate=rate)
        n = model.num_layers
        expected = [round(i * (rate / (n - 1)), 6) for i in range(n)]
        assert expected[0] == 0.0
        assert expected[-1] == pytest.approx(rate)
        assert model.drop_path_rates == expected
        actual = [l.stochastic_depth_rate for l in model.encoder_layers]
        assert actual == expected
        # And the rate really reached the sub-layer, not just the block's attribute.
        assert [
            l.attention_stochastic_depth.drop_path_rate for l in model.encoder_layers
        ] == expected
        for l in model.encoder_layers:
            assert l.use_stochastic_depth is True

    def test_mask_token_is_present_and_built_even_on_an_unmasked_forward(self):
        """The warm-start contract: ALWAYS CREATE, CONDITIONALLY USE.

        Why this can fail if the implementation is wrong: dropping the "dead" mask
        token from a backbone that never masks makes the classifier's trunk a
        different layer set from the MIM trunk, and the warm start then transfers a
        strict subset with no error at all.
        """
        model = _tiny()
        model(_images(), training=False)  # never passes a mask
        assert isinstance(model.mask_token, MaskTokenApply)
        assert model.mask_token.built
        assert model.mask_token.mask_token is not None
        assert tuple(model.mask_token.mask_token.shape) == (1, 1, model.hidden_size)
        assert "mask_token" in {l.name for l in model.layers}

    def test_cls_token_is_a_class_token_prepend(self):
        model = _tiny()
        model.build((None,) + IMG)
        assert isinstance(model.cls_token, ClassTokenPrepend)
        assert tuple(model.cls_token.cls_token.shape) == (1, 1, model.hidden_size)

    def test_absolute_position_embedding_is_off_by_default(self):
        """BEiT uses RELATIVE bias; the absolute table must not exist by default."""
        assert _tiny().pos_embed is None
        enabled = _tiny(use_absolute_position_embeddings=True)
        assert enabled.pos_embed is not None
        enabled.build((None,) + IMG)
        # And it actually changes the forward pass.
        x = _images()
        a = ops.convert_to_numpy(_tiny()(x, training=False))
        b = ops.convert_to_numpy(enabled(x, training=False))
        assert not np.allclose(a, b, atol=1e-6)

    def test_final_norm_follows_the_mean_pooling_fork(self):
        """D-007: `use_mean_pooling=True` -> the trunk applies NO final norm.

        Why this can fail if the implementation is wrong: always applying a final norm
        here is the obvious "cleanup", and at the default config it silently inserts a
        normalization the reference does not have in front of BOTH heads -- no error,
        no shape change, a plausible loss curve.
        """
        pooled = _tiny(use_mean_pooling=True)
        assert pooled.final_norm is None
        assert "final_norm" not in {l.name for l in pooled.layers}

        cls_mode = _tiny(use_mean_pooling=False)
        assert cls_mode.final_norm is not None
        assert cls_mode.final_norm.epsilon == EPS
        cls_mode.build((None,) + IMG)
        assert cls_mode.final_norm.built
        # The fork is observable in the OUTPUT, not just in the layer list: a normed
        # sequence has ~unit variance along the feature axis.
        out = ops.convert_to_numpy(cls_mode(_images(), training=False))
        assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-4)
        assert np.allclose(out.std(axis=-1), 1.0, atol=1e-2)

    def test_the_backbone_is_named_for_the_warm_start(self):
        assert create_beit_backbone('tiny', IMG, PATCH).name == BACKBONE_NAME
        assert BeitModel.from_variant('beit_tiny', IMG, PATCH).name == BACKBONE_NAME

    def test_from_variant_and_the_factory_agree(self):
        a = BeitModel.from_variant('beit_small', IMG, PATCH)
        b = create_beit_backbone('small', IMG, PATCH)
        for key in ('hidden_size', 'num_layers', 'num_heads', 'intermediate_size',
                    'layer_scale_init_value', 'scale'):
            assert getattr(a, key) == getattr(b, key)
