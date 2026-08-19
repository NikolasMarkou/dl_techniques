"""F-51: the reference epsilon must reach EVERY norm in the text tower.

``layers/fastvit/reference.py``'s interface contract says
``REFERENCE_NORM_EPSILON`` (1e-5, torch's ``nn.LayerNorm`` default) is to be
passed EXPLICITLY at every construction site in this port, because Keras'
default is 1e-3 and the mismatch is invisible to every shape assertion.

``test_norm_epsilon.py`` pinned exactly one site: ``encoder.layer_norm``, the
OpenCLIP ``ln_final``. The ``num_layers`` ``TransformerLayer``s constructed
immediately above it passed no ``attention_norm_args`` / ``ffn_norm_args``, so
their ``2 * num_layers`` norms fell through to ``layers/norms/factory.py``'s
``setdefault('epsilon', 1e-6)``.

MEASURED at ``11f971ed1``, ``num_layers=3``: six norms at 1e-06 and one at
1e-05. For ``mobileclip2_s3`` (``text_depth=24``) that is 24 wrong and 1 right.

This file widens the guard past ``encoder.layer_norm`` to a SWEEP over every
sub-layer that owns an ``epsilon``, so a future norm added anywhere in the tower
is covered without editing the test.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import keras
import pytest

from dl_techniques.layers.fastvit.reference import REFERENCE_NORM_EPSILON
from dl_techniques.models.mobile_clip.components import MobileClipTextEncoder

KERAS_DEFAULT_EPSILON = 1e-3
#: What `layers/norms/factory.py` setdefaults, i.e. what the unthreaded sites got.
FACTORY_DEFAULT_EPSILON = 1e-6


def _encoder(num_layers=3, **kwargs):
    return MobileClipTextEncoder(
        vocab_size=64,
        max_seq_len=8,
        embed_dim=16,
        num_layers=num_layers,
        num_heads=2,
        intermediate_size=32,
        projection_dim=8,
        **kwargs,
    )


def _epsilon_sites(encoder):
    """Every sub-layer in the tower that owns an ``epsilon``, by path."""
    sites = []
    for index, block in enumerate(encoder.transformer_layers):
        for sub in block._flatten_layers(include_self=False):
            if hasattr(sub, "epsilon"):
                sites.append((f"transformer_layer_{index}/{sub.name}", sub))
    sites.append(("final_layer_norm", encoder.layer_norm))
    return sites


class TestEverySiteCarriesTheReferenceEpsilon:

    @pytest.fixture(scope="class")
    def built(self):
        encoder = _encoder()
        encoder.build((None, 8))
        return encoder

    def test_the_sweep_actually_finds_the_per_block_norms(self, built):
        """STOP-IF: without this the sweep below could be vacuous."""
        sites = _epsilon_sites(built)
        per_block = [name for name, _ in sites if name.startswith("transformer_layer_")]
        assert len(per_block) == 2 * 3, per_block

    def test_no_site_is_left_at_the_factory_default(self, built):
        offenders = [
            (name, layer.epsilon) for name, layer in _epsilon_sites(built)
            if layer.epsilon == FACTORY_DEFAULT_EPSILON
        ]
        assert offenders == [], (
            "these norms fell through to layers/norms/factory.py's 1e-6 "
            f"setdefault instead of the port's reference epsilon: {offenders}"
        )

    def test_no_site_is_left_at_the_keras_default(self, built):
        offenders = [
            (name, layer.epsilon) for name, layer in _epsilon_sites(built)
            if layer.epsilon == KERAS_DEFAULT_EPSILON
        ]
        assert offenders == [], offenders

    def test_every_site_equals_the_shared_constant(self, built):
        wrong = [
            (name, layer.epsilon) for name, layer in _epsilon_sites(built)
            if layer.epsilon != REFERENCE_NORM_EPSILON
        ]
        assert wrong == [], wrong

    @pytest.mark.parametrize("num_layers", [1, 4])
    def test_it_holds_at_other_depths(self, num_layers):
        encoder = _encoder(num_layers=num_layers)
        encoder.build((None, 8))
        sites = _epsilon_sites(encoder)
        assert len(sites) == 2 * num_layers + 1
        assert all(layer.epsilon == REFERENCE_NORM_EPSILON for _, layer in sites)


class TestTheKnobIsThreadedNotHardcoded:

    def test_the_blocks_receive_the_args_dicts(self):
        encoder = _encoder(num_layers=2)
        for block in encoder.transformer_layers:
            assert block.attention_norm_args == {"epsilon": REFERENCE_NORM_EPSILON}
            assert block.ffn_norm_args == {"epsilon": REFERENCE_NORM_EPSILON}

    def test_each_block_gets_its_own_dict_instance(self):
        """A shared mutable default would let one block's edit hit them all."""
        encoder = _encoder(num_layers=3)
        seen = [id(b.attention_norm_args) for b in encoder.transformer_layers]
        seen += [id(b.ffn_norm_args) for b in encoder.transformer_layers]
        assert len(set(seen)) == len(seen)

    def test_the_constant_is_imported_not_re_declared(self):
        import dl_techniques.models.mobile_clip.components as components

        assert components.REFERENCE_NORM_EPSILON is REFERENCE_NORM_EPSILON


class TestItSurvivesSerialization:

    def test_a_config_round_trip_keeps_every_site(self):
        encoder = _encoder(num_layers=2)
        clone = MobileClipTextEncoder.from_config(encoder.get_config())
        clone.build((None, 8))
        wrong = [
            (name, layer.epsilon) for name, layer in _epsilon_sites(clone)
            if layer.epsilon != REFERENCE_NORM_EPSILON
        ]
        assert wrong == [], wrong


class TestTheEpsilonIsNotInert:
    """The knob must reach the forward pass, or the whole finding is cosmetic."""

    def test_a_large_epsilon_changes_the_tower_output(self):
        import numpy as np

        tokens = np.random.RandomState(0).randint(0, 64, size=(2, 8)).astype("int32")
        keras.utils.set_random_seed(0)
        reference = _encoder(num_layers=2)
        reference.build((None, 8))
        a = keras.ops.convert_to_numpy(reference(tokens, training=False))

        keras.utils.set_random_seed(0)
        loud = _encoder(num_layers=2)
        loud.build((None, 8))
        for _, layer in _epsilon_sites(loud):
            layer.epsilon = 1.0
        b = keras.ops.convert_to_numpy(loud(tokens, training=False))

        assert float(np.max(np.abs(a - b))) > 1e-4, (
            "epsilon does not reach the forward pass on this path, so pinning "
            "its value proves nothing"
        )
