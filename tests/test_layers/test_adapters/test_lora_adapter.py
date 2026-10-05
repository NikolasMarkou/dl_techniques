"""Tests for the extracted LoRAAdapter.

Most of these assertions were written against the layer while it lived in
``models/language/zamba2/layers.py``; the extraction moved the code and renamed
the slot-count argument, and these are the guards on that move.

TWIN NOTE. Every "nothing changed" assertion here has a twin that moves the
other way. A guard that only proves one direction passes for a layer that is
simply inert.
"""

import numpy as np
import pytest

keras = pytest.importorskip("keras")

from dl_techniques.layers.adapters.lora import (  # noqa: E402
    LEGACY_NUM_OCCURRENCES_KEY,
    LoRAAdapter,
)
from dl_techniques.layers.adapters.factory import create_adapter_layer  # noqa: E402

CONFIG = dict(output_dim=32, rank=4, alpha=8.0, num_adapters=3)


def _build(**overrides):
    layer = LoRAAdapter(**{**CONFIG, **overrides})
    layer.build((2, 5, 16))
    return layer


def _tape():
    """A fresh gradient tape, imported lazily to keep module import cheap."""
    import tensorflow as tf

    return tf.GradientTape()


class TestConstruction:
    @pytest.mark.parametrize('field,value', [
        ('output_dim', 0), ('rank', 0), ('alpha', 0.0),
        ('alpha', -1.0), ('num_adapters', 0),
    ])
    def test_a_non_positive_argument_raises(self, field, value):
        with pytest.raises(ValueError, match=field):
            LoRAAdapter(**{**CONFIG, field: value})

    def test_the_scale_is_alpha_over_rank(self):
        assert LoRAAdapter(**{**CONFIG, 'rank': 4, 'alpha': 16.0}).scale == 4.0

    def test_no_weights_exist_before_build(self):
        """``__init__`` must not create weights; the input width is not known yet."""
        layer = LoRAAdapter(**CONFIG)
        assert layer.a is None and layer.b is None
        assert layer.weights == []

    def test_compute_output_shape_works_unbuilt(self):
        assert LoRAAdapter(**CONFIG).compute_output_shape((2, 5, 16)) == (2, 5, 32)

    def test_build_creates_exactly_two_stacked_weight_tensors(self):
        layer = _build()
        assert layer.a.shape == (3, 16, 4)
        assert layer.b.shape == (3, 4, 32)
        assert len(layer.weights) == 2


class TestForward:
    def test_the_delta_is_zero_before_training(self):
        """``B`` is zero-initialised, so attaching this must not perturb a model."""
        layer = _build()
        x = keras.random.normal((2, 5, 16))
        assert float(np.abs(np.asarray(layer(x, adapter_idx=0))).max()) == 0.0

    def test_the_forward_actually_grows_a_non_zero_delta(self):
        """TWIN of the zero test: proves the zero is the INIT, not a dead layer."""
        layer = _build()
        layer.b.assign(keras.random.normal(layer.b.shape) * 0.1)
        out = layer(keras.random.normal((2, 5, 16)), adapter_idx=0)
        assert float(np.abs(np.asarray(out)).max()) > 0.0

    def test_the_delta_shape_matches_the_declared_output_dim(self):
        layer = _build(output_dim=48)
        out = layer(keras.random.normal((2, 5, 16)), adapter_idx=0)
        assert tuple(out.shape) == (2, 5, 48)

    @pytest.mark.parametrize('rank', [1, 5, 16])
    def test_the_output_is_finite_for_several_ranks(self, rank):
        layer = _build(rank=rank)
        layer.b.assign(keras.random.normal(layer.b.shape) * 0.1)
        out = layer(keras.random.normal((2, 5, 16)), adapter_idx=0)
        assert bool(np.all(np.isfinite(np.asarray(out))))

    def test_broadcasts_over_a_bare_2d_input(self):
        layer = _build()
        out = layer(keras.random.normal((7, 16)), adapter_idx=0)
        assert tuple(out.shape) == (7, 32)


class TestPerSlotIndependence:
    """D-007: every slot's ``A`` must be drawn independently.

    A shared seedless initializer instance is stateless-deterministic and
    replays the same sample at every call of the same shape, which once made
    every slot's ``A`` slice bit-identical.
    """

    def test_no_two_slots_share_a_bit_identical_a_slice(self):
        a = np.asarray(_build().a.value)
        for i in range(a.shape[0]):
            for j in range(i + 1, a.shape[0]):
                assert not np.array_equal(a[i], a[j]), (
                    f"A[{i}] == A[{j}] bit-for-bit; the per-slice clone is gone"
                )

    def test_the_slots_produce_different_deltas(self):
        layer = _build()
        layer.b.assign(keras.random.normal(layer.b.shape) * 0.1)
        x = keras.random.normal((2, 5, 16))
        assert not np.allclose(
            np.asarray(layer(x, adapter_idx=0)),
            np.asarray(layer(x, adapter_idx=1)),
        )

    def test_a_seeded_initializer_reproduces_across_two_instances(self):
        """``clone_initializer`` preserves an explicit seed; the clone must not
        consume it, so two layers with the same seed draw the same numbers."""
        first = _build(kernel_initializer=keras.initializers.RandomNormal(seed=5))
        second = _build(kernel_initializer=keras.initializers.RandomNormal(seed=5))
        assert np.allclose(np.asarray(first.a.value), np.asarray(second.a.value))

    def test_an_unseeded_initializer_is_independent_across_instances(self):
        """TWIN of the seeded test: without a seed the draws must differ."""
        first = _build()
        second = _build()
        assert not np.allclose(
            np.asarray(first.a.value), np.asarray(second.a.value)
        )


class TestSlotIndex:
    @pytest.mark.parametrize('index', [-1, 3, 99])
    def test_an_out_of_range_index_raises(self, index):
        with pytest.raises(ValueError, match='adapter_idx'):
            _build()(keras.random.normal((2, 5, 16)), adapter_idx=index)

    def test_a_missing_index_defaults_to_slot_zero(self):
        """Keras calls ``layer(x)`` in Sequential, functional models and
        ``fit()``, so requiring the keyword would make the adapter unusable there."""
        layer = _build()
        layer.b.assign(keras.random.normal(layer.b.shape) * 0.1)
        x = keras.random.normal((2, 5, 16))
        assert np.allclose(
            np.asarray(layer(x)), np.asarray(layer(x, adapter_idx=0))
        )

    def test_the_layer_is_usable_inside_a_sequential_model(self):
        """The reason the default exists, asserted end to end."""
        layer = _build()
        layer.b.assign(keras.random.normal(layer.b.shape) * 0.1)
        model = keras.Sequential([keras.Input(shape=(5, 16)), layer])
        out = model(keras.random.normal((2, 5, 16)), training=False)
        assert tuple(out.shape) == (2, 5, 32)
        assert bool(np.all(np.isfinite(np.asarray(out))))

    def test_the_deprecated_spelling_still_selects_the_same_slot(self):
        layer = _build()
        layer.b.assign(keras.random.normal(layer.b.shape) * 0.1)
        x = keras.random.normal((2, 5, 16))
        assert np.allclose(
            np.asarray(layer(x, adapter_idx=1)),
            np.asarray(layer(x, occurrence_idx=1)),
        )

    def test_both_spellings_agreeing_is_accepted(self):
        layer = _build()
        layer.b.assign(keras.random.normal(layer.b.shape) * 0.1)
        x = keras.random.normal((2, 5, 16))
        assert np.allclose(
            np.asarray(layer(x, adapter_idx=2)),
            np.asarray(layer(x, adapter_idx=2, occurrence_idx=2)),
        )

    def test_the_two_spellings_disagreeing_raises(self):
        """A silent preference for one name would hide a caller's bug."""
        with pytest.raises(ValueError, match='disagree'):
            _build()(keras.random.normal((2, 5, 16)),
                     adapter_idx=0, occurrence_idx=1)

    def test_num_occurrences_reads_the_renamed_field(self):
        assert _build().num_occurrences == 3

    def test_num_occurrences_is_read_only(self):
        """Two writable names for one value would give the slot count two
        sources of truth."""
        with pytest.raises(AttributeError):
            _build().num_occurrences = 5


class TestSerialization:
    def test_get_config_carries_every_constructor_argument(self):
        config = LoRAAdapter(
            **CONFIG, kernel_initializer='he_normal'
        ).get_config()
        for key in ('output_dim', 'rank', 'alpha', 'num_adapters',
                    'kernel_initializer'):
            assert key in config, key

    def test_get_config_does_not_emit_the_legacy_key(self):
        """Two names for one value in the round trip is not a valid config."""
        config = _build().get_config()
        assert LEGACY_NUM_OCCURRENCES_KEY not in config
        assert config['num_adapters'] == 3

    def test_from_config_round_trips(self):
        original = LoRAAdapter(**CONFIG, name='a')
        restored = LoRAAdapter.from_config(original.get_config())
        assert restored.output_dim == original.output_dim
        assert restored.num_adapters == original.num_adapters
        assert restored.scale == original.scale

    def test_a_pre_extraction_config_still_loads(self):
        """The rename must not break archives written before the extraction.

        ``legacy_packages`` fixes the registry KEY; this fixes the constructor
        ARGUMENT NAME stored inside the config dict, which no amount of aliasing
        at the registry level can reach.
        """
        legacy = LoRAAdapter(**CONFIG).get_config()
        legacy[LEGACY_NUM_OCCURRENCES_KEY] = legacy.pop('num_adapters')
        assert LEGACY_NUM_OCCURRENCES_KEY in legacy
        assert LoRAAdapter.from_config(legacy).num_adapters == CONFIG['num_adapters']

    def test_a_pre_extraction_checkpoint_still_loads_through_keras(self):
        """The end-to-end half of the rename guard.

        ``test_a_pre_extraction_config_still_loads`` exercises ``from_config``
        directly; this proves the path a real archive takes -- a config dict
        written by the pre-extraction layer, loaded through Keras -- also works.
        """
        legacy_config = {
            'module': 'dl_techniques.layers.adapters.lora',
            'class_name': 'LoRAAdapter',
            'config': {
                'name': 'lora',
                'trainable': True,
                'dtype': 'float32',
                'output_dim': CONFIG['output_dim'],
                'rank': CONFIG['rank'],
                'alpha': CONFIG['alpha'],
                'kernel_initializer': 'glorot_uniform',
                LEGACY_NUM_OCCURRENCES_KEY: CONFIG['num_adapters'],
            },
        }
        restored = keras.saving.deserialize_keras_object(legacy_config)
        assert isinstance(restored, LoRAAdapter)
        assert restored.num_adapters == CONFIG['num_adapters']

    def test_declaring_the_slot_count_under_both_names_conflicting_raises(self):
        legacy = LoRAAdapter(**CONFIG).get_config()
        legacy[LEGACY_NUM_OCCURRENCES_KEY] = legacy.pop('num_adapters')
        legacy['num_adapters'] = 99
        with pytest.raises(ValueError, match='once'):
            LoRAAdapter.from_config(legacy)

    def test_declaring_both_names_agreeing_is_accepted(self):
        legacy = LoRAAdapter(**CONFIG).get_config()
        legacy[LEGACY_NUM_OCCURRENCES_KEY] = legacy.pop('num_adapters')
        assert LoRAAdapter.from_config(legacy).num_adapters == 3


class TestRoundTripValues:
    """A shape-only round trip passes for a model that restored zero weights.

    Both tests compare VALUES, and the first compares at ``atol=0.0`` BEFORE the
    loaded model has been called once -- after a call, a build-only load path
    reads the same weight count for a correct variant and a broken one alike.
    """

    @staticmethod
    def _as_model(layer, input_width=16):
        """Wrap a bare Layer so Keras 3 will save it.

        ``save``/``load_model`` are ``Model`` methods; a bare ``Layer`` has
        neither, so the round trip has to go through a container that owns it.
        """
        return keras.Sequential([
            keras.Input(shape=(5, input_width)),
            layer,
        ])

    @staticmethod
    def _adapter_in(model):
        """Find the LoRAAdapter inside a (possibly restored) container.

        By TYPE, not by index: a restored ``Sequential`` drops the ``InputLayer``
        the original held, so ``layers[1]`` exists on one and overflows on the
        other. Index-based lookup across a save/load boundary is a latent
        IndexError dressed as a passing test.
        """
        found = [lyr for lyr in model.layers if isinstance(lyr, LoRAAdapter)]
        assert len(found) == 1, f"expected exactly one LoRAAdapter, got {found}"
        return found[0]

    def test_a_saved_adapter_restores_its_weight_values(self, tmp_path):
        original = _build()
        original.b.assign(keras.random.normal(original.b.shape) * 0.3)
        model = self._as_model(original)

        path = tmp_path / 'lora.keras'
        model.save(path)

        weights_before = [w.path for w in model.weights]
        restored = keras.models.load_model(path)
        assert [w.path for w in restored.weights] == weights_before
        assert np.allclose(
            np.asarray(self._adapter_in(model).b.value),
            np.asarray(self._adapter_in(restored).b.value),
            atol=0.0,
        )
        keras.backend.clear_session()

    def test_the_restored_forward_matches_the_original_on_values(self, tmp_path):
        original = _build()
        original.b.assign(keras.random.normal(original.b.shape) * 0.3)
        model = self._as_model(original)
        x = keras.random.normal((2, 5, 16))

        path = tmp_path / 'lora2.keras'
        model.save(path)
        restored = keras.models.load_model(path)

        # ``training=False`` explicitly: a bare ``model(x)`` is not an inference.
        assert np.allclose(
            np.asarray(model(x, training=False)),
            np.asarray(restored(x, training=False)),
            atol=1e-6, rtol=0,
        )
        keras.backend.clear_session()


class TestGradientFlow:
    """Gradients reach the adapter's weights for an exercised slot.

    The interesting fact pinned here is WHICH weights receive gradient at
    initialisation, and it is not the obvious one.

    ``delta = (x @ A) @ B * scale`` and ``B`` is zero-initialised, so ``delta``
    is exactly zero at construction. A loss quadratic in ``delta`` therefore has
    gradient zero too -- not because the graph is disconnected, but because
    ``d/dB (sum delta^2) = 2 * delta * d delta/dB`` and one factor is zero.

    That is not a defect; it is why the zero-init leaves an already-trained
    block unperturbed. It does mean the two weights come alive in ORDER:
    ``grad_B`` is nonzero at ``B = 0`` for a loss linear in the delta, while
    ``grad_A`` is zero there (``A`` only enters multiplied by ``B``) and becomes
    nonzero once ``B`` moves off zero. Both facts are asserted below, each with
    the other as its twin.
    """

    @staticmethod
    def _linear_loss(layer, slot):
        """A loss LINEAR in the delta, so it has gradient at ``B = 0``."""
        return keras.ops.sum(layer(keras.random.normal((2, 5, 16)),
                                   adapter_idx=slot))

    def test_grad_b_is_nonzero_at_the_zero_initialisation(self):
        """The bootstrap: ``B`` can learn first, because the loss is linear in it."""
        layer = _build()
        with _tape() as tape:
            loss = self._linear_loss(layer, slot=1)
        grad = tape.gradient(loss, layer.b)
        assert grad is not None
        assert float(np.abs(np.asarray(grad)).max()) > 0.0

    def test_grad_a_is_zero_at_the_zero_initialisation(self):
        """TWIN of the above: ``A`` only enters multiplied by ``B``, so it is pinned.

        If this ever goes nonzero it means the zero-init has been removed, which
        would break the "attaching this adapter does not perturb the model"
        guarantee.
        """
        layer = _build()
        with _tape() as tape:
            loss = self._linear_loss(layer, slot=1)
        grad = tape.gradient(loss, layer.a)
        assert grad is not None
        assert float(np.abs(np.asarray(grad)).max()) == 0.0

    def test_grad_a_becomes_nonzero_once_b_leaves_zero(self):
        """The converse, and the reason training is not permanently stuck."""
        layer = _build()
        layer.b.assign(keras.random.normal(layer.b.shape) * 0.1)
        with _tape() as tape:
            loss = keras.ops.sum(layer(keras.random.normal((2, 5, 16)),
                                       adapter_idx=1) ** 2)
        grads = tape.gradient(loss, [layer.a, layer.b])
        assert all(g is not None for g in grads)
        assert float(np.abs(np.asarray(grads[0])).max()) > 0.0
        assert float(np.abs(np.asarray(grads[1])).max()) > 0.0

    def test_only_the_exercised_slot_is_reachable(self):
        """``A``/``B`` are stacked; an unexercised slot must stay untouched.

        Asserted on ``A`` rather than ``B``: with ``B`` zero-initialised every
        slot's ``grad_B`` is zero, so a ``B``-only assertion would pass for a
        layer that leaked gradient into all slots at once.
        """
        layer = _build()
        layer.b.assign(keras.random.normal(layer.b.shape) * 0.1)
        with _tape() as tape:
            loss = keras.ops.sum(layer(keras.random.normal((2, 5, 16)),
                                       adapter_idx=0) ** 2)
        grad = np.asarray(tape.gradient(loss, layer.a))
        per_slot = np.abs(grad).reshape(3, -1).max(axis=1)
        assert per_slot[0] > 0.0
        assert per_slot[1] == 0.0 and per_slot[2] == 0.0


class TestViaTheFactory:
    def test_the_factory_builds_an_equivalent_layer(self):
        via_factory = create_adapter_layer('lora', name='x', **CONFIG)
        assert isinstance(via_factory, LoRAAdapter)
        assert via_factory.num_adapters == CONFIG['num_adapters']
        assert via_factory.scale == CONFIG['alpha'] / CONFIG['rank']
