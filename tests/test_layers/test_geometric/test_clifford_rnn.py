"""Tests for the Clifford geometric-algebra recurrent layer.

Covers construction, forward shapes, ``get_config`` completeness (derived from
``inspect.signature`` so it cannot drift), activation serialisation (including
the callable-activation regression that made ``get_config`` non-JSON-safe), the
``keras.layers.Layer``-as-activation rejection guard, ``.keras`` save/load value
equality, and every ``state_update`` / ``global_context_mode`` the layer accepts.
"""

import inspect
import json
import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.geometric.clifford_rnn import (
    CliffordRNN,
    CliffordRNNCell,
    _STATE_UPDATES,
    _GLOBAL_MODES,
    _CLI_MODES,
    _CTX_MODES,
)


def _ctor_params(cls) -> set:
    """Names of ``cls.__init__``'s explicit keyword parameters."""
    return {
        name
        for name, param in inspect.signature(cls.__init__).parameters.items()
        if name != "self"
        and param.kind
        not in (param.VAR_KEYWORD, param.VAR_POSITIONAL)
    }


# ===========================================================================
# TestCliffordRNNCell
# ===========================================================================


class TestCliffordRNNCell:
    """Test suite for CliffordRNNCell."""

    @pytest.fixture
    def units(self) -> int:
        return 8

    @pytest.fixture
    def step_tensor(self) -> tf.Tensor:
        """One timestep, ``(B, F)``."""
        keras.utils.set_random_seed(11)
        return tf.random.normal([2, 6])

    # ------------------------------------------------------------------

    def test_initialization_defaults(self, units):
        """Defaults land on the documented values."""
        cell = CliffordRNNCell(units=units)
        assert cell.units == units
        assert cell.shifts == [1, 2, 4]
        assert cell.cli_mode == "full"
        assert cell.ctx_mode == "diff"
        assert cell.state_update == "gated"
        assert cell.use_global_context is False
        assert cell.use_gate is True

    def test_initialization_custom(self, units):
        """Custom parameters are stored verbatim."""
        cell = CliffordRNNCell(
            units=units,
            shifts=[1, 3],
            cli_mode="wedge",
            ctx_mode="abs",
            state_update="decay",
            include_vector_grade=True,
            name="custom_cell",
        )
        assert cell.shifts == [1, 3]
        assert cell.cli_mode == "wedge"
        assert cell.ctx_mode == "abs"
        assert cell.state_update == "decay"
        assert cell.include_vector_grade is True
        assert cell.name == "custom_cell"

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"units": 0}, "units"),
            ({"units": 8, "cli_mode": "bad"}, "cli_mode"),
            ({"units": 8, "ctx_mode": "bad"}, "ctx_mode"),
            ({"units": 8, "state_update": "bad"}, "state_update"),
            ({"units": 8, "global_context_mode": "bad"}, "global_context_mode"),
            ({"units": 8, "dropout_rate": 1.5}, "dropout"),
            ({"units": 8, "recurrent_dropout_rate": 1.5}, "recurrent_dropout"),
            ({"units": 1, "use_global_context": True}, "use_global_context"),
        ],
    )
    def test_invalid_arguments(self, kwargs, match):
        """Every documented validation raises at construction."""
        with pytest.raises(ValueError, match=match):
            CliffordRNNCell(**kwargs)

    def test_forward_shape(self, units, step_tensor):
        """A single step returns ``(B, units)`` plus the carried state."""
        keras.utils.set_random_seed(11)
        cell = CliffordRNNCell(units=units)
        states = cell.get_initial_state(batch_size=step_tensor.shape[0])
        out, new_states = cell(step_tensor, states, training=False)
        assert out.shape == (step_tensor.shape[0], units)
        assert len(new_states) == len(states)
        assert new_states[0].shape == (step_tensor.shape[0], units)

    def test_forward_shape_global_context(self, units, step_tensor):
        """The global branch adds states but not output width."""
        keras.utils.set_random_seed(11)
        cell = CliffordRNNCell(units=units, use_global_context=True)
        states = cell.get_initial_state(batch_size=step_tensor.shape[0])
        assert len(states) > 1
        out, new_states = cell(step_tensor, states, training=False)
        assert out.shape == (step_tensor.shape[0], units)
        assert len(new_states) == len(states)

    def test_usable_inside_keras_rnn(self, units, step_tensor):
        """The cell drops into ``keras.layers.RNN`` unchanged."""
        keras.utils.set_random_seed(11)
        x = tf.random.normal([2, 5, 6])
        rnn = keras.layers.RNN(CliffordRNNCell(units=units))
        assert rnn(x, training=False).shape == (2, units)

    def test_gradient_flow(self, units):
        """Gradients reach the input through one step."""
        keras.utils.set_random_seed(11)
        cell = CliffordRNNCell(units=units)
        x = tf.Variable(tf.random.normal([2, 6]))
        states = cell.get_initial_state(batch_size=2)
        with tf.GradientTape() as tape:
            out, _ = cell(x, states, training=True)
            loss = tf.reduce_mean(tf.square(out))
        grads = tape.gradient(loss, x)
        assert grads is not None
        assert np.any(grads.numpy() != 0)


# ===========================================================================
# TestCliffordRNN
# ===========================================================================


class TestCliffordRNN:
    """Test suite for the CliffordRNN layer."""

    @pytest.fixture
    def units(self) -> int:
        return 8

    @pytest.fixture
    def seq_tensor(self) -> tf.Tensor:
        """``(B, T, F)`` sequence input."""
        keras.utils.set_random_seed(13)
        return tf.random.normal([2, 5, 6])

    # ------------------------------------------------------------------

    def test_forward_shape(self, units, seq_tensor):
        """Last-step output is ``(B, units)``."""
        keras.utils.set_random_seed(13)
        layer = CliffordRNN(units=units)
        assert layer(seq_tensor, training=False).shape == (2, units)

    def test_forward_shape_return_sequences(self, units, seq_tensor):
        """``return_sequences=True`` yields ``(B, T, units)``."""
        keras.utils.set_random_seed(13)
        layer = CliffordRNN(units=units, return_sequences=True)
        assert layer(seq_tensor, training=False).shape == (2, 5, units)

    def test_return_state(self, units, seq_tensor):
        """``return_state=True`` appends the state tensors."""
        keras.utils.set_random_seed(13)
        layer = CliffordRNN(units=units, return_state=True)
        result = layer(seq_tensor, training=False)
        assert isinstance(result, (list, tuple))
        assert result[0].shape == (2, units)

    @pytest.mark.parametrize("state_update", _STATE_UPDATES)
    def test_state_update_modes(self, units, seq_tensor, state_update):
        """Every ``state_update`` mode runs and moves the state.

        The all-modes-agree failure is guarded explicitly: a mode whose branch
        silently returned ``h_prev`` (or ``term``) would produce a constant or
        an output identical to another mode's.
        """
        keras.utils.set_random_seed(17)
        layer = CliffordRNN(units=units, state_update=state_update)
        out = keras.ops.convert_to_numpy(layer(seq_tensor, training=False))
        assert out.shape == (2, units)
        assert np.isfinite(out).all()
        # A dead branch (h_new = h_prev) leaves the zero initial state intact.
        assert np.abs(out).max() > 0.0, (
            f"state_update={state_update!r} produced an all-zero state, i.e. "
            "the update branch never moved the carry"
        )

    def test_state_update_modes_are_distinct(self, units, seq_tensor):
        """The three carry rules are genuinely different functions.

        Weights are seeded identically per mode, so any two modes agreeing to
        1e-6 means one branch is not the rule it claims to be.
        """
        outs = {}
        for mode in _STATE_UPDATES:
            keras.utils.set_random_seed(17)
            layer = CliffordRNN(units=units, state_update=mode)
            outs[mode] = keras.ops.convert_to_numpy(
                layer(seq_tensor, training=False)
            )
        modes = list(_STATE_UPDATES)
        for i in range(len(modes)):
            for j in range(i + 1, len(modes)):
                a, b = modes[i], modes[j]
                assert not np.allclose(outs[a], outs[b], rtol=1e-6, atol=1e-6), (
                    f"state_update={a!r} and {b!r} produced identical outputs; "
                    "one of the carry branches is not implementing its rule"
                )

    @pytest.mark.parametrize("global_mode", _GLOBAL_MODES)
    def test_global_context_modes(self, units, seq_tensor, global_mode):
        """Every ``global_context_mode`` runs end to end."""
        keras.utils.set_random_seed(19)
        layer = CliffordRNN(
            units=units,
            use_global_context=True,
            global_context_mode=global_mode,
        )
        out = keras.ops.convert_to_numpy(layer(seq_tensor, training=False))
        assert out.shape == (2, units)
        assert np.isfinite(out).all()

    def test_global_context_branch_reaches_the_output(self, units, seq_tensor):
        """The global branch's weights actually influence the output.

        Comparing ``use_global_context`` True vs False across two constructions
        is NOT a valid probe: the True variant draws extra weights from the
        shared RNG, so the two outputs differ even when the branch is dead
        (measured). Instead the branch's own projection kernel is perturbed on
        a single built instance — a dead branch leaves the output untouched.
        """
        keras.utils.set_random_seed(19)
        layer = CliffordRNN(units=units, use_global_context=True)
        before = keras.ops.convert_to_numpy(layer(seq_tensor, training=False))

        kernel = layer.cell.global_geo_prod.proj.kernel
        kernel.assign(keras.ops.multiply(kernel, 100.0))

        after = keras.ops.convert_to_numpy(layer(seq_tensor, training=False))
        assert not np.allclose(before, after, rtol=1e-6, atol=1e-6), (
            "Scaling the global branch's projection kernel by 100 left the "
            "output unchanged; the global branch never reaches g_feat"
        )

    @pytest.mark.parametrize("cli_mode", _CLI_MODES)
    def test_cli_modes(self, units, seq_tensor, cli_mode):
        """Every geometric-product grade selection runs."""
        keras.utils.set_random_seed(23)
        layer = CliffordRNN(units=units, cli_mode=cli_mode)
        assert layer(seq_tensor, training=False).shape == (2, units)

    @pytest.mark.parametrize("ctx_mode", _CTX_MODES)
    def test_ctx_modes(self, units, seq_tensor, ctx_mode):
        """Every context mode runs."""
        keras.utils.set_random_seed(23)
        layer = CliffordRNN(units=units, ctx_mode=ctx_mode)
        assert layer(seq_tensor, training=False).shape == (2, units)

    def test_inference_is_deterministic(self, units, seq_tensor):
        """Two ``training=False`` calls agree exactly, even with dropout set."""
        keras.utils.set_random_seed(29)
        layer = CliffordRNN(
            units=units, dropout_rate=0.5, recurrent_dropout_rate=0.5
        )
        a = keras.ops.convert_to_numpy(layer(seq_tensor, training=False))
        b = keras.ops.convert_to_numpy(layer(seq_tensor, training=False))
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)

    def test_gradient_flow(self, units, seq_tensor):
        """Gradients propagate through the whole recurrence."""
        keras.utils.set_random_seed(29)
        layer = CliffordRNN(units=units)
        x = tf.Variable(seq_tensor)
        with tf.GradientTape() as tape:
            out = layer(x, training=True)
            loss = tf.reduce_mean(tf.square(out))
        grads = tape.gradient(loss, x)
        assert grads is not None
        assert np.any(grads.numpy() != 0)


# ===========================================================================
# TestConfigCompleteness
# ===========================================================================


class TestConfigCompleteness:
    """``get_config`` must carry every constructor parameter."""

    @pytest.mark.parametrize("cls", [CliffordRNNCell, CliffordRNN])
    def test_every_ctor_param_is_a_config_key(self, cls):
        """Expected keys are derived from the signature, never hand-listed."""
        config = cls(units=8).get_config()
        missing = sorted(_ctor_params(cls) - set(config))
        assert not missing, (
            f"{cls.__name__}.get_config() omits constructor parameter(s) "
            f"{missing}; cls(**config) would silently fall back to defaults"
        )

    @pytest.mark.parametrize("cls", [CliffordRNNCell, CliffordRNN])
    def test_config_is_json_serializable(self, cls):
        """The default config is JSON-safe."""
        json.dumps(cls(units=8).get_config())

    @pytest.mark.parametrize("cls", [CliffordRNNCell, CliffordRNN])
    def test_from_config_round_trip_preserves_attributes(self, cls):
        """``cls(**config)`` reproduces the non-default settings."""
        original = cls(
            units=8,
            shifts=[1, 3],
            cli_mode="wedge",
            ctx_mode="abs",
            state_update="decay",
            include_vector_grade=True,
            layer_scale_init=0.5,
            use_gate=False,
            use_bias=False,
        )
        restored = cls.from_config(original.get_config())
        for attr in (
            "units",
            "cli_mode",
            "ctx_mode",
            "state_update",
        ):
            assert getattr(restored, attr) == getattr(original, attr)
        assert list(restored.shifts) == list(original.shifts)


# ===========================================================================
# TestActivationHandling
# ===========================================================================


_ACTIVATION_KWARGS = (
    "activation",
    "dot_activation",
    "gate_activation",
    "feature_activation",
)


class TestActivationHandling:
    """Regression suite for the activation-serialisation hardening.

    Before this hardening the module carried a stale local ``_resolve_activation``
    and stored/emitted activations raw, so a callable (non-string) activation
    produced a ``get_config()`` that ``json.dumps`` could not encode, and a
    ``keras.layers.Layer`` activation was accepted here while the sibling
    ``clifford_block`` rejected it.
    """

    @pytest.mark.parametrize("cls", [CliffordRNNCell, CliffordRNN])
    @pytest.mark.parametrize("kwarg", _ACTIVATION_KWARGS)
    def test_callable_activation_config_is_json_serializable(self, cls, kwarg):
        """A callable activation still yields a JSON-encodable config."""
        layer = cls(units=8, **{kwarg: keras.activations.silu})
        config = layer.get_config()
        try:
            json.dumps(config)
        except TypeError as exc:  # pragma: no cover - failure path
            pytest.fail(
                f"{cls.__name__}.get_config() is not JSON-serialisable with a "
                f"callable {kwarg}: {exc}"
            )

    @pytest.mark.parametrize("cls", [CliffordRNNCell, CliffordRNN])
    @pytest.mark.parametrize("kwarg", _ACTIVATION_KWARGS)
    def test_callable_activation_survives_from_config(self, cls, kwarg):
        """The serialised callable deserialises back into a working layer.

        ``CliffordRNN`` stores activations on its cell (it exposes convenience
        properties only for ``units``/``shifts``/``cli_mode``/``ctx_mode``/
        ``state_update``/dropout), so the owner is resolved rather than assumed.
        """
        original = cls(units=8, **{kwarg: keras.activations.silu})
        restored = cls.from_config(original.get_config())
        owner = getattr(restored, "cell", restored)
        spec = getattr(owner, kwarg)
        assert spec is not None
        resolved = keras.activations.get(spec) if isinstance(spec, str) else spec
        assert callable(resolved)
        # A dict here would mean _activation_spec never canonicalised the
        # deserialised config back into a callable.
        assert not isinstance(spec, dict), (
            f"{cls.__name__}.{kwarg} restored as a raw serialised dict"
        )

    @pytest.mark.parametrize("cls", [CliffordRNNCell, CliffordRNN])
    @pytest.mark.parametrize("kwarg", _ACTIVATION_KWARGS)
    def test_layer_activation_is_rejected(self, cls, kwarg):
        """A stateful activation *layer* must be refused at construction.

        A Layer activation would create its weights during ``call()`` instead of
        ``build()``, which does not survive a ``.keras`` round-trip.
        """
        with pytest.raises(ValueError, match="keras Layer instance"):
            cls(units=8, **{kwarg: keras.layers.ReLU()})

    def test_string_activation_passes_through_unchanged(self):
        """Strings are stored and emitted verbatim (no spurious wrapping)."""
        cell = CliffordRNNCell(units=8, activation="relu")
        assert cell.activation == "relu"
        assert cell.get_config()["activation"] == "relu"


# ===========================================================================
# TestSaveLoad
# ===========================================================================


class TestSaveLoad:
    """``.keras`` round-trips must preserve output VALUES, not just shapes."""

    @staticmethod
    def _round_trip(layer, x):
        """Build a tiny model around ``layer``, save it, and reload it.

        ``training=False`` is passed EXPLICITLY on both calls: ``training=None``
        is not inference in this repo, and a stochastic path running silently
        has produced exactly this false failure before.
        """
        inp = keras.Input(shape=x.shape[1:])
        model = keras.Model(inputs=inp, outputs=layer(inp, training=False))
        before = keras.ops.convert_to_numpy(model(x, training=False))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False))
        return before, after

    def test_save_load_preserves_values(self):
        """Default configuration round-trips value-exactly."""
        keras.utils.set_random_seed(31)
        x = tf.random.normal([2, 5, 6])
        before, after = self._round_trip(
            CliffordRNN(units=8, name="clifford_rnn"), x
        )
        assert np.abs(before).max() > 0.0, "degenerate all-zero reference output"
        np.testing.assert_allclose(
            before,
            after,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Output VALUES changed across a .keras round-trip; the "
            "reloaded layer is not carrying the saved weights",
        )

    @pytest.mark.parametrize("state_update", _STATE_UPDATES)
    def test_save_load_preserves_values_per_state_update(self, state_update):
        """Each carry rule round-trips value-exactly."""
        keras.utils.set_random_seed(37)
        x = tf.random.normal([2, 5, 6])
        before, after = self._round_trip(
            CliffordRNN(units=8, state_update=state_update, name="rnn"), x
        )
        np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-5)

    def test_save_load_with_global_context(self):
        """The global branch's extra states survive the round-trip."""
        keras.utils.set_random_seed(41)
        x = tf.random.normal([2, 5, 6])
        before, after = self._round_trip(
            CliffordRNN(units=8, use_global_context=True, name="rnn_glob"), x
        )
        np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-5)

    def test_save_load_with_callable_activation(self):
        """A callable activation survives save/load (the (a) regression)."""
        keras.utils.set_random_seed(43)
        x = tf.random.normal([2, 5, 6])
        before, after = self._round_trip(
            CliffordRNN(
                units=8, activation=keras.activations.silu, name="rnn_act"
            ),
            x,
        )
        np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-5)
