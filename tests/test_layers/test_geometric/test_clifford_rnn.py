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


# ===========================================================================
# TestDropout
# ===========================================================================


class TestDropout:
    """``dropout_rate`` / ``recurrent_dropout_rate`` must actually be LIVE.

    Added in iteration 2 (plan-2026-08-10-3649c19e step 13; review concern 6):
    the layer's variational-dropout path is its riskiest machinery -- it hangs
    off a private Keras mixin (see :class:`TestPrivateKerasImportContract`) --
    and the original 68-test suite never executed it. A silently dead dropout
    is invisible to every shape, config and round-trip test in this file.

    Each test asserts BOTH directions, which is what makes it non-vacuous:

    * train != eval           -- the mask is applied at all;
    * train call 1 != call 2  -- ``reset_dropout_mask()`` really runs between
      calls and the ``SeedGenerator`` really advances (a mask cached forever
      would make two train calls identical while still differing from eval);
    * eval call 1 == call 2   -- and nothing stochastic leaks into inference.
    """

    UNITS = 8
    RATE = 0.5

    @pytest.fixture
    def seq(self) -> tf.Tensor:
        keras.utils.set_random_seed(101)
        return tf.random.normal([4, 6, 6])

    @staticmethod
    def _np(t):
        return keras.ops.convert_to_numpy(t)

    def _deltas(self, layer, x):
        """Return ``(train_vs_eval, train_vs_train, eval_vs_eval)`` max-abs."""
        eval_a = self._np(layer(x, training=False))
        eval_b = self._np(layer(x, training=False))
        train_a = self._np(layer(x, training=True))
        train_b = self._np(layer(x, training=True))
        return (
            float(np.max(np.abs(train_a - eval_a))),
            float(np.max(np.abs(train_a - train_b))),
            float(np.max(np.abs(eval_a - eval_b))),
        )

    # ------------------------------------------------------------------

    def test_input_dropout_is_live(self, seq):
        """``dropout_rate`` changes the output under ``training=True``."""
        keras.utils.set_random_seed(103)
        layer = CliffordRNN(units=self.UNITS, dropout_rate=self.RATE, seed=7)
        d_train_eval, d_train_train, d_eval_eval = self._deltas(layer, seq)

        assert d_train_eval > 1e-4, (
            "training=True output equals training=False output at "
            f"dropout_rate={self.RATE}: the input dropout mask is never "
            f"applied (delta {d_train_eval})"
        )
        assert d_train_train > 1e-4, (
            "two training=True calls are identical: the dropout mask is "
            "cached across calls (reset_dropout_mask is not running) or the "
            f"SeedGenerator is not advancing (delta {d_train_train})"
        )
        assert d_eval_eval == 0.0, (
            f"inference is not deterministic (delta {d_eval_eval})"
        )

    def test_recurrent_dropout_is_live(self, seq):
        """``recurrent_dropout_rate`` changes the output the same way."""
        keras.utils.set_random_seed(107)
        layer = CliffordRNN(
            units=self.UNITS, recurrent_dropout_rate=self.RATE, seed=7
        )
        d_train_eval, d_train_train, d_eval_eval = self._deltas(layer, seq)

        assert d_train_eval > 1e-4, (
            "training=True output equals training=False output at "
            f"recurrent_dropout_rate={self.RATE}: the recurrent mask is never "
            f"applied (delta {d_train_eval})"
        )
        assert d_train_train > 1e-4, (
            "two training=True calls are identical: reset_recurrent_dropout_"
            f"mask is not running (delta {d_train_train})"
        )
        assert d_eval_eval == 0.0

    def test_zero_rate_is_a_no_op(self, seq):
        """The control: at rate 0 train and eval must agree EXACTLY.

        Without this, `test_input_dropout_is_live` could be passing on some
        other source of train/eval divergence (a norm layer in training mode,
        say) rather than on dropout.
        """
        keras.utils.set_random_seed(109)
        layer = CliffordRNN(
            units=self.UNITS,
            dropout_rate=0.0,
            recurrent_dropout_rate=0.0,
            seed=7,
        )
        train = self._np(layer(seq, training=True))
        evaluation = self._np(layer(seq, training=False))
        np.testing.assert_allclose(train, evaluation, rtol=0, atol=0)

    def test_the_mask_is_shared_across_timesteps(self, seq):
        """Variational dropout: ONE mask per sequence, not one per timestep.

        Measured structurally rather than statistically: the cell caches the
        mask, so a second `get_dropout_mask` on the same cell returns the
        identical tensor until `reset_dropout_mask()` is called.
        """
        keras.utils.set_random_seed(113)
        cell = CliffordRNNCell(units=self.UNITS, dropout_rate=self.RATE, seed=7)
        step = tf.random.normal([4, 6])
        first = self._np(cell.get_dropout_mask(step))
        second = self._np(cell.get_dropout_mask(step))
        np.testing.assert_array_equal(first, second)
        cell.reset_dropout_mask()
        third = self._np(cell.get_dropout_mask(step))
        assert float(np.max(np.abs(first - third))) > 0.0, (
            "reset_dropout_mask() did not clear the cache"
        )

    def test_rate_is_validated(self):
        """Out-of-range rates raise rather than silently clamping."""
        with pytest.raises(ValueError, match="dropout"):
            CliffordRNNCell(units=self.UNITS, dropout_rate=1.0)
        with pytest.raises(ValueError, match="recurrent_dropout"):
            CliffordRNNCell(units=self.UNITS, recurrent_dropout_rate=-0.1)


# ===========================================================================
# TestMasking
# ===========================================================================


class TestMasking:
    """``mask`` support is inherited from ``keras.layers.RNN`` -- prove it.

    ``CliffordRNN.call`` forwards ``mask`` to ``keras.layers.RNN.call``, and
    ``keras.layers.RNN.__init__`` sets ``supports_masking = True``, so a
    ``Masking`` / ``Embedding(mask_zero=True)`` upstream layer propagates
    automatically. "Inherited" is a claim, not a measurement, so these tests
    measure it: a masked-out timestep must not reach the state at all.
    """

    UNITS = 8

    @staticmethod
    def _np(t):
        return keras.ops.convert_to_numpy(t)

    def test_supports_masking_is_declared(self):
        assert CliffordRNN(units=self.UNITS).supports_masking is True

    def test_masked_tail_timesteps_do_not_reach_the_output(self):
        """Overwriting a MASKED tail step must not change the final output.

        This is the real masking contract: the state stops updating once the
        mask goes False, so garbage in the padded tail is inert.
        """
        keras.utils.set_random_seed(127)
        layer = CliffordRNN(units=self.UNITS)

        x = np.asarray(tf.random.normal([2, 5, 6]))
        mask = np.array(
            [[True, True, True, False, False],
             [True, True, True, True, True]]
        )
        base = self._np(layer(tf.constant(x), mask=tf.constant(mask)))

        # Perturb ONLY sample 0's masked tail.
        perturbed = x.copy()
        perturbed[0, 3:, :] += 100.0
        after = self._np(
            layer(tf.constant(perturbed), mask=tf.constant(mask))
        )

        d_masked = float(np.max(np.abs(base[0] - after[0])))
        d_unmasked = float(np.max(np.abs(base[1] - after[1])))
        assert d_masked == 0.0, (
            f"a MASKED timestep changed the output by {d_masked}; the mask is "
            "not gating the state update"
        )
        assert d_unmasked == 0.0, "sample 1 was not perturbed; sanity check"

    def test_the_mask_probe_is_not_vacuous(self):
        """Control: the SAME perturbation with no mask MUST change the output.

        Without this, `test_masked_tail_timesteps_do_not_reach_the_output`
        would pass for a layer that ignores its late timesteps entirely (or
        whose state saturates), not for a layer that honours the mask.
        """
        keras.utils.set_random_seed(127)
        layer = CliffordRNN(units=self.UNITS)
        x = np.asarray(tf.random.normal([2, 5, 6]))
        perturbed = x.copy()
        perturbed[0, 3:, :] += 100.0

        base = self._np(layer(tf.constant(x)))
        after = self._np(layer(tf.constant(perturbed)))
        assert float(np.max(np.abs(base[0] - after[0]))) > 1e-3, (
            "the unmasked control did not move: the probe cannot distinguish "
            "'mask honoured' from 'late timesteps ignored'"
        )

    def test_masking_layer_propagates(self):
        """End-to-end through ``keras.layers.Masking``, the documented path.

        The claim measured here is the strong one: a ZERO-padded length-5
        sequence run through ``Masking(0.0) -> CliffordRNN`` must give exactly
        the same answer as the unpadded length-3 prefix run through the SAME
        layer. That can only hold if the mask reaches the recurrence.
        """
        keras.utils.set_random_seed(131)
        rnn = CliffordRNN(units=self.UNITS)
        inp = keras.Input(shape=(5, 6))
        model = keras.Model(inp, rnn(keras.layers.Masking(0.0)(inp)))

        x = np.array(tf.random.normal([2, 3, 6]))
        padded = np.concatenate([x, np.zeros((2, 2, 6), dtype=x.dtype)], axis=1)

        padded_out = self._np(model(padded, training=False))
        prefix_out = self._np(rnn(tf.constant(x), training=False))
        np.testing.assert_allclose(padded_out, prefix_out, rtol=1e-6, atol=1e-6)

        # Anti-vacuity: without the mask the padding DOES move the answer.
        unmasked_out = self._np(rnn(tf.constant(padded), training=False))
        assert float(np.max(np.abs(unmasked_out - prefix_out))) > 1e-4, (
            "zero padding is inert even WITHOUT a mask, so this test cannot "
            "distinguish a working mask from a no-op one"
        )

    def test_zero_output_for_mask(self):
        """``zero_output_for_mask`` zeroes the masked positions' outputs."""
        keras.utils.set_random_seed(137)
        layer = CliffordRNN(
            units=self.UNITS, return_sequences=True, zero_output_for_mask=True
        )
        x = tf.random.normal([2, 5, 6])
        mask = tf.constant(
            [[True, True, True, False, False],
             [True, True, True, True, True]]
        )
        out = self._np(layer(x, mask=mask))
        assert float(np.max(np.abs(out[0, 3:]))) == 0.0, out[0, 3:]
        assert float(np.max(np.abs(out[1, 3:]))) > 0.0, "sample 1 must be live"


# ===========================================================================
# TestPrivateKerasImportContract
# ===========================================================================


class TestPrivateKerasImportContract:
    """``keras.src.layers.rnn.dropout_rnn_cell`` is a PRIVATE Keras path.

    ``clifford_rnn.py`` imports ``DropoutRNNCell`` from it at module scope,
    with no ``try``/``except``, so a Keras upgrade that moves or deletes that
    module makes the whole module unimportable.

    DECISION plan-2026-08-10T130454-3649c19e/D-029: this suite is the chosen
    remedy INSTEAD of a ``try``/``except`` fallback, because a fallback cannot
    keep the promise the old in-code comment made. The inheritance is not
    decorative: ``keras/src/layers/rnn/rnn.py`` gates BOTH
    ``_maybe_config_dropout_masks`` (rnn.py:436) and
    ``_maybe_reset_dropout_masks`` (rnn.py:449) on
    ``isinstance(cell, DropoutRNNCell)``. Without the base class the per-batch
    ``reset_dropout_mask()`` never fires, so ONE dropout mask would be reused
    for an entire training run -- a silent, hard-to-see degradation. A loud
    import error is strictly better than that. See decisions.md D-029.
    """

    def test_the_private_keras_module_still_exists(self):
        """Fail LEGIBLY, here, rather than as a collection error everywhere."""
        try:
            from keras.src.layers.rnn.dropout_rnn_cell import (  # noqa: F401
                DropoutRNNCell,
            )
        except ImportError as exc:  # pragma: no cover - upgrade tripwire
            pytest.fail(
                "keras.src.layers.rnn.dropout_rnn_cell.DropoutRNNCell has "
                f"moved or been removed ({exc}). This is a PRIVATE Keras path "
                "that clifford_rnn.py inherits from at module scope, so this "
                "breaks the whole module, not just dropout. Fix: re-point the "
                "import, or re-implement the mask lifecycle -- note that "
                "keras.layers.RNN gates reset_dropout_mask() on "
                "isinstance(cell, DropoutRNNCell), so simply dropping the "
                "base class leaks ONE dropout mask across every batch."
            )

    def test_the_cell_is_recognised_by_keras_rnn(self):
        """The reason the base class is inherited at all.

        Asserts the *behavioural* consequence, not just the import: an
        ``isinstance`` check that silently went False would leave every
        assertion in :class:`TestDropout` still passing (masks are created
        lazily by our local implementations) while the per-batch reset died.
        """
        from keras.src.layers.rnn.dropout_rnn_cell import DropoutRNNCell

        cell = CliffordRNNCell(units=8, dropout_rate=0.25)
        assert isinstance(cell, DropoutRNNCell), (
            "keras.layers.RNN gates its dropout-mask lifecycle on this "
            "isinstance check; CliffordRNNCell must inherit DropoutRNNCell"
        )

    def test_the_local_mask_api_is_complete(self):
        """All four mixin methods are implemented locally, not inherited.

        The class overrides them so the mask lifecycle is readable in one file.
        A future edit that deletes one would fall back to the mixin's version,
        which reads ``self.dropout`` / ``self.recurrent_dropout`` -- names this
        cell does NOT have (they are ``*_rate`` here), i.e. an AttributeError
        at the first training step.
        """
        for name in (
            "get_dropout_mask",
            "get_recurrent_dropout_mask",
            "reset_dropout_mask",
            "reset_recurrent_dropout_mask",
        ):
            assert name in CliffordRNNCell.__dict__, (
                f"{name} is no longer defined on CliffordRNNCell; the "
                "inherited mixin version reads self.dropout / "
                "self.recurrent_dropout, which do not exist on this cell"
            )


# ===========================================================================
# TestModuleDocstringExamples
# ===========================================================================


class TestModuleDocstringExamples:
    """Every code example in ``clifford_rnn.py`` must actually RUN.

    The module docstring shipped ``CliffordRNN(64, ..., dropout=0.1)`` (the
    kwarg is ``dropout_rate``) and ``from clifford_rnn import CliffordRNN``
    (not an importable path) -- both raise. Copy-pasteable examples are part of
    the public surface; this pins them.
    """

    def test_module_docstring_usage_block(self):
        x = keras.Input((None, 32))
        assert CliffordRNN(
            64, return_sequences=True, dropout_rate=0.1
        )(x).shape == (None, None, 64)
        assert keras.layers.RNN(
            CliffordRNNCell(64), return_sequences=True
        )(x).shape == (None, None, 64)
        assert keras.layers.Bidirectional(
            CliffordRNN(64, return_sequences=True)
        )(x).shape == (None, None, 128)
        assert keras.layers.RNN(
            [CliffordRNNCell(64), CliffordRNNCell(64)]
        )(x).shape == (None, 64)

    def test_class_docstring_example(self):
        x = keras.Input((None, 32))
        y = CliffordRNN(64, shifts=[1, 2, 4], return_sequences=True)(x)
        assert y.shape == (None, None, 64)

    def test_no_docstring_example_uses_a_renamed_kwarg(self):
        """Guards the whole module against the rename regressing.

        `dropout=` / `recurrent_dropout=` / `tcn_dropout=` are the three names
        this repo renamed to `*_rate`; none may reappear in a docstring here.
        """
        import re

        import dl_techniques.layers.geometric.clifford_rnn as mod

        src = open(mod.__file__).read()
        offenders = re.findall(
            r"\b(?:dropout|recurrent_dropout|tcn_dropout)=(?!\w)", src
        )
        assert not offenders, (
            f"found {len(offenders)} use(s) of a pre-rename kwarg name in "
            "clifford_rnn.py; the accepted names are dropout_rate / "
            "recurrent_dropout_rate"
        )
