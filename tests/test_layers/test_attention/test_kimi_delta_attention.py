"""Test suite for the KimiDeltaAttention (KDA) layer.

Covers initialization and argument validation, output shapes, float64
equality of the chunk form, the recurrent form and an independent NumPy
oracle of paper Eq. 1, gradients, causality, padding masks, ``mixed_float16``,
``get_config`` and the ``.keras`` round trip, factory construction, the
traced-graph paths (``fit``, ``predict``, ``jit_compile``), a lazy build
inside ``fit`` in a cold process, the carried state
``(S, conv_q, conv_k, conv_v)`` (split calls, token-by-token decode, padding
masks across calls), the symbolic ``return_state`` structure, and the fla
defaults for the short-convolution kernel and the output-norm epsilon.

All tests run on CPU with tiny dimensions.
"""

import doctest
import inspect
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.attention.factory import (
    ATTENTION_REGISTRY,
    create_attention_layer,
)
from dl_techniques.layers.attention.kimi_delta_attention import KimiDeltaAttention

# ---------------------------------------------------------------------

MODES = ("chunk", "recurrent")

# Small config shared by most tests: d_k != d_v, three heads, a chunk size that
# divides none of the sequence lengths used below.
BASE_CONFIG: Dict[str, Any] = dict(
    dim=32, num_heads=3, head_dim=8, value_head_dim=12, conv_size=4, chunk_size=7,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _make(**overrides: Any) -> KimiDeltaAttention:
    """Build an unbuilt layer from ``BASE_CONFIG`` with ``overrides`` applied."""
    config = dict(BASE_CONFIG)
    config.update(overrides)
    return KimiDeltaAttention(**config)


def _randomize(layer: KimiDeltaAttention, seed: int) -> None:
    """Overwrite every weight with seeded values large enough to exercise the math.

    The default init leaves the gates and the short convolution close to
    identity-like regimes; random weights make every term matter. Each value
    takes its variable's own dtype (``A_log``/``dt_bias`` stay float32 under
    a float64 policy).
    """
    rng = np.random.default_rng(seed)
    values = []
    for weight in layer.weights:
        path = weight.path
        shape = tuple(weight.shape)
        if path.endswith("A_log"):
            value = np.log(rng.uniform(0.3, 6.0, shape))
        elif path.endswith("dt_bias"):
            value = rng.normal(0.0, 1.0, shape)
        elif "o_norm" in path:
            value = rng.uniform(0.5, 1.5, shape)
        elif "bias" in path:
            value = rng.normal(0.0, 0.3, shape)
        else:
            value = rng.normal(0.0, 1.5 / np.sqrt(max(shape[0], 1)), shape)
        values.append(value.astype(np.asarray(weight).dtype))
    layer.set_weights(values)


def _built(seed: int, n: int, batch: int = 2, **overrides: Any) -> KimiDeltaAttention:
    """A built layer with randomized weights."""
    layer = _make(**overrides)
    layer.build((batch, n, layer.dim))
    _randomize(layer, seed)
    return layer


def _inputs(seed: int, shape, scale: float = 1.0, dtype: str = "float32") -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, scale, shape).astype(dtype)


def _close(actual, expected, atol: float) -> None:
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), atol=atol, rtol=0
    )


# ---------------------------------------------------------------------
# Traced graphs: fit, predict, jit_compile
# ---------------------------------------------------------------------


class TestTracedGraphs:
    """Both modes run in traced graphs.

    The default chunk mode used to raise under any trace (``ops.tril``), and
    training under ``jit_compile=True`` failed in both modes because
    ``keras.ops.scan`` sets no ``maximum_iterations`` ("XLA compilation
    requires a fixed tensor list size").
    """

    @staticmethod
    def _model(mode: str, jit_compile: bool) -> keras.Model:
        inp = keras.Input((None, 16))
        layer = KimiDeltaAttention(dim=16, num_heads=2, chunk_size=4, mode=mode)
        model = keras.Model(inp, layer(inp))
        model.compile(optimizer="adam", loss="mse", jit_compile=jit_compile)
        return model

    @pytest.mark.parametrize("mode", MODES)
    def test_fit(self, mode):
        model = self._model(mode, jit_compile=False)
        x = _inputs(0, (4, 10, 16))
        history = model.fit(x, _inputs(1, (4, 10, 16)), epochs=1, batch_size=2, verbose=0)
        assert np.isfinite(history.history["loss"][0])

    @pytest.mark.parametrize("mode", MODES)
    def test_predict_matches_eager(self, mode):
        model = self._model(mode, jit_compile=False)
        x = _inputs(2, (3, 10, 16))
        _close(model.predict(x, verbose=0), model(x), atol=1e-5)

    @pytest.mark.parametrize("mode", MODES)
    def test_jit_compile_matches_eager(self, mode):
        model = self._model(mode, jit_compile=True)
        x = _inputs(3, (2, 10, 16))
        _close(model.predict(x, verbose=0), model(x), atol=1e-5)
        history = model.fit(x, _inputs(4, (2, 10, 16)), epochs=1, verbose=0)
        assert np.isfinite(history.history["loss"][0])


# ---------------------------------------------------------------------
# A_log / dt_bias initialization
# ---------------------------------------------------------------------

_COLD_LAZY_BUILD_SCRIPT = """
import numpy as np
import keras
from dl_techniques.layers.attention.kimi_delta_attention import KimiDeltaAttention


class Wrapper(keras.Model):
    def __init__(self):
        super().__init__()
        self.kda = KimiDeltaAttention(dim=16, num_heads=2, mode="recurrent")
        self.head = keras.layers.Dense(1)

    def call(self, x):
        return self.head(self.kda(x))


model = Wrapper()
model.compile(optimizer="adam", loss="mse")
x = np.random.default_rng(0).normal(0.0, 1.0, (4, 9, 16)).astype("float32")
history = model.fit(x, np.zeros((4, 9, 1), "float32"), epochs=1, batch_size=2, verbose=0)
assert np.isfinite(history.history["loss"][0])
print("COLD_LAZY_BUILD_OK")
"""


class TestDecayInitialization:
    """``A_log``/``dt_bias`` are drawn with NumPy, never with ``keras.random``."""

    def test_cold_subclassed_model_builds_lazily_inside_fit(self, tmp_path):
        # A fresh interpreter is the only way to guarantee the global Keras
        # SeedGenerator has never been touched; in this process earlier tests
        # have created it, which hides the failure. Recurrent mode isolates the
        # initializer from the chunk-mode graph fix.
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT / "src")] + [p for p in [env.get("PYTHONPATH")] if p]
        )
        result = subprocess.run(
            [sys.executable, "-c", _COLD_LAZY_BUILD_SCRIPT],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600,
        )
        assert "COLD_LAZY_BUILD_OK" in result.stdout, (
            f"cold lazy build failed (exit {result.returncode}):\n"
            f"{result.stderr[-3000:]}"
        )

    def test_draw_is_reproducible_under_set_random_seed(self):
        drawn = []
        for _ in range(2):
            keras.utils.set_random_seed(1234)
            layer = KimiDeltaAttention(dim=32, num_heads=4)
            layer.build((1, 5, 32))
            drawn.append((np.asarray(layer.A_log), np.asarray(layer.dt_bias)))
        np.testing.assert_array_equal(drawn[0][0], drawn[1][0])
        np.testing.assert_array_equal(drawn[0][1], drawn[1][1])

    def test_draw_follows_the_documented_ranges(self):
        layer = KimiDeltaAttention(
            dim=64, num_heads=4, a_init_range=(2.0, 5.0), dt_min=0.01, dt_max=0.05,
        )
        layer.build((1, 5, 64))
        rate = np.exp(np.asarray(layer.A_log, dtype=np.float64))
        assert rate.shape == (4,)
        assert np.all((rate >= 2.0 - 1e-5) & (rate <= 5.0 + 1e-5))
        dt = np.log1p(np.exp(np.asarray(layer.dt_bias, dtype=np.float64)))
        assert dt.shape == (64,)
        assert np.all((dt >= 0.01 * (1 - 1e-4)) & (dt <= 0.05 * (1 + 1e-4)))
        assert layer.A_log.dtype == "float32" and layer.dt_bias.dtype == "float32"


# ---------------------------------------------------------------------
# Carried state: (S, conv_q, conv_k, conv_v)
# ---------------------------------------------------------------------


class TestStateCarry:
    """Split calls and token-by-token decode equal one full call."""

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("conv_size", (0, 1, 4))
    def test_state_structure(self, mode, conv_size):
        layer = _built(0, 9, mode=mode, conv_size=conv_size)
        _, state = layer(_inputs(0, (2, 9, 32)), return_state=True)
        assert isinstance(state, tuple) and len(state) == 4
        s, conv_q, conv_k, conv_v = state
        window = max(conv_size - 1, 0)
        assert tuple(s.shape) == (2, 3, 8, 12)
        assert tuple(conv_q.shape) == (2, window, 24)
        assert tuple(conv_k.shape) == (2, window, 24)
        assert tuple(conv_v.shape) == (2, window, 36)

    @staticmethod
    def _split_and_decode_diffs(layer, x, n_decode: int = 10):
        """Max |difference| against one full call, outputs and final state, of
        a 24+16 split call and of a token-by-token decode of the first
        ``n_decode`` tokens (each eager one-token call is slow on CPU)."""
        y_full, state_full = layer(x, return_state=True)
        y1, state1 = layer(x[:, :24], return_state=True)
        y2, state2 = layer(x[:, 24:], initial_state=state1, return_state=True)
        split = max(
            [float(np.abs(np.concatenate([y1, y2], axis=1) - np.asarray(y_full)).max())]
            + [float(np.abs(np.asarray(a) - np.asarray(b)).max(initial=0.0))
               for a, b in zip(state2, state_full)]
        )
        y_head, state_head = layer(x[:, :n_decode], return_state=True)
        state, outputs = None, []
        for t in range(n_decode):
            y_t, state = layer(x[:, t:t + 1], initial_state=state, return_state=True)
            outputs.append(np.asarray(y_t))
        decode = max(
            [float(np.abs(np.concatenate(outputs, axis=1) - np.asarray(y_head)).max())]
            + [float(np.abs(np.asarray(a) - np.asarray(b)).max(initial=0.0))
               for a, b in zip(state, state_head)]
        )
        return split, decode

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("conv_size", (0, 1, 4))
    def test_split_and_decode_equal_the_full_call_in_float32(self, mode, conv_size):
        # The default initialization, float32: the documented tolerance.
        # chunk_size=16 does not divide 40, and the 24+16 split crosses a
        # chunk boundary.
        layer = KimiDeltaAttention(
            dim=32, num_heads=2, conv_size=conv_size, chunk_size=16, mode=mode
        )
        x = _inputs(1, (2, 40, 32))
        split, decode = self._split_and_decode_diffs(layer, x)
        assert split <= 1e-5, split
        assert decode <= 1e-5, decode

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("conv_size", (0, 4))
    def test_split_and_decode_are_exact_in_float64(self, float64_policy, mode, conv_size):
        # Random weights put every term to work; float64 shows the carry is
        # exact rather than inside float32 rounding (which reaches ~1e-5 at
        # |y| ~ 3 with these weights).
        layer = _built(2, 40, mode=mode, conv_size=conv_size)
        x = _inputs(2, (2, 40, 32), dtype="float64")
        assert np.asarray(layer(x[:, :3])).dtype == np.float64
        split, decode = self._split_and_decode_diffs(layer, x)
        assert split <= 1e-12, split
        assert decode <= 1e-12, decode

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("n_real", (2, 10))
    def test_padded_segments_carry_as_if_the_pads_were_deleted(self, mode, n_real):
        # Row 0 is right-padded, row 1 left-padded, in the first segment; the
        # second segment is all padding; the third continues with real tokens.
        # Every kept row must equal the unpadded run of the real tokens.
        layer = _built(3, 30, mode=mode)
        real = _inputs(3, (2, n_real + 12, 32))
        y_ref = np.asarray(layer(real))
        width, pads = 16, 16 - n_real
        garbage = _inputs(4, (2, width, 32), scale=3.0)

        seg1, keep1 = garbage.copy(), np.zeros((2, width), "float32")
        seg1[0, :n_real], keep1[0, :n_real] = real[0, :n_real], 1.0
        seg1[1, pads:], keep1[1, pads:] = real[1, :n_real], 1.0
        y1, state = layer(seg1, attention_mask=keep1, return_state=True)

        seg2 = _inputs(5, (2, 5, 32), scale=3.0)
        y2, state = layer(seg2, attention_mask=np.zeros((2, 5), "float32"),
                          initial_state=state, return_state=True)
        y3 = layer(real[:, n_real:], initial_state=state)

        y1 = np.asarray(y1)
        _close(y1[0, :n_real], y_ref[0, :n_real], atol=1e-5)
        _close(y1[1, pads:], y_ref[1, :n_real], atol=1e-5)
        assert np.all(y1[0, n_real:] == 0.0) and np.all(y1[1, :pads] == 0.0)
        assert np.all(np.asarray(y2) == 0.0)
        _close(y3, y_ref[:, n_real:], atol=1e-5)

    @pytest.mark.parametrize("mode", MODES)
    def test_a_static_length_shorter_than_the_conv_builds(self, mode):
        # The 'valid' convolution sees conv_size - 1 carried rows in front of
        # the sequence; building it for N alone fails at N < conv_size.
        layer = _make(mode=mode, conv_size=4)
        layer.build((2, 1, 32))
        y, state = layer(_inputs(7, (2, 1, 32)), return_state=True)
        assert tuple(y.shape) == (2, 1, 32)
        assert tuple(state[1].shape) == (2, 3, 24)

    def test_initial_state_must_be_the_returned_tuple(self):
        layer = _built(4, 6)
        x = _inputs(4, (2, 6, 32))
        _, state = layer(x, return_state=True)
        with pytest.raises(ValueError, match="initial_state"):
            layer(x, initial_state=state[0])
        with pytest.raises(ValueError, match="initial_state"):
            layer(x, initial_state=state[:3])

    def test_the_docstring_example_runs_and_its_continuation_is_exact(self):
        parser = doctest.DocTestParser()
        globs: Dict[str, Any] = {}
        test = parser.get_doctest(
            KimiDeltaAttention.__doc__, globs, "KimiDeltaAttention", None, 0
        )
        assert test.examples, "the class docstring has no runnable example"
        runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
        runner.run(test, clear_globs=False)
        assert runner.failures == 0, f"{runner.failures} docstring example(s) failed"
        names = test.globs  # DocTest runs in a copy of `globs`
        for name in ("x", "y", "y2"):
            assert name in names, f"the example no longer defines {name!r}"
        _close(names["y2"], np.asarray(names["y"])[:, 64:], atol=1e-5)


# ---------------------------------------------------------------------
# Symbolic calls
# ---------------------------------------------------------------------


class TestSymbolicState:
    """A ``keras.Input`` call with ``return_state=True`` returns the state too."""

    @pytest.mark.parametrize("conv_size", (0, 4))
    def test_functional_model_returns_and_consumes_the_state(self, conv_size):
        layer = KimiDeltaAttention(dim=16, num_heads=2, conv_size=conv_size, chunk_size=4)
        inp = keras.Input((None, 16))
        out = layer(inp, return_state=True)
        assert isinstance(out, (tuple, list)) and len(out) == 2
        y, state = out
        assert isinstance(state, (tuple, list)) and len(state) == 4
        window = max(conv_size - 1, 0)
        assert tuple(y.shape) == (None, None, 16)
        assert tuple(state[0].shape) == (None, 2, 8, 8)
        assert tuple(state[1].shape) == (None, window, 16)
        assert tuple(state[3].shape) == (None, window, 16)

        prefill = keras.Model(inp, [y, *state])
        state_inputs = [
            keras.Input((2, 8, 8)),
            keras.Input((window, 16)),
            keras.Input((window, 16)),
            keras.Input((window, 16)),
        ]
        step_inp = keras.Input((None, 16))
        y_step = layer(step_inp, initial_state=tuple(state_inputs))
        decode = keras.Model([step_inp, *state_inputs], y_step)

        x = _inputs(6, (2, 9, 16))
        y_eager = np.asarray(layer(x))
        prefill_out = prefill.predict(x[:, :5], verbose=0)
        y_rest = decode.predict([x[:, 5:], *prefill_out[1:]], verbose=0)
        _close(prefill_out[0], y_eager[:, :5], atol=1e-5)
        _close(y_rest, y_eager[:, 5:], atol=1e-5)


# ---------------------------------------------------------------------
# Defaults taken from fla
# ---------------------------------------------------------------------


class TestReferenceDefaults:
    """Short-conv kernel init and ``o_norm`` epsilon follow fla / torch."""

    @pytest.mark.parametrize("conv_size", (2, 4, 9))
    def test_conv_kernel_is_torch_default_uniform(self, conv_size):
        # torch's default Conv1d init for a depthwise kernel is
        # U(-1/sqrt(fan_in), 1/sqrt(fan_in)) with fan_in = conv_size, so the
        # std is 1/sqrt(3 * conv_size): 0.289 at conv_size=4.
        layer = KimiDeltaAttention(
            dim=512, num_heads=4, head_dim=128, value_head_dim=128, conv_size=conv_size,
        )
        layer.build((1, 8, 512))
        bound = 1.0 / np.sqrt(conv_size)
        expected_std = bound / np.sqrt(3.0)
        for conv in (layer.q_conv, layer.k_conv, layer.v_conv):
            kernel = np.asarray(conv.kernel)
            assert np.abs(kernel).max() <= bound + 1e-7
            assert abs(kernel.std() / expected_std - 1.0) < 0.08
        # the three kernels draw independently
        assert not np.array_equal(np.asarray(layer.q_conv.kernel), np.asarray(layer.k_conv.kernel))

    def test_conv_kernel_std_at_the_default_conv_size(self):
        layer = KimiDeltaAttention(dim=512, num_heads=4, head_dim=128, value_head_dim=128)
        layer.build((1, 8, 512))
        std = float(np.asarray(layer.q_conv.kernel).std())
        assert 0.27 < std < 0.31, std

    def test_projection_kernels_keep_kernel_initializer(self):
        layer = KimiDeltaAttention(dim=32, num_heads=2, kernel_initializer="zeros")
        layer.build((1, 4, 32))
        assert not np.any(np.asarray(layer.q_proj.kernel))
        assert np.any(np.asarray(layer.q_conv.kernel))

    def test_output_norm_epsilon_defaults_to_1e_5(self):
        layer = KimiDeltaAttention(dim=32, num_heads=2)
        assert layer.o_norm.epsilon == 1e-5

    def test_output_norm_epsilon_can_be_overridden(self):
        layer = KimiDeltaAttention(
            dim=32, num_heads=2, output_norm_kwargs={"epsilon": 1e-3}
        )
        assert layer.o_norm.epsilon == 1e-3


# ---------------------------------------------------------------------
# Initialization and validation
# ---------------------------------------------------------------------


class TestInitialization:
    """Derived defaults and argument refusals."""

    def test_derived_defaults(self):
        layer = KimiDeltaAttention(dim=64, num_heads=4)
        assert (layer.head_dim, layer.value_head_dim, layer.gate_low_rank_dim) == (16, 16, 16)
        assert (layer.key_dim, layer.value_dim) == (64, 64)
        assert layer.mode == "chunk" and layer.chunk_size == 64 and layer.conv_size == 4
        assert layer.conv_window == 3
        assert layer.scale == pytest.approx(0.25)
        assert layer.a_init_range == (1.0, 16.0)

    def test_explicit_head_dims_need_no_divisibility(self):
        layer = KimiDeltaAttention(dim=30, num_heads=4, head_dim=8, value_head_dim=5)
        assert (layer.key_dim, layer.value_dim) == (32, 20)

    def test_optional_sublayers_are_absent_when_disabled(self):
        layer = KimiDeltaAttention(
            dim=16, num_heads=2, conv_size=0, use_output_gate=False,
            output_norm_type=None, dropout_rate=0.0,
        )
        assert layer.q_conv is None and layer.k_conv is None and layer.v_conv is None
        assert layer.g_up is None and layer.g_down is None
        assert layer.o_norm is None and layer.dropout_layer is None
        assert layer.conv_window == 0

    @pytest.mark.parametrize("kwargs", [
        dict(dim=0, num_heads=2),
        dict(dim=32, num_heads=0),
        dict(dim=30, num_heads=4),
        dict(dim=32, num_heads=4, head_dim=0),
        dict(dim=32, num_heads=4, value_head_dim=0),
        dict(dim=32, num_heads=4, conv_size=-1),
        dict(dim=32, num_heads=4, gate_low_rank_dim=0),
        dict(dim=32, num_heads=4, chunk_size=0),
        dict(dim=32, num_heads=4, mode="parallel"),
        dict(dim=32, num_heads=4, a_init_range=(0.0, 1.0)),
        dict(dim=32, num_heads=4, a_init_range=(5.0, 2.0)),
        dict(dim=32, num_heads=4, dt_min=0.0),
        dict(dim=32, num_heads=4, dt_min=0.2, dt_max=0.1),
        dict(dim=32, num_heads=4, dt_init_floor=0.0),
        dict(dim=32, num_heads=4, dropout_rate=1.5),
        dict(dim=32, num_heads=4, eps=0.0),
    ])
    def test_invalid_arguments_raise(self, kwargs):
        with pytest.raises(ValueError):
            KimiDeltaAttention(**kwargs)

    def test_build_refuses_a_wrong_input(self):
        with pytest.raises(ValueError, match="3D"):
            KimiDeltaAttention(dim=16, num_heads=2).build((2, 16))
        with pytest.raises(ValueError, match="must match"):
            KimiDeltaAttention(dim=16, num_heads=2).build((2, 5, 12))

    def test_a_rank_3_mask_is_refused(self):
        layer = KimiDeltaAttention(dim=16, num_heads=2)
        x = _inputs(0, (2, 5, 16))
        with pytest.raises(ValueError, match="rank-2"):
            layer(x, attention_mask=np.ones((2, 5, 5), "float32"))


# ---------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------


class TestShapes:
    """Output shape equals the input shape for assorted head geometries."""

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("batch, n, dim, heads, head_dim, value_head_dim", [
        (1, 1, 16, 2, None, None),
        (2, 13, 32, 4, None, None),
        (3, 9, 24, 3, 16, 4),
        (2, 70, 32, 1, 8, 8),
    ])
    def test_output_shape_and_finite(self, mode, batch, n, dim, heads, head_dim, value_head_dim):
        layer = KimiDeltaAttention(
            dim=dim, num_heads=heads, head_dim=head_dim,
            value_head_dim=value_head_dim, chunk_size=8, mode=mode,
        )
        y = np.asarray(layer(_inputs(0, (batch, n, dim))))
        assert y.shape == (batch, n, dim)
        assert np.all(np.isfinite(y))
        assert layer.compute_output_shape((batch, n, dim)) == (batch, n, dim)

    def test_disabled_options_still_give_the_input_shape(self):
        layer = KimiDeltaAttention(
            dim=16, num_heads=2, conv_size=0, use_output_gate=False,
            output_norm_type=None, use_bias=True,
        )
        assert np.asarray(layer(_inputs(1, (2, 6, 16)))).shape == (2, 6, 16)


# ---------------------------------------------------------------------
# Independent float64 NumPy oracle of paper Eq. 1 and Eq. 10
# ---------------------------------------------------------------------
# Shares no code with the layer: explicit S_t = (I - b k k^T) Diag(a) S_{t-1}
# + b k v^T, o_t = S_t^T q_t, one token, batch row and head at a time. It
# reads the layer's weights and nothing else.


def _np_silu(x):
    return x / (1.0 + np.exp(-x))


def _np_softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _np_dense(sub, x):
    y = x @ np.asarray(sub.kernel, dtype=np.float64)
    if sub.use_bias:
        y = y + np.asarray(sub.bias, dtype=np.float64)
    return y


def _np_causal_depthwise_conv(conv, x):
    # Keras grouped Conv1D kernel (k, 1, C); causal = k-1 zeros in front;
    # cross-correlation.
    kernel = np.asarray(conv.kernel, dtype=np.float64)[:, 0, :]
    k = kernel.shape[0]
    b, n, c = x.shape
    padded = np.concatenate([np.zeros((b, k - 1, c)), x], axis=1)
    return sum(padded[:, j:j + n, :] * kernel[j] for j in range(k))


def _np_oracle(layer, x, keep=None):
    """Output ``(B, N, dim)`` and final ``S`` of ``layer`` on ``x``, float64."""
    x = np.asarray(x, dtype=np.float64)
    b, n, _ = x.shape
    h, dk, dv = layer.num_heads, layer.head_dim, layer.value_head_dim
    if keep is not None:
        keep = np.asarray(keep, dtype=np.float64)
        x = x * keep[..., None]
    q, k, v = (_np_dense(layer.q_proj, x), _np_dense(layer.k_proj, x),
               _np_dense(layer.v_proj, x))
    if keep is not None:
        q, k, v = q * keep[..., None], k * keep[..., None], v * keep[..., None]
    if layer.q_conv is not None:
        q = _np_causal_depthwise_conv(layer.q_conv, q)
        k = _np_causal_depthwise_conv(layer.k_conv, k)
        v = _np_causal_depthwise_conv(layer.v_conv, v)
    q, k, v = _np_silu(q), _np_silu(k), _np_silu(v)
    f = _np_dense(layer.f_up, _np_dense(layer.f_down, x)).reshape(b, n, h, dk)
    beta = _np_sigmoid(_np_dense(layer.b_proj, x))
    q, k, v = q.reshape(b, n, h, dk), k.reshape(b, n, h, dk), v.reshape(b, n, h, dv)
    rate = np.exp(np.asarray(layer.A_log, dtype=np.float64))
    dt_bias = np.asarray(layer.dt_bias, dtype=np.float64).reshape(h, dk)
    alpha = np.exp(-rate[None, None, :, None] * _np_softplus(f + dt_bias))
    q = q / np.sqrt((q ** 2).sum(-1, keepdims=True) + layer.eps) / np.sqrt(dk)
    k = k / np.sqrt((k ** 2).sum(-1, keepdims=True) + layer.eps)
    if keep is not None:
        beta = beta * keep[..., None]
        alpha = np.where(keep[..., None, None] > 0, alpha, 1.0)
    state = np.zeros((b, h, dk, dv))
    o = np.zeros((b, n, h, dv))
    eye = np.eye(dk)
    for t in range(n):
        for i in range(b):
            for j in range(h):
                kt = k[i, t, j][:, None]
                transition = (eye - beta[i, t, j] * (kt @ kt.T)) @ np.diag(alpha[i, t, j])
                state[i, j] = transition @ state[i, j] + beta[i, t, j] * (kt @ v[i, t, j][None, :])
                o[i, t, j] = state[i, j].T @ q[i, t, j]
    if layer.o_norm is not None:
        scale = np.asarray(layer.o_norm.scale, dtype=np.float64)
        o = o / np.sqrt((o ** 2).mean(-1, keepdims=True) + layer.o_norm.epsilon) * scale
    if layer.g_up is not None:
        o = o * _np_sigmoid(_np_dense(layer.g_up, _np_dense(layer.g_down, x))).reshape(b, n, h, dv)
    out = _np_dense(layer.o_proj, o.reshape(b, n, h * dv))
    if keep is not None:
        out = out * keep[..., None]
    return out, state


class TestFloat64Equivalence:
    """Chunk form == recurrent form == NumPy oracle, float64, to 1e-12."""

    @pytest.mark.parametrize("n", (1, 17))
    @pytest.mark.parametrize("overrides", [
        dict(),
        dict(chunk_size=3),
        dict(conv_size=0),
        dict(use_bias=True, output_norm_type=None),
        dict(use_output_gate=False, head_dim=16, value_head_dim=4, num_heads=2),
    ], ids=["base", "chunk3", "conv0", "bias_nonorm", "nogate_dk16_dv4"])
    def test_chunk_recurrent_and_oracle_agree(self, float64_policy, overrides, n):
        layer = _built(10, n, **overrides)
        x = _inputs(10, (2, n, 32), dtype="float64")
        y_chunk, s_chunk = layer(x, return_state=True)
        layer.mode = "recurrent"
        y_rec, s_rec = layer(x, return_state=True)
        y_ref, s_ref = _np_oracle(layer, x)
        assert np.asarray(y_chunk).dtype == np.float64
        _close(y_chunk, y_rec, atol=1e-12)
        _close(y_chunk, y_ref, atol=1e-12)
        _close(y_rec, y_ref, atol=1e-12)
        _close(s_chunk[0], s_ref, atol=1e-12)
        _close(s_rec[0], s_ref, atol=1e-12)

    def test_strong_decay_stays_exact(self, float64_policy):
        # exp(A_log) in 50..100: the chunk form's exponents are all <= 0.
        layer = _make(a_init_range=(50.0, 100.0), chunk_size=16)
        layer.build((2, 40, 32))
        a_log, dt_bias = np.asarray(layer.A_log), np.asarray(layer.dt_bias)
        _randomize(layer, 11)
        layer.A_log.assign(a_log)
        layer.dt_bias.assign(dt_bias)
        x = _inputs(11, (2, 40, 32), dtype="float64")
        y_chunk = layer(x)
        layer.mode = "recurrent"
        _close(y_chunk, layer(x), atol=1e-12)
        _close(y_chunk, _np_oracle(layer, x)[0], atol=1e-12)

    def test_masked_run_matches_the_oracle(self, float64_policy):
        layer = _built(12, 13)
        x = _inputs(12, (2, 13, 32), dtype="float64")
        keep = np.ones((2, 13))
        keep[0, 9:] = 0.0
        keep[1, :4] = 0.0
        y, s = layer(x, attention_mask=keep, return_state=True)
        y_ref, s_ref = _np_oracle(layer, x, keep=keep)
        _close(y, y_ref, atol=1e-12)
        _close(s[0], s_ref, atol=1e-12)


# ---------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------


class TestGradients:
    """Every trainable variable gets a finite gradient; both modes agree."""

    @staticmethod
    def _gradients(layer, x):
        x = tf.convert_to_tensor(x)
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = tf.reduce_sum(tf.square(layer(x, training=True)))
        return tape.gradient(loss, [x] + layer.trainable_variables)

    @pytest.mark.parametrize("mode", MODES)
    def test_gradients_are_finite_and_complete(self, mode):
        layer = _built(13, 11, mode=mode, use_bias=True)
        grads = self._gradients(layer, _inputs(13, (2, 11, 32)))
        assert len(layer.trainable_variables) == 21
        for grad, var in zip(grads[1:], layer.trainable_variables):
            assert grad is not None, var.path
            assert np.all(np.isfinite(np.asarray(grad))), var.path
            assert np.any(np.asarray(grad) != 0.0), var.path

    def test_chunk_and_recurrent_gradients_agree_in_float64(self, float64_policy):
        layer = _built(14, 19)
        x = _inputs(14, (2, 19, 32), dtype="float64")
        g_chunk = self._gradients(layer, x)
        layer.mode = "recurrent"
        g_rec = self._gradients(layer, x)
        for a, b in zip(g_chunk, g_rec):
            a, b = np.asarray(a), np.asarray(b)
            assert np.abs(a - b).max() <= 1e-10 * max(np.abs(b).max(), 1.0)


# ---------------------------------------------------------------------
# Causality and padding
# ---------------------------------------------------------------------


class TestCausalityAndPadding:
    """Earlier outputs ignore later tokens; padding acts as deletion."""

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("n, p", [(20, 11), (20, 7), (17, 16)])
    def test_a_later_token_never_moves_an_earlier_output(self, mode, n, p):
        layer = _built(15, n, mode=mode)
        x = _inputs(15, (2, n, 32))
        y0 = np.asarray(layer(x))
        x2 = x.copy()
        x2[:, p] += _inputs(16, (2, 32), scale=5.0)
        y1 = np.asarray(layer(x2))
        np.testing.assert_array_equal(y1[:, :p], y0[:, :p])
        assert np.abs(y1[:, p:] - y0[:, p:]).max() > 1e-3

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("use_bias", (False, True))
    def test_left_and_right_padding_equal_the_unpadded_run(self, mode, use_bias):
        n, pads = 11, 6
        layer = _built(17, n + pads, mode=mode, use_bias=use_bias)
        real = _inputs(17, (2, n, 32))
        y_ref, s_ref = layer(real, return_state=True)
        garbage = _inputs(18, (2, pads, 32), scale=3.0)
        keep = np.concatenate([np.ones((2, n)), np.zeros((2, pads))], 1).astype("float32")
        cases = {
            "right": (np.concatenate([real, garbage], 1), keep, slice(0, n), slice(n, None)),
            "left": (np.concatenate([garbage, real], 1), keep[:, ::-1].copy(),
                     slice(pads, None), slice(0, pads)),
        }
        for name, (x, mask, kept, padded) in cases.items():
            y, s = layer(x, attention_mask=mask, return_state=True)
            y = np.asarray(y)
            _close(y[:, kept], y_ref, atol=1e-5)
            assert np.all(y[:, padded] == 0.0), name
            for part, whole in zip(s, s_ref):
                _close(part, whole, atol=1e-5)

    @pytest.mark.parametrize("conv_size, exact", [(0, True), (4, False)])
    def test_an_interior_hole_is_exact_only_without_a_conv_window(self, conv_size, exact):
        # Documented: a masked position between kept ones still occupies a
        # slot of the short convolution's window.
        layer = _built(19, 9, mode="recurrent", conv_size=conv_size)
        x = _inputs(19, (1, 9, 32))
        keep = np.ones((1, 9), "float32")
        keep[0, 4] = 0.0
        y_hole = np.delete(np.asarray(layer(x, attention_mask=keep)), 4, axis=1)
        y_deleted = np.asarray(layer(np.delete(x, 4, axis=1)))
        _close(y_hole[:, :4], y_deleted[:, :4], atol=1e-5)
        gap = np.abs(y_hole[:, 4:] - y_deleted[:, 4:]).max()
        assert (gap <= 1e-5) if exact else (gap > 1e-2), gap

    def test_a_boolean_mask_equals_a_float_mask(self):
        layer = _built(20, 8)
        x = _inputs(20, (2, 8, 32))
        keep = np.ones((2, 8), "float32")
        keep[:, 6:] = 0.0
        np.testing.assert_array_equal(
            np.asarray(layer(x, attention_mask=keep > 0)),
            np.asarray(layer(x, attention_mask=keep)),
        )


# ---------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------


class TestMixedPrecision:
    """Under mixed_float16 the recurrence and S stay float32, all finite."""

    @pytest.mark.parametrize("mode", MODES)
    def test_mixed_float16_is_finite(self, mixed_float16_policy, mode):
        layer = KimiDeltaAttention(dim=32, num_heads=2, chunk_size=8, mode=mode)
        x = _inputs(21, (2, 19, 32), scale=30.0)
        y, (s, conv_q, conv_k, conv_v) = layer(x, return_state=True)
        assert layer.compute_dtype == "float16"
        assert y.dtype == "float16" and conv_q.dtype == "float16"
        assert s.dtype == "float32"
        for t in (y, s, conv_q, conv_k, conv_v):
            assert np.all(np.isfinite(np.asarray(t, dtype=np.float32)))
        y2 = layer(x[:, 15:], initial_state=(s, conv_q, conv_k, conv_v))
        assert np.all(np.isfinite(np.asarray(y2, dtype=np.float32)))


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------


class TestSerialization:
    """get_config covers every argument; .keras round trips exactly."""

    NON_DEFAULT: Dict[str, Any] = dict(
        dim=24, num_heads=3, head_dim=4, value_head_dim=6, conv_size=3,
        gate_low_rank_dim=5, use_output_gate=False, output_norm_type="rms_norm",
        output_norm_kwargs={"epsilon": 1e-4}, chunk_size=5, mode="recurrent",
        a_init_range=(2.0, 8.0), dt_min=0.002, dt_max=0.05, dt_init_floor=2e-4,
        dropout_rate=0.1, use_bias=True, eps=1e-5,
        kernel_initializer="he_normal", bias_initializer="ones",
        kernel_regularizer=keras.regularizers.L2(1e-4),
        bias_regularizer=keras.regularizers.L1(1e-5),
    )

    def test_get_config_covers_every_init_argument(self):
        params = {
            name for name in inspect.signature(KimiDeltaAttention.__init__).parameters
            if name not in ("self", "kwargs")
        }
        config = KimiDeltaAttention(**self.NON_DEFAULT).get_config()
        assert params <= set(config), sorted(params - set(config))
        assert set(self.NON_DEFAULT) == params

    def test_from_config_rebuilds_the_same_config(self):
        layer = KimiDeltaAttention(**self.NON_DEFAULT)
        rebuilt = KimiDeltaAttention.from_config(layer.get_config())
        assert rebuilt.get_config() == layer.get_config()

    @pytest.mark.parametrize("mode", MODES)
    def test_keras_round_trip_gives_identical_outputs(self, tmp_path, mode):
        inp = keras.Input((None, 32))
        layer = KimiDeltaAttention(dim=32, num_heads=4, chunk_size=4, mode=mode, use_bias=True)
        model = keras.Model(inp, layer(inp))
        _randomize(layer, 22)
        x = _inputs(22, (2, 9, 32))
        path = tmp_path / "kda.keras"
        model.save(path)
        loaded = keras.models.load_model(path)
        _close(loaded(x), model(x), atol=1e-6)
        for a, b in zip(loaded.get_weights(), model.get_weights()):
            np.testing.assert_array_equal(a, b)
        assert loaded.layers[1].get_config() == layer.get_config()


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


class TestFactory:
    """``create_attention_layer('kimi_delta', ...)`` builds this layer."""

    def test_factory_builds_the_layer(self):
        layer = create_attention_layer("kimi_delta", dim=32, num_heads=4)
        assert isinstance(layer, KimiDeltaAttention)
        assert layer.dim == 32 and layer.num_heads == 4
        assert np.asarray(layer(_inputs(23, (2, 7, 32)))).shape == (2, 7, 32)

    def test_factory_forwards_optional_arguments(self):
        layer = create_attention_layer(
            "kimi_delta", name="kda", dim=32, num_heads=4, mode="recurrent", conv_size=0,
        )
        assert layer.name == "kda" and layer.mode == "recurrent" and layer.conv_size == 0

    def test_registry_defaults_match_the_constructor(self):
        signature = inspect.signature(KimiDeltaAttention.__init__).parameters
        entry = ATTENTION_REGISTRY["kimi_delta"]
        assert entry["required_params"] == ["dim", "num_heads"]
        for name, default in entry["optional_params"].items():
            assert signature[name].default == default, name
