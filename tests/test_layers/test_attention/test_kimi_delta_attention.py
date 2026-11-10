"""Test suite for the KimiDeltaAttention (KDA) layer.

Covers the traced-graph paths (``fit``, ``predict``, ``jit_compile``), a lazy
build inside ``fit`` in a cold process, the carried state
``(S, conv_q, conv_k, conv_v)`` (split calls, token-by-token decode, padding
masks across calls), the symbolic ``return_state`` structure, and the fla
defaults for the short-convolution kernel and the output-norm epsilon.

All tests run on CPU with tiny dimensions.
"""

import doctest
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import keras
import numpy as np
import pytest

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
