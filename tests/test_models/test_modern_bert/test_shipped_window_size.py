"""ModernBERT's local layers at the window size it actually ships.

Every other test in this package runs at ``TEST_WINDOW_SIZE = 16`` with
sequence lengths of 16/24/32 (``test_modern_bert.py:29``). All of those
lengths are ``<= 16**2 = 256``, so ~25 pre-existing tests exercise the local
layers as *full* attention over a padded 256-slot window without asserting
that they do — the shipped configuration is untested and the degeneracy is
invisible.

This module pins the degeneracy at the shipped size, and pins it as a COST
claim, because the cost claim in the surrounding prose used to be inverted
rather than merely approximate.

The mechanism, from ``WindowAttention._call_grid``:

* ``H = W = ceil(sqrt(N))``; the sequence is padded up to ``H * W``.
* ``pad_h = (ws - H % ws) % ws``, and likewise ``pad_w``. When ``H < ws`` this
  is ``ws - H``, so the padded grid is exactly one ``ws x ws`` tile.
* ``_window_partition`` therefore yields a SINGLE window holding ``ws**2``
  slots, of which ``ws**2 - N`` are padding, and attention inside it is dense.

Threshold: ``N <= M`` with ``M = window_size**2``. ``MODEL_VARIANTS`` ships
``window_size=128`` for ``base``/``large`` (``M = 16384``) and ``64`` for
``tiny`` (``M = 4096``), against ``DEFAULT_MAX_POSITION_EMBEDDINGS = 8192``.

**How these tests observe the partition without paying for it.** A real
forward at ``window_size=128`` would materialize a ``16384 x 16384`` score
matrix per head — ~2.7e8 entries, which is the point being made and also far
too expensive to run. So the per-window attention sublayer is replaced with an
identity recorder that captures the shape of the window tensor handed to it.
Everything up to and including the partition is the layer's own code; only the
attention math inside the window is stubbed out. The recorded shape is
``(B * num_windows, window_slots, dim)``.

``test_windowing_does_engage_for_tiny_above_its_threshold`` is the CONTROL: it
uses the same instrument on a configuration that genuinely windows, so a
"one window" result elsewhere cannot be an artefact of the recorder.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest

from dl_techniques.layers.attention.window_attention import WindowAttention
from dl_techniques.models.modern_bert.model import ModernBERT

DIM = 4
HEADS = 2


class _RecordingWindowAttention:
    """Identity stand-in for ``SingleWindowAttention`` that records shapes."""

    def __init__(self) -> None:
        self.window_shapes: list = []

    @property
    def built(self) -> bool:
        return True

    def build(self, input_shape) -> None:  # noqa: D401 - stub
        return None

    def __call__(self, windows, attention_mask=None, training=None):
        self.window_shapes.append(tuple(int(d) for d in windows.shape))
        return windows


def _partition_of(window_size: int, seq_len: int):
    """Return ``(num_windows, slots_per_window)`` for one forward pass.

    Runs ``WindowAttention``'s real grid-formation, padding and partitioning
    code; only the per-window attention is stubbed.
    """
    layer = WindowAttention(
        dim=DIM,
        window_size=window_size,
        num_heads=HEADS,
        use_relative_position_bias=False,
    )
    recorder = _RecordingWindowAttention()
    layer.attention = recorder

    out = layer(np.zeros((1, seq_len, DIM), dtype="float32"))
    assert tuple(int(d) for d in out.shape) == (1, seq_len, DIM), (
        "the layer must stay shape-preserving; if it does not, the recorded "
        "partition below is not describing the pass this test thinks it is"
    )

    assert len(recorder.window_shapes) == 1, "expected exactly one attention call"
    num_windows, slots, _ = recorder.window_shapes[0]
    return num_windows, slots


class TestShippedWindowSizeIsDenseAttention:
    """The shipped ``window_size=128`` never windows an admissible sequence."""

    @pytest.mark.parametrize("seq_len", [128, 512, 8192])
    def test_one_padded_window_at_every_admissible_length(self, seq_len):
        """One window of 16384 slots, whatever the sequence length is.

        This is the whole claim in one assertion: the partition is INDEPENDENT
        of ``seq_len``. Windowed attention that does not depend on ``N`` is not
        windowed attention.
        """
        window_size = 128
        slots_per_window = window_size ** 2

        num_windows, slots = _partition_of(window_size, seq_len)

        assert num_windows == 1, (
            f"L={seq_len} at window_size={window_size} produced {num_windows} "
            f"windows. If a 1-D sliding-window attention layer has been added "
            f"and wired in, DELETE this module rather than relaxing it."
        )
        assert slots == slots_per_window, (
            f"the single window holds {slots} slots, not window_size**2 = "
            f"{slots_per_window}"
        )
        assert slots >= seq_len, "padding, not truncation, is what fills a window"

    def test_the_cost_is_inverted_not_approximate(self):
        """A local layer costs MORE than the global attention it replaces.

        Score-matrix entries are ``slots**2`` per head per sample regardless of
        ``L``, against ``L**2`` for dense global attention over the real
        tokens. The ratio is ``(M / L)**2`` and is > 1 for every ``L <= M``.
        """
        window_size = 128
        _, slots = _partition_of(window_size, 128)

        windowed_entries = slots ** 2
        assert windowed_entries == 16384 ** 2

        for seq_len in (128, 8192):
            dense_entries = seq_len ** 2
            ratio = windowed_entries / dense_entries
            assert ratio > 1.0, (
                f"at L={seq_len} the windowed path would be cheaper than dense "
                f"attention, which contradicts the padding mechanism"
            )

        assert windowed_entries // (128 ** 2) == 16384, "≈16,384x dense at L=128"
        assert windowed_entries // (8192 ** 2) == 4, "≈4x dense at L=8192"

    def test_windowing_does_engage_for_tiny_above_its_threshold(self):
        """CONTROL: the same instrument sees a genuine multi-window partition.

        ``tiny`` ships ``window_size=64`` (``M = 4096``), so ``L = 8192 > M``
        is above its threshold and the grid really is partitioned. Without this
        arm, every "1 window" assertion above could be an artefact of the
        recorder rather than a property of the configuration.
        """
        num_windows, slots = _partition_of(window_size=64, seq_len=8192)

        assert num_windows == 4, f"expected a 2x2 partition, got {num_windows}"
        assert slots == 64 ** 2


class TestVariantTableAgainstTheThreshold:
    """Arithmetic over the shipped constants, not over hand-copied numbers."""

    def test_base_and_large_can_never_window(self):
        max_pos = ModernBERT.DEFAULT_MAX_POSITION_EMBEDDINGS
        for variant in ("base", "large"):
            window_size = ModernBERT.MODEL_VARIANTS[variant][
                "local_attention_window_size"
            ]
            threshold = window_size ** 2
            assert threshold >= max_pos, (
                f"variant '{variant}': window_size**2 = {threshold} vs "
                f"max_position_embeddings = {max_pos}. The docstring claim "
                f"that no admissible length is ever windowed depends on this."
            )

    def test_tiny_is_the_only_variant_where_windowing_can_engage(self):
        max_pos = ModernBERT.DEFAULT_MAX_POSITION_EMBEDDINGS
        threshold = (
            ModernBERT.MODEL_VARIANTS["tiny"]["local_attention_window_size"] ** 2
        )
        assert threshold < max_pos, "tiny's threshold must sit below max position"
        assert threshold == 4096

    def test_the_rest_of_the_suite_runs_in_the_degenerate_regime_too(self):
        """``TEST_WINDOW_SIZE = 16`` with L in 16/24/32 is one padded window.

        Recorded so a reader does not mistake ~25 green local-attention tests
        for evidence that windowing works.
        """
        for seq_len in (16, 24, 32):
            num_windows, slots = _partition_of(window_size=16, seq_len=seq_len)
            assert num_windows == 1
            assert slots == 256


class TestModelWiresTheShippedWindowSize:
    """The model hands ``local_attention_window_size`` straight to the layer."""

    def test_local_layer_receives_the_window_size_unmodified(self):
        model = ModernBERT(
            vocab_size=32,
            hidden_size=16,
            num_layers=1,
            num_heads=2,
            intermediate_size=24,
            global_attention_interval=999,  # no layer is global
            local_attention_window_size=128,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        layer = model.encoder_layers[0]
        assert layer.attention_type == "window"
        assert isinstance(layer.attention, WindowAttention)
        assert layer.attention.window_size == 128, (
            "the degeneracy proven above only applies to the model if the "
            "model's own window size reaches the layer unchanged"
        )
