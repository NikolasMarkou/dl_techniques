"""Single-claim guard: GPT-2's residual-path projections are initialized at
``initializer_range / sqrt(2 * n_layer)``, and NOTHING else in the block is.

The claim has three inseparable halves and this module pins all three, because
each of them alone is satisfiable by a wrong implementation:

1. The residual projections (attention output projection, FFN contracting
   projection) shrink relative to ``initializer_range``.
2. Q/K/V and the FFN EXPANSION do **not** shrink. The tempting one-line
   "fix" -- scaling the block's single ``kernel_initializer`` -- passes (1) and
   fails here, and it is the reason ``models/gpt2/gpt2.py`` carried this as a
   disclosed departure rather than shipping that shortcut.
3. The shrink factor TRACKS DEPTH as ``1/sqrt(2 * depth)``. A constant factor
   passes (1) and (2) and fails here.

Tolerance, and why it is not a round number
-------------------------------------------
Every assertion below is on a RATIO of two sample standard deviations, never on
an absolute std. That is deliberate:

* ``TruncatedNormal(stddev=s)`` does not produce samples with std ``s``. Keras
  truncates at 2 sigma, so the realized std is ``0.87964 * s`` (measured:
  ``0.017592`` for the shipped ``initializer_range=0.02``). An absolute
  assertion at the reference's nominal ``0.02 / sqrt(2 * n_layer)`` would be
  RED for a correct implementation; a tolerance loose enough to swallow the
  12% truncation gap would be too loose to discriminate anything. The
  truncation factor is IDENTICAL on both sides of a ratio, so it cancels
  exactly.
* The remaining error is sampling noise. For ``N`` iid samples the sample std
  has relative standard error ``1/sqrt(2N)``; the ratio of two independent
  sample stds therefore has relative standard error
  ``sqrt(1/(2*N_a) + 1/(2*N_b))``. ``_ratio_tolerance`` returns EIGHT of those,
  which at the sizes used here is ~4-5% relative -- comfortably below the
  quantities being separated (a factor of 2 between the two depths, and a
  factor of ~3.5-7 between scaled and unscaled). The discrimination margin is
  asserted explicitly by ``test_the_tolerance_discriminates`` so that loosening
  the tolerance later cannot silently defeat this file.
"""

import math

import keras
import numpy as np
import pytest

from dl_techniques.layers.transformers.text_decoder import TextDecoder
from dl_techniques.models.gpt2.gpt2 import GPT2

# Empirical std of ``TruncatedNormal(stddev=1.0)``: 2-sigma truncation.
TRUNCATION_FACTOR = 0.87964


def _kernel_stds(layer_or_model):
    """Pool every block kernel by architectural role and return their stds.

    :param layer_or_model: A built ``TextDecoder`` or ``GPT2``.
    :return: ``{'qkv', 'attn_proj', 'ffn_fc1', 'ffn_fc2'} -> (std, n_samples)``.
    :rtype: dict
    """
    buckets = {}
    for w in layer_or_model.weights:
        path = w.path
        if "/proj/kernel" in path:
            key = "attn_proj"
        elif "/fc2/kernel" in path:
            key = "ffn_fc2"
        elif "/qkv/kernel" in path:
            key = "qkv"
        elif "/fc1/kernel" in path:
            key = "ffn_fc1"
        else:
            continue
        buckets.setdefault(key, []).append(np.asarray(w).ravel())
    assert set(buckets) == {"qkv", "attn_proj", "ffn_fc1", "ffn_fc2"}, (
        f"weight-path taxonomy drifted; found {sorted(buckets)}"
    )
    pooled = {k: np.concatenate(v) for k, v in buckets.items()}
    return {k: (float(v.std()), int(v.size)) for k, v in pooled.items()}


def _ratio_tolerance(n_a: int, n_b: int, sigmas: float = 8.0) -> float:
    """Relative tolerance for a ratio of two sample standard deviations."""
    return sigmas * math.sqrt(1.0 / (2 * n_a) + 1.0 / (2 * n_b))


def _build_decoder(depth: int, scaled: bool, seed: int = 3) -> TextDecoder:
    keras.utils.set_random_seed(seed)
    dec = TextDecoder(
        vocab_size=64,
        embed_dim=64,
        depth=depth,
        num_heads=4,
        max_seq_len=16,
        initializer_range=0.02,
        scale_residual_initializer_by_depth=scaled,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        name=f"dec_d{depth}_{int(scaled)}",
    )
    dec.build((None, 8))
    return dec


DEPTHS = (4, 16)


@pytest.mark.parametrize("depth", DEPTHS)
def test_the_residual_projections_shrink_by_one_over_sqrt_two_n(depth):
    """(1) + (3): both residual projections track ``1/sqrt(2 * depth)``."""
    stds = _kernel_stds(_build_decoder(depth, scaled=True))
    expected = 1.0 / math.sqrt(2.0 * depth)

    for residual, reference in (("attn_proj", "qkv"), ("ffn_fc2", "ffn_fc1")):
        std_r, n_r = stds[residual]
        std_ref, n_ref = stds[reference]
        ratio = std_r / std_ref
        tol = _ratio_tolerance(n_r, n_ref)
        assert ratio == pytest.approx(expected, rel=tol), (
            f"depth={depth}: {residual}/{reference} std ratio {ratio:.6f} "
            f"is not 1/sqrt(2*{depth})={expected:.6f} within rel={tol:.4f}"
        )


@pytest.mark.parametrize("depth", DEPTHS)
def test_the_qkv_and_expansion_kernels_are_not_touched(depth):
    """(2): the shrink reaches the residual projections and NOTHING else.

    Pinned against the UNSCALED arm of the same construction, so this fails the
    moment someone "simplifies" the fix into scaling the block's shared
    ``kernel_initializer``.
    """
    scaled = _kernel_stds(_build_decoder(depth, scaled=True))
    plain = _kernel_stds(_build_decoder(depth, scaled=False))

    for role in ("qkv", "ffn_fc1"):
        std_s, n_s = scaled[role]
        std_p, n_p = plain[role]
        tol = _ratio_tolerance(n_s, n_p)
        assert std_s / std_p == pytest.approx(1.0, rel=tol), (
            f"depth={depth}: {role} std moved from {std_p:.7f} to {std_s:.7f} "
            f"when only the residual projections should have"
        )
        # ... and it is still the plain truncated `initializer_range`.
        assert std_p == pytest.approx(0.02 * TRUNCATION_FACTOR, rel=0.02)

    for role in ("attn_proj", "ffn_fc2"):
        assert scaled[role][0] < 0.5 * plain[role][0], (
            f"depth={depth}: {role} did not shrink at all"
        )


def test_the_tolerance_discriminates():
    """The tolerance separates scaled from unscaled AND depth from depth.

    A std assertion with a tolerance loose enough to pass either way is the
    classic worthless guard; this test measures the margin instead of trusting
    it.
    """
    ratios, tols = {}, {}
    for depth in DEPTHS:
        stds = _kernel_stds(_build_decoder(depth, scaled=True))
        std_r, n_r = stds["attn_proj"]
        std_ref, n_ref = stds["qkv"]
        ratios[depth] = std_r / std_ref
        tols[depth] = _ratio_tolerance(n_r, n_ref)

    # The tolerance is never anywhere near 1.0, so an UNSCALED build (ratio ~1)
    # can never pass the scaled assertion.
    for depth in DEPTHS:
        expected = 1.0 / math.sqrt(2.0 * depth)
        assert tols[depth] < 0.20, f"tolerance {tols[depth]:.4f} is too loose"
        assert abs(1.0 - expected) / expected > 10 * tols[depth], (
            f"depth={depth}: an unscaled ratio of 1.0 is not separated from "
            f"{expected:.6f} by the tolerance {tols[depth]:.4f}"
        )

    # And the two depths are separated from each other by far more than the
    # tolerance, so a depth-INDEPENDENT constant factor cannot satisfy both.
    lo, hi = DEPTHS
    sep = abs(ratios[lo] - ratios[hi]) / ratios[hi]
    assert sep > 10 * max(tols.values()), (
        f"depth {lo} ratio {ratios[lo]:.6f} and depth {hi} ratio "
        f"{ratios[hi]:.6f} are not separated by 10x the tolerance "
        f"{max(tols.values()):.4f}"
    )
    assert ratios[lo] / ratios[hi] == pytest.approx(
        math.sqrt(hi / lo), rel=max(tols.values())
    )


def test_gpt2_opts_in_and_its_blocks_are_scaled():
    """The model this exists for actually turns the rule on."""
    keras.utils.set_random_seed(3)
    depth = 6
    model = GPT2(
        vocab_size=64, embed_dim=64, depth=depth, num_heads=4, max_seq_len=16,
    )
    model(keras.ops.zeros((1, 8), dtype="int32"))

    assert model.decoder.scale_residual_initializer_by_depth is True

    stds = _kernel_stds(model)
    expected = 1.0 / math.sqrt(2.0 * depth)
    for residual, reference in (("attn_proj", "qkv"), ("ffn_fc2", "ffn_fc1")):
        std_r, n_r = stds[residual]
        std_ref, n_ref = stds[reference]
        tol = _ratio_tolerance(n_r, n_ref)
        assert std_r / std_ref == pytest.approx(expected, rel=tol)


def test_every_transformer_layer_of_a_default_decoder_is_unscaled():
    """The parameter is OFF by default, so no other consumer is affected."""
    dec = _build_decoder(4, scaled=False)
    assert dec.scale_residual_initializer_by_depth is False
    for block in dec.decoder_layers:
        assert block.residual_output_kernel_initializer is None


def test_the_parameter_round_trips_through_get_config():
    """``get_config`` carries it on both the block and the decoder."""
    from dl_techniques.layers.transformers.transformer import TransformerLayer

    init = keras.initializers.TruncatedNormal(stddev=0.00408)
    block = TransformerLayer(
        hidden_size=32, num_heads=4, intermediate_size=64,
        residual_output_kernel_initializer=init, name="blk",
    )
    cfg = block.get_config()
    assert cfg["residual_output_kernel_initializer"] is not None
    restored = TransformerLayer.from_config(cfg)
    assert isinstance(
        restored.residual_output_kernel_initializer,
        keras.initializers.TruncatedNormal,
    )
    assert restored.residual_output_kernel_initializer.stddev == pytest.approx(0.00408)

    # None survives as None, not as a materialized default.
    plain = TransformerLayer(hidden_size=32, num_heads=4, intermediate_size=64)
    assert plain.get_config()["residual_output_kernel_initializer"] is None
    assert TransformerLayer.from_config(
        plain.get_config()
    ).residual_output_kernel_initializer is None

    dec = _build_decoder(4, scaled=True)
    dec_cfg = dec.get_config()
    assert dec_cfg["scale_residual_initializer_by_depth"] is True
    assert TextDecoder.from_config(
        dec_cfg
    ).scale_residual_initializer_by_depth is True


def test_asking_for_it_on_an_unsupported_ffn_type_raises_instead_of_no_op():
    """A silently ignored initializer request is the defect being removed."""
    from dl_techniques.layers.transformers.transformer import TransformerLayer

    with pytest.raises(ValueError):
        TransformerLayer(
            hidden_size=32, num_heads=4, intermediate_size=64,
            ffn_type="swiglu",
            residual_output_kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=0.004
            ),
            name="unsupported_ffn",
        )
