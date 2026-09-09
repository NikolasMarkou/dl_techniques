"""Guards for H-Net's isotropic stack: the mask builder, the mixer dispatch, the block.

What this file pins
-------------------
1. **The causal keep predicate.** Shape, rank, dtype, ``{0, 1}``-valuedness (it is a keep
   predicate, NEVER an additive bias), the ``j <= i AND i - j <= window`` semantics, the
   ``window_size = -1`` sentinel, and that it survives ``tf.function`` at
   ``jit_compile`` both False and True. Plus a source ban on ``keras.ops.tril``/``triu``:
   those RAISE the moment the function is traced, and their EAGER result is bitwise equal
   to the ``arange`` form, so a behavioural test alone cannot see the wrong call until it
   is traced (D-010(b)).
2. **Mixer dispatch by letter and by case**, and the three explicitly-passed values whose
   defaults are wrong or have moved: RMSNorm ``epsilon=1e-5``, Mamba-2
   ``norm_before_gate=False``, SwiGLU ``(4, 128)`` with no ``hidden_dim``.
3. **D-005**: ``rope_percentage == rotary_emb_dim / head_dim`` for every stage of every
   one of the six shipped variants, and that RoPE is LIVE rather than a silent no-op.
4. **Pre-norm, not post-norm**, checked as arithmetic against the block's own sub-layers.
5. **Causality**, three-armed, per mixer family, with a non-causal control on the same
   weights proving the probe can report movement.

Numerical tolerances
--------------------
Every ``assert_allclose`` here passes ``rtol=0`` so the bound is purely absolute.
``tests/numerics.reassociation_atol`` is deliberately NOT used: plan decision D-012
measured it UNDER-counting against a float64 oracle and missing normalize chains
entirely, and a bound below a correct implementation's own noise floor can never pass.
Each bound below is derived at its call site and its attained value is recorded there.

Most of what is pinned here is structural (shapes, dtypes, dispatch, exact zeros from a
mask) and D-010 measured the two device families agreeing on all of it. The ONE exception
is the traced-vs-eager comparison, which is arithmetic: its bound is
:func:`traced_parity_atol`, which MEASURES the active device's matmul precision instead of
assuming float32. That claim was wrong here until 2026-09-09 -- the XLA arm was RED on a
TF32 GPU at 2.69e-03 against a 1.24e-05 float32 bound -- and the wrongness was invisible
because the file said no GPU arm was needed.
"""

import inspect
import re

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.ffn.factory import create_ffn_layer
from dl_techniques.layers.norms.factory import create_normalization_layer
from dl_techniques.models.language.hnet.components import (
    NORM_EPSILON,
    ROPE_THETA,
    SWIGLU_EXPANSION_FACTOR,
    SWIGLU_MULTIPLE_OF,
    HNetBlock,
    HNetIsotropic,
    build_causal_keep_mask,
    build_mixer,
    build_mlp,
    parse_isotropic_layout,
)
from dl_techniques.models.language.hnet import components as components_module
from dl_techniques.models.language.hnet.config import MODEL_VARIANTS
from dl_techniques.models.language.mamba.components_v2 import Mamba2Layer
from dl_techniques.layers.attention.group_query_attention import GroupedQueryAttention
from tests.numerics import matmul_precision_atol, matmul_unit_roundoff

from ..test_sam.dead_component_oracle import fit_one_step_moved_variables


# ---------------------------------------------------------------------
# Hand-built oracles. None of these calls the code under test.
# ---------------------------------------------------------------------


def numpy_causal_keep(seq_len: int, window_size: int = -1) -> np.ndarray:
    """The keep predicate, written out in NumPy with two explicit loops.

    Loops rather than broadcasts on purpose: the subject builds the mask by broadcasting,
    and an oracle that broadcasts the same way would reproduce a broadcasting mistake.
    """
    keep = np.zeros((seq_len, seq_len), dtype=np.int32)
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i and (window_size < 0 or (i - j) <= window_size):
                keep[i, j] = 1
    return keep


def reference_swiglu_hidden(d_model: int) -> int:
    """``hnet/modules/mlp.py:19-23``, transcribed."""
    d_intermediate = int(8 * d_model / 3)
    return (d_intermediate + 128 - 1) // 128 * 128


def perturbation_profile(forward, seq_len=16, d_model=64, batch=2, position=9, seed=11):
    """Three-armed causality probe: before / at / after a single perturbed position.

    :returns: ``(max|delta| before position, at position, after position)``.
    """
    base = np.array(keras.random.normal((batch, seq_len, d_model), seed=seed))
    bumped = base.copy()
    bumped[:, position, :] += 3.7
    delta = np.abs(np.array(forward(bumped)) - np.array(forward(base)))
    return (
        float(delta[:, :position].max()),
        float(delta[:, position].max()),
        float(delta[:, position + 1:].max()),
    )


# ---------------------------------------------------------------------
# 1. The causal keep predicate
# ---------------------------------------------------------------------


class TestCausalKeepMask:
    """``build_causal_keep_mask`` is a rank-3 integral keep predicate, and it is causal."""

    def test_causal_only_matches_the_hand_written_lower_triangle(self):
        """MAIN: ``window_size=-1`` is exactly ``j <= i``, diagonal included."""
        got = np.array(build_causal_keep_mask(9, -1))
        assert got.shape == (1, 9, 9)
        np.testing.assert_array_equal(got[0], numpy_causal_keep(9, -1))

    def test_the_lower_triangle_is_not_every_other_triangle(self):
        """TWIN for the above: the comparison can say no.

        Without this, an all-ones or a strictly-lower mask would satisfy an equality
        against an oracle that made the same mistake.
        """
        got = np.array(build_causal_keep_mask(9, -1))[0]
        assert not np.array_equal(got, np.ones((9, 9), dtype=np.int32))
        assert not np.array_equal(got, np.tril(np.ones((9, 9), dtype=np.int32), -1))
        assert not np.array_equal(got, got.T), "a symmetric mask is not causal"

    @pytest.mark.parametrize("window_size", [0, 1, 2, 5])
    def test_the_window_bands_the_lower_triangle(self, window_size):
        """MAIN: ``j <= i AND i - j <= window_size``, for four widths."""
        got = np.array(build_causal_keep_mask(9, window_size))[0]
        np.testing.assert_array_equal(got, numpy_causal_keep(9, window_size))

    def test_a_window_actually_removes_keys_the_causal_mask_kept(self):
        """TWIN: the band is not a no-op at these sizes."""
        causal = np.array(build_causal_keep_mask(9, -1))[0]
        banded = np.array(build_causal_keep_mask(9, 2))[0]
        assert banded.sum() < causal.sum()
        # Every key the band keeps was already causal; it only ever removes.
        assert np.all(banded <= causal)

    def test_window_minus_one_is_the_unlimited_context_sentinel(self):
        """``-1`` means no band, which is NOT the same as ``window_size=0``."""
        unlimited = np.array(build_causal_keep_mask(7, -1))
        np.testing.assert_array_equal(
            unlimited, np.array(build_causal_keep_mask(7, 6))
        )  # at L=7 a 6-step lookback reaches everything causal
        assert not np.array_equal(
            unlimited, np.array(build_causal_keep_mask(7, 0))
        )

    def test_it_is_a_keep_predicate_and_never_an_additive_bias(self):
        """Values are exactly ``{0, 1}`` in an INTEGER dtype.

        An additive ``-1e9`` bias would pass every shape and rank check above and then
        produce ``0 * -inf = NaN`` at every unmasked position under ``mixed_float16``
        (SYSTEM.md). The polarity matters too: ``1`` must mean *attend*, which the
        diagonal pins -- a token always attends to itself.
        """
        mask = build_causal_keep_mask(6, 2)
        assert keras.backend.standardize_dtype(mask.dtype) == "int32"
        values = np.array(mask)
        assert set(np.unique(values).tolist()) <= {0, 1}
        assert np.all(np.diag(values[0]) == 1), "1 must mean ATTEND"

    def test_the_leading_axis_is_one_so_it_broadcasts_over_the_batch(self):
        assert np.array(build_causal_keep_mask(5)).shape == (1, 5, 5)

    @pytest.mark.parametrize("bad", [-2, -17])
    def test_a_window_below_minus_one_raises(self, bad):
        with pytest.raises(ValueError, match="window_size"):
            build_causal_keep_mask(4, bad)

    @pytest.mark.parametrize("bad", [1.5, "3", True])
    def test_a_non_integer_window_raises(self, bad):
        with pytest.raises(TypeError, match="window_size"):
            build_causal_keep_mask(4, bad)

    def test_the_mask_builder_survives_tracing_and_xla(self):
        """The whole reason for the ``tril`` ban: this function must TRACE.

        ``keras.ops.tril`` raises ``TypeError: ('pred must not be a Python bool', True)``
        here at ``jit_compile=False`` as well as True (D-010(b)), while its eager result
        is bitwise equal to the ``arange`` form -- so only a traced arm can tell them
        apart.
        """
        eager = np.array(build_causal_keep_mask(8, 3))

        @tf.function
        def graph_build():
            return build_causal_keep_mask(8, 3)

        @tf.function(jit_compile=True)
        def xla_build():
            return build_causal_keep_mask(8, 3)

        np.testing.assert_array_equal(np.array(graph_build()), eager)
        np.testing.assert_array_equal(np.array(xla_build()), eager)

    def test_the_module_never_calls_tril_or_triu(self):
        """A source ban, because the behavioural difference is invisible eagerly.

        This is deliberately a text guard and is honest about it: eager ``tril(ones)`` is
        BITWISE equal to ``arange <= arange`` (measured, D-010(b) anti-vacuity line), so
        no value assertion can distinguish the banned call. The traced arm above is the
        behavioural half of the same claim.
        """
        source = inspect.getsource(components_module)
        offenders = re.findall(r"ops\.(?:tril|triu)\s*\(", source)
        assert offenders == [], (
            f"{offenders} found: keras.ops.tril/triu raise under tf.function on this "
            f"stack. Build masks from arange broadcasts."
        )


class TestKeyValidityComposition:
    """A rank-2 padding predicate ANDs into the rank-3 causal one."""

    def test_padding_keys_are_removed_for_every_query(self):
        keras.utils.set_random_seed(0)
        stack = HNetIsotropic(
            d_model=32, layout="T1", num_heads=4, rotary_emb_dim=4, name="pad_iso"
        )
        stack.build((None, 8, 32))
        x = keras.random.normal((2, 8, 32), seed=1)

        valid = np.ones((2, 8), dtype="int32")
        valid[:, 5:] = 0  # the last three positions are padding

        y_masked = np.array(
            stack(x, padding_mask=keras.ops.convert_to_tensor(valid), training=False)
        )

        # Changing a PADDED key must not move any output.
        x2 = np.array(x).copy()
        x2[:, 6, :] += 9.1
        y_masked2 = np.array(
            stack(x2, padding_mask=keras.ops.convert_to_tensor(valid), training=False)
        )
        # Position 6 itself is a query too and its own output changes; the real claim is
        # about the surviving (non-padded) queries.
        np.testing.assert_allclose(
            y_masked2[:, :5], y_masked[:, :5], rtol=0, atol=0.0
        )

    def test_without_the_padding_mask_that_key_does_move_the_output(self):
        """TWIN: the exact zero above is the mask's doing, not a dead probe."""
        keras.utils.set_random_seed(0)
        stack = HNetIsotropic(
            d_model=32, layout="T1", num_heads=4, rotary_emb_dim=4, name="pad_iso"
        )
        stack.build((None, 8, 32))
        x = np.array(keras.random.normal((2, 8, 32), seed=1))
        x2 = x.copy()
        x2[:, 6, :] += 9.1
        moved = np.max(
            np.abs(
                np.array(stack(x2, training=False))[:, :5]
                - np.array(stack(x, training=False))[:, :5]
            )
        )
        # Position 6 is in the FUTURE of queries 0..4, so causality alone already pins
        # this to zero. The honest twin is therefore a PAST key.
        assert moved == 0.0
        x3 = x.copy()
        x3[:, 1, :] += 9.1
        moved_past = np.max(
            np.abs(
                np.array(stack(x3, training=False))[:, 2:5]
                - np.array(stack(x, training=False))[:, 2:5]
            )
        )
        assert moved_past > 1e-3, "the probe cannot see any movement at all"


# ---------------------------------------------------------------------
# 2. Mixer dispatch, and the explicitly-passed values
# ---------------------------------------------------------------------


class TestMixerDispatch:
    """The letter picks the mixer; the CASE picks the MLP."""

    @pytest.mark.parametrize("letter", ["m", "M"])
    def test_m_letters_build_the_mamba_mixer(self, letter):
        mixer = build_mixer(letter, d_model=64, headdim=16, d_state=8)
        assert isinstance(mixer, Mamba2Layer)

    @pytest.mark.parametrize("letter", ["t", "T"])
    def test_t_letters_build_the_attention_mixer(self, letter):
        mixer = build_mixer(letter, d_model=64, num_heads=4, rotary_emb_dim=8)
        assert isinstance(mixer, GroupedQueryAttention)

    def test_the_two_families_are_not_the_same_class(self):
        """TWIN: the two assertions above would both hold if one class served both."""
        assert not isinstance(
            build_mixer("m", 64, headdim=16, d_state=8), GroupedQueryAttention
        )
        assert not isinstance(
            build_mixer("t", 64, num_heads=4, rotary_emb_dim=8), Mamba2Layer
        )

    def test_attention_is_the_mha_special_case_of_gqa(self):
        mixer = build_mixer("T", 64, num_heads=4, rotary_emb_dim=8)
        assert mixer.num_kv_heads == mixer.num_heads == 4
        assert mixer.rope_theta == ROPE_THETA == 10000.0
        assert mixer.use_bias is False

    @pytest.mark.parametrize("letter", ["x", "M4", "", "s"])
    def test_an_unknown_letter_raises(self, letter):
        with pytest.raises(ValueError, match="layout letter"):
            build_mixer(letter, 64)

    def test_attention_without_heads_raises(self):
        with pytest.raises(ValueError, match="num_heads"):
            build_mixer("t", 64, num_heads=0, rotary_emb_dim=4)

    def test_indivisible_width_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            build_mixer("t", 65, num_heads=4, rotary_emb_dim=4)

    def test_rotary_wider_than_the_head_raises(self):
        with pytest.raises(ValueError, match="rotary_emb_dim"):
            build_mixer("t", 64, num_heads=4, rotary_emb_dim=17)

    @pytest.mark.parametrize(
        "letter,expected", [("m", False), ("M", True), ("t", False), ("T", True)]
    )
    def test_case_selects_the_mlp(self, letter, expected):
        block = HNetBlock(
            d_model=32, kind=letter, num_heads=4, rotary_emb_dim=4,
            headdim=16, d_state=8, name=f"blk_{letter}",
        )
        assert block.has_mlp is expected
        assert (block.mlp is not None) is expected
        assert (block.norm2 is not None) is expected

    @pytest.mark.parametrize("letter", ["m", "t"])
    def test_a_lowercase_block_owns_no_mlp_weights(self, letter):
        """Kills "the MLP was added for lowercase letters too" on the WEIGHT signature.

        A weight-name check, not an output check: two random inits differ in output for
        reasons that have nothing to do with an extra sub-layer.
        """
        block = HNetBlock(
            d_model=32, kind=letter, num_heads=4, rotary_emb_dim=4,
            headdim=16, d_state=8, name=f"low_{letter}",
        )
        block.build((None, 8, 32))
        names = [w.path for w in block.weights]
        assert not any("mlp" in n for n in names), names

    @pytest.mark.parametrize("letter", ["M", "T"])
    def test_an_uppercase_block_does_own_mlp_weights(self, letter):
        """TWIN of the above."""
        block = HNetBlock(
            d_model=32, kind=letter, num_heads=4, rotary_emb_dim=4,
            headdim=16, d_state=8, name=f"up_{letter}",
        )
        block.build((None, 8, 32))
        names = [w.path for w in block.weights]
        assert any("mlp" in n for n in names), names


class TestExplicitlyPassedValues:
    """Three values whose defaults are wrong, or have moved, or both."""

    def test_every_norm_in_the_stack_carries_epsilon_1e_5(self):
        """Asserted on the CONSTRUCTED sub-layer, not on the call we made.

        Walks every norm the stack owns -- both norms of every block and the final one --
        because "we passed epsilon" is not the same claim as "epsilon arrived".
        """
        stack = HNetIsotropic(
            d_model=32, layout="m1T1", num_heads=4, rotary_emb_dim=4,
            headdim=16, d_state=8, name="eps_iso",
        )
        norms = [stack.final_norm]
        for block in stack.blocks:
            norms.append(block.norm1)
            if block.norm2 is not None:
                norms.append(block.norm2)
        assert len(norms) == 1 + 2 + 1  # final + two block norm1s + one block norm2
        for norm in norms:
            assert norm.epsilon == NORM_EPSILON == 1e-5

    def test_1e_5_is_not_what_the_factory_would_have_given_us(self):
        """TWIN: the assertion above is not satisfied by doing nothing."""
        default_norm = create_normalization_layer("rms_norm")
        assert default_norm.epsilon == 1e-6
        assert default_norm.epsilon != NORM_EPSILON

    def test_the_mamba_mixer_carries_the_reference_norm_knobs(self):
        mixer = build_mixer("m", 64, headdim=16, d_state=8)
        assert mixer.norm_epsilon == 1e-5
        assert mixer.norm_before_gate is False

    def test_the_mamba_norm_knobs_are_passed_EXPLICITLY(self):
        """A source guard, and it is the only instrument that can see this claim.

        ``Mamba2Layer``'s own defaults for these two happen to equal the reference's
        values TODAY, so a value assertion is satisfied whether or not we pass them --
        it cannot distinguish "passed" from "inherited". The reason to pass them anyway
        is historical: ``norm_before_gate`` defaulted to ``True`` in this repository
        until 2026-08-15. A value that has moved once is not a value to inherit, and this
        guard is what keeps the call site explicit after the next move.
        """
        source = inspect.getsource(build_mixer)
        assert "norm_epsilon=NORM_EPSILON" in source
        assert "norm_before_gate=False" in source

    @pytest.mark.parametrize(
        "d_model,expected", [(1024, 2816), (1536, 4096), (2048, 5504)]
    )
    def test_swiglu_width_reproduces_the_reference_at_every_shipped_d_model(
        self, d_model, expected
    ):
        """``round_up(8 * d_model / 3, 128)``, cross-checked against a transcription.

        The literals come from ``hnet/modules/mlp.py:19-23`` evaluated by hand; the
        oracle re-derives them. Both agree, so a typo in either is visible.
        """
        assert reference_swiglu_hidden(d_model) == expected
        assert build_mlp(d_model).hidden_dim == expected

    @pytest.mark.parametrize("d_model", [64, 128, 320, 1024, 1536, 2048, 4096])
    def test_the_sizing_rule_holds_away_from_the_shipped_widths_too(self, d_model):
        assert build_mlp(d_model).hidden_dim == reference_swiglu_hidden(d_model)

    def test_the_factory_defaults_would_have_been_wrong(self):
        """TWIN, and it is the case that matters: ``(4, 256)`` agrees TWICE and then lies.

        Two of the three shipped widths agree under the factory's own defaults. A probe
        that stopped at 1024 and 1536 would have passed.
        """
        assert SWIGLU_EXPANSION_FACTOR == 4 and SWIGLU_MULTIPLE_OF == 128
        default_widths = [
            create_ffn_layer("swiglu", output_dim=d).hidden_dim
            for d in (1024, 1536, 2048)
        ]
        assert default_widths == [2816, 4096, 5632]
        assert default_widths != [2816, 4096, 5504]

    def test_the_swiglu_is_bias_free(self):
        mlp = build_mlp(64, name="swiglu_probe")
        mlp.build((None, 8, 64))
        paths = [w.path for w in mlp.weights]
        assert paths, "the probe found no weights at all"
        assert [p for p in paths if p.endswith("/bias")] == [], paths


# ---------------------------------------------------------------------
# 2b. D-031: `d_intermediate` is CONSUMED
# ---------------------------------------------------------------------


#: The SwiGLU width of every stage of every shipped variant, MEASURED on the
#: pre-D-031 code (where `d_intermediate` reached nothing) and re-measured after the
#: wiring. Hand-written here as an INDEPENDENT second source: it is deliberately NOT
#: recomputed from `build_mlp`, `reference_swiglu_hidden` or the config, because the
#: whole claim of D-031 is that wiring a previously-dead field did not move one single
#: shipped model. A table derived from the code under test could not make that claim.
#: `(variant, stage) -> hidden width`.
SHIPPED_SWIGLU_WIDTHS = {
    ("hnet_1stage_L", 0): 2816,
    ("hnet_1stage_L", 1): 4096,
    ("hnet_1stage_XL", 0): 2816,
    ("hnet_1stage_XL", 1): 5504,
    ("hnet_2stage_L", 0): 2816,
    ("hnet_2stage_L", 1): 2816,
    ("hnet_2stage_L", 2): 4096,
    ("hnet_2stage_XL", 0): 2816,
    ("hnet_2stage_XL", 1): 4096,
    ("hnet_2stage_XL", 2): 5504,
    ("hnet_2stage_XL_chinese", 0): 2816,
    ("hnet_2stage_XL_chinese", 1): 4096,
    ("hnet_2stage_XL_chinese", 2): 5504,
    ("hnet_2stage_XL_code", 0): 2816,
    ("hnet_2stage_XL_code", 1): 4096,
    ("hnet_2stage_XL_code", 2): 5504,
}


class TestGuardSixIntermediateWidth:
    """``build_mlp`` reads ``d_intermediate``. For one iteration it did not.

    The anchor in :func:`build_mlp` names this class. It is its guard.

    The defect this replaces was invisible on the entire population that was checked:
    every shipped variant's JSON ``d_intermediate`` EQUALS the width the 2/3 rule
    derives, so ``[0, 0]``, ``[0, 999]`` and ``[0, 4096]`` all built the identical
    128-wide SwiGLU and no variant test could tell. So the tests below are in two
    halves that must BOTH hold: an explicit value must MOVE the width (which is what
    was broken), and no shipped variant's width may move (which is what the repair must
    not break).
    """

    # -- half one: a non-derived value actually changes the built width ------------

    @pytest.mark.parametrize(
        "d_model,d_intermediate,expected",
        [
            # The reviewer's own reproducer shapes, at the width they used.
            (16, 0, 128),      # derived: round_up(8*16/3 = 42, 128)
            (16, 999, 1024),   # honoured, rounded up
            (16, 4096, 4096),  # honoured, already a multiple
            # A value BELOW the derived width, so "bigger wins" cannot pass this.
            (1024, 256, 256),  # derived would be 2816
            # A value that is not a multiple of 128 and is not adjacent to one.
            (256, 130, 256),
        ],
    )
    def test_an_explicit_intermediate_width_is_honoured(
        self, d_model, d_intermediate, expected
    ):
        """MAIN. RED against the pre-D-031 `build_mlp`, which returned 128/128/128."""
        assert build_mlp(d_model, d_intermediate).hidden_dim == expected

    def test_the_three_reviewer_values_do_not_all_build_the_same_layer(self):
        """ANTI-VACUITY twin: the measured symptom, stated as the symptom.

        The reviewer's finding was literally "``[0,0]``, ``[0,999]`` and ``[0,4096]``
        all build an identical 128-wide SwiGLU". This asserts the negation directly,
        so the finding cannot silently come back in a form the parametrized arm above
        happens not to cover.
        """
        widths = {build_mlp(16, di).hidden_dim for di in (0, 999, 4096)}
        assert len(widths) == 3, widths

    def test_zero_derives_and_matches_the_reference_transcription(self):
        """``0`` is the port's spelling of upstream's ``None``, not a zero-width MLP."""
        for d_model in (64, 128, 320, 1024, 1536, 2048):
            assert (
                build_mlp(d_model, 0).hidden_dim
                == build_mlp(d_model).hidden_dim
                == reference_swiglu_hidden(d_model)
            )

    def test_the_rounding_is_upstreams_rounding_of_an_EXPLICIT_value(self):
        """``mlp.py:24`` rounds a value it was GIVEN, not only a value it derived."""
        multiple = SWIGLU_MULTIPLE_OF
        for requested in (1, 127, 128, 129, 2815, 2816, 5503):
            expected = (requested + multiple - 1) // multiple * multiple
            assert build_mlp(64, requested).hidden_dim == expected

    def test_a_negative_intermediate_width_is_refused(self):
        with pytest.raises(ValueError, match="d_intermediate"):
            build_mlp(64, -1)

    # -- the value must REACH the layers, not just the builder ---------------------

    def test_the_width_reaches_HNetBlock_and_survives_get_config(self):
        block = HNetBlock(d_model=16, kind="T", d_intermediate=512, num_heads=2)
        assert block.mlp.hidden_dim == 512
        assert block.get_config()["d_intermediate"] == 512
        assert HNetBlock(d_model=16, kind="T", num_heads=2).mlp.hidden_dim == 128

    def test_the_width_reaches_every_block_of_an_HNetIsotropic_stack(self):
        stack = HNetIsotropic(
            d_model=16, layout="T2t1T1", d_intermediate=384, num_heads=2
        )
        widths = [b.mlp.hidden_dim for b in stack.blocks if b.mlp is not None]
        assert widths == [384, 384, 384], widths
        # The lowercase letter really has no MLP, so the sweep above is not vacuous.
        assert [b.mlp is None for b in stack.blocks] == [False, False, True, False]
        assert stack.get_config()["d_intermediate"] == 384

    # -- half two: the shipped models did not move --------------------------------

    def test_no_shipped_variants_swiglu_width_moved(self):
        """The claim D-031 must earn: wiring a dead field changed no shipped model.

        Compared against :data:`SHIPPED_SWIGLU_WIDTHS`, a hand-written table measured
        on the code BEFORE the wiring. 16 (variant, stage) pairs.
        """
        measured = {
            (name, stage): build_mlp(
                cfg.d_model[stage], cfg.d_intermediate[stage]
            ).hidden_dim
            for name, cfg in MODEL_VARIANTS.items()
            for stage in range(cfg.num_stages)
        }
        assert measured == SHIPPED_SWIGLU_WIDTHS

    def test_the_shipped_configs_are_the_reason_the_defect_was_invisible(self):
        """TWIN, and it is the finding: every POSITIVE shipped value equals the
        derived one, which is exactly why no variant test could have caught this.

        If a future variant is added whose ``d_intermediate`` genuinely diverges from
        the 2/3 rule, this test goes RED -- and that is correct: it is the signal that
        the population which used to hide the defect no longer does, and that
        :data:`SHIPPED_SWIGLU_WIDTHS` must be re-measured rather than edited.
        """
        agreeing = 0
        for name, cfg in MODEL_VARIANTS.items():
            for stage in range(cfg.num_stages):
                width = cfg.d_intermediate[stage]
                if width == 0:
                    continue
                assert width == reference_swiglu_hidden(cfg.d_model[stage]), (
                    name, stage, width
                )
                agreeing += 1
        assert agreeing == 10, agreeing


# ---------------------------------------------------------------------
# 3. D-005: the RoPE decision, and its two guards
# ---------------------------------------------------------------------


class TestDecisionD005RoPE:
    """The anchor in ``build_mixer`` names these two tests. They are its guards."""

    def test_rope_percentage_is_the_ratio_for_every_stage_of_every_variant(self):
        """MAIN: ``rope_percentage == rotary_emb_dim / head_dim``, 6 variants x all stages.

        Constructed, not built: ``GroupedQueryAttention`` allocates nothing until
        ``build``, so this sweeps ``d_model=2048`` stages for free.
        """
        checked = 0
        for name, cfg in MODEL_VARIANTS.items():
            for stage in range(cfg.num_stages):
                d_model = cfg.d_model[stage]
                num_heads = cfg.attn_cfg.num_heads[stage]
                rotary = cfg.attn_cfg.rotary_emb_dim[stage]
                head_dim = d_model // num_heads
                mixer = build_mixer(
                    "T", d_model, num_heads=num_heads, rotary_emb_dim=rotary
                )
                assert mixer.rope_percentage == rotary / head_dim, (
                    f"{name} stage {stage}: d_model={d_model} heads={num_heads} "
                    f"rotary={rotary} head_dim={head_dim}"
                )
                checked += 1
        assert checked == 16, (
            f"expected 2 + 2 + 3 + 3 + 3 + 3 = 16 stages across the six variants, "
            f"swept {checked}"
        )

    def test_the_shipped_variants_are_all_PARTIAL_rotary(self):
        """TWIN: a hardcoded ``rope_percentage=1.0`` would be wrong at every stage.

        Every shipped stage rotates exactly half its head width (32 of 64, 48 of 96,
        64 of 128), so the equality above is not vacuously satisfied by the factory
        default of 1.0.
        """
        seen = set()
        for cfg in MODEL_VARIANTS.values():
            for stage in range(cfg.num_stages):
                head_dim = cfg.d_model[stage] // cfg.attn_cfg.num_heads[stage]
                seen.add(cfg.attn_cfg.rotary_emb_dim[stage] / head_dim)
        assert seen == {0.5}
        assert 1.0 not in seen

    @pytest.mark.parametrize("rotary,head_dim", [(16, 16), (8, 16), (4, 16)])
    def test_rope_is_live(self, rotary, head_dim):
        """MAIN: RoPE moves the output. Asserted as ``> 1e-3``, never as ``!= 0.0``.

        Instrument: with no mask, self-attention is exactly permutation-equivariant along
        the sequence axis, so ``out(roll(x)) == roll(out(x))`` IFF nothing position-aware
        is applied. Threshold derivation: step 2(d) measured the signal at ``7.33e-01``
        and the ``rope_percentage=0.0`` FLOOR at ``~3e-07`` on CPU and GPU alike -- that
        floor is float32 matmul non-associativity under a permuted reduction order, not
        residual RoPE, so a ``!= 0.0`` assertion would be flaky by construction. ``1e-3``
        sits four orders above the floor and three below the signal. Attained here:
        8.06e-01 / 7.32e-01 / 5.58e-01 for rotary 16 / 8 / 4 of head_dim 16.
        """
        keras.utils.set_random_seed(4242)
        mixer = build_mixer(
            "t", head_dim * 4, num_heads=4, rotary_emb_dim=rotary, max_seq_len=16
        )
        mixer.build((None, 16, head_dim * 4))
        x = keras.random.normal((2, 16, head_dim * 4), seed=3)
        rolled_out = mixer(keras.ops.roll(x, 5, axis=1), training=False)
        out_rolled = keras.ops.roll(mixer(x, training=False), 5, axis=1)
        delta = float(np.max(np.abs(np.array(rolled_out) - np.array(out_rolled))))
        assert delta > 1e-3, f"RoPE looks inert: max|delta| = {delta:.6e}"

    def test_the_liveness_probe_reads_near_zero_when_rope_is_OFF(self):
        """TWIN: the probe is measuring RoPE, not measuring nothing.

        At ``rotary_emb_dim=0`` the same instrument reads ``2.38e-07`` here (2.38e-07 /
        3.58e-07 measured at step 2(d) on GPU / CPU) -- four orders BELOW the 1e-3 bar,
        which is exactly why that bar is expressed as an inequality.
        """
        keras.utils.set_random_seed(4242)
        mixer = build_mixer("t", 64, num_heads=4, rotary_emb_dim=0, max_seq_len=16)
        mixer.build((None, 16, 64))
        assert mixer.rope_percentage == 0.0
        x = keras.random.normal((2, 16, 64), seed=3)
        rolled_out = mixer(keras.ops.roll(x, 5, axis=1), training=False)
        out_rolled = keras.ops.roll(mixer(x, training=False), 5, axis=1)
        delta = float(np.max(np.abs(np.array(rolled_out) - np.array(out_rolled))))
        assert delta < 1e-5, f"something position-aware survived rope off: {delta:.6e}"

    def test_the_decision_anchor_is_present_at_the_call_site(self):
        """The anchor must not be deleted by a future edit; the two tests above are it."""
        source = inspect.getsource(build_mixer)
        assert "DECISION plan-2026-09-09T042752-6d66ac56/D-005" in source
        assert "rope_percentage = rotary_emb_dim / head_dim" in source


# ---------------------------------------------------------------------
# 4. The block is PRE-norm
# ---------------------------------------------------------------------


class TestPreNormBlock:
    """The residual convention, checked as arithmetic against the block's own layers."""

    def _block(self, kind="T"):
        keras.utils.set_random_seed(5)
        block = HNetBlock(
            d_model=32, kind=kind, num_heads=4, rotary_emb_dim=4,
            headdim=16, d_state=8, name=f"prenorm_{kind}",
        )
        block.build((None, 8, 32))
        return block

    def test_the_block_computes_x_plus_mixer_of_norm_x(self):
        """MAIN, on an attention block WITHOUT an MLP.

        atol = 0.0: this recomputes the identical ops on the identical tensors in the
        identical order, so any nonzero difference is a different formula, not noise.
        Attained: 0.0.
        """
        block = self._block("t")
        x = keras.random.normal((2, 8, 32), seed=6)
        mask = build_causal_keep_mask(8, -1)
        expected = x + block.mixer(
            block.norm1(x, training=False), attention_mask=mask, training=False
        )
        got = block(x, attention_mask=mask, training=False)
        np.testing.assert_allclose(
            np.array(got), np.array(expected), rtol=0, atol=0.0
        )

    def test_post_norm_would_have_given_a_different_answer(self):
        """TWIN: pre-norm and post-norm are not the same number on this input.

        Without this, the equality above could hold for a block that normalized in the
        wrong place and happened to agree.
        """
        block = self._block("t")
        x = keras.random.normal((2, 8, 32), seed=6)
        mask = build_causal_keep_mask(8, -1)
        post_norm = block.norm1(
            x + block.mixer(x, attention_mask=mask, training=False), training=False
        )
        got = block(x, attention_mask=mask, training=False)
        assert np.max(np.abs(np.array(got) - np.array(post_norm))) > 1e-3

    def test_the_mlp_branch_is_a_second_pre_norm_residual(self):
        """MAIN, on an uppercase block: ``h = h + mlp(norm2(h))`` on top of the mixer."""
        block = self._block("T")
        x = keras.random.normal((2, 8, 32), seed=6)
        mask = build_causal_keep_mask(8, -1)
        hidden = x + block.mixer(
            block.norm1(x, training=False), attention_mask=mask, training=False
        )
        expected = hidden + block.mlp(
            block.norm2(hidden, training=False), training=False
        )
        got = block(x, attention_mask=mask, training=False)
        np.testing.assert_allclose(
            np.array(got), np.array(expected), rtol=0, atol=0.0
        )

    def test_dropping_the_mlp_branch_would_have_given_a_different_answer(self):
        """TWIN for the MLP branch."""
        block = self._block("T")
        x = keras.random.normal((2, 8, 32), seed=6)
        mask = build_causal_keep_mask(8, -1)
        mixer_only = x + block.mixer(
            block.norm1(x, training=False), attention_mask=mask, training=False
        )
        got = block(x, attention_mask=mask, training=False)
        assert np.max(np.abs(np.array(got) - np.array(mixer_only))) > 1e-3


class TestIsotropicStack:
    """Layout expansion, the final norm, and shapes."""

    def test_the_layout_expands_to_one_block_per_letter(self):
        stack = HNetIsotropic(
            d_model=32, layout="m2T1", num_heads=4, rotary_emb_dim=4,
            headdim=16, d_state=8, name="expand_iso",
        )
        assert [b.kind for b in stack.blocks] == ["m", "m", "T"]
        assert stack.arch_full == ("m", "m", "T")
        assert parse_isotropic_layout("m2T1").arch_full == ("m", "m", "T")

    def test_a_different_layout_expands_differently(self):
        """TWIN: the expansion is read from the string, not hardcoded."""
        stack = HNetIsotropic(d_model=32, layout="m1", headdim=16, d_state=8, name="one")
        assert [b.kind for b in stack.blocks] == ["m"]

    def test_the_stack_is_its_blocks_followed_by_the_final_norm(self):
        """MAIN: the reference's trailing ``Isotropic.rmsnorm`` is present.

        atol = 0.0 for the same reason as the block arithmetic above. Attained: 0.0.
        """
        keras.utils.set_random_seed(8)
        stack = HNetIsotropic(
            d_model=32, layout="m1T1", num_heads=4, rotary_emb_dim=4,
            window_size=3, headdim=16, d_state=8, name="final_iso",
        )
        stack.build((None, 8, 32))
        x = keras.random.normal((2, 8, 32), seed=9)
        mask = build_causal_keep_mask(8, 3)
        hidden = x
        for block in stack.blocks:
            hidden = block(hidden, attention_mask=mask, training=False)
        expected = stack.final_norm(hidden, training=False)
        np.testing.assert_allclose(
            np.array(stack(x, training=False)), np.array(expected), rtol=0, atol=0.0
        )

    def test_without_the_final_norm_the_answer_differs(self):
        """TWIN: the final norm is not an identity on this tensor."""
        keras.utils.set_random_seed(8)
        stack = HNetIsotropic(
            d_model=32, layout="m1T1", num_heads=4, rotary_emb_dim=4,
            window_size=3, headdim=16, d_state=8, name="final_iso",
        )
        stack.build((None, 8, 32))
        x = keras.random.normal((2, 8, 32), seed=9)
        mask = build_causal_keep_mask(8, 3)
        hidden = x
        for block in stack.blocks:
            hidden = block(hidden, attention_mask=mask, training=False)
        assert np.max(
            np.abs(np.array(stack(x, training=False)) - np.array(hidden))
        ) > 1e-3

    def test_shape_is_preserved(self):
        stack = HNetIsotropic(d_model=32, layout="m1", headdim=16, d_state=8, name="sh")
        y = stack(keras.random.normal((3, 11, 32)))
        assert y.shape == (3, 11, 32)
        assert stack.compute_output_shape((None, 11, 32)) == (None, 11, 32)

    def test_a_width_mismatch_raises_at_build(self):
        block = HNetBlock(d_model=32, kind="m", headdim=16, d_state=8, name="mm")
        with pytest.raises(ValueError, match="d_model"):
            block.build((None, 8, 64))

    def test_a_malformed_layout_raises(self):
        with pytest.raises(ValueError, match="malformed"):
            HNetIsotropic(d_model=32, layout="m4 xyz", name="bad")

    def test_a_window_below_minus_one_raises_at_construction(self):
        with pytest.raises(ValueError, match="window_size"):
            HNetIsotropic(d_model=32, layout="T1", num_heads=4, window_size=-2,
                          name="badw")


# ---------------------------------------------------------------------
# 5. Causality, three-armed, per mixer family
# ---------------------------------------------------------------------


class TestCausality:
    """Perturb position ``t``; the past must be untouched and the future must move."""

    def test_an_attention_stack_never_looks_forward(self):
        """MAIN: before-``t`` is EXACTLY 0.0, not merely small.

        The strict form is what step 2(e) measured on the bare layer (0.0 before, 6.37
        after) and it survives the full stack: attained here 0.0 / 2.78 / 0.54 for
        before / at / after.
        """
        keras.utils.set_random_seed(7)
        stack = HNetIsotropic(
            d_model=64, layout="T2", num_heads=4, rotary_emb_dim=16,
            window_size=-1, name="causal_iso",
        )
        stack.build((None, 16, 64))
        before, at, after = perturbation_profile(
            lambda z: stack(z, training=False)
        )
        assert before == 0.0, f"future leaked into the past: {before:.6e}"
        assert at > 1e-3
        assert after > 1e-3, "the probe sees no movement anywhere"

    def test_the_same_weights_DO_leak_when_the_mask_is_all_ones(self):
        """TWIN / control: the exact zero above is the mask's doing.

        Same layer, same weights, same tensors -- only the keep predicate changes. An
        all-ones rank-3 mask moves the before-``t`` positions by 7.6e-01.
        """
        keras.utils.set_random_seed(7)
        stack = HNetIsotropic(
            d_model=64, layout="T2", num_heads=4, rotary_emb_dim=16,
            window_size=-1, name="causal_iso",
        )
        stack.build((None, 16, 64))
        block = stack.blocks[0]
        ones = keras.ops.ones((1, 16, 16), dtype="int32")
        before, _, _ = perturbation_profile(
            lambda z: block(z, attention_mask=ones, training=False)
        )
        assert before > 1e-3, "the control cannot report movement, so the probe is dead"

    def test_a_windowed_stack_forgets_beyond_the_band(self):
        """MAIN: with ``window_size=2`` and ONE block, ``t`` reaches ``t+1`` and ``t+2``.

        One block on purpose: two stacked windowed blocks propagate 2 * window, and a
        guard written against the wrong reach would pass for the wrong reason.
        Attained: before 0.0, in-band 1.36, beyond-band 0.0.
        """
        keras.utils.set_random_seed(7)
        stack = HNetIsotropic(
            d_model=64, layout="T1", num_heads=4, rotary_emb_dim=16,
            window_size=2, name="window_iso",
        )
        stack.build((None, 16, 64))
        base = np.array(keras.random.normal((2, 16, 64), seed=11))
        bumped = base.copy()
        bumped[:, 9, :] += 3.7
        delta = np.abs(
            np.array(stack(bumped, training=False))
            - np.array(stack(base, training=False))
        )
        assert delta[:, :9].max() == 0.0, "the band is not causal"
        assert delta[:, 10:12].max() > 1e-3, "the band does not reach t+1..t+2"
        assert delta[:, 12:].max() == 0.0, "the band reaches beyond window_size"

    def test_an_unwindowed_stack_does_reach_beyond_that_point(self):
        """TWIN: the beyond-band zero above belongs to the WINDOW, not to the layer."""
        keras.utils.set_random_seed(7)
        stack = HNetIsotropic(
            d_model=64, layout="T1", num_heads=4, rotary_emb_dim=16,
            window_size=-1, name="nowindow_iso",
        )
        stack.build((None, 16, 64))
        base = np.array(keras.random.normal((2, 16, 64), seed=11))
        bumped = base.copy()
        bumped[:, 9, :] += 3.7
        delta = np.abs(
            np.array(stack(bumped, training=False))
            - np.array(stack(base, training=False))
        )
        assert delta[:, 12:].max() > 1e-3

    def test_a_mamba_stack_never_looks_forward(self):
        """The SSM family carries its own causality; nothing masks it.

        Attained: before 0.0, at 2.87, after 2.28.
        """
        keras.utils.set_random_seed(7)
        stack = HNetIsotropic(
            d_model=64, layout="m2", headdim=16, d_state=8, name="mamba_iso"
        )
        stack.build((None, 16, 64))
        before, at, after = perturbation_profile(lambda z: stack(z, training=False))
        assert before == 0.0, f"the SSM scan leaked the future: {before:.6e}"
        assert at > 1e-3 and after > 1e-3


# ---------------------------------------------------------------------
# 6. Training, serialization, graph and XLA
# ---------------------------------------------------------------------


def build_mixed_model(seq_len=12, d_model=32):
    """A small stack with BOTH mixer families and an MLP, wrapped in a functional model."""
    keras.utils.set_random_seed(3)
    inputs = keras.Input(shape=(seq_len, d_model))
    stack = HNetIsotropic(
        d_model=d_model, layout="m1T1", num_heads=4, rotary_emb_dim=4,
        window_size=3, headdim=16, d_state=8, max_seq_len=seq_len, name="iso",
    )
    return keras.Model(inputs, stack(inputs))


def traced_parity_atol(scale, d_model=32):
    """Bound on ``|traced forward - eager forward|`` for :func:`build_mixed_model`.

    ONE definition, so the assertion and its anti-vacuity twin below cannot drift apart.

    Two terms, and the larger wins -- the same shape as
    ``test_routing_module.routing_parity_atol``:

    1. **Reassociation, at true float32.** The traced kernels are free to reorder each
       length-``d_model`` contraction, bounding the error by
       ``d_model * eps_float32 * max|y|`` = ``32 * 1.19e-07 * max|y|``.
    2. **A narrower matmul format.** :func:`~tests.numerics.matmul_precision_atol`
       MEASURES the active device's matmul unit roundoff and allows 4 of them. On a
       tensor-core GPU with TF32 enabled -- the default -- that is ``4 * 2**-11``, i.e.
       ~4100x term 1, and it is a change of arithmetic rather than a defect.

    MEASURED 2026-09-09, ``max|traced - eager|`` relative to ``max|y| ~ 3.24``, swept
    over ten input scalings ``c`` (the round-off diagnostic: an error FLAT in ``c`` and
    bit-identical at powers of two is round-off, an error growing or decaying with ``c``
    is an additive-bias defect):

    ==============================  =============  =============  ===========
    regime                          graph          XLA            bound
    ==============================  =============  =============  ===========
    CPU (true float32)              1.19e-06       8.94e-07       1.24e-05
    GPU 4090, TF32 DISABLED         1.31e-06       1.07e-06       1.24e-05
    GPU 4090, TF32 ON (default)     1.31e-06       2.69e-03       6.33e-03
    ==============================  =============  =============  ===========

    The TF32-disabled GPU row is what rules out a real defect: XLA on the SAME device
    attains the float32 bound with a 12x margin once TF32 is off, and the relative error
    under TF32 is flat at 3.7e-04 to 9.7e-04 across ``c`` -- ~2 TF32 unit roundoffs,
    which is round-off, not bias. Before this term existed the XLA arm read 2.69e-03
    against 1.24e-05 on GPU 0 and was RED.

    :param scale: ``max|eager output|``.
    :type scale: float
    :param d_model: Contraction length of the traced path.
    :type d_model: int
    :return: Absolute tolerance, valid in whichever matmul regime is active.
    :rtype: float
    """
    reassociation = d_model * float(np.finfo(np.float32).eps) * float(scale)
    return max(reassociation, matmul_precision_atol(scale))


class TestTrainingAndSerialization:

    def test_every_weight_moves_after_one_real_optimizer_step(self):
        """Gradient flow AFTER a step, never at init.

        A gradient that exists at init proves nothing about a weight the optimizer never
        reaches; the instrument reports moved variables BY NAME so a dead sub-layer is
        named rather than counted.
        """
        model = build_mixed_model()
        model.compile(optimizer=keras.optimizers.SGD(1.0), loss="mse")
        x = np.array(keras.random.normal((4, 12, 32), seed=1))
        y = np.array(keras.random.normal((4, 12, 32), seed=2))
        report = fit_one_step_moved_variables(model, x, y)
        assert report.total == 19, f"weight population changed: {report.total}"
        assert report.unmoved == (), report.summary()

    def test_keras_round_trip_preserves_VALUES(self, tmp_path):
        """Saved and reloaded, the model returns the same numbers.

        ``training=False`` is passed EXPLICITLY on both sides: the default differs
        between a direct call and a call inside a reloaded functional graph, and a
        round-trip that silently compared train-mode against inference-mode output would
        be measuring dropout, not serialization. atol = 0.0, and 0.0 is attained -- the
        reload reconstructs the identical graph and reads back the identical weights.
        """
        model = build_mixed_model()
        x = np.array(keras.random.normal((3, 12, 32), seed=5))
        before = np.array(model(x, training=False))

        path = tmp_path / "hnet_components.keras"
        model.save(path)
        reloaded = keras.models.load_model(path)
        after = np.array(reloaded(x, training=False))

        np.testing.assert_allclose(after, before, rtol=0, atol=0.0)

    def test_the_round_trip_comparison_can_fail(self, tmp_path):
        """TWIN: 0.0 above is not an artifact of comparing a tensor with itself."""
        model = build_mixed_model()
        x = np.array(keras.random.normal((3, 12, 32), seed=5))
        other = np.array(keras.random.normal((3, 12, 32), seed=6))
        path = tmp_path / "hnet_components.keras"
        model.save(path)
        reloaded = keras.models.load_model(path)
        assert np.max(
            np.abs(
                np.array(reloaded(other, training=False))
                - np.array(model(x, training=False))
            )
        ) > 1e-3

    def test_get_config_round_trips_every_constructor_argument(self):
        stack = HNetIsotropic(
            d_model=32, layout="m1T1", num_heads=4, rotary_emb_dim=4,
            window_size=3, max_seq_len=64, d_state=8, d_conv=3, expand=2,
            headdim=16, norm_epsilon=1e-5, name="cfg_iso",
        )
        config = stack.get_config()
        rebuilt = HNetIsotropic.from_config(config)
        for key in (
            "d_model", "layout", "num_heads", "rotary_emb_dim", "window_size",
            "max_seq_len", "d_state", "d_conv", "expand", "headdim", "norm_epsilon",
        ):
            assert getattr(rebuilt, key) == getattr(stack, key), key

    @pytest.mark.parametrize("jit_compile", [False, True])
    def test_the_forward_pass_traces_in_graph_mode_and_under_xla(self, jit_compile):
        """Both mixer families, traced.

        Tolerance is :func:`traced_parity_atol` -- derived, regime-aware, and shared with
        the anti-vacuity twin below. It is NOT ``tests/numerics.reassociation_atol``,
        which D-012 measured under-counting on this path.
        """
        model = build_mixed_model()
        x = np.array(keras.random.normal((3, 12, 32), seed=5))
        eager = np.array(model(x, training=False))

        traced = tf.function(
            lambda z: model(z, training=False), jit_compile=jit_compile
        )
        got = np.array(traced(tf.constant(x)))

        atol = traced_parity_atol(float(np.max(np.abs(eager))))
        assert atol > 0.0
        np.testing.assert_allclose(got, eager, rtol=0, atol=atol)

    def test_that_traced_bound_is_tight_enough_to_reject_a_wrong_answer(self):
        """TWIN: the derived bound is not so loose that anything passes.

        Runs in whichever matmul regime is active, so it re-proves discriminating
        power on a TF32 GPU (where the bound is 510x looser) and not only on CPU.
        MEASURED on GPU 0 under TF32: a different input moves the output by 4.3906e+00
        against a bound of 6.3335e-03 -- a factor of 693.
        """
        model = build_mixed_model()
        x = np.array(keras.random.normal((3, 12, 32), seed=5))
        other = np.array(keras.random.normal((3, 12, 32), seed=6))
        eager = np.array(model(x, training=False))
        atol = traced_parity_atol(float(np.max(np.abs(eager))))
        assert np.max(
            np.abs(np.array(model(other, training=False)) - eager)
        ) > atol

    def test_the_traced_bound_is_the_reassociation_term_on_a_true_float32_device(self):
        """The regime term must be INERT where the arithmetic really is float32.

        Without this, a future edit that loosened the allowance would silently
        loosen every CPU run too, and no CPU test would notice. On a true-float32
        device the bound must be exactly the ``32 * eps * scale`` reassociation term
        it always was; only on a reduced-precision device may it be larger.
        """
        scale = 3.2428
        reassociation = 32 * float(np.finfo(np.float32).eps) * scale
        if matmul_unit_roundoff() > float(np.finfo(np.float32).eps) / 2.0:
            assert traced_parity_atol(scale) > reassociation
        else:
            assert traced_parity_atol(scale) == reassociation
