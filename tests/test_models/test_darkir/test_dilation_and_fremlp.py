"""DarkIR's two central mechanisms, measured: dilated support and FreMLP's phase.

Why this file exists
--------------------
DarkIR is 1,675 source lines against 143 lines of test -- the worst ratio in
`tests/test_models/`, and neither of the two ideas the model is built on is
checked anywhere. The existing tests build the model, run a forward, and assert
shapes.

1. **Parallel dilated branches.** A 3x3 depthwise kernel at dilation `d` touches
   exactly the offsets `{-d, 0, +d}` on each axis -- support `3 + 2(d - 1)` at
   the parameter cost of 3x3. The claim is therefore two-sided and exact:
   the pixel at offset `d` reaches the centre, and the pixels at offsets `d-1`,
   `d+1` and `2d` do NOT.

   MEASURED 2026-08-18 (17x17 input, 4 channels, seed 1), influence on the
   centre output of bumping the input at offset o along one axis:

       dilation 1:  o=1 -> 1.53733,  o=2 -> 0.0
       dilation 4:  o=1 -> 0.0,      o=4 -> 1.53733,  o=5 -> 0.0,  o=8 -> 0.0

   A dense 3x3 (dilation 1) is the dead component for the `d=4` case: it scores
   0.0 at the offset where the dilated branch scores 1.537, and vice versa.

2. **FreMLP edits MAGNITUDE only.** It transforms to the frequency domain,
   passes the magnitude through a 1x1-conv MLP and reattaches the input's
   phasor. The consequence that no shape test can see: the per-bin ratio
   `FFT(out) / FFT(in)` must be REAL. MEASURED: the relative imaginary part of
   that ratio is 8.76e-06 for FreMLP and exactly 1.0 for a `DilatedBranch` --
   a spatial convolution, whose frequency response is genuinely complex. That
   control is what makes the FreMLP number mean something.

   A REFINEMENT the model's own docstring does not state: the phase is
   preserved only UP TO SIGN. Of the significant bins, 57.4% come back with an
   unchanged phase and 42.6% with a phase flipped by exactly pi -- because the
   magnitude MLP is a plain 1x1 convolution with no non-negativity constraint,
   so its output can be negative and multiplies the unit phasor by a negative
   real number. "Reattaches the original phase" is therefore true only in the
   projective sense measured here.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.darkir.model import DilatedBranch, FreMLP


SIZE, CHANNELS, CENTRE = 17, 4, 8
SPATIAL_INPUT = (
    np.random.default_rng(0).normal(size=(1, SIZE, SIZE, CHANNELS)).astype("float32")
)
SPECTRAL_INPUT = (
    np.random.default_rng(0).random((1, 16, 16, CHANNELS)).astype("float32")
)


def _centre_influence(layer, offset: int) -> float:
    clean = np.asarray(
        keras.ops.convert_to_numpy(
            layer(keras.ops.convert_to_tensor(SPATIAL_INPUT), training=False)
        )
    )
    bumped_input = SPATIAL_INPUT.copy()
    bumped_input[0, CENTRE + offset, CENTRE, :] += 5.0
    bumped = np.asarray(
        keras.ops.convert_to_numpy(
            layer(keras.ops.convert_to_tensor(bumped_input), training=False)
        )
    )
    return float(np.max(np.abs(clean[0, CENTRE, CENTRE] - bumped[0, CENTRE, CENTRE])))


def _branch(dilation: int) -> DilatedBranch:
    keras.utils.set_random_seed(1)
    return DilatedBranch(channels=CHANNELS, expansion=1, dilation=dilation)


class TestDilatedBranchSupport:
    @pytest.mark.parametrize("dilation", [1, 2, 4])
    def test_only_the_dilated_taps_reach_the_centre(self, dilation):
        branch = _branch(dilation)
        assert _centre_influence(branch, dilation) > 1e-3, (
            f"dilation {dilation}: the tap at offset {dilation} does not reach "
            f"the centre; the depthwise kernel is not dilated as configured"
        )
        for offset in {dilation + 1, 2 * dilation} - {dilation}:
            assert _centre_influence(branch, offset) == 0.0, (
                f"dilation {dilation}: offset {offset} is BETWEEN or BEYOND the "
                f"3x3 kernel's taps, yet it influences the centre. The support "
                f"is not the claimed 3 + 2*(d - 1)."
            )

    def test_a_dense_kernel_and_a_dilated_one_are_distinguishable(self):
        """The dead-component pairing, stated as one comparison."""
        dense, dilated = _branch(1), _branch(4)
        assert _centre_influence(dense, 1) > 1e-3
        assert _centre_influence(dense, 4) == 0.0
        assert _centre_influence(dilated, 4) > 1e-3
        assert _centre_influence(dilated, 1) == 0.0, (
            "a dilation-4 branch responds to its immediate neighbour, so it is "
            "behaving like a dense 3x3 -- the dilation_rate is not reaching the "
            "convolution"
        )


def _relative_imaginary_part(output: np.ndarray) -> float:
    """max |Im(r)| / |r| over the significant bins, where r = FFT(out)/FFT(in)."""
    spectrum_in = np.fft.fft2(SPECTRAL_INPUT, axes=(1, 2))
    spectrum_out = np.fft.fft2(output, axes=(1, 2))
    significant = np.abs(spectrum_in) > 1e-2
    ratio = spectrum_out[significant] / spectrum_in[significant]
    return float(np.max(np.abs(ratio.imag) / np.maximum(np.abs(ratio), 1e-12)))


class TestFreMLPEditsMagnitudeOnly:
    @staticmethod
    def _fremlp_output() -> np.ndarray:
        keras.utils.set_random_seed(3)
        layer = FreMLP(channels=CHANNELS, expansion=2)
        return np.asarray(
            keras.ops.convert_to_numpy(
                layer(keras.ops.convert_to_tensor(SPECTRAL_INPUT), training=False)
            )
        )

    def test_the_per_bin_ratio_is_real(self):
        # Measured 8.76e-06 -- float32 FFT noise, not a phase rotation.
        imaginary = _relative_imaginary_part(self._fremlp_output())
        assert imaginary < 1e-4, (
            f"FreMLP rotated the phase: the per-bin ratio FFT(out)/FFT(in) has "
            f"a relative imaginary part of {imaginary:.3e}. It is supposed to "
            f"scale the magnitude and reattach the input's phasor."
        )

    def test_a_spatial_convolution_fails_that(self):
        """RED proof: a `DilatedBranch` -- the model's other path -- scores 1.0."""
        keras.utils.set_random_seed(1)
        branch = DilatedBranch(channels=CHANNELS, expansion=1, dilation=4)
        output = np.asarray(
            keras.ops.convert_to_numpy(
                branch(keras.ops.convert_to_tensor(SPECTRAL_INPUT), training=False)
            )
        )
        imaginary = _relative_imaginary_part(output)
        assert imaginary > 0.1, (
            f"the control is not a control: a spatial convolution's per-bin "
            f"ratio should be genuinely complex, but measured {imaginary:.3e}"
        )

    def test_the_phase_is_preserved_only_up_to_sign(self):
        """The measured refinement of the docstring's 'phase unchanged'.

        The magnitude MLP is an unconstrained 1x1 convolution, so its output can
        be negative and flip the phasor by pi. MEASURED: 57.4% of significant
        bins unchanged, 42.6% flipped by exactly pi, and NOTHING in between --
        which is the same fact as the real-ratio claim above, seen from the
        phase side.
        """
        spectrum_in = np.fft.fft2(SPECTRAL_INPUT, axes=(1, 2))
        spectrum_out = np.fft.fft2(self._fremlp_output(), axes=(1, 2))
        significant = np.abs(spectrum_in) > 1e-2
        delta = np.angle(spectrum_out) - np.angle(spectrum_in)
        delta = np.abs((delta + np.pi) % (2 * np.pi) - np.pi)[significant]

        unchanged = float((delta < 1e-3).mean())
        flipped = float((np.abs(delta - np.pi) < 1e-3).mean())
        assert unchanged + flipped == pytest.approx(1.0, abs=1e-6), (
            f"only {100 * (unchanged + flipped):.1f}% of bins are at 0 or pi; "
            f"the rest carry a genuine phase rotation, which FreMLP must not "
            f"introduce"
        )
        assert unchanged > 0.0 and flipped > 0.0, (
            f"measured {100 * unchanged:.1f}% unchanged / {100 * flipped:.1f}% "
            f"flipped; both were non-zero when this was written, and a run with "
            f"zero flips would mean the magnitude MLP had become sign-constrained"
        )
