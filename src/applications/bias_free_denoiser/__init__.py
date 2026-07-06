"""Bias-free-denoiser inverse-problem application (public API).

One unified interface that solves all of Kadkhodaie & Simoncelli (2021)'s problems
using the prior implicit in a bias-free denoiser: (0) unconstrained prior sampling
[Algorithm 1], (1) block inpainting, (2) random missing pixels, (3) super-resolution,
(4) spectral deblurring, (5) compressive sensing [Algorithm 2]. Every problem runs
through the SAME :class:`UniversalInverseSolver` loop; only the
:class:`MeasurementOperator` changes.

The GUI-free core is exported here (INV-7): :class:`DenoiserPrior` (the frozen
denoiser wrapped as an implicit prior), :class:`UniversalInverseSolver` (the unified
Algorithm-1/2 stochastic-ascent loop), and the :class:`MeasurementOperator` family.
The optional Streamlit GUI lives in ``streamlit_app.py`` and is the only module that
imports streamlit; the CLI demo lives in ``main.py``.

All pixels live in the ``[-0.5, +0.5]`` domain the checkpoint was trained on
(INV-1 / D-002); use :meth:`DenoiserPrior.ingest` / :meth:`DenoiserPrior.denorm` to
move in and out of it.
"""

from .denoiser_prior import DenoiserPrior
from .solver import UniversalInverseSolver
from .operators import (
    MeasurementOperator,
    MaskOperator,
    NullOperator,
    InpaintingOperator,
    RandomPixelsOperator,
    SuperResolutionOperator,
    SpectralDeblurOperator,
    CompressiveSensingOperator,
)

__all__ = [
    "DenoiserPrior",
    "UniversalInverseSolver",
    "MeasurementOperator",
    "MaskOperator",
    "NullOperator",
    "InpaintingOperator",
    "RandomPixelsOperator",
    "SuperResolutionOperator",
    "SpectralDeblurOperator",
    "CompressiveSensingOperator",
]
