"""Image restoration — a LITERATURE SURVEY, not an implementation.

This directory holds no Python. It sits under ``models/`` because that is where a
reader looks for "what does this repo know about image restoration", but it ships
exactly two documents and no code:

* ``BENCHMARKS.md`` — consolidated PSNR/SSIM tables for all-in-one image restoration
  models, transcribed from 2023-2026 papers reporting on the same standard benchmarks
  (BSD68, Rain100L, SOTS, GoPro, LOL, CDD11, Urban100, Kodak24, WeatherBench).
* ``README.md`` — a pointer to the above.

**Every number in those tables is quoted from a paper. Not one of them was measured
here, and no model in this repository produced any of them.** They are useful as a
target board and as a citation index; they are not a claim about this code. For the
denoising architectures this repo actually implements and can measure, see
``dl_techniques.models.vision.bias_free_denoisers`` and ``src/train/bfunet/``.

This module is deliberately empty of exports. It defines no ``keras.Model``, no
``create_*`` factory and no ``__all__``, so it resolves to no main module and is a
named exemption in
``tests/test_models/test_package_api_contract.py::_PACKAGES_WITHOUT_MAIN_MODULE``
rather than an R-039 offender. If an implementation ever lands here, delete that
exemption in the same commit that adds the model.
"""
