"""Image restoration — three model packages plus a literature survey.

This subfamily groups the vision models whose task is recovering a clean image from a
degraded one: denoising, deblurring, dehazing, low-light enhancement, and the all-in-one
variants that handle several degradations with one network.

* ``darkir/`` — DarkIR, low-light image restoration.
* ``pw_fnet/`` — a 2-level U-Net with FFT token mixing and multi-scale supervision. Its
  name misattributes on two of three words; see ``models/CLAUDE.md`` § Names that
  misattribute before trusting it.
* ``scunet/`` — SCUNet, a swin-conv U-Net denoiser.

It also carries two documents that predate the three packages above and are not
superseded by them:

* ``BENCHMARKS.md`` — consolidated PSNR/SSIM tables for all-in-one image restoration
  models, transcribed from 2023-2026 papers reporting on the same standard benchmarks
  (BSD68, Rain100L, SOTS, GoPro, LOL, CDD11, Urban100, Kodak24, WeatherBench). It records
  what the field claims and where to read it.
* ``README.md`` — a pointer to the above.

Every number in those tables comes from a paper. None of them was measured here, and no
model in this repository produced any of them. A reader who finds ``scunet/`` next to a
SCUNet row in ``BENCHMARKS.md`` should not read that row as this package's score. For
denoising architectures this repo measures end to end, see
``dl_techniques.models.vision.bias_free_denoisers`` and ``src/train/bfunet/``.

This module holds no re-exports, like every other container under ``models/``. Import
from the leaf package:

    from dl_techniques.models.vision.image_restoration.scunet import create_scunet

It defines no ``keras.Model``, no ``create_*`` factory and no ``__all__``. This directory
is a container, not a package with its own model; its entry in
``tests/test_models/test_package_api_contract.py::_PACKAGES_WITHOUT_MAIN_MODULE`` reflects
that.
"""
