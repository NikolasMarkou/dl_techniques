"""Image restoration — three model packages plus the literature survey that predates them.

This subfamily groups the vision models whose task is recovering a clean image from a
degraded one — denoising, deblurring, dehazing, low-light enhancement and the all-in-one
variants that address several degradations with one network:

* ``darkir/`` — DarkIR, low-light image restoration.
* ``pw_fnet/`` — a 2-level U-Net with FFT token mixing and multi-scale supervision. Its
  name misattributes on two of three words; see ``models/CLAUDE.md`` § Names that
  misattribute before trusting it.
* ``scunet/`` — SCUNet, a swin-conv U-Net denoiser.

It also carries the two documents that were this directory's original and, until the
2026-08-24 restructure, its only reason for existing. They are not leftovers and they are
not superseded by the three packages above:

* ``BENCHMARKS.md`` — consolidated PSNR/SSIM tables for all-in-one image restoration
  models, transcribed from 2023-2026 papers reporting on the same standard benchmarks
  (BSD68, Rain100L, SOTS, GoPro, LOL, CDD11, Urban100, Kodak24, WeatherBench). It is a
  target board and a citation index: it says what the field claims and where to read it.
* ``README.md`` — a pointer to the above.

**Every number in those tables is quoted from a paper. Not one of them was measured here,
and no model in this repository produced any of them.** That warning applies with more
force now that implementations sit beside the tables: a reader who finds ``scunet/`` next
to a SCUNet row in ``BENCHMARKS.md`` must not read the row as this package's score. For the
denoising architectures this repo can actually measure end to end, see
``dl_techniques.models.vision.bias_free_denoisers`` and ``src/train/bfunet/``.

This module is deliberately free of re-exports, like every other container under
``models/``. Import from the leaf package:

    from dl_techniques.models.vision.image_restoration.scunet import create_scunet

It therefore defines no ``keras.Model``, no ``create_*`` factory and no ``__all__``. Under
the leaf-package walk introduced on 2026-08-25 this directory is a CONTAINER, not a
population member, so its entry in
``tests/test_models/test_package_api_contract.py::_PACKAGES_WITHOUT_MAIN_MODULE`` is inert
rather than load-bearing.
"""
