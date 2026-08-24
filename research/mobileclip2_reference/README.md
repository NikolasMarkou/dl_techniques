# MobileCLIP2 upstream reference material

Third-party reference material, supplied by the repository owner and committed VERBATIM so that
the tests of `src/dl_techniques/models/fastvit/` and `models/mobile_clip/mobile_clip_v2.py` have a **real** oracle instead of a second
hand transcription of the same numbers.

## Contents

| Path | What it is |
| --- | --- |
| `mobileclip2.py` | Apple's MobileCLIP2 image-encoder definition file, verbatim. Defines `fastvit_mci3`, `fastvit_mci4`, the monkey-patched `convolutional_stem_timm` and `LayerNormChannel`. |
| `model_configs/MobileCLIP-S3.json`, `MobileCLIP-S4.json` | The MobileCLIP (v1) S3/S4 open_clip model configs. |
| `model_configs/MobileCLIP2-S0.json`, `-S2.json`, `-S3.json`, `-S4.json` | The MobileCLIP2 S0/S2/S3/S4 open_clip model configs. |

Provenance: Apple's MobileCLIP2 release. Nothing here was written by this repository.

## Rules

- **Not library code.** This directory is NOT a Python package, is NOT on the import path, and is
  NOT imported by anything in `src/`. It sits under `research/` deliberately.
- **Not executable here.** `mobileclip2.py` is PyTorch/`timm` code. Neither `torch` nor `timm` is
  installed in this environment, so it cannot be run or imported. The tests that use it PARSE it
  with `ast`; importing it would raise `ModuleNotFoundError`.
- **Do not edit.** Any change makes the oracle agree with the port by construction, which is the
  exact defect this directory exists to remove. If upstream changes, replace the file wholesale
  and say so.

## What this directory does and does NOT cover

Covered — these tests read the files here and compare them field by field against the port:

- `tests/test_models/test_fastvit/test_model.py::TestMciVariantTable::test_mci3_mci4_match_supplied_source`
  parses `mobileclip2.py` and checks `MCI_VARIANTS['mci3']` / `['mci4']`.
- `tests/test_models/test_mobile_clip/test_mobile_clip_v2.py::TestModelVariants::test_model_variants_match_supplied_json_configs`
  reads all six JSONs and checks all six `MODEL_VARIANTS` rows.

**NOT covered — `mci0`, `mci1` and `mci2` are not defined here.** The supplied source defines only
`fastvit_mci3` and `fastvit_mci4`; the JSON configs merely NAME `fastvit_mci0` / `fastvit_mci2`
without giving their architecture. Those three variant tables were transcribed from `timm`
upstream, `timm` is not installed, and so **they still have no local oracle** — deviation X-3 in
`src/dl_techniques/models/fastvit/README.md` and `src/dl_techniques/models/mobile_clip/README.md`.

The three ViT-tower configs of the release (`MobileCLIP-L-14`, `MobileCLIP2-B`,
`MobileCLIP2-L-14`) are out of scope for this port and are deliberately not committed.
