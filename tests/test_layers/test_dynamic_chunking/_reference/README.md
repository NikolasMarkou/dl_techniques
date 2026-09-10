# Vendored H-Net reference sources — EVIDENCE, do not edit

These `.py` files are copied **byte-for-byte** from the upstream PyTorch H-Net
reference implementation. They exist so that
`tests/test_layers/test_dynamic_chunking/hnet_reference_numpy.py` can be read
side-by-side with the source it was transcribed from, and so that every
`file:line` citation in that module can be checked without leaving this repo.

| vendored file | upstream path | sha256 |
|---|---|---|
| `dc.py` | `hnet/modules/dc.py` | `b0a30a75245a0f6911a92a56e6be80a3725fc27aa859bd0e9905c239fe87bb38` |
| `train.py` | `hnet/utils/train.py` | `83534fe9ddf769f78bac331fbb7aacd8a9412a30e2f5cdf180f4d25434c3d27e` |

- Upstream: <https://github.com/goombalab/hnet.git>, commit
  `3673fe1217ebeb0d1438c7c71d58d32bdd190ec2` (2025-09-30). Local clone read at
  `/media/arxwn/data_fast/repositories/hnet`.
- Licence: MIT (`LICENSE`, copied here alongside the sources).

## Three rules

1. **Never edit these files.** Not to reformat, not to lint, not to "fix" an
   import. They are the ground truth a transcription is graded against; an
   edited oracle source is not an oracle. The line numbers cited in
   `hnet_reference_numpy.py` are the line numbers of THESE files.
2. **There is deliberately no `__init__.py` in this directory.** That is what
   keeps `_reference/` un-importable as a package: nothing under `src/` can
   reach it, it cannot be executed by the suite (it imports `torch`,
   `einops` and `mamba_ssm`, none of which this repo depends on), and it can
   therefore never quietly become a second copy of our own code.
3. **Pytest must not collect anything here.** No file in this directory is
   named `test_*.py`, and the sibling transcription is named
   `hnet_reference_numpy.py` for the same reason.
