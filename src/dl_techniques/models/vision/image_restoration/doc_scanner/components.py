"""DocScanner's rectification-stage parts, and the width table the whole port reads.

This module will hold the RAFT-lineage building blocks of DocScanner's second
stage -- the ``align_corners`` coordinate adapter, ``coords_grid``, the convex
``8x`` upsample, the instance-norm helper, the residual block, the feature
encoder, ``SepConvGRU``, the motion encoder, the flow/mask heads and the update
block. At this point it carries only the two things every one of those parts
depends on and nothing else depends on: the module-level CONSTANTS and the
single ``_VARIANT_SPEC`` width table.

Why the width table lives HERE and not in ``model.py``
------------------------------------------------------
The package's import graph runs ``model.py -> components.py -> (nothing in this
package)``. ``components.py`` needs every channel width, because the encoder,
the GRU and the motion encoder are all built here; if the table lived in
``model.py`` those widths would have to travel back down the import edge, which
is a cycle. So the table is defined at the bottom of the graph and ``model.py``
imports it, rather than the reverse. Nothing about the table is private to
``model.py``: it is the port's single source of every channel count, which is
exactly why it has to sit where every consumer can reach it.

Registration convention
-----------------------
Classes added to this module register under
``dl_techniques.models.doc_scanner.components`` -- the key strips BOTH the
``vision`` family and the ``image_restoration`` subfamily, per the repo-wide
convention. It is NOT the full import path. Nothing is registered yet; this
module defines no class.

The upstream reference
----------------------
Every constant below cites the line of the upstream PyTorch release it was read
from, at ``/media/arxwn/data_fast/repositories/DocScanner`` (commit as cloned
2026-09-10). One constant -- :data:`SEQUENCE_LOSS_GAMMA` -- has no upstream line
to cite, because the release ships inference code only and contains no training
loop at all; it is cited to the paper instead, and labelled as such.

References:
    - Feng et al., 2021. DocScanner: Robust Document Image Rectification with
      Progressive Learning. (https://arxiv.org/abs/2110.14968), v2.
    - Upstream release: https://github.com/fh2019ustc/DocScanner --
      ``model.py``, ``update.py``, ``extractor.py``, ``seg.py``,
      ``inference.py``.
"""

from typing import Any, Dict, List

# ---------------------------------------------------------------------
# Constants.
#
# Every value below is transcribed from the upstream release, and the citation
# beside it is the line that USES the value, never a constructor default that
# no call site ever reaches. See the `_VARIANT_SPEC` anchor at the bottom of
# this file for why that distinction is load-bearing here.
# ---------------------------------------------------------------------

# `extractor.py:21` and `:93` construct `nn.InstanceNorm2d(...)`, whose torch
# default is `eps=1e-5`. The Keras stand-in for per-sample instance norm is
# `keras.layers.GroupNormalization(groups=C)`, whose OWN default epsilon is
# `1e-3` -- 100x larger, with no shape symptom and no warning. Every
# construction site in this port therefore passes this value explicitly.
# `norms/factory.py`'s 18 registry keys contain neither instance nor group
# norm, so the layer is constructed directly rather than through
# `create_normalization_layer`; `layers/CLAUDE.md` requires exactly this when a
# normalization layer is built by hand -- state the epsilon, cite the source.
INSTANCE_NORM_EPSILON: float = 1e-5

# The rectifier runs its refinement loop on a 1/8-resolution coordinate field
# and convex-upsamples back. `model.py:49-50` builds `coords_grid(N, H // 8,
# W // 8)`; `model.py:65` reshapes the upsampled flow to `(N, 2, 8 * H, 8 * W)`.
# The feature encoder's total stride is the same 8 (stem stride 2 at
# `extractor.py:96`, then `layer2`/`layer3` at stride 2, `extractor.py:100-101`).
SPATIAL_DIVISOR: int = 8

# The number of GRU refinement iterations. `model.py:67` declares `iters=12` as
# the forward-pass default AND `inference.py:28` passes `iters=12` explicitly at
# the only call site, so the two agree -- this is not a dead default.
REFINE_ITERATIONS: int = 12

# The exponential decay of the K-iteration sequence loss,
# `L = sum_{k=1..K} gamma^(K-k) * L^(k)`, so the LAST iteration carries weight
# exactly 1 and earlier ones decay backwards.
#
# NOT FROM THE CODE. The upstream release ships inference only -- it contains no
# training loop, no loss and no optimizer -- so there is no file:line to cite.
# The value is the paper's: arXiv:2110.14968v2, Eq. 9 (`gamma = 0.85`, `K = 12`).
# Anyone reconciling this port against the upstream checkout will not find it
# there, and that is expected rather than a transcription error.
SEQUENCE_LOSS_GAMMA: float = 0.85

# The composite pipeline binarizes the segmenter's confidence map before using
# it as a multiplicative background mask: `inference.py:25`,
# `msk = (msk > 0.5).float()`.
SEG_MASK_THRESHOLD: float = 0.5

# The backward map the rectifier emits is in PIXEL units; the upstream
# inference wrapper rescales it to the `[-1, 1]` range its sampler wants with
# `bm = (2 * (bm / 286.8) - 1) * 0.99` -- `inference.py:29`, one line carrying
# both numbers. The divisor is NOT the 288 training resolution: it is 286.8,
# and the trailing 0.99 shrinks the range slightly inside the image domain.
# Neither number is explained upstream or in the paper; both are transcribed
# exactly rather than "corrected" to 288 / 1.0.
BM_CALIBRATION_DIVISOR: float = 286.8
BM_CALIBRATION_SCALE: float = 0.99

# ---------------------------------------------------------------------


# DECISION plan-2026-09-10T065432-05fcb6dd/D-006: read these widths from the CALL
# SITES (`model.py:35-39`, `update.py:88`). Do NOT read them from `update.py:18,36`'s
# `hidden_dim=128, input_dim=192+128` defaults, which no call site reaches: the GRU is
# width-agnostic, so a 128-wide hidden state passes EVERY shape test. Worse, `192+128`
# equals the live 320 (D-007), so the wrong reading looks half-confirmed. No width
# literal belongs anywhere else in this package. See decisions.md D-006, D-007.
_VARIANT_SPEC: Dict[str, Dict[str, Any]] = {
    "docscanner-l": {
        # `model.py:35-36`: `self.hidden_dim = hdim = 160`, `self.context_dim = 160`.
        # The encoder's 320 output channels are split in half into the GRU's
        # hidden state (`tanh`) and its context input (`relu`), `model.py:74-76`,
        # which is why these two are equal and sum to `fnet_output_dim`.
        "hidden_dim": 160,
        "context_dim": 160,
        # `update.py:88`: `SepConvGRU(hidden_dim=hidden_dim, input_dim=160+160)`.
        # This overrides `update.py:36`'s dead `input_dim=192+128` default.
        "gru_input_dim": 320,
        # `model.py:38`: `BasicEncoder(output_dim=320, norm_fn='instance')`, and
        # `extractor.py:104`'s `conv2` maps 240 -> `output_dim` with a 1x1.
        "fnet_output_dim": 320,
        # `extractor.py:96`: the 7x7 stride-2 stem emits 80 channels. The
        # reference then declares `norm1 = nn.InstanceNorm2d(64)` at
        # `extractor.py:93` and applies it to those 80 channels -- inert under
        # torch's `affine=False` (no parameters are allocated), but a Keras
        # affine normalization allocates by shape and would CRASH. 80 is the
        # true width; 64 is a latent upstream bug.
        "encoder_stem_channels": 80,
        # `extractor.py:98-101`: `in_planes = 80`, then `_make_layer(80,
        # stride=1)`, `_make_layer(160, stride=2)`, `_make_layer(240, stride=2)`.
        "encoder_stage_channels": (80, 160, 240),
    },
}

# The motion encoder's, flow head's and mask head's internal widths
# (`update.py:66-70`, `:89`, `:91-94`) are DERIVED from the row above rather
# than independent -- e.g. the motion encoder's 158 is `hidden_dim - 2`, and the
# mask head's 576 is `SPATIAL_DIVISOR ** 2 * 9`. They are added to this table by
# the step that builds those blocks, expressed as derivations, so that no width
# literal ever appears at a construction site.

__all__: List[str] = [
    "INSTANCE_NORM_EPSILON",
    "SPATIAL_DIVISOR",
    "REFINE_ITERATIONS",
    "SEQUENCE_LOSS_GAMMA",
    "SEG_MASK_THRESHOLD",
    "BM_CALIBRATION_DIVISOR",
    "BM_CALIBRATION_SCALE",
]
