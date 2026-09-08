r"""Run a trained DocRes on real pages: pad, predict, post-process, write.

Usage::

    # what the CLI offers -- allocates nothing, touches no GPU
    MPLBACKEND=Agg .venv/bin/python -m train.doc_res.infer_doc_res --help

    # one page, one task, a trained checkpoint
    MPLBACKEND=Agg .venv/bin/python -m train.doc_res.infer_doc_res \
        --task binarization --input page.png --output-dir results/docres_infer \
        --checkpoint results/doc_res_binarization_.../best_model.keras --gpu 1

    # a directory of pages, and the three prompt channels alongside each output
    ... --input /path/to/pages/ --save-dtsprompt

This is the counterpart of :mod:`train.doc_res.train_doc_res`: the training
entry point owns ``fit()``, this one owns everything between an image file and
a restored image file. Both parse FIRST and touch a GPU only afterwards.

Where the padding lives, and why it lives here
----------------------------------------------
:class:`~dl_techniques.models.vision.image_restoration.doc_res.model.DocRes`
downsamples three times and **refuses** an input whose height or width is not a
multiple of 8, naming the axis and the remedy (D-016). That refusal is
deliberate: a model that padded silently could not un-pad for a caller who was
never told it happened. So the padding is the caller's job, and this module is
the caller. :func:`pad_to_multiple` replicates the border at the TOP and the
LEFT exactly as upstream's ``stride_integral`` does, and
:func:`crop_padding` takes the same rows and columns back off the prediction,
so the written image has the input's size to the pixel.

**The unused tiler is NOT ported.** Upstream's
``data/preprocess/crop_merge_image.py`` defines ``split_img`` / ``combine_imgs``
for patch-wise inference, and *nothing calls them* -- not ``inference.py``, not
``eval.py``. Only ``stride_integral`` is used. Porting the tiler would be
porting dead code, with a seam-artefact class of bug attached.

No task string is compared anywhere below
-----------------------------------------
``dl_techniques.datasets.document_restoration.tasks.TASKS`` is the single home
of DocRes task specificity, and this module reads it rather than re-stating it.
Both dispatch tables here -- :data:`INPUT_BUILDERS` and :data:`POSTPROCESSORS`
-- are keyed by the table's own ``POSTPROCESS_*`` constants, and the supervised
channel slice comes from :attr:`TaskSpec.supervised_slice`. Change a row of
``TASKS`` and the behaviour of this script changes with it; that is asserted,
not asserted-about, in ``tests/test_train/test_doc_res/test_inference.py``.

The one composite name, :data:`END2END`, is not a task and is not in the table.
It is a *sequence* of three table keys (:data:`END2END_STAGES`), each resolved
through ``get_task`` like any other.

Deviations from upstream, stated
--------------------------------
Each of these was a choice, not an oversight.

``end2end`` CHAINS IN MEMORY.
    Upstream writes ``restorted/step1.jpg`` and re-reads it between stages
    (``inference.py:305-312``), baking a JPEG encode/decode -- a lossy,
    quality-75 quantisation -- into the middle of the pipeline. This port
    passes the array. The output therefore differs from upstream's, in the
    direction of *not* having been degraded twice by an artefact of how the
    reference implementation happened to plumb its stages.

NUMPY + SCIPY ONLY, no OpenCV (D-002).
    ``cv2.remap`` becomes :func:`remap_bilinear`, ``cv2.blur`` becomes a
    ``uniform_filter(mode='nearest')``, ``cv2.resize`` becomes
    :func:`resize_bilinear`. OpenCV is undeclared in ``pyproject.toml``, and
    the prompt module it would have to agree with is already numpy+scipy.
    Bit-parity with upstream is not claimed and is not needed: no DocRes
    checkpoint is ported (``pretrained=True`` raises ``NotImplementedError``),
    so only train/inference self-consistency matters.

THE PROMPT IS COMPUTED ON THE UNPADDED PAGE, for every task.
    Upstream is inconsistent: appearance and deshadowing prompt the original
    and pad afterwards, while binarization and deblurring pad first and prompt
    the padded page. This port always prompts the original. The reason is not
    tidiness -- ``prepare_doc_res_data.py`` precomputes the *training* prompt
    sidecars on the unpadded page, so prompting a padded page at inference
    would feed the network a prompt it was never trained against.

RESAMPLING STAYS IN FLOAT.
    Upstream round-trips through ``uint8`` between its resize and its
    normalisation; this port normalises once (``/255``, the same single site
    ``common.py`` uses) and resamples the float. The difference is below one
    grey level and it removes a quantisation, not a behaviour.

THE 1600 px CAP APPLIES TO EVERY ``clamp_image`` TASK.
    Upstream applies it to appearance and deshadowing but not to deblurring,
    which is a per-task-string branch. The ``TASKS`` table binds all three to
    ``POSTPROCESS_CLAMP_IMAGE`` and that constant's own docstring already
    attaches the homomorphic re-composition to the *mode*. Keying the cap off
    the mode is the only way to honour "no task-string branching outside
    ``TASKS``"; deblurring consequently gains the cap. Nothing observes the
    difference today (the deblurring corpus host is dead and no deblurring
    checkpoint exists), and the alternative is the branch the table exists to
    forbid. See D-030.

``argmax`` IS TAKEN DIRECTLY, WITHOUT THE SOFTMAX.
    Upstream computes ``argmax(softmax(logits))``. Softmax is strictly
    increasing, so it cannot move an argmax; the call is arithmetic with no
    effect on the result.
    ``test_inference.py::test_the_softmax_upstream_applies_cannot_move_the_argmax``
    measures that rather than asserting it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import keras
import numpy as np
from scipy import ndimage

from dl_techniques.datasets.document_restoration.tasks import (
    POSTPROCESS_ARGMAX_BINARY,
    POSTPROCESS_CLAMP_IMAGE,
    POSTPROCESS_FLOW_REMAP,
    TaskSpec,
    get_task,
    task_names,
)
from dl_techniques.datasets.document_restoration.dtsprompt import (
    apply_document_mask,
    base_coordinate_grid,
)
from dl_techniques.models.vision.image_restoration.doc_res.model import (
    SPATIAL_DIVISOR,
    create_doc_res,
)
from dl_techniques.utils.logger import logger
from train.common import setup_gpu
from train.doc_res.common import (
    BINARY_INK_CLASS_INDEX,
    N_RGB_CHANNELS,
    PIXEL_SCALE,
)

__all__ = [
    "DEWARP_INPUT_SIZE",
    "END2END",
    "END2END_STAGES",
    "FLOW_SMOOTHING_KERNEL",
    "FLOW_SMOOTHING_PASSES",
    "INPUT_BUILDERS",
    "MAX_INPUT_SIZE",
    "NON_CONFIG_DESTS",
    "POSTPROCESSORS",
    "InputPlan",
    "RestorationResult",
    "as_rgb",
    "build_input_plan",
    "build_parser",
    "collect_input_images",
    "crop_padding",
    "load_docres_model",
    "load_mask",
    "load_page",
    "main",
    "pad_to_multiple",
    "parse_arguments",
    "postprocess",
    "predict",
    "remap_bilinear",
    "resize_bilinear",
    "restore",
    "restore_end2end",
    "restore_image_file",
    "save_image",
]


PROGRAM_NAME: str = "infer_doc_res.py"
"""``prog`` for the parser, so ``--help`` names the script rather than
``__main__`` when the module is run with ``python -m``."""

NON_CONFIG_DESTS: FrozenSet[str] = frozenset({"gpu"})
"""``--gpu`` acts on the process and is consumed by ``setup_gpu``; it is not
part of any restoration call. Named for symmetry with
``train_doc_res.NON_CONFIG_DESTS``."""

END2END: str = "end2end"
"""The composite mode. **Not** a key of ``TASKS`` and deliberately not added to
it: it has no prompt, no loss and no post-processing of its own."""

END2END_STAGES: Tuple[str, ...] = ("dewarping", "deshadowing", "appearance")
"""The three ``TASKS`` keys :data:`END2END` runs, in order
(``inference.py:305-312``). Every one is resolved through ``get_task``."""

DEWARP_INPUT_SIZE: int = 256
"""Fixed square resolution the network runs at when the post-processing is a
flow remap (``inference.py:96``). A flow field is a coordinate map, so it is
resolution-independent: it is predicted small, smoothed, and scaled up to the
source resolution before the warp."""

MAX_INPUT_SIZE: int = 1600
"""Above this longest edge, a ``clamp_image`` task runs at ``1600x1600`` and
the full-resolution output is rebuilt by homomorphic re-composition rather than
by upsampling the prediction (``inference.py:141-166``). The threshold is
inclusive: ``max(h, w) >= MAX_INPUT_SIZE`` takes the resized branch, matching
upstream's ``if max(w,h) < MAX_SIZE: pad``."""

FLOW_SMOOTHING_PASSES: int = 15
"""Box-blur passes over the predicted flow before the warp
(``inference.py:121-122``). Reproduced, not omitted: fifteen 3x3 box passes are
a wide, near-Gaussian smoothing, and dropping them leaves the warp following
per-pixel prediction noise."""

FLOW_SMOOTHING_KERNEL: int = 3
"""Extent of each box-blur pass. ``cv2.blur((3,3), BORDER_REPLICATE)`` is
``uniform_filter(size=3, mode='nearest')``."""

BACKGROUND_LEVEL_U8: int = 255
"""Value written for the *background* class of a binarized page."""

INK_LEVEL_U8: int = 0
"""Value written for the *ink* class of a binarized page. Ink is black and
background is white, which is the DIBCO convention the ground truth uses; which
predicted channel means ink is not restated here but read from
``common.BINARY_INK_CLASS_INDEX``."""

IMAGE_SUFFIXES: Tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
)
"""Suffixes ``--input`` accepts when it names a directory."""

DEFAULT_OUTPUT_DIR: str = "results/doc_res_inference"
"""Repo-root ``results/`` as everywhere else in ``src/train``. Nothing under
``results/`` is ever deleted by anything in this repository."""


# ---------------------------------------------------------------------------
# Resampling primitives (numpy + scipy; see D-002)
# ---------------------------------------------------------------------------


def resize_bilinear(array: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resample in float, with OpenCV's ``INTER_LINEAR`` conventions.

    Interface contract: pure. Half-pixel sample centres
    (``src = (dst + 0.5) * scale - 0.5``) and a replicated border, evaluated in
    float64 with **no quantisation** -- unlike
    ``dtsprompt._resize_bilinear``, which is the same arithmetic followed by a
    round-and-clip to uint8. The two are pinned together by
    ``test_inference.py::test_the_float_resize_rounds_to_the_prompt_modules_resize``,
    so this is a widening of that function's dtype, not a second convention.

    The float form is required, not preferred: the homomorphic re-composition
    resizes a *ratio* field whose values are not grey levels, and rounding it
    to uint8 mid-computation would quantise an illumination map.

    :param array: ``(H, W)`` or ``(H, W, C)`` array of any numeric dtype.
    :param out_h: Target height in pixels.
    :param out_w: Target width in pixels.
    :return: float64 array of shape ``(out_h, out_w[, C])``.
    :raises ValueError: If either target extent is not positive.
    """
    if out_h < 1 or out_w < 1:
        raise ValueError(f"resize target must be positive, got {(out_h, out_w)}")
    src = np.asarray(array, dtype=np.float64)
    in_h, in_w = src.shape[:2]
    if (in_h, in_w) == (out_h, out_w):
        return src.copy()

    y = (np.arange(out_h, dtype=np.float64) + 0.5) * (in_h / out_h) - 0.5
    x = (np.arange(out_w, dtype=np.float64) + 0.5) * (in_w / out_w) - 0.5
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    fy = (y - y0)[:, None]
    fx = (x - x0)[None, :]
    y0c = np.clip(y0, 0, in_h - 1)
    y1c = np.clip(y0 + 1, 0, in_h - 1)
    x0c = np.clip(x0, 0, in_w - 1)
    x1c = np.clip(x0 + 1, 0, in_w - 1)
    if src.ndim == 3:
        fy = fy[..., None]
        fx = fx[..., None]

    top = src[y0c][:, x0c] * (1.0 - fx) + src[y0c][:, x1c] * fx
    bottom = src[y1c][:, x0c] * (1.0 - fx) + src[y1c][:, x1c] * fx
    return top * (1.0 - fy) + bottom * fy


def remap_bilinear(
        source: np.ndarray, map_x: np.ndarray, map_y: np.ndarray
) -> np.ndarray:
    """Bilinear backward-warp: ``out[y, x] = source[map_y[y,x], map_x[y,x]]``.

    Interface contract: pure. Reproduces
    ``cv2.remap(source, map_x, map_y, INTER_LINEAR)`` at OpenCV's default
    border mode, ``BORDER_CONSTANT`` with value 0 -- a tap that falls outside
    the source contributes zero, so a warp that reaches past the page edge
    darkens rather than smearing the border inward. Arithmetic is float64;
    OpenCV interpolates with 5-bit fixed-point weights, so agreement is close
    but not bit-exact (D-002 / assumption A2).

    :param source: ``(H, W)`` or ``(H, W, C)`` array; the image being sampled.
    :param map_x: ``(out_h, out_w)`` float array of source COLUMN coordinates.
    :param map_y: ``(out_h, out_w)`` float array of source ROW coordinates.
    :return: uint8 array of shape ``map_x.shape[+ (C,)]``.
    :raises ValueError: If the two coordinate maps do not share one shape.
    """
    if map_x.shape != map_y.shape:
        raise ValueError(
            f"remap coordinate maps must share a shape, got {map_x.shape} "
            f"and {map_y.shape}"
        )
    src = np.asarray(source, dtype=np.float64)
    in_h, in_w = src.shape[:2]

    xf = np.asarray(map_x, dtype=np.float64)
    yf = np.asarray(map_y, dtype=np.float64)
    x0 = np.floor(xf).astype(np.int64)
    y0 = np.floor(yf).astype(np.int64)
    fx = xf - x0
    fy = yf - y0

    accumulator = np.zeros(
        xf.shape + (() if src.ndim == 2 else (src.shape[2],)), dtype=np.float64
    )
    for dy, wy in ((0, 1.0 - fy), (1, fy)):
        for dx, wx in ((0, 1.0 - fx), (1, fx)):
            yi = y0 + dy
            xi = x0 + dx
            inside = (yi >= 0) & (yi < in_h) & (xi >= 0) & (xi < in_w)
            weight = wy * wx * inside
            tap = src[np.clip(yi, 0, in_h - 1), np.clip(xi, 0, in_w - 1)]
            if src.ndim == 3:
                weight = weight[..., None]
            accumulator += tap * weight
    return np.clip(np.rint(accumulator), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Padding -- the whole reason this shim exists (D-016)
# ---------------------------------------------------------------------------


def pad_to_multiple(
        array: np.ndarray, divisor: int = SPATIAL_DIVISOR
) -> Tuple[np.ndarray, int, int]:
    """Replicate-pad the TOP and LEFT until both extents divide ``divisor``.

    Interface contract: pure. The port of upstream's ``stride_integral``
    (``crop_merge_image.py:116-131``). The side matters and is not a detail:
    upstream pads top/left and crops with ``pred[padding_h:, padding_w:]``,
    so padding at the bottom/right instead would return an image shifted by up
    to seven pixels while keeping the correct SHAPE -- a defect no shape
    assertion can see. :func:`crop_padding` is its exact inverse and the two
    are tested as a round trip.

    :param array: ``(H, W)`` or ``(H, W, C)`` array.
    :param divisor: Required factor of both extents; the model's
        ``SPATIAL_DIVISOR`` (8).
    :return: ``(padded, pad_h, pad_w)``.
    :raises ValueError: If ``divisor`` is not positive.
    """
    if divisor < 1:
        raise ValueError(f"divisor must be positive, got {divisor}")
    height, width = array.shape[:2]
    pad_h = (-height) % divisor
    pad_w = (-width) % divisor
    if pad_h == 0 and pad_w == 0:
        return array, 0, 0
    widths = [(pad_h, 0), (pad_w, 0)] + [(0, 0)] * (array.ndim - 2)
    return np.pad(array, widths, mode="edge"), pad_h, pad_w


def crop_padding(array: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """Take back exactly what :func:`pad_to_multiple` added.

    Interface contract: pure; a view, not a copy.

    :param array: The padded array, or a prediction of the same spatial shape.
    :param pad_h: Rows added at the top.
    :param pad_w: Columns added at the left.
    :return: The array without the padding.
    """
    return array[pad_h:, pad_w:]


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputPlan:
    """One page, prepared for the network, plus what it takes to undo that.

    :param array: ``(1, H, W, 6)`` float32 ``rgb ++ prompt`` model input, in
        ``[0, 1]``.
    :param prompt: ``(H, W, 3)`` float32 prompt at the model's resolution, in
        ``[0, 1]``; written out by ``--save-dtsprompt``.
    :param original: The ``(h, w, 3)`` uint8 page as it was read from disk.
        Every post-processor sizes its output from this, so a restored page is
        the size of the page that was handed in.
    :param pad_h: Rows :func:`pad_to_multiple` added at the top.
    :param pad_w: Columns it added at the left.
    :param resized: Whether the page was RESIZED to reach a legal input size
        rather than padded. Only a resized ``clamp_image`` page takes the
        homomorphic branch, so this flag is what makes that branch reachable.
    """

    array: np.ndarray
    prompt: np.ndarray
    original: np.ndarray
    pad_h: int
    pad_w: int
    resized: bool


def _prompt_channels(
        page: np.ndarray, spec: TaskSpec, mask: Optional[np.ndarray]
) -> np.ndarray:
    """Run the task's prompt generator and normalise it to float32 ``[0, 1]``.

    Interface contract: the ONE call site of ``spec.prompt_fn`` in this module.
    ``requires_mask`` and ``prompt_dtype`` are read off the spec, so neither
    the mask argument nor the ``/255`` is decided by a task name.

    :param page: ``(H, W, 3)`` uint8 page the prompt is computed from.
    :param spec: The task spec.
    :param mask: ``(H, W)`` uint8 document mask; required iff
        ``spec.requires_mask``.
    :return: ``(H, W, 3)`` float32 prompt in ``[0, 1]``.
    :raises ValueError: If the task needs a mask and none was supplied.
    """
    if spec.requires_mask:
        if mask is None:
            raise ValueError(
                f"task {spec.name!r} needs a document mask and none was given. "
                "Upstream obtains it from the separate MBD network, which this "
                "port does not include; pass one with --mask (a single-channel "
                "image, non-zero inside the page)."
            )
        prompt = spec.prompt_fn(page, mask)
    else:
        prompt = spec.prompt_fn(page)
    prompt = np.asarray(prompt, dtype=np.float32)
    if spec.prompt_dtype == "uint8":
        prompt = prompt / np.float32(PIXEL_SCALE)
    return prompt


def _stack(page: np.ndarray, prompt: np.ndarray) -> np.ndarray:
    """``rgb/255 ++ prompt`` as one ``(H, W, 6)`` float32 array.

    THE single normalisation site of the inference path, the mirror of
    ``common.decode_triplet``'s. Do not rescale again downstream.
    """
    rgb = np.asarray(page, dtype=np.float32) / np.float32(PIXEL_SCALE)
    return np.concatenate([rgb, prompt], axis=-1).astype(np.float32)


def _plan_fixed_size(
        page: np.ndarray, spec: TaskSpec, mask: Optional[np.ndarray]
) -> InputPlan:
    """Flow-remap preparation: mask the page, then run at ``256x256``.

    The RGB the network sees is the MASKED page (``inference.py:22``), not the
    raw one, and the prompt is built at the same reduced resolution so its base
    coordinate grid matches the flow the network predicts.
    """
    masked = apply_document_mask(page, mask) if mask is not None else page
    size = DEWARP_INPUT_SIZE
    small_page = np.clip(
        np.rint(resize_bilinear(masked, size, size)), 0, 255
    ).astype(np.uint8)
    small_mask = (
        None
        if mask is None
        else np.clip(np.rint(resize_bilinear(mask, size, size)), 0, 255).astype(
            np.uint8
        )
    )
    prompt = _prompt_channels(small_page, spec, small_mask)
    stack = _stack(small_page, prompt)
    return InputPlan(
        array=stack[None, ...], prompt=prompt, original=page,
        pad_h=0, pad_w=0, resized=True,
    )


def _plan_capped(
        page: np.ndarray, spec: TaskSpec, mask: Optional[np.ndarray]
) -> InputPlan:
    """Clamp-image preparation: pad below :data:`MAX_INPUT_SIZE`, resize above.

    Above the cap the network runs at ``1600x1600`` and the post-processor
    rebuilds full resolution homomorphically; below it the page is padded and
    the prediction is cropped, so the network sees every original pixel.
    """
    height, width = page.shape[:2]
    prompt = _prompt_channels(page, spec, mask)
    if max(height, width) >= MAX_INPUT_SIZE:
        size = MAX_INPUT_SIZE
        small_page = np.clip(
            np.rint(resize_bilinear(page, size, size)), 0, 255
        ).astype(np.uint8)
        small_prompt = resize_bilinear(prompt, size, size).astype(np.float32)
        return InputPlan(
            array=_stack(small_page, small_prompt)[None, ...],
            prompt=small_prompt, original=page,
            pad_h=0, pad_w=0, resized=True,
        )
    padded, pad_h, pad_w = pad_to_multiple(_stack(page, prompt))
    return InputPlan(
        array=padded[None, ...], prompt=prompt, original=page,
        pad_h=pad_h, pad_w=pad_w, resized=False,
    )


def _plan_padded(
        page: np.ndarray, spec: TaskSpec, mask: Optional[np.ndarray]
) -> InputPlan:
    """Argmax-binary preparation: pad to a multiple of 8, full resolution.

    A binarization decision is per-pixel and has no resolution cap upstream;
    the page goes in whole.
    """
    prompt = _prompt_channels(page, spec, mask)
    padded, pad_h, pad_w = pad_to_multiple(_stack(page, prompt))
    return InputPlan(
        array=padded[None, ...], prompt=prompt, original=page,
        pad_h=pad_h, pad_w=pad_w, resized=False,
    )


# DECISION plan-2026-09-08T111844-de235227/D-030
# Both dispatch tables are keyed by the TASKS table's own POSTPROCESS_*
# constants. Do NOT re-key either one on `spec.name`, and do NOT add an
# `if spec.name == "deblurring"` to restore upstream's uncapped deblurring
# path: a task-string comparison anywhere outside TASKS is exactly the
# duplication the table was built to end, and the mode constant's docstring in
# tasks.py already binds the homomorphic re-composition to the MODE. The
# consequence is stated and accepted -- deblurring inherits the 1600 px cap it
# does not have upstream. See D-030 in decisions.md.
INPUT_BUILDERS: Dict[
    str, Callable[[np.ndarray, TaskSpec, Optional[np.ndarray]], InputPlan]
] = {
    POSTPROCESS_FLOW_REMAP: _plan_fixed_size,
    POSTPROCESS_CLAMP_IMAGE: _plan_capped,
    POSTPROCESS_ARGMAX_BINARY: _plan_padded,
}
"""Post-processing mode -> input preparation. Complete over
``POSTPROCESS_MODES``; a new mode is a ``KeyError`` at the call site rather
than a silent fallthrough to the wrong preparation."""


def build_input_plan(
        page: np.ndarray, spec: TaskSpec, mask: Optional[np.ndarray] = None
) -> InputPlan:
    """Prepare one page for the network, table-driven.

    :param page: ``(H, W, 3)`` uint8 RGB page.
    :param spec: The task spec, from ``get_task``.
    :param mask: ``(H, W)`` uint8 document mask, for a task that needs one.
    :return: The plan.
    :raises KeyError: If the spec names a post-processing mode with no
        preparation.
    :raises ValueError: If the page is not an ``(H, W, 3)`` uint8 array.
    """
    page = np.asarray(page)
    if page.ndim != 3 or page.shape[2] != N_RGB_CHANNELS or page.dtype != np.uint8:
        raise ValueError(
            f"page must be an (H, W, {N_RGB_CHANNELS}) uint8 array, got shape "
            f"{page.shape} dtype {page.dtype}"
        )
    return INPUT_BUILDERS[spec.postprocess](page, spec, mask)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def _postprocess_clamp_image(
        prediction: np.ndarray, plan: InputPlan, spec: TaskSpec
) -> np.ndarray:
    """``clip(pred, 0, 1) -> uint8``, then crop -- or re-compose homomorphically.

    Below the cap this is upstream's ``clamp`` + ``*255`` + ``pred[ph:, pw:]``.
    At or above it, the low-resolution prediction is used as an illumination
    FIELD rather than as a picture: ``shadow = resize(original, 1600) / pred``
    is the per-pixel factor the network removed, and dividing the FULL-
    resolution original by that field upsampled restores full resolution
    without upsampling the prediction's detail. Upsampling ``pred`` directly
    would return a 1600 px-detailed image stretched to page size; this returns
    the page's own detail with the illumination divided out.
    """
    clamped = np.clip(np.asarray(prediction, dtype=np.float64), 0.0, 1.0)
    predicted = np.clip(np.rint(clamped * PIXEL_SCALE), 0, 255).astype(np.uint8)
    if not plan.resized:
        return crop_padding(predicted, plan.pad_h, plan.pad_w)

    height, width = plan.original.shape[:2]
    size = predicted.shape[0]
    # Upstream's `pred[pred==0] = 1`: the prediction is a DIVISOR here, and a
    # predicted zero is a division by zero, not a black pixel.
    denominator = predicted.astype(np.float64)
    denominator[denominator == 0.0] = 1.0
    reference = resize_bilinear(plan.original, size, size)
    shadow_map = resize_bilinear(reference / denominator, height, width)
    shadow_map[shadow_map == 0.0] = 1e-5
    return np.clip(
        plan.original.astype(np.float64) / shadow_map, 0, 255
    ).astype(np.uint8)


def _postprocess_argmax_binary(
        prediction: np.ndarray, plan: InputPlan, spec: TaskSpec
) -> np.ndarray:
    """``argmax`` over the supervised logit pair -> a ``{0, 255}`` page.

    The two supervised channels are a 2-class LOGIT PAIR, decided against each
    other. They are NOT one channel thresholded: a sigmoid-and-threshold reads
    one channel and invents a cut point, and would disagree with this wherever
    the pair is (say) ``(-3.0, -1.0)`` -- both "negative", but class 1 wins.
    Which class means ink is read from ``common.BINARY_INK_CLASS_INDEX``, the
    same constant the training target is built from, so the two cannot drift.
    """
    logits = np.asarray(prediction)[..., spec.supervised_slice]
    classes = np.argmax(logits, axis=-1)
    page = np.where(
        classes == BINARY_INK_CLASS_INDEX, INK_LEVEL_U8, BACKGROUND_LEVEL_U8
    ).astype(np.uint8)
    return crop_padding(page, plan.pad_h, plan.pad_w)


def _postprocess_flow_remap(
        prediction: np.ndarray, plan: InputPlan, spec: TaskSpec
) -> np.ndarray:
    """Predicted flow + base grid -> smoothed -> scaled -> backward warp.

    The network predicts an OFFSET from the normalised base-coordinate grid, so
    the grid is added back before anything else. Channel 0 is x and scales by
    the width, channel 1 is y and scales by the height -- swapping them keeps
    the shape, the dtype and the value range and transposes the page, which is
    why ``base_coordinate_grid`` carries its own anchor (D-021) and why this
    has a dedicated guard.
    """
    flow = np.asarray(prediction, dtype=np.float64)[..., spec.supervised_slice]
    field = flow + base_coordinate_grid(flow.shape[0], flow.shape[1])
    for _ in range(FLOW_SMOOTHING_PASSES):
        field = ndimage.uniform_filter(
            field, size=(FLOW_SMOOTHING_KERNEL, FLOW_SMOOTHING_KERNEL, 1),
            mode="nearest",
        )
    height, width = plan.original.shape[:2]
    field = resize_bilinear(field, height, width)
    return remap_bilinear(
        plan.original, field[..., 0] * width, field[..., 1] * height
    )


POSTPROCESSORS: Dict[
    str, Callable[[np.ndarray, InputPlan, TaskSpec], np.ndarray]
] = {
    POSTPROCESS_CLAMP_IMAGE: _postprocess_clamp_image,
    POSTPROCESS_ARGMAX_BINARY: _postprocess_argmax_binary,
    POSTPROCESS_FLOW_REMAP: _postprocess_flow_remap,
}
"""Post-processing mode -> the function that turns a prediction into a picture.
Keyed by the ``TASKS`` table's constants; see the D-030 anchor above
:data:`INPUT_BUILDERS`."""


def postprocess(
        prediction: np.ndarray, plan: InputPlan, spec: TaskSpec
) -> np.ndarray:
    """Turn one raw prediction into a written-out-able uint8 image.

    :param prediction: The network's ``(H, W, 3)`` output for one page, with
        the batch axis already removed.
    :param plan: The plan that produced the input.
    :param spec: The task spec.
    :return: uint8 array -- ``(h, w, 3)`` for the image tasks, ``(h, w)`` for
        binarization -- at the ORIGINAL page's spatial size.
    :raises KeyError: If the spec names a mode with no post-processor.
    """
    return POSTPROCESSORS[spec.postprocess](prediction, plan, spec)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_docres_model(
        checkpoint: Optional[str] = None, variant: str = "docres"
) -> keras.Model:
    """Load a trained DocRes, or say LOUDLY that there is not one.

    Interface contract: the single model-construction site of this module. An
    absent checkpoint is NOT an error -- the padding, prompt and
    post-processing paths are worth exercising on a fresh model -- but it is
    never silent, because a randomly-initialised restoration model produces
    plausible-looking garbage that no downstream assertion can distinguish from
    a bad checkpoint.

    :param checkpoint: Path to a ``.keras`` file, or ``None``.
    :param variant: A key of ``DocRes.MODEL_VARIANTS``, used only when there is
        no checkpoint.
    :return: The model.
    :raises FileNotFoundError: If ``checkpoint`` is given and does not exist.
    """
    if checkpoint is None:
        logger.warning(
            "NO CHECKPOINT GIVEN: running a FRESHLY INITIALISED %r. The output "
            "is untrained noise, not a restoration. Pass --checkpoint "
            "<run>/best_model.keras to restore anything.",
            variant,
        )
        return create_doc_res(variant=variant)
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"no DocRes checkpoint at {path}")
    logger.info("loading DocRes checkpoint %s", path)
    return keras.models.load_model(path, compile=False)


def predict(model: keras.Model, plan: InputPlan) -> np.ndarray:
    """Run one prepared page through the network.

    :param model: A DocRes model.
    :param plan: The prepared input.
    :return: The ``(H, W, 3)`` float32 prediction, batch axis removed.
    """
    outputs = model(plan.array, training=False)
    return np.asarray(keras.ops.convert_to_numpy(outputs))[0]


# ---------------------------------------------------------------------------
# Restoration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RestorationResult:
    """What one restoration produced.

    :param image: The restored page, uint8, at the input page's spatial size.
        ``(h, w, 3)`` for the image tasks, ``(h, w)`` for binarization.
    :param prompt: The ``(H, W, 3)`` float32 prompt actually fed to the
        network, at the model's resolution.
    :param stages: The ``TASKS`` keys executed, in execution order. One entry
        for a single task, three for :data:`END2END`.
    """

    image: np.ndarray
    prompt: np.ndarray
    stages: Tuple[str, ...]


def as_rgb(image: np.ndarray) -> np.ndarray:
    """Widen a single-channel page to 3 channels; pass RGB through unchanged.

    Interface contract: pure. Needed only because binarization returns a 2-D
    page while every other task returns RGB, and a chained stage must be handed
    an ``(H, W, 3)`` uint8 page.

    :param image: ``(H, W)`` or ``(H, W, 3)`` uint8.
    :return: ``(H, W, 3)`` uint8.
    """
    if image.ndim == 2:
        return np.repeat(image[..., None], N_RGB_CHANNELS, axis=-1)
    return image


def restore(
        model: keras.Model,
        page: np.ndarray,
        task: str,
        mask: Optional[np.ndarray] = None,
) -> RestorationResult:
    """Prepare, predict and post-process ONE page for ONE task.

    Interface contract: the single per-task restoration path. ``main`` calls
    it, :func:`restore_end2end` calls it once per stage, and the tests
    instrument it to observe stage ORDER -- so there is no second path an
    end-to-end run could take.

    :param model: A DocRes model.
    :param page: ``(H, W, 3)`` uint8 RGB page.
    :param task: A key of ``TASKS``.
    :param mask: ``(H, W)`` uint8 document mask, for a task that needs one.
    :return: The result.
    :raises ValueError: On an unknown task, a malformed page, or a missing mask.
    """
    spec = get_task(task)
    plan = build_input_plan(page, spec, mask)
    prediction = predict(model, plan)
    return RestorationResult(
        image=postprocess(prediction, plan, spec),
        prompt=plan.prompt,
        stages=(task,),
    )


def restore_end2end(
        model: keras.Model, page: np.ndarray, mask: Optional[np.ndarray] = None
) -> RestorationResult:
    """Chain :data:`END2END_STAGES` in memory, each stage on the last's output.

    Upstream re-invokes itself through ``cv2.imwrite`` / ``cv2.imread`` between
    stages; this passes the array. See the module docstring's deviation list --
    the disk round trip is a JPEG quantisation, not a step of the algorithm.

    The mask is offered to every stage and consumed only by the ones whose spec
    asks for it, which today is the dewarping stage alone. Note that each stage
    recomputes its OWN prompt from the previous stage's output, which is the
    point of chaining: deshadowing prompts the dewarped page, not the original.

    :param model: A DocRes model.
    :param page: ``(H, W, 3)`` uint8 RGB page.
    :param mask: ``(H, W)`` uint8 document mask for the dewarping stage.
    :return: The final result, carrying every stage name in order.
    :raises ValueError: If a stage needs a mask and none was supplied.
    """
    current = page
    prompt: Optional[np.ndarray] = None
    for name in END2END_STAGES:
        result = restore(model, current, name, mask=mask)
        current = as_rgb(result.image)
        prompt = result.prompt
        # The mask describes the ORIGINAL page's geometry. Dewarping changes
        # that geometry, so carrying the mask into the later stages would
        # describe the wrong picture; they do not need one.
        mask = None
    return RestorationResult(image=current, prompt=prompt, stages=END2END_STAGES)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def load_page(path: Path) -> np.ndarray:
    """Read an image file as an ``(H, W, 3)`` uint8 RGB array.

    Decoding is PIL's, matching ``common._decode_triplet_numpy`` -- the staged
    corpus is full of TIFF, which ``tf.io.decode_image`` cannot read (D-028).

    :param path: The image file.
    :return: The page.
    """
    from PIL import Image

    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8)


def load_mask(path: Path) -> np.ndarray:
    """Read a document mask as an ``(H, W)`` uint8 array.

    :param path: The mask file.
    :return: The mask, non-zero inside the page.
    """
    from PIL import Image

    with Image.open(path) as handle:
        return np.asarray(handle.convert("L"), dtype=np.uint8)


def save_image(array: np.ndarray, path: Path) -> Path:
    """Write a uint8 array to ``path``, creating the parent directory.

    :param array: ``(H, W)`` or ``(H, W, 3)`` uint8.
    :param path: Destination.
    :return: ``path``.
    """
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)
    return path


def collect_input_images(target: Path) -> List[Path]:
    """The pages ``--input`` names: itself if a file, its images if a directory.

    :param target: A file or a directory.
    :return: Sorted list of image paths.
    :raises FileNotFoundError: If ``target`` does not exist, or is a directory
        holding no readable image.
    """
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(f"--input names nothing that exists: {target}")
    found = sorted(
        p for p in target.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not found:
        raise FileNotFoundError(
            f"--input directory {target} holds no image with a suffix in "
            f"{list(IMAGE_SUFFIXES)}"
        )
    return found


def restore_image_file(
        model: keras.Model,
        image_path: Path,
        output_dir: Path,
        task: str,
        mask: Optional[np.ndarray] = None,
        save_dtsprompt: bool = False,
) -> Path:
    """Read one page, restore it, write the result (and optionally its prompt).

    :param model: A DocRes model.
    :param image_path: The page to restore.
    :param output_dir: Directory the outputs are written into.
    :param task: A ``TASKS`` key or :data:`END2END`.
    :param mask: Optional document mask.
    :param save_dtsprompt: Whether to write the three prompt channels too.
    :return: The path of the restored image.
    """
    page = load_page(image_path)
    if task == END2END:
        result = restore_end2end(model, page, mask=mask)
    else:
        result = restore(model, page, task, mask=mask)

    out_path = output_dir / f"{image_path.stem}_{task}.png"
    save_image(result.image, out_path)
    logger.info(
        "%s -> %s (stages: %s, in %s -> out %s)",
        image_path.name, out_path, ", ".join(result.stages),
        page.shape, result.image.shape,
    )
    if save_dtsprompt and result.prompt is not None:
        channels = np.clip(
            np.rint(np.asarray(result.prompt, dtype=np.float64) * PIXEL_SCALE),
            0, 255,
        ).astype(np.uint8)
        for index in range(channels.shape[-1]):
            save_image(
                channels[..., index],
                output_dir / f"{image_path.stem}_{task}_prompt{index + 1}.png",
            )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the inference parser.

    Interface contract: pure. Constructs and returns a parser; parses nothing,
    reads no environment and touches no filesystem. The single parser
    :func:`parse_arguments` and the CLI guard both drive.

    :return: The parser.
    """
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description=(
            "Restore document images with a trained DocRes. Pads to a multiple "
            f"of {SPATIAL_DIVISOR} and crops the padding back off, then applies "
            "the task's post-processing from the TASKS table."
        ),
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="An image file, or a directory of images to restore.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Directory the restored images are written into.",
    )
    parser.add_argument(
        "--task", type=str, default="binarization",
        choices=list(task_names()) + [END2END],
        help=(
            "Which restoration to run. 'end2end' chains "
            + " -> ".join(END2END_STAGES) + "."
        ),
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help=(
            "Trained .keras checkpoint. WITHOUT one the model is freshly "
            "initialised and the output is noise; the run says so loudly."
        ),
    )
    parser.add_argument(
        "--model-variant", type=str, default="docres",
        help="DocRes variant key, used only when there is no checkpoint.",
    )
    parser.add_argument(
        "--mask", type=str, default=None,
        help=(
            "Document mask for dewarping (and for end2end's dewarping stage). "
            "Upstream gets it from the MBD network, which this port does not "
            "include, so it must be supplied."
        ),
    )
    parser.add_argument(
        "--save-dtsprompt", action="store_true",
        help="Also write the three prompt channels beside each output.",
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help=(
            "GPU index to use (e.g. 1). Omit to let the process see whatever "
            "CUDA_VISIBLE_DEVICES exposes."
        ),
    )
    return parser


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse ``argv`` with the real parser.

    :param argv: Tokens without the program name. ``None`` reads
        ``sys.argv[1:]``.
    :return: The namespace.
    :raises SystemExit: As argparse does, on ``--help`` or a bad flag.
    """
    return build_parser().parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse the CLI, set the process up, restore every input page.

    The statement ORDER is the contract the CLI guard's sentinels measure:

    1. parse -- so ``--help`` costs nothing and allocates nothing;
    2. collect the inputs -- so a bad ``--input`` fails before a GPU is
       claimed;
    3. ``setup_gpu`` -- process-level, from the non-config ``--gpu`` dest;
    4. load or construct the model, and restore each page.

    :param argv: Tokens without the program name. ``None`` reads
        ``sys.argv[1:]``.
    :raises FileNotFoundError: If ``--input`` or ``--checkpoint`` names nothing.
    """
    args = parse_arguments(argv)

    images = collect_input_images(Path(args.input))

    setup_gpu(gpu_id=args.gpu)

    model = load_docres_model(args.checkpoint, args.model_variant)
    mask = load_mask(Path(args.mask)) if args.mask else None
    output_dir = Path(args.output_dir)

    logger.info(
        "DocRes inference: task %r over %d image(s) -> %s",
        args.task, len(images), output_dir,
    )
    for image_path in images:
        restore_image_file(
            model, image_path, output_dir, args.task,
            mask=mask, save_dtsprompt=args.save_dtsprompt,
        )
    logger.info("DocRes inference complete; %d image(s) in %s", len(images), output_dir)


if __name__ == "__main__":
    main()
