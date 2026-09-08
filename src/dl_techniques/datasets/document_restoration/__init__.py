"""Document-restoration data utilities for the DocRes port.

DocRes conditions one Restormer backbone on five restoration tasks purely by
concatenating three classical-CV "prompt" channels onto the RGB input. This
subpackage owns both halves of that scheme:

* :mod:`~dl_techniques.datasets.document_restoration.dtsprompt` — the five
  prompt generators and the ONE shared background-estimation primitive they
  are built from, implemented on **numpy + scipy only** (no ``cv2``, no
  ``skimage``: neither is a declared dependency of this library).
* :mod:`~dl_techniques.datasets.document_restoration.tasks` — the single
  ``TASKS`` table binding each task name to its prompt generator, supervised
  output-channel count, loss name and post-processing mode. Nothing else in
  the tree branches on a task string.

Public surface:
    * :func:`estimate_background` — dilate/median/absdiff/normalise; returns
      both the background estimate and the normalised page.
    * :func:`deshadow_prompt`, :func:`appearance_prompt`, :func:`deblur_prompt`,
      :func:`binarization_prompt`, :func:`dewarp_prompt` — the generators.
    * :func:`sauvola_mod_binarization` — the two-pass Sauvola.
    * :func:`base_coordinate_grid`, :func:`apply_document_mask` — dewarping
      helpers (the dewarping mask is a caller input; the MBD network that
      produces it upstream is not part of this port).
    * :class:`TaskSpec`, :data:`TASKS`, :func:`get_task`, :func:`task_names`.
    * :data:`N_OUTPUT_CHANNELS`, :data:`N_PROMPT_CHANNELS`, and the ``LOSS_*``
      / ``POSTPROCESS_*`` name constants.
"""

from .dtsprompt import (
    BACKGROUND_WORKING_SIZE,
    DILATE_KERNEL_SIZE,
    MEDIAN_KERNEL_SIZE,
    SAUVOLA_BINARY_THRESHOLD,
    appearance_prompt,
    apply_document_mask,
    base_coordinate_grid,
    binarization_prompt,
    deblur_prompt,
    deshadow_prompt,
    dewarp_prompt,
    estimate_background,
    sauvola_mod_binarization,
)
from .tasks import (
    LOSS_CATEGORICAL_CROSSENTROPY,
    LOSS_L1,
    LOSSES,
    N_OUTPUT_CHANNELS,
    N_PROMPT_CHANNELS,
    POSTPROCESS_ARGMAX_BINARY,
    POSTPROCESS_CLAMP_IMAGE,
    POSTPROCESS_FLOW_REMAP,
    POSTPROCESS_MODES,
    TASKS,
    TaskSpec,
    get_task,
    task_names,
)

__all__ = [
    "BACKGROUND_WORKING_SIZE",
    "DILATE_KERNEL_SIZE",
    "MEDIAN_KERNEL_SIZE",
    "SAUVOLA_BINARY_THRESHOLD",
    "LOSSES",
    "LOSS_CATEGORICAL_CROSSENTROPY",
    "LOSS_L1",
    "N_OUTPUT_CHANNELS",
    "N_PROMPT_CHANNELS",
    "POSTPROCESS_ARGMAX_BINARY",
    "POSTPROCESS_CLAMP_IMAGE",
    "POSTPROCESS_FLOW_REMAP",
    "POSTPROCESS_MODES",
    "TASKS",
    "TaskSpec",
    "appearance_prompt",
    "apply_document_mask",
    "base_coordinate_grid",
    "binarization_prompt",
    "deblur_prompt",
    "deshadow_prompt",
    "dewarp_prompt",
    "estimate_background",
    "get_task",
    "sauvola_mod_binarization",
    "task_names",
]
