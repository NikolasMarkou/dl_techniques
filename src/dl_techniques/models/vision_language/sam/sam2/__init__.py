"""
SAM 2 (Segment Anything in Images and Videos): the image and streaming paths.

A Hiera trunk with an FPN neck, a streaming memory (memory attention, memory
encoder, memory bank), and a mask decoder that also emits an object score and
an object pointer. No pretrained weights ship here and no released SAM 2
checkpoint has been loaded in this repository, so this package makes no
accuracy claim. ``SAM2.call`` is the traceable image path and
``SAM2.stream_step`` the streaming video path; ``SAM2TrainingModel``, in the
``training_model`` submodule, is the traceable multi-frame wrapper ``fit()``
can train.

This ``__init__`` imports ``memory_bank`` and ``model`` only, never
``training_model``, so importing this package does not register
``SAM2TrainingModel`` under either its current or legacy key. Loading a
``.keras`` file whose top-level class is ``SAM2TrainingModel`` needs an
explicit ``import dl_techniques.models.vision_language.sam.sam2.training_model``
first. ``README.md`` section 7 has what training is measured to do (the
trainer memorises one batch; before plan-2026-10-14T062616-81938e81 D-260 the
mask head did not learn, because the encoder was fed unstandardized pixels).

References:
    - Ravi et al., 2024. SAM 2: Segment Anything in Images and Videos.

Example:

.. code-block:: python

    from dl_techniques.models.vision_language.sam.sam2 import SAM2, SAM2MemoryBank, create_sam2
    model = create_sam2("tiny")
    outputs = model({"image": images, "points": (coords, labels)})
"""

from .memory_bank import SAM2MemoryBank
from .model import SAM2, create_sam2

__all__ = [
    'SAM2',
    'SAM2MemoryBank',
    'create_sam2',
]
