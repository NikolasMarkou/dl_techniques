"""SAM 2 — promptable image and video segmentation.

This package implements the SAM 2.1 architecture: a Hiera trunk with an FPN
neck, a streaming memory (memory attention, memory encoder, memory bank) and a
mask decoder that additionally emits an object score and an object pointer.

**The exported surface is deliberately minimal**, mirroring
``dl_techniques.models.SAM.SAM1``'s: the model, its factory, and the one plain-Python
state container a caller must construct itself to drive the streaming API.
Everything else is an implementation component and is imported from its own
submodule::

    from dl_techniques.models.sam2 import SAM2, create_sam2, SAM2MemoryBank
    from dl_techniques.models.sam2.hiera import Hiera
    from dl_techniques.models.sam2.neck import SAM2ImageEncoder
    from dl_techniques.models.sam2.memory_attention import SAM2MemoryAttention
    from dl_techniques.models.sam2.memory_encoder import SAM2MemoryEncoder
    from dl_techniques.models.sam2.mask_decoder import SAM2MaskDecoder

Widening ``__all__`` is a deliberate act, not a convenience: it is pinned by
``tests/test_models/test_sam2/test_package_surface.py``, which asserts the exact
contents so the surface cannot drift open one re-export at a time.

Example::

    model = create_sam2('tiny')
    outputs = model({'image': images, 'points': (coords, labels)})
"""

from .memory_bank import SAM2MemoryBank
from .model import SAM2, create_sam2

__all__ = [
    'SAM2',
    'SAM2MemoryBank',
    'create_sam2',
]
