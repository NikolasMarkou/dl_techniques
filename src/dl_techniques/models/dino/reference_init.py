"""
DINO's published weight-initialization convention, in one place.

Reference (fetched):
    https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
    The same ``std=.02`` ViT/``Mlp`` convention carries through ``dinov2`` and
    ``dinov3``:
    https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/vision_transformer.py

Why this module exists
----------------------
``dino_v1.py`` and ``dino_v3.py` previously spelled this **four** ways across two
files: a bare ``"truncated_normal"`` string (twice) and ``"glorot_uniform"``
(once), plus a second bare string on a classifier head. The bare string is the
subtle one -- it names the right *distribution family* while silently carrying
Keras' own default scale:

    keras/src/initializers/random_initializers.py -- TruncatedNormal(mean=0.0, stddev=0.05)

which is **2.5x wider** than the reference. A string that looks correct and is
not is exactly the kind of value that must have one home rather than four.

Why a dict rather than an ``Initializer`` instance
--------------------------------------------------
A seedless ``RandomInitializer`` resolves ``seed=None`` to a concrete seed **at
construction time** (``random_initializers.py:12-14``), so one instance REPLAYS
the identical draw on every call. MEASURED: two calls of a single
``TruncatedNormal(stddev=0.02)`` instance at the same shape differ by exactly
``0.0``; two instances resolved from this dict differ by ``6.1e-02``. A module
constant that is an instance, used as a default argument (evaluated once at
import), would therefore hand every model built in the process the same weights.
``keras.initializers.get`` resolves an inert dict to a fresh instance per
consumer. Same hazard as D-072 / D-481.
"""

from typing import Any, Dict

# ---------------------------------------------------------------------

#: DINO's published initializer: ``trunc_normal_(std=.02)``.
#: Realized (post-truncation) std is ``0.02 * 0.87964 ~= 0.0176`` -- assert the
#: realized figure, never the nominal 0.02.
DINO_KERNEL_INITIALIZER: Dict[str, Any] = {
    "class_name": "TruncatedNormal",
    "config": {"stddev": 0.02},
}

#: The published nominal std, for guards that want the reference number itself.
DINO_INITIALIZER_STDDEV: float = 0.02

# ---------------------------------------------------------------------
