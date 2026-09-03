"""DINO's published weight-initialization convention, in one place.

This module holds `DINO_KERNEL_INITIALIZER`, a config dict for
`trunc_normal_(std=.02)`, the initializer the reference DINO, DINOv2 and
DINOv3 implementations all use for every `nn.Linear`. `dino_v1.py` and
`dino_v3.py` previously spelled this four different ways across two files,
including a bare `"truncated_normal"` string, which names the right
distribution family but silently carries Keras' own default scale
(`stddev=0.05`, 2.5x wider than the reference).

This is a dict, not an `Initializer` instance, because a seedless
`RandomInitializer` resolves its seed once at construction and then replays
the identical draw on every call: a module-level instance used as a default
argument would hand every model built in the process the same weights.
`keras.initializers.get` resolves this dict to a fresh instance per consumer.

References:
    - https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
    - https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/vision_transformer.py
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
