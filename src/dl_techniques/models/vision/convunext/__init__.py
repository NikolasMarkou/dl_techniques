"""ConvUNext public API.

Re-exports the stem layer, the attention layer, the variant config table and the
two functional builders. Internal callers may still import from
``dl_techniques.models.vision.convunext.model`` directly.

Deliberately NOT re-exported: ``create_inference_model_from_training_model``.
``create_convunext`` and ``create_convunext_variant`` do take an
``enable_deep_supervision`` flag, so this package is a plausible place to reach
for that helper -- but the helper is model-agnostic (it slices any Keras
functional graph down to output 0), ``model.py`` never defined it, and it was
imported at the file tail purely so this line could pass it through. Nothing in
the repo ever imported it by this path. It now comes only from its canonical
home::

    from dl_techniques.utils.deep_supervision import (
        create_inference_model_from_training_model,
    )

Do not re-add a pass-through here.
"""

from .model import (
    ConvUNextStem,
    SpatialLinearAttention,
    CONVUNEXT_CONFIGS,
    create_convunext,
    create_convunext_variant,
)

__all__ = [
    'ConvUNextStem',
    'SpatialLinearAttention',
    'CONVUNEXT_CONFIGS',
    'create_convunext',
    'create_convunext_variant',
]
