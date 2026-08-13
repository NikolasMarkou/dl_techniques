"""FastViT / MobileCLIP2 (MCi) backbone primitives.

Channels-last Keras 3 transcriptions of the blocks that make up timm's FastViT
MCi image tower, the backbone of Apple's MobileCLIP / MobileCLIP2 image encoder.

The classes here are deliberately prefixed ``FastVit*`` where their reference
name is generic (``ConvMlp``, ``PatchEmbed``, ``Stage``, ...): the repo's
serialization registry is keyed by bare class name, so a generic name silently
shadows — or is shadowed by — an unrelated class depending on import order.
Names that are already distinctive in the reference (``ReparamLargeKernelConv``,
``RepConditionalPosEnc``) are kept unprefixed.

.. note::
   ``FastVitRepMixerBlock`` is NOT the same class as the pre-existing standalone
   ``dl_techniques.layers.repmixer_block.RepMixerBlock``. The two are different
   architectures that happen to share a name; the standalone one is consumed by
   ``models/fastvlm/`` and is left untouched.
"""

from .conv_mlp import FastVitConvMlp
from .patch_embed import FastVitPatchEmbed
from .rep_conditional_pos_enc import RepConditionalPosEnc
from .rep_mixer import FastVitRepMixer, FastVitRepMixerBlock
from .reparam_large_kernel_conv import ReparamLargeKernelConv

__all__ = [
    "FastVitConvMlp",
    "FastVitPatchEmbed",
    "RepConditionalPosEnc",
    "FastVitRepMixer",
    "FastVitRepMixerBlock",
    "ReparamLargeKernelConv",
]
