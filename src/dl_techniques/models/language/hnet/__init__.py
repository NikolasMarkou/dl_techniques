"""H-Net: byte-level language modelling with hierarchical dynamic chunking.

The package assembles the three chunking layers of
:mod:`dl_techniques.layers.dynamic_chunking` into the recursive encoder / chunk /
inner-network / dechunk / decoder hierarchy of Hwang et al. (2025), with a byte
embedding at the bottom and a 256-way language-modelling head at the top. There is
no tokenizer anywhere in it: the input is raw bytes and the chunk boundaries are
learned.

.. code-block:: python

    from dl_techniques.models.language.hnet import create_hnet

    model = create_hnet("hnet_1stage_L", max_seq_len=1024)

Six named variants ship, each transcribed from the reference repository's
``configs/*.json``; :data:`~dl_techniques.models.language.hnet.config.MODEL_VARIANTS`
is their one home and :meth:`HNet.from_variant` is the one place a name becomes a
model. **No pretrained weights are distributed** -- ``pretrained=True`` raises
``NotImplementedError`` at every entry point.

The port diverges from the reference in six recorded ways, all of them stated in
``README.md`` beside the consequence each one carries. Read that file before
comparing anything here against the paper or the upstream code.

Importing this package is what registers ``HNet``, ``HNetStage``, ``HNetBlock`` and
``HNetIsotropic`` for deserialization; the chunking layers register themselves from
``layers/dynamic_chunking/``, which this package imports transitively.

References:
    - Hwang et al., 2025. Dynamic Chunking for End-to-End Hierarchical Sequence
      Modeling. (https://arxiv.org/abs/2507.07955)
    - Reference implementation: https://github.com/goombalab/hnet
"""

from .config import MODEL_VARIANTS, HNetArchConfig, get_variant_config
from .model import HNet, create_hnet

__all__ = [
    "MODEL_VARIANTS",
    "HNet",
    "HNetArchConfig",
    "create_hnet",
    "get_variant_config",
]
