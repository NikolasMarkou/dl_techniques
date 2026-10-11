"""TopoLM — a language model whose units are laid out on a 2D grid of tissue.

The package assembles a GPT-2-shaped causal decoder whose attention and
feed-forward branch outputs each carry a spatial smoothness penalty, so that
functional clusters emerge from the training objective rather than from a
post-hoc clustering of whatever the model happened to learn. Every unit of every
tapped tensor is assigned a distinct cell of an ``h x w`` grid with ``h * w ==
embed_dim``, and the penalty makes nearby units co-activate across the batch.

.. code-block:: python

    from dl_techniques.models.language.topolm import create_topolm

    topographic = create_topolm("paper")            # alpha = 2.5
    control = create_topolm("paper", alpha=0.0)     # identical weights, no loss
    ablation = create_topolm("paper", permute=False)  # the paper's Fig. 12

The ``alpha = 0`` arm is a genuine control, not the absence of one: its taps are
still created, still built, and still derive their layouts from the same seeds,
so it differs from the trained model in its objective alone.

Training goes through the shared CLM head, which must be told to fold the
backbone's own ``add_loss`` contribution into the scalar it backpropagates:

.. code-block:: python

    from dl_techniques.models.common.masked_language_model import CausalLanguageModel

    head = CausalLanguageModel(
        backbone=topographic,
        vocab_size=topographic.vocab_size,
        skip_head=True,
        output_key="logits",
        aggregate_backbone_losses=True,
    )

``aggregate_backbone_losses=True`` is load-bearing. Without it the smoothness term
is computed every step and thrown away, and the run looks healthy.

The three shipped variants differ in PROVENANCE, and that matters when quoting
any of them: ``paper`` is transcribed from Rathi et al. (2025) and is pinned
against its source, while ``small`` and ``tiny`` are repo-authored scales that
exist nowhere in the paper and are deliberately not pinned. See
:mod:`~dl_techniques.models.topolm.config`.

**No pretrained weights are distributed** — ``pretrained=True`` raises
``NotImplementedError`` at every entry point.

Importing this package registers ``TopoLM`` and ``TopoLMBlock`` for
deserialization; the taps register themselves from
``layers/regularization/spatial_smoothness.py``, which this package imports
transitively.

References:
    - Rathi, Mehrer, AlKhamissi, Binhuraib, Blauch & Schrimpf, 2025. TopoLM:
      Brain-like spatio-functional organization in a topographic language model.
      ICLR 2025. (https://arxiv.org/abs/2410.11516)
"""

from .components import TAP_SITES, TopoLMBlock
from .config import (
    DEFAULT_ALPHA,
    DEFAULT_NUM_NEIGHBORHOODS,
    DEFAULT_RADIUS,
    MODEL_VARIANTS,
    grid_shape_of,
    validate_variant,
)
from .model import TopoLM, create_topolm, create_topolm_with_head

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_NUM_NEIGHBORHOODS",
    "DEFAULT_RADIUS",
    "MODEL_VARIANTS",
    "TAP_SITES",
    "TopoLM",
    "TopoLMBlock",
    "create_topolm",
    "create_topolm_with_head",
    "grid_shape_of",
    "validate_variant",
]