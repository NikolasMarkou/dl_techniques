"""TopoLM — a language model whose units are laid out on a grid of tissue (2D; 3D is an extension).

The package assembles a GPT-2-shaped causal decoder whose attention and
feed-forward branch outputs each carry a spatial smoothness penalty, so that
functional clusters emerge from the training objective rather than from a
post-hoc clustering of whatever the model happened to learn. Every unit of every
tapped tensor is assigned a distinct cell of an ``h x w`` grid with ``h * w ==
embed_dim``, and the penalty makes nearby units co-activate across the batch.
With ``grid_dim=3`` the grid is an ``h x w x d`` box (``h <= w <= d``, ``h * w *
d == embed_dim``) and a neighbourhood is a cube. The paper is 2D; the 3D layout is
repo-authored. The trainer and the post-hoc analysis take both ranks; the Gaussian
readout stays 2D-only.

.. code-block:: python

    from dl_techniques.models.language.topolm import create_topolm

    topographic = create_topolm("paper")            # alpha = 2.5
    control = create_topolm("paper", alpha=0.0)     # identical weights, no loss
    ablation = create_topolm("paper", permute=False)  # the paper's Fig. 12

The ``alpha = 0`` arm is a genuine control, not the absence of one: its taps are
still created, still built, and still derive their layouts from the same seeds,
so it differs from the trained model in its objective alone.

The block's defaults follow the authors' released code: one feed-forward MLP with
no bias, LayerNorm without a beta, the exact GELU, residual projections scaled by
``1 / sqrt(2 * depth)``, and the taps on the branch outputs (``tap_placement=
'branch'``; ``'residual'`` taps the post-add tensor). ``use_bias``, ``norm_center``,
``activation``, ``tap_placement``, ``ffn_type`` and ``normalization_type`` are
serialized constructor options of ``TopoLM`` and ``from_variant``.

Training goes through the shared CLM head, which must be told to fold the
backbone's own ``add_loss`` contribution into the scalar it backpropagates:

.. code-block:: python

    from dl_techniques.models.common.lm_pretraining import CausalLanguageModel

    head = CausalLanguageModel(
        backbone=topographic,
        vocab_size=topographic.vocab_size,
        skip_head=True,
        output_key="logits",
        aggregate_backbone_losses=True,
    )

``aggregate_backbone_losses=True`` is load-bearing. Without it the smoothness term
is computed every step and thrown away, and the run looks healthy.

The five shipped variants differ in PROVENANCE, and that matters when quoting
any of them: ``paper`` is transcribed from Rathi et al. (2025) and is pinned
against its source, while ``small``, ``tiny``, ``paper3d`` (1000 units on a
10 x 10 x 10 box, radius 4) and ``tiny3d`` (343 units on a 7 x 7 x 7 box, radius 2)
are repo-authored, exist nowhere in the paper and are deliberately not pinned. The
two 3D rows carry their own ``grid_dim`` and ``radius`` keys. See
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
from .model import TopoLM, create_topolm

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_NUM_NEIGHBORHOODS",
    "DEFAULT_RADIUS",
    "MODEL_VARIANTS",
    "TAP_SITES",
    "TopoLM",
    "TopoLMBlock",
    "create_topolm",
    "grid_shape_of",
    "validate_variant",
]