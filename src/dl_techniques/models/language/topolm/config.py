"""Variant registry and topographic hyperparameters for TopoLM.

Provenance of every row below, because a variant table is quoted data whenever it
claims to be somebody else's architecture:

* ``paper`` is transcribed from Rathi et al. (2025), Section 3, and is pinned by
  ``tests/test_variant_tables_match_upstream_references.py``. 784 units is not a
  free choice -- it is the only plausible hidden size near GPT-2 small that is a
  product of two equal factors, 28 x 28, so the unit grid is square.
* ``small`` and ``tiny`` are **repo-authored**, not upstream. They are the paper's
  shape scaled down for smoke runs and CI, and they exist nowhere in the paper.
  They are deliberately NOT pinned by the upstream guard: quoting a derived
  number as if it were published is exactly the false citation that guard exists
  to prevent.

References:
    - Rathi, Mehrer, AlKhamissi, Binhuraib, Blauch & Schrimpf, 2025. TopoLM.
      ICLR 2025. (https://arxiv.org/abs/2410.11516)
"""

from typing import Any, Dict, Tuple

#: Every variant must satisfy ``embed_dim >= (2 * radius + 1) ** 2`` or the
#: neighbourhood cannot fit on its grid, and ``embed_dim % num_heads == 0``.
#: :func:`validate_variant` is the single place that is enforced.

#: Weight on the added spatial loss, per tap. The paper's value after a
#: hyperparameter search; lower values under-develop topography and higher values
#: cost task performance.
DEFAULT_ALPHA = 2.5

#: Neighbourhoods sampled per step. The paper averages over five.
DEFAULT_NUM_NEIGHBORHOODS = 5

#: Radius the paper trains with: an 11x11 patch of 121 units.
DEFAULT_RADIUS = 5


MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
    # Quoted from Rathi et al. (2025), Section 3. d_model 784 = 28 x 28,
    # 12 blocks, 16 heads (head dim 49), MLP width 3136, context 1024.
    "paper": {
        "embed_dim": 784,
        "depth": 12,
        "num_heads": 16,
        "ffn_intermediate_size": 3136,
        "max_seq_len": 1024,
        "description": "TopoLM (Rathi et al. 2025, Sec. 3): 28x28 unit grid",
    },
    # Repo-authored: the paper's shape at roughly a quarter of the width and
    # depth, for smoke runs. 512 = 16 x 32, a deliberately NON-SQUARE grid, so the
    # orientation probes have a transposed stride to miss.
    "small": {
        "embed_dim": 512,
        "depth": 6,
        "num_heads": 8,
        "ffn_intermediate_size": 2048,
        "max_seq_len": 512,
        "description": "Repo-authored scale of the paper shape, 16x32 grid",
    },
    # Repo-authored: smallest configuration whose grid still holds a radius-5
    # neighbourhood. 256 = 16 x 16, comfortably above the 121-unit floor.
    "tiny": {
        "embed_dim": 256,
        "depth": 4,
        "num_heads": 4,
        "ffn_intermediate_size": 1024,
        "max_seq_len": 256,
        "description": "Repo-authored smoke scale, 16x16 grid",
    },
}


def grid_shape_of(embed_dim: int) -> Tuple[int, int]:
    """The unit grid a given width sits on.

    Delegates to the layer's own resolver so there is exactly one factoring rule
    in the tree: a model variant and the tap it builds cannot disagree about what
    grid a width implies, and a disagreement here would surface as a tap that
    raises at build time deep inside the model.
    """
    from dl_techniques.layers.regularization.spatial_smoothness import (
        resolve_grid_shape,
    )

    return resolve_grid_shape(embed_dim)


def validate_variant(name: str, config: Dict[str, Any], radius: int) -> None:
    """Check one variant against the constraints a tap imposes.

    Called for every shipped preset at class construction, not just the one a
    caller happens to ask for: a variant table with one entry sitting on a
    degenerate boundary is the shape of defect that ships and is never exercised.

    :param name: Variant name, for the error message.
    :type name: str
    :param config: The variant's parameter dict.
    :type config: Dict[str, Any]
    :param radius: The neighbourhood radius the model will build taps with.
    :type radius: int
    :raises ValueError: If the width cannot host the grid, cannot fill a
        neighbourhood, or does not divide among the heads -- naming the variant
        and the offending value in each case.
    """
    embed_dim = int(config["embed_dim"])
    num_heads = int(config["num_heads"])
    min_units = (2 * int(radius) + 1) ** 2

    if embed_dim % num_heads != 0:
        raise ValueError(
            f"variant {name!r}: embed_dim ({embed_dim}) must be divisible by "
            f"num_heads ({num_heads})"
        )
    if embed_dim < min_units:
        raise ValueError(
            f"variant {name!r}: embed_dim ({embed_dim}) is below the "
            f"{(2 * radius + 1)}x{(2 * radius + 1)} = {min_units} units a "
            f"radius-{radius} neighbourhood needs. Lower the radius, or pick a "
            f"wider variant."
        )


def validate_all_variants(radius: int = DEFAULT_RADIUS) -> None:
    """Sweep every shipped preset against :func:`validate_variant`.

    Run at import of the model module so a table edit that breaks a variant fails
    on the import that introduced it, not on whichever training run happens to
    select that row first.
    """
    for name, config in MODEL_VARIANTS.items():
        validate_variant(name, config, radius)