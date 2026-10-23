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

from typing import Any, Dict, Optional, Tuple

#: Every variant must satisfy ``embed_dim >= (2 * radius + 1) ** grid_dim`` and
#: have every axis of its resolved grid ``>= 2 * radius + 1``, or the
#: neighbourhood cannot fit on its grid, and ``embed_dim % num_heads == 0``.
#: :func:`validate_variant` enforces it for the shipped rows; the model's
#: constructor enforces the same axis rule (3D) through :func:`axes_fit_problem`.

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


def grid_shape_of(embed_dim: int, grid_dim: int = 2) -> Tuple[int, ...]:
    """The unit grid a given width sits on.

    Delegates to the layer's own resolver so there is exactly one factoring rule
    in the tree: a model variant and the tap it builds cannot disagree about what
    grid a width implies, and a disagreement here would surface as a tap that
    raises at build time deep inside the model.

    :param embed_dim: Model width, i.e. the number of units on the grid.
    :type embed_dim: int
    :param grid_dim: Number of grid axes, 2 (the default, the paper's layout) or 3.
    :type grid_dim: int
    :return: The grid, ``(height, width)`` for 2 and a three-axis tuple for 3.
    :rtype: Tuple[int, ...]
    """
    from dl_techniques.layers.regularization.spatial_smoothness import (
        resolve_grid_shape,
    )

    return resolve_grid_shape(embed_dim, grid_dim=grid_dim)


def radius_advice(grid: Tuple[int, ...]) -> str:
    """Name the largest radius a grid can host, or say that none can.

    A patch of side ``2 * radius + 1`` has to fit along the shortest axis, so the
    answer is ``(min(grid) - 1) // 2`` and it is at least 1 only when that axis
    has 3 or more cells. Shared by :func:`validate_variant` and the model's own
    construction check, so the two cannot word the advice differently.

    :param grid: A resolved unit grid, any number of axes.
    :type grid: Tuple[int, ...]
    :return: One sentence, ending in a period.
    :rtype: str
    """
    min_axis = min(grid)
    if min_axis >= 3:
        return f"The largest radius this grid can use is {(min_axis - 1) // 2}."
    return (
        "No radius can work on this grid (the shortest axis is below 3); "
        "pick a different width."
    )


def axes_fit_problem(grid: Tuple[int, ...], radius: int) -> Optional[str]:
    """Say why a radius-``radius`` patch does not fit on ``grid``, or ``None``.

    The patch has side ``2 * radius + 1`` and must fit along EVERY axis, which is
    exactly the condition ``build_patch_tables`` enforces. The sentence names the
    grid, each short axis with its length, and the largest usable radius
    (:func:`radius_advice`). It starts with a verb so a caller can put its own
    subject in front: ``validate_variant`` prefixes the variant and the width,
    the model's constructor only the width.

    :param grid: A resolved unit grid, any number of axes.
    :type grid: Tuple[int, ...]
    :param radius: The neighbourhood radius.
    :type radius: int
    :return: The sentence, or ``None`` when every axis is at least the side.
    :rtype: Optional[str]
    """
    side = 2 * int(radius) + 1
    short = [(k, n) for k, n in enumerate(grid) if n < side]
    if not short:
        return None
    offenders = "; ".join(f"axis {k} has {n}" for k, n in short)
    return (
        f"resolves to the {'x'.join(str(n) for n in grid)} grid, but a "
        f"radius-{radius} neighbourhood needs every axis >= {side}: "
        f"{offenders}. {radius_advice(grid)} Lower the radius, or pick a wider "
        f"variant."
    )


def validate_variant(
    name: str, config: Dict[str, Any], radius: int, grid_dim: int = 2
) -> None:
    """Check one variant against the constraints a tap imposes.

    Called for every shipped preset at class construction, not just the one a
    caller happens to ask for: a variant table with one entry sitting on a
    degenerate boundary is the shape of defect that ships and is never exercised.

    Two checks bound the neighbourhood. The floor ``(2 * radius + 1) ** grid_dim``
    on the total unit count is necessary but not sufficient: a patch has to fit
    along EVERY axis of the resolved grid, so the grid is resolved and each axis
    is compared against ``2 * radius + 1``. That axis test is exactly the
    condition :func:`~dl_techniques.layers.regularization.spatial_smoothness.build_patch_tables`
    enforces, so a configuration that builds is never rejected here.

    :param name: Variant name, for the error message.
    :type name: str
    :param config: The variant's parameter dict.
    :type config: Dict[str, Any]
    :param radius: The neighbourhood radius the model will build taps with.
    :type radius: int
    :param grid_dim: Number of grid axes, 2 or 3.
    :type grid_dim: int
    :raises ValueError: If the width does not divide among the heads, cannot fill
        a neighbourhood, or resolves to a grid with an axis shorter than the
        patch side -- naming the variant and the offending value in each case.
    """
    embed_dim = int(config["embed_dim"])
    num_heads = int(config["num_heads"])
    side = 2 * int(radius) + 1
    min_units = side ** grid_dim

    if embed_dim % num_heads != 0:
        raise ValueError(
            f"variant {name!r}: embed_dim ({embed_dim}) must be divisible by "
            f"num_heads ({num_heads})"
        )
    if embed_dim < min_units:
        # The 2D wording is pinned verbatim; only the 3D message carries the
        # radius advice, because a cube below the default radius's floor is the
        # common 3D mistake and the floor is the check that catches it first.
        hint = (
            f" {radius_advice(grid_shape_of(embed_dim, grid_dim))}"
            if grid_dim != 2
            else ""
        )
        raise ValueError(
            f"variant {name!r}: embed_dim ({embed_dim}) is below the "
            f"{'x'.join([str(side)] * grid_dim)} = {min_units} units a "
            f"radius-{radius} neighbourhood needs. Lower the radius, or pick a "
            f"wider variant.{hint}"
        )

    grid = grid_shape_of(embed_dim, grid_dim)
    # DECISION plan-2026-10-23T141429-390f3815/D-007: the unit count alone is not
    # enough. Do NOT validate only the total (the floor above): 784 units at
    # radius 4 clear the 3D floor of 729 yet resolve to (7, 8, 14), whose first
    # axis cannot hold a 9-wide patch, so the error would fire deep inside the tap
    # build. The same hole exists in 2D (997 = 1 x 997 clears the floor and dies at
    # tap build); for grid_dim=2 this check only moves WHERE an already
    # unbuildable configuration fails, never what builds.
    problem = axes_fit_problem(grid, radius)
    if problem is not None:
        raise ValueError(f"variant {name!r}: embed_dim ({embed_dim}) {problem}")


def validate_all_variants(radius: int = DEFAULT_RADIUS) -> None:
    """Sweep every shipped preset against :func:`validate_variant`.

    Run at import of the model module so a table edit that breaks a variant fails
    on the import that introduced it, not on whichever training run happens to
    select that row first.

    A row may carry its own ``"radius"`` and ``"grid_dim"``; each is judged by its
    own values and falls back to ``radius`` and 2 when absent, so the rows that
    predate the keys behave exactly as before.

    :param radius: Radius used for every row that does not carry its own.
    :type radius: int
    :raises ValueError: From :func:`validate_variant`, for the first bad row.
    """
    for name, config in MODEL_VARIANTS.items():
        validate_variant(
            name,
            config,
            int(config.get("radius", radius)),
            int(config.get("grid_dim", 2)),
        )
