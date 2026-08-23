"""Ceiling gates on the shared-initializer-INSTANCE collision population.

Why this file exists
====================

A single ``keras.initializers.Initializer`` **instance** handed to two weights of
the same shape produces **bit-identical** tensors. MEASURED here 2026-08-22, on
two ``Dense(4)`` built from ``(None, 6)`` after ``keras.utils.set_random_seed``:

======================================================  ==================
how the initializer is passed                           ``max|delta|``
======================================================  ==================
one shared instance, **no** ``seed=``                   **0.0**
one shared instance, **explicit** ``seed=42``           **0.0**
two *separate* ``GlorotUniform(seed=42)`` instances     **0.0**
======================================================  ==================

So ``seed=`` is **NOT** the discriminator -- **instance identity is**. Any belief
that "only explicitly-seeded sharing is risky" is refuted by row 1. See
``decisions.md`` D-200 and ``src/dl_techniques/initializers/clone.py``.

Whether a collision is a *defect* is per-site (D-057, prior plan
``plan-2026-08-19T163559-499b6f0e``): identical weights that play the **same**
architectural role are harmless and sometimes wanted; identical weights whose
**difference is the architecture** (query vs key, a positive vs a negative
branch, a mu head vs a sigma head) are a training pathology. That judgement
cannot be automated, which is why this file pins **population ceilings** instead
of asserting a per-site verdict.

What this static census CANNOT see
==================================

The sweep below groups by ``(file, function scope, initializer expression)`` and
keeps groups with **2+ syntactic call sites**. A ``for`` loop that hands ONE
instance to N blocks is a single syntactic site, so it scores 1 and is invisible
here -- no matter how many weights it collides. MEASURED 2026-08-23: 96 such
loop-carried sites in 40 files, including the three this step fixed. ``vit`` was
visible only through its three *unrolled* ``__init__`` sites; ``swin_transformer``
and ``dino_v3`` never appeared in this census at all, yet a seeded build of each
showed 16 and 6 bit-identical same-shape kernel pairs respectively.

That is why ``TestPortedViTsDoNotShareOneInitializerInstance`` below is a
**runtime** guard: it reads the built weights, which is the only instrument that
sees a loop. Do NOT treat the ceilings above as coverage.

The remedy, already shipped and DRY, is
``dl_techniques.initializers.clone.clone_initializer`` -- do not write a second
one.
"""

import ast
import os
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pytest

SRC_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
    "dl_techniques",
)

#: Ceiling on *collision groups*: one bound name or ``self.<x>_initializer``
#: attribute passed to 2+ weight-creating calls inside a single function scope.
#: MEASURED at HEAD 2026-08-23 after this plan's step 9: **157** groups / 533
#: participating call sites / 99 files, down from 159/539/100 (which was itself
#: down from 175/609/108 before step 19.1's eight fixes). Step 9 cloned
#: ``models/vit/model.py``'s two ``__init__`` groups away; ``swin_transformer``
#: and ``dino_v3`` were fixed in the same step but never appeared here at all --
#: see the loop blind spot noted below.
#:
#: The carried census figure of "378 sites / 118 files" does **not** reproduce
#: under this instrument, nor under either of the two instruments D-144 already
#: tried, nor under the 178/565/102 variant recorded in
#: ``findings/initializer-and-norm-census.md``. Do not re-quote 378/118.
_COLLISION_GROUP_CEILING = 157

#: Ceiling on the *defect-shaped* subset: a collision group in which 2+ of the
#: colliding calls have the SAME callee AND the SAME size expression (so their
#: weights have the same shape, so they really are bit-identical) but are
#: assigned to DIFFERENT attributes (so they plausibly play different roles).
#: This is the triage list. MEASURED at HEAD 2026-08-23: **57**, down from **79**
#: before step 19.1 (the eight fixes removed 22 of them). Step 9's three fixes
#: did NOT move this number: none of the six colliding groups it removed was
#: same-callee-same-size-different-target, which is precisely the blind spot the
#: runtime guard below exists to cover.
#: Every new entry needs a per-site role probe before it is called benign.
_ROLE_SUSPECT_CEILING = 57

#: Sites fixed by step 19.1 with ``clone_initializer``, each one *measured*
#: bit-identical before and *measured* non-identical after. Named so a
#: regression that unwraps one of them is loud.
_FIXED_MODULES = (
    "layers/attention/ring_attention.py",
    "layers/attention/linear_attention.py",
    "layers/attention/hopfield_attention.py",
    "layers/attention/capsule_routing_attention.py",
    "layers/attention/gated_attention.py",
    "layers/ffn/diff_ffn.py",
    "layers/statistics/mdn_layer.py",
    "layers/ffn/residual_block.py",
)


def _expr_key(node: ast.AST) -> Optional[str]:
    """Return a dotted source name for a ``Name``/``Attribute``, else ``None``.

    A ``Call`` value -- notably ``clone_initializer(...)`` -- deliberately
    returns ``None``: a cloned site creates an independent instance and so
    cannot collide, which is exactly why wrapping a site removes it from the
    population this file counts.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _expr_key(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _callee(call: ast.Call) -> str:
    return getattr(call.func, "attr", getattr(call.func, "id", "?"))


def _sweep() -> Dict[Tuple[str, str, str], List[dict]]:
    """Return ``{(relpath, scope, initializer_expr): [site, ...]}`` for groups >= 2."""
    groups: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for dirpath, dirnames, filenames in os.walk(SRC_ROOT):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            rel = os.path.relpath(path, SRC_ROOT)
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            # No `except SyntaxError: continue` here -- a silent skip is a
            # silent shrink of every number below (D-070).
            tree = ast.parse(source, filename=path)

            assigned: Dict[int, str] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                    assigned[id(node.value)] = _expr_key(node.targets[0]) or "?"

            for scope in ast.walk(tree):
                if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for call in ast.walk(scope):
                    if not isinstance(call, ast.Call):
                        continue
                    size = None
                    for kw in call.keywords:
                        if kw.arg in ("units", "filters", "output_dim", "input_dim"):
                            size = f"{kw.arg}={ast.unparse(kw.value)}"
                    if size is None and call.args:
                        size = f"pos0={ast.unparse(call.args[0])}"
                    for kw in call.keywords:
                        if kw.arg is None or not kw.arg.endswith("_initializer"):
                            continue
                        key = _expr_key(kw.value)
                        if key is None:
                            continue
                        groups[(rel, scope.name, key)].append(
                            {
                                "line": kw.value.lineno,
                                "kwarg": kw.arg,
                                "callee": _callee(call),
                                "target": assigned.get(id(call), "?"),
                                "size": size,
                            }
                        )
    return {k: v for k, v in groups.items() if len(v) >= 2}


def _role_suspects(groups: Dict[Tuple[str, str, str], List[dict]]) -> List[str]:
    suspects: List[str] = []
    for (rel, scope, expr), sites in sorted(groups.items()):
        by_shape: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        for site in sites:
            by_shape[(site["callee"], str(site["size"]))].append(site)
        for (callee, size), matched in sorted(by_shape.items()):
            targets = {site["target"] for site in matched}
            if len(matched) >= 2 and len(targets) >= 2:
                suspects.append(
                    f"{rel}::{scope}::{expr} -> {callee}({size}) "
                    f"reaches {sorted(targets)}"
                )
    return suspects


@pytest.fixture(scope="module")
def collision_groups() -> Dict[Tuple[str, str, str], List[dict]]:
    return _sweep()


class TestSharedInitializerPopulationIsPinned:
    """Growth, not silent fallback, is the residual risk this family carries."""

    def test_the_collision_group_population_has_not_grown(self, collision_groups):
        assert len(collision_groups) <= _COLLISION_GROUP_CEILING, (
            "shared-initializer collision groups grew from "
            f"{_COLLISION_GROUP_CEILING} to {len(collision_groups)}. A new group "
            "means one initializer INSTANCE now reaches 2+ weights in one scope, "
            "which is bit-identical whenever their shapes match (seed= is NOT "
            "the discriminator). Probe the new site's roles; if they differ, wrap "
            "it in dl_techniques.initializers.clone.clone_initializer. Do NOT "
            "raise this ceiling to make a red test green. New groups: "
            f"{sorted(set(collision_groups) )[:20]}"
        )

    # R-038 closure -- plan-2026-08-22T035419-a11304c8 / D-303. Same shape as
    # the norm-factory census guard's opt-out (D-251): this test REPORTS the
    # 57-entry triage list through `warnings.warn` on purpose, so that the
    # residue D-200 closed by documented default is re-stated on every run and
    # cannot rot silently. Step 5 later landed a repo-wide `error::UserWarning`
    # (D-252), which turned that deliberate report into a failure -- this ONE
    # test opts back out. Do NOT silence the warning itself: the report IS the
    # instrument that keeps the 57 visible.
    @pytest.mark.filterwarnings("always::UserWarning")
    def test_the_role_suspect_subset_has_not_grown(self, collision_groups):
        suspects = _role_suspects(collision_groups)
        if suspects:
            warnings.warn(
                f"same-shape different-target initializer collisions "
                f"({len(suspects)}), each needing a per-site role probe: "
                f"{suspects}",
                UserWarning,
                stacklevel=2,
            )
        assert len(suspects) <= _ROLE_SUSPECT_CEILING, (
            "the defect-shaped subset of initializer collisions grew from "
            f"{_ROLE_SUSPECT_CEILING} to {len(suspects)}. Each of these is a "
            "single instance reaching 2+ SAME-SHAPE weights assigned to "
            "DIFFERENT attributes -- the exact signature of the eight defects "
            "step 19.1 measured and fixed (q/k/v projections built equal, an "
            "MDN mu head equal to its sigma head, a positive branch equal to "
            "its negative branch). Probe the new one and either clone it or "
            "record why its two roles are genuinely the same role. "
            f"Suspects: {suspects}"
        )

    @pytest.mark.parametrize("module", _FIXED_MODULES)
    def test_a_fixed_module_still_clones(self, module):
        path = os.path.join(SRC_ROOT, module)
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        assert "clone_initializer(" in source, (
            f"{module} no longer calls clone_initializer. Step 19.1 MEASURED "
            "bit-identical weights (max|delta| = 0.0) between weights of "
            "different architectural roles in this module and fixed them by "
            "cloning the initializer per site. Unwrapping restores the defect."
        )


# ---------------------------------------------------------------------------
# Runtime guard -- the instrument that sees a loop (D-540)
# ---------------------------------------------------------------------------

import itertools

import numpy as np


def _tiny_vit():
    import keras

    from dl_techniques.models.vit.model import ViT

    keras.utils.set_random_seed(1234)
    model = ViT(
        input_shape=(32, 32, 3), num_classes=10, scale="tiny", patch_size=8,
        include_top=True, name="vit",
    )
    model.build((None, 32, 32, 3))
    return model


def _tiny_swin():
    import keras

    from dl_techniques.models.swin_transformer.model import SwinTransformer

    keras.utils.set_random_seed(1234)
    return SwinTransformer(
        num_classes=10, embed_dim=24, depths=(2, 2, 2, 2),
        num_heads=(2, 2, 2, 2), window_size=2, patch_size=2,
        input_shape=(32, 32, 3), name="swin",
    )


def _tiny_dino_v3():
    import keras

    from dl_techniques.models.dino.dino_v3 import DINOv3

    keras.utils.set_random_seed(1234)
    model = DINOv3(
        image_size=32, patch_size=8, embed_dim=48, depth=3, num_heads=4,
        num_classes=10, include_top=True, name="dino3",
    )
    model.build((None, 32, 32, 3))
    return model


#: ``(builder, minimum number of SAME-SHAPE kernel pairs the model must expose)``.
#: The second element is the ANTI-VACUITY floor: a guard that compares kernels of
#: different shapes proves nothing, and a guard whose model stopped exposing any
#: same-shape pair at all would pass while measuring NOTHING. These floors are the
#: pair counts MEASURED 2026-08-23 (vit 264, swin 44, dino_v3 12), so a config or
#: architecture change that stops producing them fails loudly instead of silently
#: going green.
_RUNTIME_SUBJECTS = (
    ("vit", _tiny_vit, 264),
    ("swin_transformer", _tiny_swin, 44),
    ("dino_v3", _tiny_dino_v3, 12),
)


def _identical_same_shape_kernel_pairs(model):
    """Return ``(identical_pairs, total_same_shape_pairs)`` for a built model.

    Only weights with ``ndim >= 2`` and a non-zero maximum are considered: a
    1-D vector is a bias/gain (``zeros``/``ones`` initialized, legitimately
    identical everywhere), and an all-zero tensor is a deliberate constant.
    Comparison is strictly WITHIN a shape group, so no pair is ever judged on
    the strength of a shape difference.
    """
    by_shape: Dict[Tuple[int, ...], List[Tuple[str, "np.ndarray"]]] = defaultdict(list)
    for weight in model.weights:
        array = np.array(weight)
        if array.ndim >= 2 and np.abs(array).max() > 0:
            by_shape[array.shape].append((weight.path, array))

    identical: List[str] = []
    total = 0
    for members in by_shape.values():
        for (path_a, a), (path_b, b) in itertools.combinations(members, 2):
            total += 1
            if float(np.abs(a - b).max()) == 0.0:
                identical.append(f"{path_a} == {path_b}")
    return identical, total


class TestPortedViTsDoNotShareOneInitializerInstance:
    """No two same-shape kernels of a freshly built port may be bit-identical.

    MEASURED at HEAD before this guard landed (seeded builds, CPU):

    ==================  =========================================
    model               bit-identical same-shape kernel pairs
    ==================  =========================================
    ``vit``             **132** of 264 (all 12 blocks' qkv, all 12 proj)
    ``swin_transformer``  **16** of 44 (both blocks of every stage)
    ``dino_v3``            **6** of 12 (all 3 encoder layers)
    ==================  =========================================

    Every one of those pairs spans DIFFERENT blocks, i.e. the model started as N
    copies of one block. For ``swin_transformer`` the two colliding blocks of a
    stage additionally differ by ``shift_size`` (regular vs shifted window), so
    they play different architectural roles outright.

    Not covered by this guard, and deliberately: Q vs K vs V. All three ports use
    a FUSED ``qkv`` Dense, so one draw supplies all three roles and they were
    MEASURED distinct (``max|delta|`` 0.073..0.080) even before the fix.
    """

    @pytest.mark.parametrize("name,builder,min_pairs", _RUNTIME_SUBJECTS)
    def test_no_two_same_shape_kernels_are_bit_identical(self, name, builder, min_pairs):
        model = builder()
        identical, total = _identical_same_shape_kernel_pairs(model)

        # Anti-vacuity FIRST: prove the guard had same-shape pairs to judge.
        assert total >= min_pairs, (
            f"{name} exposed only {total} same-shape kernel pairs, below the "
            f"measured floor of {min_pairs}. This guard is now weaker than when "
            "it was written -- it cannot detect a shared initializer instance in "
            "kernels it never compares. Fix the subject config, do not lower the "
            "floor."
        )

        assert not identical, (
            f"{name}: {len(identical)} of {total} same-shape kernel pairs are "
            f"BIT-IDENTICAL (max|delta| = 0.0). One seedless "
            "keras.initializers.Initializer INSTANCE reaches them all and "
            "replays its draw -- seed= is NOT the discriminator, instance "
            "identity is. Wrap each consumer in "
            "dl_techniques.initializers.clone_initializer. Colliding pairs: "
            f"{identical[:10]}"
        )
