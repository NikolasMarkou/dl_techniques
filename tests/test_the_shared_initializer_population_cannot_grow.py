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
#: MEASURED at HEAD 2026-08-22 after step 19.1's eight fixes: **159** groups /
#: 539 participating call sites / 100 files, down from 175/609/108 before them.
#:
#: The carried census figure of "378 sites / 118 files" does **not** reproduce
#: under this instrument, nor under either of the two instruments D-144 already
#: tried, nor under the 178/565/102 variant recorded in
#: ``findings/initializer-and-norm-census.md``. Do not re-quote 378/118.
_COLLISION_GROUP_CEILING = 159

#: Ceiling on the *defect-shaped* subset: a collision group in which 2+ of the
#: colliding calls have the SAME callee AND the SAME size expression (so their
#: weights have the same shape, so they really are bit-identical) but are
#: assigned to DIFFERENT attributes (so they plausibly play different roles).
#: This is the triage list. MEASURED at HEAD 2026-08-22: **57**, down from **79**
#: before step 19.1 (the eight fixes removed 22 of them).
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
