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
sees a loop. Do NOT treat the group ceiling as coverage.

Since step 10 (D-560) there is also a SECOND static census, ``_loop_carried_sites``,
written specifically for this shape: it flags every ``*_initializer=<bound name>``
keyword written inside a ``for``/``while`` body or a comprehension, with no 2+
requirement at all. MEASURED at HEAD: **119 sites in 44 files**. That population
is large, and the number is reported rather than tuned down -- it is the honest
size of the idiom, not a defect count. A site collides only if the instance
actually reaches two SAME-shape weights, which is a runtime question; of the six
sites step 10 probed, two measured ZERO collisions and were left alone.

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
#: MEASURED at HEAD 2026-08-23 after this plan's step 10: **151** groups / 521
#: participating call sites / 97 files, down from 157/533/99 after step 9 (itself
#: down from 159/539/100, and from 175/609/108 before step 19.1's eight fixes).
#: Step 9 cloned ``models/vit/model.py``'s two ``__init__`` groups away; step 10
#: cloned ``vit_hmlp``, ``vit_siglip``, ``vision_encoder`` and ``beit_attention``.
#: ``swin_transformer``, ``dino_v3`` and ``hrm_reasoning_module`` were fixed in
#: those same two steps but never appeared here at all -- see the loop blind spot
#: noted below, which since step 10 has its own ceiling
#: (``_LOOP_CARRIED_SITE_CEILING``) instead of only a prose warning.
#:
#: The carried census figure of "378 sites / 118 files" does **not** reproduce
#: under this instrument, nor under either of the two instruments D-144 already
#: tried, nor under the 178/565/102 variant recorded in
#: ``findings/initializer-and-norm-census.md``. Do not re-quote 378/118.
_COLLISION_GROUP_CEILING = 151

#: Ceiling on the *defect-shaped* subset: a collision group in which 2+ of the
#: colliding calls have the SAME callee AND the SAME size expression (so their
#: weights have the same shape, so they really are bit-identical) but are
#: assigned to DIFFERENT attributes (so they plausibly play different roles).
#: This is the triage list. MEASURED at HEAD 2026-08-23: **57**, down from **79**
#: before step 19.1 (the eight fixes removed 22 of them). NEITHER step 9's three
#: fixes NOR step 10's six moved this number, and that is the finding, not an
#: oversight: none of the groups those steps removed was
#: same-callee-same-size-different-target. Every defect both steps measured was
#: loop-carried, i.e. invisible to this subset by construction -- which is
#: precisely the blind spot the runtime guard and ``_LOOP_CARRIED_SITE_CEILING``
#: below exist to cover. Re-measured after step 10 and still exactly 57.
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

#: Modules fixed by step 10 (D-560), the six siblings D-540 named. Each one was
#: MEASURED bit-identical before and MEASURED non-identical after; the runtime
#: subjects below re-measure them on every run, and this text check additionally
#: catches an unwrap in a code path the runtime subjects do not reach.
_STEP_10_FIXED_MODULES = (
    "models/vit_hmlp/model.py",
    "models/vit_siglip/model.py",
    "layers/transformers/vision_encoder.py",
    "layers/reasoning/hrm_reasoning_module.py",
    "layers/attention/beit_attention.py",
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


_LOOP_BODIES = (ast.For, ast.AsyncFor, ast.While)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _loop_carried_sites() -> List[Tuple[str, int, str, str, str]]:
    """Return every ``*_initializer=<bound name>`` kwarg written inside a loop.

    This is the census ``_sweep`` structurally CANNOT be. ``_sweep`` groups by
    ``(file, scope, expression)`` and keeps groups of 2+ *syntactic* sites, so a
    ``for`` loop handing one instance to N blocks scores 1 and disappears.
    MEASURED: ``swin_transformer`` (16 real collisions), ``dino_v3`` (6),
    ``vit_hmlp`` (132), ``vit_siglip`` (132), ``vision_encoder`` (12) and
    ``hrm[multi_head]`` (12) never appeared in ``_sweep`` at all.

    A site counts when an initializer keyword argument is given a *bound name* --
    a ``Name`` or a dotted ``self.x`` attribute -- and the call sits lexically
    inside a ``for``/``while`` body or a comprehension. A ``Call`` value returns
    ``None`` from :func:`_expr_key` and is therefore excluded, which is exactly
    what makes ``clone_initializer(...)`` remove a site from this population:
    MEASURED, step 10's five source edits dropped ``vit_hmlp``, ``vit_siglip``,
    ``vision_encoder``, ``hrm_reasoning_module`` and ``beit_attention`` out of
    this list.

    :returns: ``(relpath, lineno, kwarg, expression, loop node type)`` per site.
    :rtype: list of 5-tuples
    """
    sites: List[Tuple[str, int, str, str, str]] = []
    for dirpath, dirnames, filenames in os.walk(SRC_ROOT):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            rel = os.path.relpath(path, SRC_ROOT)
            with open(path, encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)

            parents: Dict[int, ast.AST] = {}
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    parents[id(child)] = parent

            for call in ast.walk(tree):
                if not isinstance(call, ast.Call):
                    continue
                for kw in call.keywords:
                    if kw.arg is None or not kw.arg.endswith("_initializer"):
                        continue
                    key = _expr_key(kw.value)
                    if key is None:
                        continue
                    node: Optional[ast.AST] = call
                    enclosing = None
                    while node is not None:
                        parent = parents.get(id(node))
                        if isinstance(parent, _COMPREHENSIONS):
                            enclosing = type(parent).__name__
                            break
                        if isinstance(parent, _LOOP_BODIES) and node in parent.body:
                            enclosing = type(parent).__name__
                            break
                        node = parent
                    if enclosing is not None:
                        sites.append((rel, kw.value.lineno, kw.arg, key, enclosing))
    return sites


#: Ceiling on the LOOP-CARRIED population -- the shape ``_sweep`` is blind to.
#: MEASURED at HEAD 2026-08-23 after step 10: **119 sites in 44 files**. That
#: number is deliberately not made to look small. It is the honest size of the
#: construction idiom "resolve one initializer instance, then hand it to every
#: block a loop builds", and it is LARGER than the 96/40 quoted in D-540 because
#: this instrument also counts ``while`` bodies and comprehensions, not only
#: ``for`` bodies.
#:
#: A site in this list is a *candidate*, not a verdict: it collides only if the
#: instance actually reaches two weights of the SAME shape. Two of the six sites
#: step 10 probed measured ZERO collisions and were correctly left alone --
#: ``models/dino/dino_v1.py`` (the initializer is stored as an inert dict, so
#: every consumer resolves its own instance) and ``models/beit/model.py`` (the
#: ``'beit'`` attention branch of ``TransformerLayer`` does not forward
#: ``kernel_initializer`` at all). Both are still counted here, because the
#: census must not encode a per-site verdict it cannot re-derive.
_LOOP_CARRIED_SITE_CEILING = 119


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

    # Reported through `warnings.warn` for the same reason as the role-suspect
    # test above (D-252 opt-out): the population is large and must stay visible.
    @pytest.mark.filterwarnings("always::UserWarning")
    def test_the_loop_carried_population_has_not_grown(self):
        sites = _loop_carried_sites()
        files = sorted({site[0] for site in sites})
        warnings.warn(
            f"loop-carried initializer sites ({len(sites)} in {len(files)} "
            f"files), each a candidate for the D-540/D-560 defect: {files}",
            UserWarning,
            stacklevel=2,
        )
        # Anti-vacuity FIRST: an AST walk that silently matched nothing would
        # pass the ceiling assertion below while measuring NOTHING.
        assert len(sites) >= 50, (
            f"the loop-carried census found only {len(sites)} sites, far below "
            "the 119 measured when it was written. This instrument has almost "
            "certainly stopped matching (a walk change, a renamed kwarg suffix), "
            "not the repo suddenly stopped using the idiom. Fix the instrument."
        )
        assert len(sites) <= _LOOP_CARRIED_SITE_CEILING, (
            "loop-carried initializer sites grew from "
            f"{_LOOP_CARRIED_SITE_CEILING} to {len(sites)}. A new one means one "
            "initializer INSTANCE is again being handed to every layer a loop "
            "builds. This is the population the static group census CANNOT see "
            "(it needs 2+ syntactic sites; a loop is one), and it is where every "
            "collision D-540 and D-560 measured actually lived. Build the new "
            "site seeded and compare its same-shape kernels: if any pair is "
            "bit-identical, wrap it in "
            "dl_techniques.initializers.clone.clone_initializer and add it to "
            "_RUNTIME_SUBJECTS. Do NOT raise this ceiling to make a red test "
            f"green. Sites: {sites[:20]}"
        )

    @pytest.mark.parametrize("module", _FIXED_MODULES + _STEP_10_FIXED_MODULES)
    def test_a_fixed_module_still_clones(self, module):
        path = os.path.join(SRC_ROOT, module)
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        assert "clone_initializer(" in source, (
            f"{module} no longer calls clone_initializer. Step 19.1/9/10 MEASURED "
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


def _tiny_beit():
    import keras

    from dl_techniques.models.beit.model import BeitModel

    keras.utils.set_random_seed(1234)
    model = BeitModel(
        input_shape=(32, 32, 3), patch_size=8, hidden_size=48, num_layers=4,
        num_heads=4, intermediate_size=96, layer_scale_init_value=0.1,
        name="beit",
    )
    model.build((None, 32, 32, 3))
    return model


def _tiny_vit_hmlp():
    import keras

    from dl_techniques.models.vit_hmlp.model import ViTHMLP

    keras.utils.set_random_seed(1234)
    model = ViTHMLP(
        input_shape=(32, 32, 3), num_classes=10, scale="tiny", patch_size=8,
        include_top=True, name="vit_hmlp",
    )
    model.build((None, 32, 32, 3))
    return model


def _tiny_vit_siglip():
    import keras

    from dl_techniques.models.vit_siglip.model import SigLIPVisionTransformer

    keras.utils.set_random_seed(1234)
    model = SigLIPVisionTransformer(
        input_shape=(32, 32, 3), num_classes=10, scale="tiny", patch_size=8,
        include_top=True, name="siglip",
    )
    model.build((None, 32, 32, 3))
    return model


def _tiny_dino_head():
    """DINOv1's projection head -- the NEGATIVE control of this family.

    It builds ``nlayers`` same-shape ``Dense`` layers in a ``for`` loop from one
    ``self.kernel_initializer``, the exact syntactic shape of the five defects
    around it, and it MEASURED **0 of 15** identical pairs at HEAD. The reason is
    that ``DINOHead`` stores the argument RAW -- ``DINO_KERNEL_INITIALIZER`` is an
    inert ``{"class_name": ..., "config": ...}`` dict (see
    ``models/dino/reference_init.py``) -- so ``keras.initializers.get`` resolves a
    FRESH instance inside every ``Dense``. It is guarded anyway: adding the
    apparently harmless ``self.kernel_initializer = keras.initializers.get(...)``
    normalization to ``__init__`` would turn all 15 pairs identical in one line.
    """
    import keras

    from dl_techniques.models.dino.dino_v1 import DINOHead

    keras.utils.set_random_seed(1234)
    model = DINOHead(
        in_dim=48, out_dim=64, nlayers=6, hidden_dim=48, bottleneck_dim=48,
        name="dino_head",
    )
    model.build((None, 48))
    return model


def _tiny_vision_encoder():
    import keras

    from dl_techniques.layers.transformers.vision_encoder import VisionEncoder

    keras.utils.set_random_seed(1234)
    model = VisionEncoder(
        img_size=32, patch_size=8, embed_dim=48, depth=4, num_heads=4,
        name="vision_encoder",
    )
    model.build((None, 32, 32, 3))
    return model


def _tiny_hrm(attention_type: str):
    import keras

    from dl_techniques.layers.reasoning.hrm_reasoning_module import (
        HierarchicalReasoningModule,
    )

    keras.utils.set_random_seed(1234)
    model = HierarchicalReasoningModule(
        num_layers=4, embed_dim=48, num_heads=4, attention_type=attention_type,
        ffn_type="mlp" if attention_type == "multi_head" else "swiglu",
        name=f"hrm_{attention_type}",
    )
    model.build([(None, 16, 48), (None, 16, 48)])
    return model


def _tiny_hrm_group_query():
    return _tiny_hrm("group_query")


def _tiny_hrm_multi_head():
    """HRM at its DECLARED, non-default ``attention_type``.

    Both configurations are subjects on purpose. At the default 'group_query' the
    module MEASURED 0 colliding trainable pairs even before step 10, because
    ``GroupedQueryAttention`` already clones per projection -- the masking is that
    layer's property, not this one's. Flip the knob to the equally supported
    'multi_head' and the same source line MEASURED 12 of 24 pairs at
    ``max|delta| = 0.0``. A guard that only ever built the default would have
    called this site clean.
    """
    return _tiny_hrm("multi_head")


#: ``(builder, minimum number of SAME-SHAPE kernel pairs the model must expose)``.
#: The second element is the ANTI-VACUITY floor: a guard that compares kernels of
#: different shapes proves nothing, and a guard whose model stopped exposing any
#: same-shape pair at all would pass while measuring NOTHING. These floors are the
#: pair counts MEASURED 2026-08-23 (vit 264, swin 44, dino_v3 12), so a config or
#: architecture change that stops producing them fails loudly instead of silently
#: going green.
#:
#: Step 10 (D-560) added the six siblings D-540 named. Their pre-fix / post-fix
#: identical-pair counts, all MEASURED on seeded CPU builds:
#:
#: ==========================  ==============  ==============
#: subject                     identical BEFORE  identical AFTER
#: ==========================  ==============  ==============
#: ``beit``                    24 of 133       0 of 133
#: ``vit_hmlp``                132 of 264      0 of 264
#: ``vit_siglip``              132 of 264      0 of 264
#: ``dino_v1_head``            **0 of 15**     0 of 15  (never defective)
#: ``vision_encoder``          12 of 24        0 of 24
#: ``hrm[group_query]``        **0 of 154**    0 of 154 (masked by its attention)
#: ``hrm[multi_head]``         12 of 24        0 of 24
#: ==========================  ==============  ==============
_RUNTIME_SUBJECTS = (
    ("vit", _tiny_vit, 264),
    ("swin_transformer", _tiny_swin, 44),
    ("dino_v3", _tiny_dino_v3, 12),
    ("beit", _tiny_beit, 133),
    ("vit_hmlp", _tiny_vit_hmlp, 264),
    ("vit_siglip", _tiny_vit_siglip, 264),
    ("dino_v1_head", _tiny_dino_head, 15),
    ("vision_encoder", _tiny_vision_encoder, 24),
    ("hrm_group_query", _tiny_hrm_group_query, 154),
    ("hrm_multi_head", _tiny_hrm_multi_head, 24),
)


def _identical_same_shape_kernel_pairs(model):
    """Return ``(identical_pairs, total_same_shape_pairs)`` for a built model.

    Only TRAINABLE weights with ``ndim >= 2`` and a non-zero maximum are
    considered: a 1-D vector is a bias/gain (``zeros``/``ones`` initialized,
    legitimately identical everywhere), and an all-zero tensor is a deliberate
    constant. Comparison is strictly WITHIN a shape group, so no pair is ever
    judged on the strength of a shape difference.

    ``trainable`` is the filter that separates a drawn weight from a COMPUTED
    table. MEASURED on ``hrm``: its four blocks' ``attention/rope/cos_cached``
    and ``sin_cached`` are pairwise ``max|delta| = 0.0`` (12 pairs) and always
    will be -- they are closed-form sinusoid caches, not initializer draws, and
    identical across depth is their CORRECT value. Dropping them leaves 154
    trainable pairs at ``min max|delta| = 0.38``. MEASURED that this filter does
    not weaken the three subjects that predate it: ``vit`` 264, ``swin`` 44,
    ``dino_v3`` 12, all unchanged (none of them has a non-trainable ndim>=2
    weight; ``swin``'s relative-position-bias tables ARE trainable and stay in).
    Do NOT replace this with a name filter such as "paths ending in /kernel":
    MEASURED, that one silently drops ``swin``'s floor from 44 to 16.
    """
    by_shape: Dict[Tuple[int, ...], List[Tuple[str, "np.ndarray"]]] = defaultdict(list)
    for weight in model.weights:
        if not weight.trainable:
            continue
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

    Step 10 (D-560) added the six siblings D-540 named; see ``_RUNTIME_SUBJECTS``
    for their measured before/after table.

    Most of these pairs span DIFFERENT blocks, i.e. the model started as N copies
    of one block -- the weaker, same-role-different-depth form. Two subjects are
    sharper. For ``swin_transformer`` the two colliding blocks of a stage differ
    by ``shift_size`` (regular vs shifted window), so they play different
    architectural roles outright. For ``beit`` the collision is INSIDE one block:
    ``BeitAttention`` builds four separate ``(dim, dim)`` projections -- MEASURED
    ``q == k == v == proj`` at ``max|delta| = 0.0`` in every block -- so ``Q == K``
    made the pre-softmax score matrix ``x W W^T x^T`` exactly symmetric at step 0.

    Q vs K vs V is NOT a collision in ``vit``/``swin``/``dino_v3``/``vit_hmlp``/
    ``vit_siglip``/``vision_encoder``: those route through a FUSED ``qkv`` Dense,
    so one draw supplies all three roles from different column blocks and they
    were MEASURED distinct (``max|delta|`` 0.073..0.080) even before the fix.
    Check the layout before calling a q/k/v pair a collision -- ``beit`` collides
    because its projections are genuinely separate layers, not because it is a
    transformer.
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
