"""Ceiling gate on the raw-``activation``-in-``get_config`` family (N-1).

The defect
==========

A class whose constructor accepts ``activation: Union[str, Callable]`` and whose
``get_config`` stores that value **raw** is broken for every non-string value.
MEASURED by step 12 on ``vit_siglip`` and reproduced on ``vit`` / ``vit_hmlp``:

* the default ``"gelu"`` round-trips bit-identically, so the defect is latent for
  every shipped config;
* a **registered** callable also already round-trips, because Keras's generic
  object encoder resolves it -- **a guard written against a registered callable
  is vacuous and passes with and without the fix**;
* an **UNREGISTERED** callable saves fine and then raises at load with
  ``ValueError: Could not interpret activation function identifier: {...}``;
  supplied with ``custom_objects`` it loads, but leaves ``.activation`` a raw
  dict which ``get_config`` then propagates.

The repair is a symmetric pair -- ``keras.activations.serialize`` in
``get_config`` and ``keras.activations.deserialize`` in ``from_config``. One side
alone breaks the other direction, which is why both are RED-proven separately for
each fixed class.

The carried "47 sites" does not reproduce
=========================================

AST at HEAD 2026-08-23 under three instruments:

============================================================  =====  =====
instrument                                                    sites  files
============================================================  =====  =====
any ``get_config`` dict key containing "activation", no
``serialize`` in the value                                    120     67
key matching ``(^|_)activation$`` whose value is a **bare**
``self.X``                                                     69     51
...and whose ``__init__`` types that argument as accepting
a **Callable** -- the actual defect precondition                14     12
============================================================  =====  =====

**None of the three is 47.** The third row is the defect-shaped one and the only
one worth acting on: a ``self.activation_type`` / ``self.activation_args`` entry
is a factory key or a kwargs dict, and storing it raw is correct.

Terminal state (decisions.md D-205): ``vit`` and ``vit_hmlp`` SHIPPED, joining
``vit_siglip`` (D-012). The remaining 14 are CLOSED-by-documented-default and
listed by name in ``_KNOWN_RAW_ACTIVATION_SITES`` below, so the residue is a
roster rather than a number.
"""

import ast
import os
import re
from typing import Dict, List

import pytest

SRC_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
)

_ACTIVATION_KEY = re.compile(r"(^|_)activation$")
_BARE_SELF_ATTR = re.compile(r"self\.\w+")

#: Ceiling on the LOOSE population: a ``get_config`` entry whose key ends in
#: ``activation`` and whose value is a bare ``self.X`` with no ``serialize``.
#: MEASURED 2026-08-23 after D-205's two fixes: **69 sites / 51 files** (was
#: 71/53). Most are string factory keys and are NOT defects; this ceiling exists
#: so a genuinely new one cannot appear unnoticed.
_RAW_ACTIVATION_ENTRY_CEILING = 69

#: Ceiling on the DEFECT-SHAPED subset: the same, restricted to classes whose
#: ``__init__`` types that argument as accepting a ``Callable``. Those are the
#: ones where a non-string value is reachable and therefore breakable.
#: MEASURED 2026-08-23 after D-205: **14** (was 16).
_CALLABLE_TYPED_RAW_CEILING = 14

#: The 14 by name. A roster, not a number: "the count happens to match" and "the
#: same sites are still the ones defaulting" are different claims, and only the
#: second is what a documented default asserts.
_KNOWN_RAW_ACTIVATION_SITES = {
    "dl_techniques/layers/attention/non_local_attention.py::NonLocalAttention::intermediate_activation",
    "dl_techniques/layers/attention/non_local_attention.py::NonLocalAttention::output_activation",
    "dl_techniques/layers/convolutional_kan.py::KANvolution::activation",
    "dl_techniques/layers/hierarchical_mlp_stem.py::HierarchicalMLPStem::activation",
    "dl_techniques/layers/repmixer_block.py::ConvolutionalStem::activation",
    "dl_techniques/layers/time_series/mixed_sequential_block.py::MixedSequentialBlock::activation",
    "dl_techniques/layers/time_series/nbeats_blocks.py::NBeatsBlock::activation",
    "dl_techniques/layers/transformers/gated_linear_attention_block.py::GatedLinearAttentionBlock::activation",
    "dl_techniques/layers/transformers/text_decoder.py::TextDecoder::activation",
    "dl_techniques/models/depth_anything/components.py::DPTDecoder::activation",
    "dl_techniques/models/depth_anything/components.py::DPTDecoder::output_activation",
    "dl_techniques/models/dino/dino_v3.py::DINOv3::activation",
    "dl_techniques/models/time_series/nbeats/nbeats.py::NBeatsNet::activation",
    "dl_techniques/models/time_series/nbeats/nbeatsx.py::NBeatsXNet::activation",
}

#: The three classes already repaired. Named so an unwrap is loud.
_FIXED_SITES = {
    "dl_techniques/models/vit/model.py": "ViT",
    "dl_techniques/models/vit_hmlp/model.py": "ViTHMLP",
    "dl_techniques/models/vit_siglip/model.py": "SigLIPVisionTransformer",
}


def _sweep() -> Dict[str, List[dict]]:
    """Return ``{"loose": [...], "callable_typed": [...]}``."""
    loose: List[dict] = []
    callable_typed: List[dict] = []
    for dirpath, dirnames, filenames in os.walk(SRC_ROOT):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            rel = os.path.relpath(path, SRC_ROOT)
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            # No `except SyntaxError: continue` -- a silent skip shrinks both
            # ceilings silently (D-070).
            tree = ast.parse(source, filename=path)
            for cls in ast.walk(tree):
                if not isinstance(cls, ast.ClassDef):
                    continue
                annotations = {}
                for member in cls.body:
                    if not (isinstance(member, ast.FunctionDef)
                            and member.name == "__init__"):
                        continue
                    for arg in list(member.args.args) + list(member.args.kwonlyargs):
                        if arg.annotation is not None:
                            annotations[arg.arg] = ast.unparse(arg.annotation)
                for member in cls.body:
                    if not (isinstance(member, ast.FunctionDef)
                            and member.name == "get_config"):
                        continue
                    for node in ast.walk(member):
                        if not isinstance(node, ast.Dict):
                            continue
                        for key, value in zip(node.keys, node.values):
                            if not (isinstance(key, ast.Constant)
                                    and isinstance(key.value, str)):
                                continue
                            if not _ACTIVATION_KEY.search(key.value):
                                continue
                            expr = ast.unparse(value)
                            if "serialize" in expr:
                                continue
                            if not _BARE_SELF_ATTR.fullmatch(expr):
                                continue
                            record = {
                                "id": f"{rel}::{cls.name}::{key.value}",
                                "file": rel,
                                "line": value.lineno,
                            }
                            loose.append(record)
                            annotation = annotations.get(key.value, "")
                            if "allable" in annotation:
                                callable_typed.append(record)
    return {"loose": loose, "callable_typed": callable_typed}


@pytest.fixture(scope="module")
def population() -> Dict[str, List[dict]]:
    return _sweep()


class TestTheRawActivationPopulationIsPinned:

    def test_the_loose_population_has_not_grown(self, population):
        loose = population["loose"]
        assert len(loose) <= _RAW_ACTIVATION_ENTRY_CEILING, (
            f"raw activation-keyed get_config entries grew from "
            f"{_RAW_ACTIVATION_ENTRY_CEILING} to {len(loose)}. Check whether the "
            "new one's constructor can receive a CALLABLE; if it can, it needs "
            "keras.activations.serialize in get_config AND deserialize in "
            "from_config. See decisions.md D-205."
        )

    def test_the_defect_shaped_subset_has_not_grown(self, population):
        typed = population["callable_typed"]
        assert len(typed) <= _CALLABLE_TYPED_RAW_CEILING, (
            f"the CALLABLE-typed raw-activation population grew from "
            f"{_CALLABLE_TYPED_RAW_CEILING} to {len(typed)}. Every entry here "
            "accepts a callable activation and stores it raw, so an unregistered "
            "callable will fail at load with 'Could not interpret activation "
            f"function identifier'. New: {sorted({r['id'] for r in typed} - _KNOWN_RAW_ACTIVATION_SITES)}"
        )

    def test_the_defaulted_residue_is_still_the_same_sites(self, population):
        observed = {record["id"] for record in population["callable_typed"]}
        assert observed == _KNOWN_RAW_ACTIVATION_SITES, (
            "the roster of sites CLOSED-by-documented-default under D-205 no "
            "longer matches what the tree contains.\n"
            f"  appeared: {sorted(observed - _KNOWN_RAW_ACTIVATION_SITES)}\n"
            f"  gone:     {sorted(_KNOWN_RAW_ACTIVATION_SITES - observed)}\n"
            "A site that disappeared was probably fixed -- move it to "
            "_FIXED_SITES. A site that appeared has never been ruled."
        )

    @pytest.mark.parametrize("relpath,classname", sorted(_FIXED_SITES.items()))
    def test_a_fixed_class_still_serializes_and_deserializes(self, relpath, classname):
        with open(os.path.join(SRC_ROOT, relpath), encoding="utf-8") as handle:
            source = handle.read()
        assert "activations.serialize(self.activation)" in source, (
            f"{classname} no longer serializes its activation in get_config. "
            "An unregistered callable now fails at load again."
        )
        assert "activations.deserialize(" in source, (
            f"{classname} no longer deserializes its activation in from_config. "
            "The pair is symmetric: one side alone breaks the other direction."
        )
