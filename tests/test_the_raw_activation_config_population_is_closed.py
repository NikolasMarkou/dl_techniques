"""The raw-``activation``-in-``get_config`` family (N-1) is CLOSED, not capped.

What the defect was
===================

A class that accepts ``activation: Union[str, Callable]`` and stores that value
**raw** in ``get_config()`` is broken for every non-string value. MEASURED:

* the default ``"gelu"`` round-trips bit-identically, so the defect is latent
  for every shipped config;
* a **registered** callable round-trips too on forward output -- Keras' generic
  encoder resolves it -- but leaves ``get_config()`` non-JSON-serializable and
  ``.activation`` a raw dict. **A guard written with a registered callable is
  vacuous**;
* an **UNREGISTERED** callable saves fine and then raises at load with
  ``ValueError: Could not interpret activation function identifier: {...}``;
  supplied with ``custom_objects`` it loads but leaves ``.activation`` a raw
  dict which ``get_config`` propagates onward.

The repair is a symmetric pair, centralised by D-400 in
``dl_techniques/utils/activation_serialization.py``: ``serialize_activation``
in ``get_config`` and ``deserialize_activation`` in ``__init__``. Its behavioural
guard, including the two separate RED-proofs, is
``tests/test_utils/test_activation_serialization.py``.

Why this file no longer carries a ceiling
=========================================

Its predecessor (``..._cannot_grow.py``) pinned a **ceiling** of 69 loose sites
and 14 ``Callable``-annotated ones and declared the rest
CLOSED-by-documented-default. That ruling was rejected, and the ``Callable``
filter it rested on does not survive contact with the tree: **Python does not
enforce annotations**, and the ``str``-annotated sites hand the value to
``keras.layers.Dense``, ``keras.layers.Activation``, ``keras.activations.get``
or ``resolve_activation_layer`` -- every one of which accepts a callable. The
14 was a subset of the annotation vocabulary, not of the reachable defect.

Re-derived at HEAD 2026-08-23: **72 sites / 54 files** (69 with a bare
``self.X`` value plus 3 that already special-cased ``keras.layers.Layer`` but
let a plain callable through raw). 65 were repaired. The 7 below are
**provably** unreachable, pinned by name with the evidence that makes them so --
not by a count.
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

#: The residue, by name, each with the expression that makes a callable
#: unreachable. These are NOT "defaulted": in every one of them the attribute
#: placed in ``get_config`` is provably a ``str`` or a ``bool`` at every
#: reachable assignment, so there is nothing for
#: ``serialize_activation`` to serialize and adding the call would be pure
#: noise. The evidence string is re-checked against the source below, so the
#: ruling rots loudly if someone drops the coercion.
_PROVABLY_NOT_AN_ACTIVATION_OBJECT = {
    "dl_techniques/layers/yolo12_blocks.py::ConvBlock::activation":
        "activation: bool = True",
    "dl_techniques/layers/transformers/energy_transformer.py::HopfieldNetwork::activation":
        "self.activation = str(activation)",
    "dl_techniques/layers/transformers/energy_transformer.py::EnergyTransformer::hopfield_activation":
        "self.hopfield_activation = str(hopfield_activation)",
    "dl_techniques/models/SAM/SAM3/decoder.py::Sam3DecoderLayer::activation":
        "self.activation = str(activation)",
    "dl_techniques/models/SAM/SAM3/decoder.py::Sam3TransformerDecoder::activation":
        "self.activation = str(activation)",
    "dl_techniques/models/energy_transformer/model.py::EnergyTransformerBackbone::hopfield_activation":
        "self.hopfield_activation = str(hopfield_activation)",
    "dl_techniques/models/graph_energy_transformer/model.py::GraphEnergyTransformerBackbone::hopfield_activation":
        "self.hopfield_activation = str(hopfield_activation)",
}


def _sweep() -> List[dict]:
    """Every ``get_config`` entry keyed ``*activation`` that is NOT routed
    through ``serialize_activation``."""
    found: List[dict] = []
    for dirpath, dirnames, filenames in os.walk(SRC_ROOT):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            rel = os.path.relpath(path, SRC_ROOT)
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            # No `except SyntaxError: continue` -- a silent skip shrinks the
            # population silently (D-070).
            tree = ast.parse(source, filename=path)
            for cls in ast.walk(tree):
                if not isinstance(cls, ast.ClassDef):
                    continue
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
                            # `serialize_activation` is D-400's shared helper.
                            # `activations.serialize` is the canonical Keras
                            # idiom used by 60 OTHER classes in this tree that
                            # normalise with `keras.activations.get` in
                            # `__init__` and were never part of N-1: measured at
                            # HEAD, 52 pair it with an `activations.get(...)`
                            # assignment and the other 8 inline the `get(...)`
                            # inside the `serialize(...)` call, so both
                            # directions are already symmetric there.
                            if ("serialize_activation" in expr
                                    or "activations.serialize" in expr):
                                continue
                            if "self." not in expr:
                                continue
                            found.append({
                                "id": f"{rel}::{cls.name}::{key.value}",
                                "file": rel,
                                "line": value.lineno,
                                "expr": expr,
                            })
    return found


@pytest.fixture(scope="module")
def population() -> List[dict]:
    return _sweep()


class TestTheRawActivationFamilyIsClosed:

    def test_nothing_outside_the_provable_residue_stores_a_raw_activation(
            self, population):
        observed = {record["id"] for record in population}
        extra = sorted(observed - set(_PROVABLY_NOT_AN_ACTIVATION_OBJECT))
        assert not extra, (
            "a `get_config` entry keyed *activation* stores its value without "
            "`serialize_activation`. An UNREGISTERED callable will then fail at "
            "load with 'Could not interpret activation function identifier'.\n"
            f"  new: {extra}\n"
            "Fix: call `serialize_activation` in `get_config` and "
            "`deserialize_activation` in `__init__` "
            "(dl_techniques.utils.activation_serialization). See D-400. Adding "
            "the site to _PROVABLY_NOT_AN_ACTIVATION_OBJECT is only legitimate "
            "if the stored value is a str/bool at EVERY reachable assignment."
        )

    def test_the_provable_residue_is_still_provable(self, population):
        """The roster is only as good as its evidence.

        A name-only exemption rots the moment someone deletes the ``str()``
        coercion that justified it, and the exemption would then silently cover
        a live defect. Each entry therefore re-greps for the expression it
        claims.
        """
        observed = {record["id"]: record for record in population}
        for site, evidence in sorted(_PROVABLY_NOT_AN_ACTIVATION_OBJECT.items()):
            relpath = site.split("::", 1)[0]
            with open(os.path.join(SRC_ROOT, relpath), encoding="utf-8") as handle:
                source = handle.read()
            assert evidence in source, (
                f"{site} is exempt because of `{evidence}`, which is no longer "
                "in the file. Either restore it or repair the site with "
                "serialize_activation/deserialize_activation."
            )
            assert site in observed, (
                f"{site} is listed as a provable non-activation but no longer "
                "appears in the sweep -- it was probably repaired. Drop it from "
                "_PROVABLY_NOT_AN_ACTIVATION_OBJECT."
            )

    def test_the_residue_has_not_grown(self, population):
        """A count assertion, kept only as a second, cheaper signal.

        The roster test above is the real gate: 'the count still matches' and
        'the same sites are still the ones exempt' are different claims and only
        the second is what an exemption asserts.
        """
        assert len(population) == len(_PROVABLY_NOT_AN_ACTIVATION_OBJECT) == 7
