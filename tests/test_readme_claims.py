"""Executable enforcement of the factual claims in two hand-written READMEs.

`src/dl_techniques/models/qwen/README.md` and
`src/dl_techniques/models/superpoint/README.md` were written from scratch by
plan-2026-08-10-3649c19e/iter-1/step-8 (decisions.md D-019). Every path, symbol,
class attribute and variant-key set they name was verified once, mechanically —
by a script in a scratch directory that was then thrown away. `REPO_MAP.md`
already records what happens next: `verify_map.py` rotted the same way, and the
map it guarded drifted for weeks.

This file is that script, landed as a test so it runs with the suite.

Scope is deliberately narrow — the two READMEs above, and only the claims that
are decidable without a GPU or a training run:

* every repo-relative path they name exists;
* every module-level symbol they name imports;
* every class attribute / method they name is present;
* every ``MODEL_VARIANTS`` key set they tabulate matches the code;
* the two bare prose claims (``qwen/__init__.py`` is 0 bytes;
  ``SuperPoint.DETECTOR_CHANNELS == 65``).

It does NOT execute the READMEs' code blocks — those construct models and are
covered by the packages' own suites (`tests/test_models/test_qwen/`,
`tests/test_models/test_superpoint/`).

Adding a claim here is cheap; leaving one unenforced is what costs.
Landed 2026-08-10 by plan-2026-08-10-3649c19e/iter-2/step-13 (decisions.md
D-032).
"""

from __future__ import annotations

import importlib
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------
# Claims
# --------------------------------------------------------------------------

# Repo-relative paths named by either README.
PATHS = [
    # qwen README
    "src/dl_techniques/models/qwen/qwen3.py",
    "src/dl_techniques/models/qwen/qwen3_next.py",
    "src/dl_techniques/models/qwen/components.py",
    "src/dl_techniques/models/qwen/qwen3_embeddings.py",
    "src/dl_techniques/models/qwen/QWEN3.md",
    "src/dl_techniques/models/qwen/QWEN3_next.md",
    "src/dl_techniques/models/qwen/__init__.py",
    "tests/test_models/test_qwen/test_qwen3.py",
    "tests/test_models/test_qwen/test_qwen3_next.py",
    "tests/test_models/test_qwen/test_components.py",
    "tests/test_models/test_qwen/test_qwen3_embeddings.py",
    # superpoint README
    "src/dl_techniques/models/superpoint/model.py",
    "src/dl_techniques/models/superpoint/__init__.py",
    "src/dl_techniques/losses/superpoint_loss.py",
    "src/dl_techniques/datasets/synthetic_shapes.py",
    "src/dl_techniques/utils/homography.py",
    "src/train/superpoint/train_magicpoint.py",
    "src/train/superpoint/train_superpoint.py",
    "src/train/superpoint/homographic_adaptation.py",
    "tests/test_models/test_superpoint/test_model.py",
    "tests/test_losses/test_superpoint_loss.py",
    "tests/test_train/test_superpoint",
]

# (module, module-level symbol) named by either README.
SYMBOLS = [
    ("dl_techniques.models.qwen.qwen3", "Qwen3"),
    ("dl_techniques.models.qwen.qwen3", "create_qwen3"),
    ("dl_techniques.models.qwen.qwen3", "create_qwen3_generation"),
    ("dl_techniques.models.qwen.qwen3", "create_qwen3_classification"),
    ("dl_techniques.models.qwen.qwen3_next", "Qwen3Next"),
    ("dl_techniques.models.qwen.qwen3_next", "create_qwen3_next"),
    ("dl_techniques.models.qwen.qwen3_next", "create_qwen3_next_generation"),
    ("dl_techniques.models.qwen.qwen3_next", "create_qwen3_next_classification"),
    ("dl_techniques.models.qwen.components", "Qwen3NextBlock"),
    ("dl_techniques.models.qwen.qwen3_embeddings", "Qwen3EmbeddingLayer"),
    ("dl_techniques.models.qwen.qwen3_embeddings", "Qwen3RerankerLayer"),
    ("dl_techniques.models.qwen.qwen3_embeddings", "Qwen3EmbeddingModel"),
    ("dl_techniques.models.qwen.qwen3_embeddings", "Qwen3RerankerModel"),
    ("dl_techniques.models.superpoint", "SuperPoint"),
    ("dl_techniques.models.superpoint", "create_superpoint"),
    ("dl_techniques.models.superpoint.model", "SuperPoint"),
    ("dl_techniques.models.superpoint.model", "create_superpoint"),
    ("dl_techniques.losses.superpoint_loss", "SuperPointDetectorLoss"),
    ("dl_techniques.losses.superpoint_loss", "SuperPointDescriptorLoss"),
    ("dl_techniques.layers.sequence_pooling", "SequencePooling"),
    ("dl_techniques.layers.transformers", "GatedLinearAttentionBlock"),
    ("dl_techniques.layers.attention.gated_attention", "GatedAttention"),
    ("dl_techniques.layers.transformers", "TransformerLayer"),
]

# (module, class, attribute-or-method) named by either README.
ATTRS = [
    ("dl_techniques.models.qwen.qwen3", "Qwen3", "MODEL_VARIANTS"),
    ("dl_techniques.models.qwen.qwen3", "Qwen3", "from_variant"),
    ("dl_techniques.models.qwen.qwen3_next", "Qwen3Next", "MODEL_VARIANTS"),
    ("dl_techniques.models.qwen.qwen3_next", "Qwen3Next", "from_variant"),
    ("dl_techniques.models.superpoint.model", "SuperPoint", "MODEL_VARIANTS"),
    ("dl_techniques.models.superpoint.model", "SuperPoint", "from_variant"),
    ("dl_techniques.models.superpoint.model", "SuperPoint", "from_config"),
    ("dl_techniques.models.superpoint.model", "SuperPoint", "get_config"),
    (
        "dl_techniques.models.superpoint.model",
        "SuperPoint",
        "compute_output_shape",
    ),
    ("dl_techniques.losses.superpoint_loss", "SuperPointDescriptorLoss", "compute"),
]

# The variant tables the READMEs print, as EXACT key sets. An inequality in
# either direction is a defect: a missing key means the README promises a
# variant that does not exist, an extra one means it under-documents.
VARIANTS = [
    (
        "dl_techniques.models.qwen.qwen3",
        "Qwen3",
        {"tiny", "small", "medium", "30b-coder"},
    ),
    (
        "dl_techniques.models.qwen.qwen3_next",
        "Qwen3Next",
        {"tiny", "small", "80b", "80b_a3b"},
    ),
    (
        "dl_techniques.models.superpoint.model",
        "SuperPoint",
        {"tiny", "base", "large"},
    ),
]


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rel_path", PATHS)
def test_readme_path_exists(rel_path):
    assert os.path.exists(os.path.join(REPO_ROOT, rel_path)), (
        f"{rel_path} is named by a README but does not exist. Fix the README "
        "or restore the file — a doc naming a path that is not there is the "
        "defect class this checker exists for."
    )


@pytest.mark.parametrize("module,symbol", SYMBOLS, ids=[f"{m}.{s}" for m, s in SYMBOLS])
def test_readme_symbol_imports(module, symbol):
    mod = importlib.import_module(module)
    assert hasattr(mod, symbol), (
        f"{module}.{symbol} is named by a README but is not importable from "
        f"that module. Public names in {sorted(getattr(mod, '__all__', []))!r}"
    )


@pytest.mark.parametrize(
    "module,cls,attr", ATTRS, ids=[f"{c}.{a}" for _, c, a in ATTRS]
)
def test_readme_class_attribute_exists(module, cls, attr):
    obj = getattr(importlib.import_module(module), cls)
    assert hasattr(obj, attr), f"{cls}.{attr} is named by a README but is absent"


@pytest.mark.parametrize(
    "module,cls,expected", VARIANTS, ids=[c for _, c, _ in VARIANTS]
)
def test_readme_variant_table_matches_code(module, cls, expected):
    actual = set(getattr(importlib.import_module(module), cls).MODEL_VARIANTS)
    assert actual == expected, (
        f"{cls}.MODEL_VARIANTS is {sorted(actual)} but its README table lists "
        f"{sorted(expected)}. Update BOTH the table and this list."
    )


def test_qwen_package_init_is_empty():
    """The qwen README states there is no curated package API.

    Measured as a byte count, not by reading: a future `__init__.py` that
    starts re-exporting names makes the README's "import from the submodules"
    instruction wrong.
    """
    init = os.path.join(REPO_ROOT, "src/dl_techniques/models/qwen/__init__.py")
    size = os.path.getsize(init)
    assert size == 0, (
        f"models/qwen/__init__.py is {size} bytes; its README states the "
        "package init is empty and that imports must come from the submodules"
    )


def test_superpoint_detector_channels_is_65():
    """65 = 8x8 cells + 1 dustbin — the README's /8-stride detector claim."""
    sp = importlib.import_module(
        "dl_techniques.models.superpoint.model"
    ).SuperPoint
    assert sp.DETECTOR_CHANNELS == 65, (
        f"SuperPoint.DETECTOR_CHANNELS is {sp.DETECTOR_CHANNELS}; the README "
        "documents 65 (an 8x8 cell plus one dustbin channel)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
