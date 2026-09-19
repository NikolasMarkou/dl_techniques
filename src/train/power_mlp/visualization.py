"""Compatibility alias for :mod:`train.common.classification_viz`.

The plots moved to ``train/common/`` when a second classification trainer (ConvNeXt)
needed them. This module IS that module (same object, not a copy of its names), so
``import train.power_mlp.visualization as viz`` keeps every public and private name
and ``monkeypatch.setattr(viz, ...)`` still reaches the code that reads the name.
"""

import sys

import train.common.classification_viz as _classification_viz

# DECISION plan-2026-09-19T040641-db6932ec/D-010: alias the module object; do NOT turn
# this into ``from train.common.classification_viz import *`` (or a hand-written
# re-export list). A copy of the names leaves ``monkeypatch.setattr(viz, "render_training_dashboard", ...)``
# and ``viz._save_and_close`` patching a name nothing reads, and the dashboard/figure
# tests of tests/test_train/test_power_mlp then pass without testing anything.
# Guard: test_the_power_mlp_visualization_module_is_the_shared_module.
sys.modules[__name__] = _classification_viz
