"""The forecasting models build their heads with ``create_time_series_head``.

A model that constructs a head class, imports one from its defining module, or
routes through ``create_head`` fails here. A planted source proves the walk
sees each of the three.
"""

import ast
from pathlib import Path
from typing import List

import pytest

from dl_techniques.layers.heads.time_series import __all__ as _TIME_SERIES_ALL

_SRC = Path(__file__).resolve().parents[2] / "src" / "dl_techniques" / "models" / "time_series"
_HEAD_CLASSES = {name for name in _TIME_SERIES_ALL if name.endswith("Head")}
_CONSUMERS = [
    "adaptive_ema/model.py",
    "deepar/model.py",
    "prism/model.py",
    "timestral/model.py",
    "tirex/model.py",
    "tirex/model_extended.py",
    "xlstm/forecaster.py",
]


def _violations(source: str) -> List[str]:
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _HEAD_CLASSES or node.func.id == "create_head":
                found.append(f"calls {node.func.id}")
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "dl_techniques.layers.heads.time_series."):
            found.append(f"imports from {node.module}")
    return found


@pytest.mark.parametrize("relative", _CONSUMERS)
def test_the_model_uses_the_factory_and_only_the_factory(relative: str) -> None:
    source = (_SRC / relative).read_text(encoding="utf-8")
    assert _violations(source) == []
    assert "create_time_series_head(" in source


def test_the_list_names_every_module_that_uses_the_factory() -> None:
    users = {
        path.relative_to(_SRC).as_posix() for path in _SRC.rglob("*.py")
        if "create_time_series_head(" in path.read_text(encoding="utf-8")
    }
    assert users == set(_CONSUMERS)


def test_the_walk_sees_each_violation() -> None:
    planted = (
        "from dl_techniques.layers.heads.time_series.likelihood_heads import GaussianLikelihoodHead\n"
        "a = GaussianLikelihoodHead(units=1)\n"
        "b = create_head('time_series', 'quantile_sequence', num_quantiles=3)\n"
    )
    assert len(_violations(planted)) == 3
