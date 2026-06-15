"""Smoke-test placeholder for the jepa family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).

`models/jepa/` exposes only JEPAEncoder / JEPAPredictor layers (encoder.py) plus
config/utilities; there is NO top-level keras.Model or factory to forward-pass
without bespoke assembly. Per the plan this family is SKIP-ONLY.
"""

import pytest


@pytest.mark.skip(
    reason="models/jepa exposes only JEPAEncoder/JEPAPredictor layers; "
    "no top-level keras.Model or factory to forward-pass"
)
def test_smoke_build_and_forward():
    pass
