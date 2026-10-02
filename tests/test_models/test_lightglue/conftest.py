"""Fixtures for the LightGlue model suite.

``tf32_disabled`` is the repo's single TF32-off fixture; it lives in
``tests/test_layers/conftest.py`` and is not collected from ``tests/test_models/``. It is
RE-EXPORTED here, not copied: a second copy is a second harness whose restore logic can drift.
The float32 parity bounds in this package are derived from eps32 arithmetic and are only
valid in that regime (the layer suites measured 3.6e-3 against the oracle with TF32 on,
5.8e-7 off).
"""

from tests.test_layers.conftest import tf32_disabled  # noqa: F401
