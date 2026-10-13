"""Shared instruments for the integer-weight placement guards.

Why this module exists
----------------------
TensorFlow keeps an int32 resource variable in host memory. An XLA-compiled
step on a GPU cannot read or assign a host resource, so a layer with an int32
weight fails ``fit(jit_compile=True)`` -- the Keras default on a GPU -- with::

    InvalidArgumentError: Trying to access resource .../patch_table located in
    device CPU:0 from device GPU:0

and passes every CPU test. ``SpatialSmoothness`` (``perm``, ``patch_table``,
``triu_flat``), ``GaussianReadout`` (``cell_to_unit_flat``, ``perm``) and
``SparsePuzzleEmbedding`` (``local_ids``) carried such weights until
plan-2026-10-13T035756-5dd4d94d/D-010 moved them to
``dl_techniques.utils.dtype_policy.INDEX_WEIGHT_DTYPE`` (int64).

Interface
---------
- :data:`GPU_VISIBLE` / :data:`requires_gpu`: the skip for the arms that need
  the real device. The defect is a placement rule, so it cannot occur on a CPU;
  the device-independent arm is :func:`int32_variables`.
- :func:`int32_variables` ``(obj) -> List[str]``: the path of every int32
  variable reachable from a layer or a model (seed-generator state included).
  An empty list is the pass condition.
- :func:`built_as_int32` ``(*modules)``: a context manager under which each
  given module builds its integer weights as int32, which is what the code did
  before the fix, so a test can write an archive in the old format. It raises
  ``AttributeError`` for a module that has no ``INDEX_WEIGHT_DTYPE`` binding,
  so it cannot silently patch nothing.
"""

import contextlib
from typing import Any, Iterator, List

import pytest
import tensorflow as tf

#: True when TensorFlow can see at least one GPU.
GPU_VISIBLE = bool(tf.config.list_physical_devices("GPU"))

#: Skip marker for an arm that needs the device the defect lives on.
requires_gpu = pytest.mark.skipif(
    not GPU_VISIBLE,
    reason=(
        "needs a visible GPU: an int32 variable is host-pinned, and only an "
        "XLA step on a GPU fails to read it"
    ),
)


def int32_variables(obj: Any) -> List[str]:
    """Return the path of every int32 variable of a layer or a model."""
    return sorted(v.path for v in obj.variables if v.dtype == "int32")


@contextlib.contextmanager
def built_as_int32(*modules: Any) -> Iterator[None]:
    """Make each module build its integer weights as int32 inside the block."""
    with pytest.MonkeyPatch.context() as patch:
        for module in modules:
            patch.setattr(module, "INDEX_WEIGHT_DTYPE", "int32")
        yield
