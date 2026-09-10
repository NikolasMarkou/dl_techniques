"""Guards for ``dl_techniques.layers.dynamic_chunking.indexing``.

Why this module exists
----------------------
``indexing.py`` is the shared index arithmetic step 6.1 introduced so that
``ChunkLayer`` and ``DeChunkLayer`` could stop holding the same partition rule
in two hand-copied blocks. Its two gather helpers are exercised heavily and
indirectly by ``test_chunk_layer.py`` and ``test_dechunk_layer.py`` -- mutating
either reds those suites hard (the filler index: 4 failed; the row offsets:
45 failed).

:func:`~dl_techniques.layers.dynamic_chunking.indexing.dim` is the exception,
and this module is its home. Review pass 2 forced it to always return the
dynamic shape and measured **234 passed on CPU** and green on GPU 0 under
``jit_compile="auto"``, then asked the right question: is the static path
load-bearing and unguarded, or is its stated rationale wrong?

**It is the rationale.** MEASURED here (D-032, TF 2.18 / Keras 3):
``keras.ops.shape`` already returns a Python ``int`` for every statically known
axis -- eagerly AND inside a traced ``tf.function`` -- and a tensor only for an
axis that is genuinely unknown. So ``dim``'s static branch computes what the
backend was going to hand back anyway, mutation S-10 is **EQUIVALENT on this
backend**, and no test can be written that fails without the branch. Claiming
otherwise would be a guard that cannot fail, which is the exact defect class
this plan has now hit ten times.

What is pinned here instead is that EQUIVALENCE, plus the result contract the
callers actually consume. A future reader who deletes the branch will find a
test that agrees with them and explains why; a future reader who re-invents the
"XLA may refuse a shape tensor" rationale will find it refuted in the same
place. The docstring was corrected to match.
"""

from __future__ import annotations

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.dynamic_chunking.indexing import (
    batched_gather,
    dim,
    pad_permutation_to_width,
)


class TestDimIsTheBackendsOwnAnswer:
    """``dim`` == ``keras.ops.shape(x)[axis]`` on this backend, measured.

    These arms are deliberately NOT advertised as catching S-10. They cannot,
    and the reason they cannot is the finding.
    """

    @pytest.mark.parametrize("axis,expected", [(0, 2), (1, 5), (2, 3), (-1, 3), (-2, 5)])
    def test_a_static_axis_is_a_python_int(self, axis, expected):
        """The result contract: ``type(...) is int``, not merely ``== 5``.

        ``pad_permutation_to_width`` documents its ``width`` as "a Python
        ``int`` whenever the caller knows it statically"; this is the promise
        behind that sentence.
        """
        x = keras.ops.zeros((2, 5, 3), dtype="float32")
        value = dim(x, axis)

        assert type(value) is int, f"axis {axis} returned {type(value)!r}"
        assert value == expected

    @pytest.mark.parametrize("axis", [0, 1, 2, -1, -2])
    def test_it_returns_EXACTLY_what_keras_ops_shape_returns(self, axis):
        """The equivalence, stated where a reader will look for it.

        This is why the static branch is not load-bearing: eagerly,
        ``keras.ops.shape`` is already int-valued per static axis.
        """
        x = keras.ops.zeros((2, 5, 3), dtype="float32")
        backend_answer = keras.ops.shape(x)[axis]

        assert dim(x, axis) == backend_answer
        assert type(dim(x, axis)) is type(backend_answer) is int

    def test_the_equivalence_ALSO_holds_inside_a_traced_function(self):
        """The regime the refuted rationale actually named.

        Tracing is where a shape "becomes a tensor" if it is ever going to.
        MEASURED: at a fully static ``TensorSpec`` every entry is still a
        Python ``int``, and only the ``None`` axis becomes a tensor.
        """
        seen = {}

        @tf.autograph.experimental.do_not_convert
        def probe(t):
            seen["shape"] = keras.ops.shape(t)
            seen["dims"] = [dim(t, a) for a in range(3)]
            return tf.reduce_sum(t)

        traced = tf.function(probe)

        traced.get_concrete_function(tf.TensorSpec([2, 5, 3], tf.float32))
        assert seen["shape"] == (2, 5, 3)
        assert [type(v) for v in seen["dims"]] == [int, int, int]
        assert seen["dims"] == [2, 5, 3]

        traced.get_concrete_function(tf.TensorSpec([2, None, 3], tf.float32))
        assert type(seen["dims"][0]) is int and type(seen["dims"][2]) is int
        assert not isinstance(seen["dims"][1], int), (
            "an axis the tracer does not know cannot come back as an int; "
            "returning one would bake a placeholder length into the graph"
        )
        # And the wrapper agrees with the backend on the dynamic axis too.
        assert type(seen["dims"][1]) is type(seen["shape"][1])

    def test_a_symbolic_KerasTensor_does_not_invent_a_length(self):
        """``build((None, None))`` is how every H-Net here is built."""
        symbolic = keras.Input(shape=(None, 3), dtype="float32")
        assert symbolic.shape[1] is None, "the fixture must be dynamic on axis 1"

        assert dim(symbolic, 1) is None
        assert dim(symbolic, 2) == 3


class TestWhatTheStaticIntBuysDownstream:
    """The consequence the callers depend on, asserted as behaviour."""

    def test_the_padded_permutation_has_a_STATIC_width(self):
        indices = keras.ops.convert_to_tensor(np.array([[0, 2, 1]], dtype="int32"))
        width = dim(keras.ops.zeros((1, 6, 4), dtype="float32"), 1)

        padded = pad_permutation_to_width(indices, width)
        assert padded.shape == (1, 6), padded.shape
        assert padded.shape[1] is not None, (
            "the width came back dynamic, so the traced graph no longer knows "
            "the inner sequence length"
        )

    def test_batched_gather_still_agrees_with_a_numpy_take_along_axis(self):
        """An end-to-end sanity arm, so this file is not purely about types."""
        rng = np.random.default_rng(11)
        params = rng.standard_normal((3, 7, 4)).astype("float32")
        indices = np.stack([rng.permutation(7) for _ in range(3)]).astype("int32")

        gathered = keras.ops.convert_to_numpy(
            batched_gather(
                keras.ops.convert_to_tensor(params),
                keras.ops.convert_to_tensor(indices),
            )
        )
        expected = np.take_along_axis(params, indices[:, :, None], axis=1)
        np.testing.assert_array_equal(gathered, expected)
