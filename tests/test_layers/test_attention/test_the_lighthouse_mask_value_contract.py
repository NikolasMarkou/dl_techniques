"""``_mask_value``'s contract: the dtypes it really receives, and the one it must not.

**Why this file exists.** ``plan-2026-09-02T170923-1285ed83`` D-018 replaced
``keras.backend.standardize_dtype(d)`` with ``getattr(d, "name", None) or str(d)``
across ``src/``. The two agree on every dtype *spelling* (47/47 measured) but not
on non-dtype objects: the old symbol RAISED ``TypeError`` on a
``keras.DTypePolicy`` and normalized ``None`` to ``floatx()``; the idiom returns
``"mixed_float16"`` and ``"None"`` respectively.

At almost every one of the 15 attention sites that divergence is benign, because
the name is immediately fed to ``common.mask_dtype``, which returns ``float32``
for anything that is not exactly ``"float64"``. ``_mask_value`` is the exception:
its name goes into ``_MASK_SENTINEL.get(name, -1.0e9)``, so an unrecognised name
returns the float32 sentinel, and ``keras.ops.cast(-1e9, "float16")`` is
``-inf``. A fully-masked row then softmaxes to NaN. The old symbol would have
raised; this one is silent.

**Live or latent?** LATENT, measured two ways. Statically, ``_mask_value`` is
module-private and has exactly two call sites in the entire repo,
``lighthouse_attention.py:1052`` and ``:1115``, and both pass a backend tensor's
``.dtype``. Dynamically, a spy over the whole of
``tests/test_layers/test_attention/`` (2060 passed / 34 skipped / 1 xfailed)
recorded **640 calls, every argument a ``tf.DType``** (all ``tf.float32``, which
is what that suite exercises), and zero policies, ``None``s or numpy scalar
types. So nothing is broken today, and no numeric path is corrected here. What is
missing is anything that BREAKS if the precondition stops holding, which is what
the arms below add.
"""

import ast
import pathlib

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.attention.lighthouse_attention import (
    _MASK_SENTINEL,
    _mask_value,
)

MODULE_PATH = pathlib.Path(
    "src/dl_techniques/layers/attention/lighthouse_attention.py"
)
if not MODULE_PATH.is_file():  # running from anywhere but the repo root
    MODULE_PATH = (
        pathlib.Path(__file__).resolve().parents[3]
        / "src/dl_techniques/layers/attention/lighthouse_attention.py"
    )

#: The population the two call sites can actually deliver: a backend tensor's
#: ``.dtype``, in both of the spellings a backend produces for it.
SUPPORTED = ("float16", "bfloat16", "float32", "float64")


class TestTheContractForTheDtypesItActuallyReceives:
    """What ``_mask_value`` promises for a real tensor dtype."""

    @pytest.mark.parametrize("name", SUPPORTED)
    def test_a_backend_dtype_returns_that_dtypes_table_entry(self, name: str) -> None:
        """A ``tf.DType`` and its name must resolve to the same table entry.

        This is the D-018 idiom's whole job here: ``str(tf.float16)`` is
        ``"<dtype: 'float16'>"``, which is NOT a key, so a bare ``str`` would
        take the ``-1e9`` fallback for every dtype and reintroduce the fp16
        ``-inf`` this table exists to remove.
        """
        expected = _MASK_SENTINEL[name]
        assert _mask_value(getattr(tf, name)) == expected
        assert _mask_value(name) == expected
        assert _mask_value(np.dtype(name)) == expected

    @pytest.mark.parametrize("name", SUPPORTED)
    def test_the_value_is_finite_in_its_own_dtype_and_exponentiates_to_zero(
        self, name: str
    ) -> None:
        """The two properties the sentinel exists for, asserted after the cast.

        Finite: ``cast(-1e9, "float16")`` is ``-inf``, and ``-inf`` in an
        additive mask makes ``0 * -inf -> NaN`` reachable. Underflowing: a
        sentinel that ``exp`` does not send to exactly 0.0 leaks probability
        into a masked position.
        """
        cast = float(keras.ops.cast(_mask_value(getattr(tf, name)), name))
        assert np.isfinite(cast), f"{name}: sentinel is not finite after the cast"
        assert cast < 0.0
        assert float(keras.ops.exp(keras.ops.cast(cast, name))) == 0.0

    def test_the_table_covers_every_floating_dtype_a_layer_can_compute_in(
        self,
    ) -> None:
        """Anti-vacuity for the parametrization: the table is the population.

        If a dtype is added to ``_MASK_SENTINEL`` and not to ``SUPPORTED``, the
        arms above silently stop covering it.
        """
        assert set(_MASK_SENTINEL) == set(SUPPORTED)


class TestThePreconditionThatMakesTheIdiomSafeHere:
    """The call sites must keep passing a tensor dtype, not a policy."""

    def test_every_call_site_passes_a_dtype_attribute(self) -> None:
        """AST-pinned: ``_mask_value(<something>.dtype)`` at every call site.

        RED-proved by rewriting either call to ``_mask_value(self.dtype_policy)``
        -- the exact argument the old ``keras.backend.standardize_dtype`` would
        have rejected with a ``TypeError`` and this idiom accepts silently.
        """
        tree = ast.parse(MODULE_PATH.read_text())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_mask_value"
        ]
        assert len(calls) == 2, (
            "the call-site population moved; re-read the anchor at _mask_value "
            f"before widening this pin. Found {len(calls)} call(s)."
        )
        for call in calls:
            arg = call.args[0]
            assert isinstance(arg, ast.Attribute) and arg.attr == "dtype", (
                f"lighthouse_attention.py:{call.lineno}: _mask_value must be "
                "given a tensor's `.dtype`. A dtype POLICY resolves to a name "
                "that is not in _MASK_SENTINEL, takes the -1e9 fallback, and "
                "becomes -inf in float16 -- silently. See the D-007 anchor."
            )

    def test_a_dtype_policy_really_would_be_silently_wrong(self) -> None:
        """Anti-vacuity for the pin above: it guards a real, measured hazard.

        Without this, the AST arm asserts a shape whose consequence nobody has
        checked. ``keras.DTypePolicy("mixed_float16")`` names itself
        ``"mixed_float16"``, misses the table, takes ``-1e9``, and casts to
        ``-inf`` in the very dtype it claims to be safe for. If a future change
        makes the fallback dtype-aware, this arm goes red -- delete it then, and
        the AST pin above with it.
        """
        policy = keras.DTypePolicy("mixed_float16")
        assert (getattr(policy, "name", None) or str(policy)) == "mixed_float16"
        assert "mixed_float16" not in _MASK_SENTINEL
        assert _mask_value(policy) == -1.0e9
        assert float(keras.ops.cast(_mask_value(policy), "float16")) == -np.inf
        # ... where the value the compute dtype's own entry gives is finite.
        assert float(keras.ops.cast(_mask_value(tf.float16), "float16")) == -60000.0

    def test_the_old_symbol_would_have_raised_on_that_argument(self) -> None:
        """The divergence, pinned against the symbol D-018 removed.

        ``keras.backend.standardize_dtype`` is banned in ``src/`` and permitted
        in ``tests/`` precisely so that a replacement can be measured against
        it. This arm is why the anchor says "the old symbol raised loudly": if
        Keras ever makes it accept a policy, that sentence is stale.
        """
        with pytest.raises(TypeError):
            keras.backend.standardize_dtype(keras.DTypePolicy("mixed_float16"))
