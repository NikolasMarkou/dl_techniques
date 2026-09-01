"""Contract for ``dl_techniques.utils.dtype_policy``.

This file opens with an anti-vacuity class, and the order is deliberate. Both
functions under test exist to defend against a numeric hazard that is invisible
in float32, so every assertion below is worthless unless the hazard itself is
first shown to be REAL in this interpreter, at these library versions:

* ``np.float16(-1e9)`` really is ``-inf``, and ``0 * -inf`` really is ``NaN`` --
  the additive-mask defect.
* ``np.float16(1e-8)`` really is exactly ``0.0`` -- the stability-floor defect --
  while ``np.float16(1e-7)`` is NOT. The boundary matters: a first-pass review of
  this defect class believed ``1e-7`` was also a hard zero, which inflated the
  defect count roughly twofold. ``1e-7`` floors are LOSSY (subnormal), not
  catastrophic.

If ``TestTheHazardsAreReal`` ever goes red, nothing else in this file is testing
anything and the whole file must be re-derived rather than repaired.

``bfloat16`` has no numpy dtype; it comes from ``ml_dtypes``, which ships as a
hard dependency of TensorFlow. Same source as
``tests/test_layers/test_activations/test_sparsemax.py:29``.
"""

import ast
import pathlib
import warnings
from typing import List, Tuple

import keras
import ml_dtypes
import numpy as np
import pytest

from dl_techniques.utils import dtype_policy
from dl_techniques.utils.dtype_policy import (
    accumulation_dtype,
    mask_sentinel,
    stability_floor,
)

# ---------------------------------------------------------------------
# Fixtures and shared tables
# ---------------------------------------------------------------------

#: The four floating-point compute dtypes this repository ships layers for.
DTYPES = ("float16", "bfloat16", "float32", "float64")

#: Name -> a numpy-compatible dtype, so a returned Python float can be
#: materialized and inspected in the dtype it claims to be safe in.
NUMPY_DTYPE = {
    "float16": np.dtype(np.float16),
    "bfloat16": np.dtype(ml_dtypes.bfloat16),
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
}

#: The float32-era epsilons this repository actually writes at guard sites.
REQUESTED_EPSILONS = (1e-7, 1e-8, 1e-10, 1e-12, 1e-30)

#: The incumbent ``layers.attention.common.MASK_BIAS_VALUE``. Imported as a
#: LITERAL on purpose: importing the symbol would make this assertion tautological
#: once ``common.py`` starts sourcing the value from the policy.
INCUMBENT_MASK_BIAS_VALUE = -1e9


def _finfo(name: str):
    """``finfo`` for any of the four dtypes, including bfloat16."""
    numpy_dtype = NUMPY_DTYPE[name]
    if name == "bfloat16":
        return ml_dtypes.finfo(ml_dtypes.bfloat16)
    return np.finfo(numpy_dtype)


def _materialize(value: float, name: str):
    """Round a Python float onto ``name``'s own grid."""
    return np.array(value, dtype=NUMPY_DTYPE[name])


@pytest.fixture
def floatx_float16():
    """Set the global ``floatx`` to float16 and restore it afterwards."""
    original = keras.config.floatx()
    keras.config.set_floatx("float16")
    try:
        yield
    finally:
        keras.config.set_floatx(original)


# ---------------------------------------------------------------------
# Anti-vacuity: the hazards are real before any defense is asserted.
# ---------------------------------------------------------------------


class TestTheHazardsAreReal:
    """Both defect classes must reproduce here, or nothing below discriminates."""

    def test_float16_of_the_incumbent_sentinel_is_negative_infinity(self):
        with np.errstate(over="ignore"):
            in_fp16 = np.float16(INCUMBENT_MASK_BIAS_VALUE)
        assert np.isneginf(in_fp16), (
            f"anti-vacuity FAILED: np.float16({INCUMBENT_MASK_BIAS_VALUE}) == "
            f"{in_fp16}, not -inf. The mask-sentinel tests below are then vacuous."
        )

    def test_zero_times_that_infinity_is_nan(self):
        """The exact arithmetic the additive mask form performs."""
        with np.errstate(over="ignore", invalid="ignore"):
            product = np.float16(0.0) * np.float16(INCUMBENT_MASK_BIAS_VALUE)
        assert np.isnan(product), (
            "anti-vacuity FAILED: `0 * float16(-1e9)` is not NaN, so the "
            "`x + (1 - keep) * sentinel` failure mode is not reproducible here."
        )

    @pytest.mark.parametrize("epsilon", [1e-8, 1e-10, 1e-12, 1e-30])
    def test_the_small_epsilons_are_exactly_zero_in_float16(self, epsilon):
        with np.errstate(under="ignore"):
            in_fp16 = np.float16(epsilon)
        assert float(in_fp16) == 0.0, (
            f"anti-vacuity FAILED: np.float16({epsilon}) == {in_fp16}, not 0.0. "
            "The stability-floor tests below are then vacuous."
        )

    def test_one_e_minus_seven_is_subnormal_but_NOT_zero_in_float16(self):
        """The corrected boundary. A first-pass review had this one wrong.

        ``1e-7`` survives float16 as a subnormal, so a ``1e-7`` floor is lossy,
        not a no-op. Only floors below roughly ``6e-8`` vanish entirely. Pinning
        this stops the defect class from being re-inflated by the same error.
        """
        with np.errstate(under="ignore"):
            in_fp16 = np.float16(1e-7)
        assert float(in_fp16) != 0.0
        assert float(in_fp16) < float(np.finfo(np.float16).tiny), (
            "1e-7 should land in the SUBNORMAL range of float16, below tiny."
        )

    def test_the_protective_looking_cast_is_the_defect(self):
        """``ops.cast(1e-10, x.dtype)`` casts the LITERAL (ema_layer.py:198)."""
        with np.errstate(under="ignore"):
            floored = np.maximum(np.float16(0.0), np.float16(1e-10))
        assert float(floored) == 0.0, (
            "anti-vacuity FAILED: a float16-cast 1e-10 floor is not a no-op "
            "here, so the ema_layer.py:198 hazard is not reproducible."
        )


# ---------------------------------------------------------------------
# mask_sentinel
# ---------------------------------------------------------------------


class TestMaskSentinelIsSafeInItsOwnDtype:
    """The three properties every returned sentinel must have."""

    @pytest.mark.parametrize("name", DTYPES)
    def test_is_finite_in_that_dtype(self, name):
        with np.errstate(over="ignore"):
            value = _materialize(mask_sentinel(name), name)
        assert np.isfinite(value), (
            f"mask_sentinel({name!r}) overflows to {value} in {name}; that is "
            "the exact defect this function exists to remove."
        )

    @pytest.mark.parametrize("name", DTYPES)
    def test_exp_of_it_is_exactly_zero_in_that_dtype(self, name):
        with np.errstate(under="ignore", over="ignore"):
            probability = np.exp(_materialize(mask_sentinel(name), name))
        assert float(probability) == 0.0, (
            f"exp(mask_sentinel({name!r})) == {probability} in {name}, not 0.0; "
            "a masked position would keep probability mass."
        )

    @pytest.mark.parametrize("name", DTYPES)
    def test_round_trips_through_the_dtype_unchanged(self, name):
        value = mask_sentinel(name)
        assert float(_materialize(value, name)) == value, (
            f"mask_sentinel({name!r}) == {value} is not exactly representable "
            f"in {name}; the documented value and the running value differ."
        )

    @pytest.mark.parametrize("name", DTYPES)
    def test_is_negative(self, name):
        assert mask_sentinel(name) < 0.0

    @pytest.mark.parametrize("name", DTYPES)
    def test_the_positive_counterpart_is_representable_too(self, name):
        """``min``-reductions negate the sentinel (sequence_pooling.py:596)."""
        positive = -mask_sentinel(name)
        assert positive > 0.0
        with np.errstate(over="ignore"):
            materialized = _materialize(positive, name)
        assert np.isfinite(materialized)
        assert float(materialized) == positive


class TestMaskSentinelValuesAndTheirGrounds:
    """The values themselves, and the reason each one is what it is."""

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("float16", -10000.0),
            ("bfloat16", -9984.0),
            ("float32", -1e9),
            ("float64", -1e9),
        ],
    )
    def test_pinned_values(self, name, expected):
        assert mask_sentinel(name) == expected

    def test_float32_reproduces_the_incumbent_mask_bias_value(self):
        """Step 10 replaces the literal with this call; the value must not move."""
        assert mask_sentinel("float32") == INCUMBENT_MASK_BIAS_VALUE

    def test_the_incumbent_value_is_rejected_for_float16(self):
        """The float16 answer must differ from the float32 one, or the policy is a constant."""
        assert mask_sentinel("float16") != INCUMBENT_MASK_BIAS_VALUE

    @pytest.mark.parametrize("name", ["float16", "bfloat16"])
    def test_reduced_precision_dtypes_share_the_float16_safe_magnitude(self, name):
        """bfloat16 is capped on NARROWING grounds, not on a range claim.

        bfloat16 carries float32's exponent range, so ``-1e9`` is finite there
        too and neither the ``gating.py`` (-1e4) nor the ``lighthouse`` (-1e9)
        spelling is unsafe. The policy sides with the float16-safe magnitude
        because a bfloat16 tensor is one cast away from float16.
        """
        with np.errstate(over="ignore"):
            in_float16 = np.float16(mask_sentinel(name))
        assert np.isfinite(in_float16)

    def test_bfloat16_is_the_bfloat16_grid_point_at_one_e_four(self):
        """-9984.0, not -10000.0: bfloat16 has 7 mantissa bits."""
        assert mask_sentinel("bfloat16") == float(
            -np.array(1e4, dtype=ml_dtypes.bfloat16)
        )
        assert mask_sentinel("bfloat16") != -1e4

    @pytest.mark.parametrize("name", DTYPES)
    def test_sits_well_below_the_dtype_maximum(self, name):
        """The overflow-headroom ground: a doubled sentinel is still finite."""
        magnitude = abs(mask_sentinel(name))
        assert 2.0 * magnitude < float(_finfo(name).max)


class TestMaskSentinelAcceptsEverySpellingTheRepoPasses:
    """A dtype reaches this function as a name, a policy, a numpy dtype or None."""

    @pytest.mark.parametrize(
        "spelling",
        [
            "float16",
            "mixed_float16",
            np.float16,
            np.dtype(np.float16),
        ],
    )
    def test_float16_spellings_agree(self, spelling):
        assert mask_sentinel(spelling) == mask_sentinel("float16")

    def test_a_dtype_policy_object_resolves_to_its_compute_dtype(self):
        policy = keras.DTypePolicy("mixed_float16")
        assert policy.variable_dtype == "float32"
        assert mask_sentinel(policy) == mask_sentinel("float16")

    def test_mixed_bfloat16_resolves_to_bfloat16(self):
        assert mask_sentinel("mixed_bfloat16") == mask_sentinel("bfloat16")

    def test_none_means_the_current_floatx(self):
        assert mask_sentinel(None) == mask_sentinel(keras.config.floatx())

    def test_none_follows_a_changed_floatx(self, floatx_float16):
        assert mask_sentinel(None) == mask_sentinel("float16")

    def test_the_default_argument_is_none(self):
        assert mask_sentinel() == mask_sentinel(None)

    @pytest.mark.parametrize("bad", ["int32", "not-a-dtype", 3, object()])
    def test_a_non_float_dtype_is_rejected_loudly(self, bad):
        with pytest.raises(ValueError):
            mask_sentinel(bad)


# ---------------------------------------------------------------------
# stability_floor
# ---------------------------------------------------------------------


class TestStabilityFloorIsStrictlyPositive:
    """SC5: 20/20 (dtype, requested) combinations, strictly positive."""

    @pytest.mark.parametrize("name", DTYPES)
    @pytest.mark.parametrize("requested", REQUESTED_EPSILONS)
    def test_is_strictly_positive(self, name, requested):
        assert stability_floor(name, requested) > 0.0

    @pytest.mark.parametrize("name", DTYPES)
    @pytest.mark.parametrize("requested", REQUESTED_EPSILONS)
    def test_is_still_nonzero_once_materialized_in_the_dtype(self, name, requested):
        with np.errstate(under="ignore"):
            materialized = _materialize(stability_floor(name, requested), name)
        assert float(materialized) > 0.0, (
            f"stability_floor({name!r}, {requested}) vanishes in {name}; the "
            "guard reads as protection and provides none."
        )

    @pytest.mark.parametrize("name", DTYPES)
    @pytest.mark.parametrize("requested", REQUESTED_EPSILONS)
    def test_is_normal_not_subnormal_in_the_dtype(self, name, requested):
        floored = stability_floor(name, requested)
        assert floored >= float(_finfo(name).tiny), (
            f"stability_floor({name!r}, {requested}) == {floored} is subnormal "
            f"in {name}; precision there is already degraded."
        )

    @pytest.mark.parametrize("name", DTYPES)
    @pytest.mark.parametrize("requested", REQUESTED_EPSILONS)
    def test_never_weakens_the_requested_floor(self, name, requested):
        assert stability_floor(name, requested) >= requested


class TestStabilityFloorLiftsOnlyWhereItMust:
    """It is a floor, not a rewrite: wide dtypes get the requested value back."""

    @pytest.mark.parametrize("requested", REQUESTED_EPSILONS)
    def test_float16_lifts_every_repo_epsilon_to_tiny(self, requested):
        assert stability_floor("float16", requested) == float(
            np.finfo(np.float16).tiny
        )

    @pytest.mark.parametrize("name", ["float32", "float64", "bfloat16"])
    @pytest.mark.parametrize("requested", REQUESTED_EPSILONS)
    def test_wide_exponent_dtypes_get_the_request_back_unchanged(
        self, name, requested
    ):
        assert stability_floor(name, requested) == requested

    def test_a_large_request_is_honored_verbatim(self):
        assert stability_floor("float16", 1e-2) == 1e-2

    @pytest.mark.parametrize(
        "spelling", ["float16", "mixed_float16", np.float16, np.dtype(np.float16)]
    )
    def test_float16_spellings_agree(self, spelling):
        assert stability_floor(spelling, 1e-8) == stability_floor("float16", 1e-8)

    def test_none_means_the_current_floatx(self, floatx_float16):
        assert stability_floor(None, 1e-8) == stability_floor("float16", 1e-8)

    @pytest.mark.parametrize("bad", [0.0, -1e-8, float("nan"), float("inf")])
    def test_a_non_positive_or_nonfinite_request_is_rejected(self, bad):
        with pytest.raises(ValueError):
            stability_floor("float32", bad)

    def test_a_request_beyond_the_dtype_maximum_is_rejected(self):
        """It would materialize as ``inf``, which is not a floor."""
        with pytest.raises(ValueError):
            stability_floor("float16", 1e30)

    def test_a_non_float_dtype_is_rejected_loudly(self):
        with pytest.raises(ValueError):
            stability_floor("int32", 1e-8)


# ---------------------------------------------------------------------
# I-6: utils/ stays cycle-free.
# ---------------------------------------------------------------------

#: The one file under ``utils/`` that imports ``dl_techniques.models`` at module
#: level, with its reason. It is NOT a licence to add another: the two-sided
#: assertion below fails both when a new file joins this set and when this file
#: leaves it, so the allowlist cannot rot into a silent pass.
MODULE_LEVEL_MODELS_IMPORT_ALLOWLIST = {
    "inference.py": (
        "Pre-existing at the base commit of this plan (a11fdb19c): "
        "`from dl_techniques.models.vision.yolo12.multitask import ...` at "
        "line 36, needed for the YOLO12 multitask inference entry point. Not "
        "introduced, and not fixed, by this plan -- pinned so it cannot grow."
    ),
}

#: Deferred (function-local) ``models`` imports are legal: they do not run at
#: import time and so cannot form an import cycle. ``multiplicative_miyasawa.py``
#: has one at line 814. Nothing pins that population; only module-level ones are
#: a cycle hazard.

UTILS_DIR = pathlib.Path(dtype_policy.__file__).resolve().parent


def _imported_modules(source: str, package: str) -> List[Tuple[str, bool]]:
    """Return ``(absolute_module_name, is_module_level)`` for every import.

    Relative imports are resolved against ``package`` so that a future
    ``from ..layers import x`` cannot slip past a walk that only reads absolute
    names. An ``ast`` walk, never a text search: this repository has been burned
    three times by a grep counting a docstring as a call site.

    :param source: Python source text.
    :param package: dotted package the source lives in, e.g.
        ``dl_techniques.utils.masking``.
    :return: one entry per imported module name.
    """
    tree = ast.parse(source)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    found: List[Tuple[str, bool]] = []
    for node in ast.walk(tree):
        module_level = isinstance(parents.get(node), ast.Module)
        if isinstance(node, ast.Import):
            found.extend((alias.name, module_level) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                prefix = package.split(".")
                prefix = prefix[: len(prefix) - node.level + 1]
                module = ".".join([*prefix, module]) if module else ".".join(prefix)
            found.append((module, module_level))
    return found


def _package_of(path: pathlib.Path) -> str:
    """Dotted package name for a file under ``src/dl_techniques/utils``."""
    relative = path.resolve().relative_to(UTILS_DIR).parent
    return ".".join(["dl_techniques", "utils", *relative.parts])


class TestTheImportDetectorCanSeeAnImport:
    """Anti-vacuity for the walk below: a blind census reports zero forever."""

    POSITIVE_SOURCE = (
        "from dl_techniques.layers.attention import common\n"
        "from dl_techniques.models.vision.resnet.model import ResNet\n"
        "from ..layers.norms import factory\n"
        "def f():\n"
        "    import dl_techniques.layers.ffn as ffn\n"
        "    from dl_techniques.models.language.bert import model\n"
    )

    CLEAN_SOURCE = (
        "import keras\n"
        "import numpy as np\n"
        "from dl_techniques.utils.logger import logger\n"
        "def f():\n"
        "    from dl_techniques.utils.tensors import reshape_to_2d\n"
    )

    def test_it_sees_every_form_in_a_synthetic_positive_source(self):
        found = _imported_modules(self.POSITIVE_SOURCE, "dl_techniques.utils")
        layers = sorted(m for m, _ in found if m.startswith("dl_techniques.layers"))
        models = sorted(m for m, _ in found if m.startswith("dl_techniques.models"))
        assert layers == [
            "dl_techniques.layers.attention",
            "dl_techniques.layers.ffn",
            "dl_techniques.layers.norms",
        ], f"saw {layers}; the third entry is the RELATIVE import, resolved"
        assert len(models) == 2, f"expected 2 models imports, saw {models}"

    def test_it_distinguishes_module_level_from_deferred(self):
        found = _imported_modules(self.POSITIVE_SOURCE, "dl_techniques.utils")
        module_level = {m for m, top in found if top and "dl_techniques.models" in m}
        deferred = {m for m, top in found if not top and "dl_techniques.models" in m}
        assert len(module_level) == 1
        assert len(deferred) == 1

    def test_it_reports_nothing_on_a_clean_source(self):
        found = _imported_modules(self.CLEAN_SOURCE, "dl_techniques.utils")
        assert not [
            m
            for m, _ in found
            if "dl_techniques.layers" in m or "dl_techniques.models" in m
        ]


class TestUtilsStaysCycleFree:
    """I-6 / SC2, asserted over the whole ``utils/`` tree, not just the new file."""

    def test_the_walk_actually_sees_the_tree(self):
        """Anti-vacuity floor: a walk over an empty file list passes everything."""
        files = list(UTILS_DIR.rglob("*.py"))
        assert len(files) >= 20, f"only {len(files)} files walked under {UTILS_DIR}"
        assert (UTILS_DIR / "dtype_policy.py") in files

    def test_no_file_under_utils_imports_dl_techniques_layers(self):
        offenders = []
        for path in sorted(UTILS_DIR.rglob("*.py")):
            for module, module_level in _imported_modules(
                path.read_text(encoding="utf-8"), _package_of(path)
            ):
                if module.startswith("dl_techniques.layers"):
                    scope = "module-level" if module_level else "deferred"
                    offenders.append(f"{path.name} ({scope}): {module}")
        assert not offenders, (
            "utils/ must not import layers/ at all -- not even inside a "
            f"function. Offenders: {offenders}"
        )

    def test_module_level_models_imports_are_exactly_the_allowlist(self):
        found = set()
        for path in sorted(UTILS_DIR.rglob("*.py")):
            for module, module_level in _imported_modules(
                path.read_text(encoding="utf-8"), _package_of(path)
            ):
                if module_level and module.startswith("dl_techniques.models"):
                    found.add(path.name)
        assert found == set(MODULE_LEVEL_MODELS_IMPORT_ALLOWLIST), (
            "the module-level `dl_techniques.models` population under utils/ "
            f"moved. Found {sorted(found)}, allowlist "
            f"{sorted(MODULE_LEVEL_MODELS_IMPORT_ALLOWLIST)}. A NEW entry is an "
            "import cycle; a MISSING one means the allowlist is now vacuous and "
            "must be shrunk."
        )

    def test_every_allowlist_entry_carries_a_written_reason(self):
        for name, reason in MODULE_LEVEL_MODELS_IMPORT_ALLOWLIST.items():
            assert reason.strip(), f"{name} is allowlisted with no reason"

    def test_the_policy_module_itself_imports_neither(self):
        source = pathlib.Path(dtype_policy.__file__).read_text(encoding="utf-8")
        for module, _ in _imported_modules(source, "dl_techniques.utils"):
            assert not module.startswith("dl_techniques.layers"), module
            assert not module.startswith("dl_techniques.models"), module


# ---------------------------------------------------------------------
# accumulation_dtype: the promotion, and its two boundaries.
# ---------------------------------------------------------------------


class TestAccumulationDtypeNeverNarrows:
    """The promotion must widen or do nothing -- never the reverse.

    The house exemplar this function generalizes,
    ``models/language/colbert/components.py``'s ``_safe_l2_normalize``, casts to
    ``"float32"`` unconditionally, which SILENTLY HALVES the precision of a
    float64 computation. That is the same defect as the one being fixed, in the
    other direction, so it is pinned rather than copied.
    """

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("float16", "float32"),
            ("bfloat16", "float32"),
            ("mixed_float16", "float32"),
            ("mixed_bfloat16", "float32"),
            ("float32", "float32"),
            ("float64", "float64"),
        ],
    )
    def test_the_promotion_table(self, name, expected):
        assert accumulation_dtype(name) == expected

    def test_no_dtype_is_ever_narrowed(self):
        for name in DTYPES:
            promoted = accumulation_dtype(name)
            assert _finfo(promoted).nmant >= _finfo(name).nmant, name
            assert _finfo(promoted).max >= _finfo(name).max, name

    def test_the_requested_epsilon_survives_the_promotion(self):
        """The whole point: at the promoted dtype the floor is the identity.

        This is what makes a promoted site bit-neutral at float32 and float64 --
        the casts are identities there and the epsilon is not lifted -- and what
        makes it EXACT under a half policy, where the pair
        ``(compute_dtype, stability_floor(compute_dtype, eps))`` would otherwise
        have coarsened ``1e-8`` to ``6.10e-05``.
        """
        for name in DTYPES:
            for requested in (1e-7, 1e-8, 1e-10, 1e-12):
                accum = accumulation_dtype(name)
                assert stability_floor(accum, requested) == requested
        # ...and the sibling call it replaces does NOT, in exactly the dtypes
        # the promotion exists for. Anti-vacuity: without this the assertion
        # above is satisfied by a function that returns its argument.
        assert stability_floor("float16", 1e-8) == pytest.approx(6.104e-05, rel=1e-3)
        assert stability_floor("float16", 1e-8) > 6000 * 1e-8

    def test_it_accepts_every_spelling_mask_sentinel_accepts(self):
        assert accumulation_dtype(None) == accumulation_dtype(keras.config.floatx())
        assert accumulation_dtype(np.float16) == "float32"
        assert accumulation_dtype(keras.DTypePolicy("mixed_float16")) == "float32"

    def test_it_rejects_a_non_float_dtype(self):
        with pytest.raises(ValueError):
            accumulation_dtype("int32")
        with pytest.raises(ValueError):
            accumulation_dtype("not-a-dtype")


# ---------------------------------------------------------------------
# SC1: the module's public surface.
# ---------------------------------------------------------------------


class TestTheModuleSurfaceStaysPureFunctions:
    """The module's public surface, pinned.

    This pin READ ``["mask_sentinel", "stability_floor"]`` -- the plan's SC1,
    which fixes the count at 2 -- until the 7.1 completion fix added
    ``accumulation_dtype``. SC1 is therefore graded FAIL at 3/2 and is
    deliberately NOT rewritten to match the result (D-005: a criterion missed as
    written is a FAIL reported as a FAIL). The third function is here because
    the alternative was a ``"float32" if <reduced precision> else <name>``
    ternary hand-copied to five call sites in two packages -- the same
    knowledge-in-N-files leak D-001 created this module to remove. What SC1
    was really protecting is unchanged and is still asserted below: no class,
    no decorator, no Keras registration, no tensor.
    """

    @staticmethod
    def _tree() -> ast.Module:
        return ast.parse(
            pathlib.Path(dtype_policy.__file__).read_text(encoding="utf-8")
        )

    def test_exactly_three_public_module_level_functions(self):
        public = [
            node.name
            for node in self._tree().body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
        ]
        assert sorted(public) == [
            "accumulation_dtype",
            "mask_sentinel",
            "stability_floor",
        ]

    def test_no_classes(self):
        assert not [
            node.name
            for node in ast.walk(self._tree())
            if isinstance(node, ast.ClassDef)
        ]

    def test_no_decorators_and_therefore_no_keras_registration(self):
        decorated = [
            node.name
            for node in ast.walk(self._tree())
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.decorator_list
        ]
        assert not decorated, (
            f"{decorated} carry decorators; this module defines no serializable "
            "object and must register nothing."
        )

    def test_it_does_not_import_from_keras_ops_directly(self):
        """House style: ``import keras`` only, qualified at the call site."""
        source = pathlib.Path(dtype_policy.__file__).read_text(encoding="utf-8")
        for module, _ in _imported_modules(source, "dl_techniques.utils"):
            assert module != "keras.ops", "use `keras.ops.x`, not `from keras import ops`"

    def test_every_function_is_importable_from_the_package_path(self):
        for function in (mask_sentinel, stability_floor, accumulation_dtype):
            assert callable(function)
            assert function.__doc__


def test_no_warning_is_emitted_on_the_common_paths():
    """A policy call is folded at trace time; it must be silent."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in DTYPES:
            mask_sentinel(name)
            stability_floor(name, 1e-8)
            accumulation_dtype(name)
