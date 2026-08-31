"""Repo-wide census: the additive mask-bias form is extinct, and every local
large-magnitude mask sentinel that remains is one somebody wrote a reason for.

.. warning::

   **THIS FILE IS EXPECTED TO BE RED until step 6 of
   ``plans/plan-2026-08-31T134711-6271592d`` completes.**

   It was written BEFORE the fixes it guards, deliberately, so that the real
   unfixed tree is its RED proof. At the plan's base commit ``043a59a52`` the
   additive-form census reports exactly seven violations::

       layers/heads/nlp/factory.py:461                    BaseNLPHead._pool_sequence
       layers/heads/nlp/factory.py:1255                   QuestionAnsweringHead.call
       layers/heads/nlp/factory.py:1256                   QuestionAnsweringHead.call
       layers/sequence_pooling/sequence_pooling.py:588    SequencePooling._apply_single_strategy
       layers/sequence_pooling/sequence_pooling.py:596    SequencePooling._apply_single_strategy
       models/language/tree_transformer/components.py:508 TreeMHA.call
       models/neural_computer/nam/cell.py:457             NAMCell.call

   That list is history, not an expectation: the assertion is ``== 0``, and the
   file is NOT marked ``xfail`` or ``skip``. A guard that cannot go red is
   worthless, and a guard permanently excused from going green is worse.

The defect being made extinct
-----------------------------
The **additive mask-bias form** writes a mask as arithmetic::

    logits = logits * mask + (1 - mask) * -1e9        # forbidden
    logits = logits + (1.0 - mask) * -1e9             # forbidden
    logits = logits + inv_mask * NEG                  # forbidden

Under ``mixed_float16`` the sentinel is materialized in the compute dtype, and
``float16`` tops out at ``65504``: ``-1e9`` and ``-1e8`` both become ``-inf``.
The multiplication then evaluates ``0 * -inf`` at every **KEPT** position, which
is ``NaN``. The failure is therefore not confined to the masked entries -- it
poisons the whole tensor, silently, on a code path a green float32 suite never
exercises. ``-1e4`` happens to survive float16, which is why several sites in
this tree are written in the forbidden shape and are nonetheless fine today;
they are still forbidden, because their safety rests on a literal nobody is
obliged to preserve.

The safe form is **selection**, never multiplication::

    logits = keras.ops.where(keep, logits, sentinel)   # required

No sentinel is ever multiplied, so an overflowed sentinel can only reach the
positions it was meant to reach. ``dl_techniques.utils.dtype_policy.mask_sentinel``
supplies a value that is finite in the target dtype;
``dl_techniques.layers.attention.common.apply_attention_mask`` supplies the full
softmax contract (``keep`` predicate, ``mask_dtype`` floor, fully-masked-row
rescue) for sites that are attention rather than a bare reduction.

Why this file exists at ``tests/`` root and not beside a package
---------------------------------------------------------------
The pre-existing guard,
``tests/test_layers/test_attention/test_common.py::TestTheAdopterCountsAreMechanical``,
enumerates its population with ``pathlib.Path(common.__file__).parent.glob("*.py")``
-- ``layers/attention/`` only, and NON-recursively. It is structurally incapable
of failing for a site anywhere else in the tree, and six of the seven violations
above live outside that directory. This file walks ``src/dl_techniques/``
entirely. It follows the house convention for repo-wide closure guards
(``tests/test_the_shared_initializer_population_cannot_grow.py``).

Why AST and never a text grep
-----------------------------
Every ``# DECISION`` anchor and module docstring in this tree that WARNS about
the additive form spells the additive form out verbatim, as does this docstring.
A textual scan counts those warnings as defects and this file as the worst
offender in the repository. The census walks ``ast`` nodes, so a docstring, a
comment and a type annotation are all invisible to it by construction.

Two magnitude thresholds, and the ground for each
-------------------------------------------------
``_ADDITIVE_MAGNITUDE = 1e4``. The additive predicate fires on a factor whose
magnitude is at least ``1e4``. That is the smallest value any masking site in
this tree uses for the purpose (``layers/moe/gating.py`` and the pooling family
converged on ``-1e4`` independently), and it is four orders of magnitude above
any ordinary activation scale, so an arithmetic expression reaching it is a
sentinel and not a coefficient. Sign is NOT part of the test: the magnitude is
taken absolute, because ``sequence_pooling.py:596`` masks a ``keras.ops.min``
reduction with ``+1e9`` -- the same hazard mirrored, and a negative-only
predicate would be blind to it.

``_SENTINEL_MAGNITUDE = 1e3``. The population predicate fires on a negative
numeric literal at or below ``-1e3``. The live values in this tree span
``-1024.0``, ``-1e4``, ``-9984``, ``-60000``, ``-1e8`` and ``-1e9``; the floor
is set just under the smallest of them (sam2's ``NO_OBJ_SCORE = -1024.0``) so
none escapes. Values derived from ``numpy.finfo(dtype).min`` at runtime carry no
literal and are invisible to this predicate; the two such sites are allowlisted
anyway, so a rename still trips the allowlist rot check.

Name resolution is deliberately file-flat
-----------------------------------------
An indirected sentinel (``NEG = -1e9`` at module scope, or
``neg_inf = -1e4 if ... else -1e9`` inside a method) is resolved by collecting
every numeric binding in the file into one flat table, with no regard for scope.
That over-approximates: a name bound to a large constant ANYWHERE in the file
counts wherever it appears. The bias is toward false positives, which are loud
and are fixed by allowlisting with a reason, and never toward blindness, which
is silent. That direction is the whole point.
"""

import ast
import pathlib
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import pytest

import dl_techniques

# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

#: The package tree the census covers. Derived from the imported package rather
#: than from this file's own location, so it follows `pyproject.toml`'s
#: `pythonpath = ["src"]` to whichever tree is actually under test.
SRC_ROOT = pathlib.Path(dl_techniques.__file__).resolve().parent

#: Measured 803 at the plan's base commit. The floor is a vacuity guard: a walk
#: that visits zero files satisfies every "no violations" assertion in this file.
MINIMUM_FILES_WALKED = 700

_ADDITIVE_MAGNITUDE = 1e4
_SENTINEL_MAGNITUDE = 1e3

#: Names that, on their own, denote the complement of a keep-mask. A factor
#: bearing one of these is treated as a `(1 - mask)` even when the subtraction
#: happened on an earlier line.
_COMPLEMENT_NAME_FRAGMENTS = (
    "inv_mask",
    "inverse_mask",
    "mask_complement",
    "complement_mask",
    "not_mask",
    "one_minus",
    "neg_mask",
    "anti_mask",
)


# ---------------------------------------------------------------------------
# The allowlist
# ---------------------------------------------------------------------------
#
# Keyed by (repo-relative path under `src/dl_techniques/`, enclosing symbol),
# never by line number. A line-keyed allowlist is wrong within a week: this
# repository measured 78% of line-cited anchors drifting after two steps, while
# an enclosing symbol survives every edit that does not rename it -- and a
# rename is exactly the event that SHOULD force a re-read of the reason.
#
# Every entry is a site that legitimately holds a LOCAL large-magnitude value
# and must NOT be converted. Each carries one written reason, asserted non-empty
# by `TestTheAllowlistItselfIsWellFormed`.

SENTINEL_ALLOWLIST: Dict[Tuple[str, str], str] = {
    # -- Verified safe: the value is chosen per dtype, or the form is selection.
    ("layers/moe/gating.py", "_mask_neg_inf"): (
        "dtype-branched helper: returns -1e4 under float16/bfloat16 and -1e9 "
        "otherwise, so the value is always finite in the dtype it is "
        "materialized in. Correct as written and deliberately not edited -- "
        "editing it would pull tests/test_layers/test_moe/ (451 tests, real "
        "multi-epoch model.fit loops) into a defect-scoped plan."
    ),
    ("layers/attention/lighthouse_attention.py", "<module>"): (
        "_MASK_SENTINEL is a per-dtype table; every entry is finite in its own "
        "dtype (float16 -> -6.0e4 < 65504). Its bfloat16 entry (-1e9) "
        "disagrees with gating.py's (-1e4), but bfloat16 carries float32's "
        "exponent range so neither overflows; the claim that lighthouse is 'the "
        "numerically correct one' is UNSUPPORTED by any measurement in this "
        "tree (decisions.md D-002)."
    ),
    ("layers/attention/lighthouse_attention.py", "_mask_value"): (
        "the lookup function reading _MASK_SENTINEL, with -1e9 as its float32 "
        "fallback. Same reason as the table above."
    ),
    ("layers/attention/ideogram4_attention.py", "<module>"): (
        "_MASK_NEG = -1e9 is consumed only through keras.ops.where with an "
        "explicit cast, never multiplied, so an fp16 -inf reaches only the "
        "positions it was meant to. The file's own anchor records the measured "
        "fp16 softmax that proves it."
    ),
    ("layers/attention/attention_routing_capsule.py", "AttentionRoutingCapsule._apply_top_k_mask"): (
        "where-form top-k mask. At least k entries are always kept, so a fully "
        "-inf row cannot occur and the fp16 overflow is provably harmless here. "
        "The site carries its own WHAT-NOT-TO-DO anchor against the additive "
        "rewrite."
    ),
    ("layers/attention/progressive_focused_attention.py", "ProgressiveFocusedAttention._apply_sparsity"): (
        "where-form, inside the documented-inert 'threshold' sparsity branch. "
        "Selection, not multiplication, so the additive hazard cannot arise."
    ),
    ("layers/attention/common.py", "<module>"): (
        "MASK_BIAS_VALUE = -1e9 is THE canonical constant, and it is contracted "
        "to be used only inside a mask_dtype(...) chain (>= float32), never in "
        "the raw compute dtype. Its own anchor forbids the (1 - keep) * "
        "MASK_BIAS_VALUE form explicitly."
    ),
    ("layers/blt_blocks.py", "PatchPooling._max_pooling"): (
        "where-form (ops.where(mask, hiddens, -1e9)) so the fp16 -inf is never "
        "multiplied and cannot NaN a kept position. NOT in the plan's findings "
        "inventory -- discovered by this census at step 3. FLAGGED, NOT FIXED: "
        "under real float16 an all-masked patch yields max = -inf, finite "
        "garbage rather than NaN, which is a separate and lesser hazard than "
        "the one this plan closes. A future plan should route it through "
        "utils.dtype_policy.mask_sentinel."
    ),
    ("models/vision_language/sam/sam2/model.py", "<module>"): (
        "NO_OBJ_SCORE = -1024.0 is a TRAINED constant, not a maskable sentinel: "
        "the memory encoder's sigmoid(x) * 20 - 10 saturates against it and the "
        "loss side pins the same number. It is deliberately representable in "
        "float16, so the suppression survives mixed_float16 unchanged. "
        "Replacing it would change what the model was trained against."
    ),
    ("models/vision_language/sam/sam1/training_model.py", "<module>"): (
        "OUTSIDE_REGION_LOGIT = -1e9 feeds keras.random.categorical, where -inf "
        "is the WRONG value: an all--inf row returns NaN-driven garbage, while "
        "an all-LOW row degenerates to a uniform draw that EMPTY_REGION_LABEL "
        "then neutralizes. A dtype-safe sentinel would not preserve that."
    ),
    ("models/vision_language/sd3_mmdit/text_encoders.py", "<module>"): (
        "derives its sentinel from np.finfo(dtype).min at call time, so it is "
        "dtype-safe by construction and carries no literal for this census to "
        "see. Allowlisted so a rename still trips the rot check. "
        "Mechanistically DIFFERENT from non_local_attention.py despite the "
        "shared finfo.min origin: this one is a where-form selection."
    ),
    ("layers/attention/non_local_attention.py", "<module>"): (
        "also finfo(dtype).min-derived, but the other mechanism: an additive "
        "ops.maximum clamp at finfo.min / 2.0 (D-015), not a selection. Both "
        "are dtype-safe by construction; they are listed separately because "
        "collapsing them into one 'finfo.min site' would lose that distinction."
    ),
    ("optimization/sled_supervision.py", "sled_builder"): (
        "a config-default fallback (inactive_logit_value, -1e9) threaded into a "
        "builder, not a live tensor bias. Nothing multiplies it here."
    ),
    ("models/common/power_sampling/sampler.py", "PowerSampler._sample_token"): (
        "plain Python: working_logits is a list and the assignment is a list "
        "item mutation during sampling. No Keras tensor, no compute dtype, so "
        "no dtype policy applies."
    ),
    ("utils/alignment/metrics.py", "_compute_nearest_neighbors"): (
        "-1e8 diagonal exclusion in a CKA/kNN metric. FLAGGED, NOT FIXED: -1e8 "
        "DOES overflow to -inf under real float16 and this site is genuinely "
        "unsafe there. It sits outside this plan's named defect set (an "
        "analysis metric, not a forward path), and is recorded here rather than "
        "silently converted so the next plan finds the hazard already measured "
        "instead of re-deriving it."
    ),
    # -- Verified NOT a mask sentinel at all.
    ("utils/tensors.py", "<module>"): (
        "-1259.1392167224028 is a Lanczos approximation coefficient in the "
        "gamma-function series, not a mask sentinel. It is caught only because "
        "the population predicate is a magnitude filter, and is listed so the "
        "filter can stay simple rather than grow a special case."
    ),
    # -- Verified safe: already the where-form with a float16-representable value.
    ("layers/sequence_pooling/attention_pooling.py", "AttentionPooling.call"): (
        "already the target idiom: ops.where with a cast -1e4, plus its own "
        "anchor forbidding a revert to the additive -1e9 form."
    ),
    ("layers/sequence_pooling/weighted_pooling.py", "WeightedPooling.call"): (
        "already the target idiom: ops.where with a cast -1e4 over a "
        "broadcast-to-mask-shape weight vector."
    ),
    ("layers/sequence_pooling/sequence_pooling.py", "SequencePooling._select_top_k"): (
        "already the target idiom: ops.where with a cast -1e4 on the top-k norm "
        "scores."
    ),
    ("models/language/colbert/components.py", "<module>"): (
        "DEFAULT_MAXSIM_MASK_VALUE = -1e4 is exactly representable in float16 "
        "(binary16 spacing at 1e4 is 8, and 10000 / 8 = 1250), is consumed via "
        "where-form, and is covered by dedicated mixed_float16 and XLA tests."
    ),
    ("models/language/tree_transformer/components.py", "GroupAttention.call"): (
        "the where-form sibling of the TreeMHA.call defect: dtype-branched to "
        "-1e4 under float16 AND selection rather than multiplication. Safe on "
        "both counts."
    ),
    # -- Sites this plan is fixing. Present so the census reports only NEW
    #    problems while steps 4-6 are in flight; each entry names the step that
    #    removes the constant, and the additive-form census (which has NO
    #    allowlist) is what actually holds them to account.
    ("layers/heads/nlp/factory.py", "BaseNLPHead._pool_sequence"): (
        "ADDITIVE-FORM DEFECT, converted by step 4 of "
        "plan-2026-08-31T134711-6271592d. Allowlisted for the population "
        "predicate only; TestTheAdditiveMaskBiasFormIsExtinct has no allowlist "
        "and fails on this site until the conversion lands."
    ),
    ("layers/heads/nlp/factory.py", "QuestionAnsweringHead.call"): (
        "ADDITIVE-FORM DEFECT (start and end logits), converted by step 4. Same "
        "accounting as above. The head is NOT dead code: it is bound in the "
        "factory dispatch table for QUESTION_ANSWERING and SPAN_EXTRACTION."
    ),
    ("layers/sequence_pooling/sequence_pooling.py", "SequencePooling._apply_single_strategy"): (
        "ADDITIVE-FORM DEFECT in the 'max' and 'min' strategies, converted by "
        "step 5. The 'min' strategy needs the POSITIVE sentinel. The same "
        "symbol also holds an already-correct where-form -1e4 in the "
        "'top_k_max' branch, which is why this entry survives step 5."
    ),
    ("models/language/tree_transformer/components.py", "TreeMHA.call"): (
        "ADDITIVE-FORM DEFECT, converted by step 6. Currently dtype-guarded to "
        "-1e4 under float16 and therefore safe in practice; converted for "
        "defect-class closure, not because it is broken."
    ),
    ("models/neural_computer/nam/cell.py", "NAMCell.call"): (
        "ADDITIVE-FORM DEFECT, converted by step 6. Same dtype guard and same "
        "reasoning as TreeMHA.call above."
    ),
    ("utils/masking/factory.py", "apply_mask"): (
        "mask_value: float = -1e9 is a function default on dead, "
        "opposite-polarity code (its 'mask' means positions to MASK, the "
        "inverse of apply_attention_mask's 'keep'). DELETED by step 9; this "
        "entry then becomes inert and the allowlist rot check tolerates it "
        "because the enclosing file survives."
    ),
}


# ---------------------------------------------------------------------------
# Census machinery
# ---------------------------------------------------------------------------


class Site(NamedTuple):
    """One census hit.

    :ivar path: repo-relative path under ``src/dl_techniques/``.
    :ivar lineno: 1-based source line.
    :ivar symbol: dotted enclosing class/function path, or ``"<module>"``.
    :ivar detail: short human-readable description of what matched.
    """

    path: str
    lineno: int
    symbol: str
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.lineno}  [{self.symbol}]  {self.detail}"


def _numeric_value(node: ast.AST) -> Optional[float]:
    """Return the float value of a numeric literal, unary-negated or not.

    ``bool`` is excluded: ``True`` is an ``int`` in Python and is never a
    sentinel.

    :param node: any AST node.
    :return: the value, or ``None`` if the node is not a numeric literal.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return None
        if isinstance(node.value, (int, float)):
            return float(node.value)
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _numeric_value(node.operand)
        return None if inner is None else -inner
    return None


def _numeric_bindings(tree: ast.AST) -> Dict[str, float]:
    """Collect ``name -> largest bound magnitude`` over the whole module.

    Flat by design (see the module docstring): no scope tracking, so a name
    bound to a large constant anywhere in the file counts everywhere. Handles
    the two indirection shapes this tree actually uses -- a plain assignment
    (``_MASK_NEG = -1e9``) and a dtype-branching conditional
    (``neg_inf = -1e4 if ... else -1e9``) -- plus dict literals, whose values
    carry the per-dtype sentinel tables.

    :param tree: a parsed module.
    :return: mapping from bound name to the largest magnitude it ever holds.
    """
    bindings: Dict[str, float] = {}

    def _record(name: str, value: Optional[float]) -> None:
        if value is not None:
            bindings[name] = max(bindings.get(name, 0.0), abs(value))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue

        if isinstance(value, ast.IfExp):
            candidates: List[ast.AST] = [value.body, value.orelse]
        elif isinstance(value, ast.Dict):
            candidates = list(value.values)
        else:
            candidates = [value]

        for target in targets:
            if isinstance(target, ast.Name):
                for candidate in candidates:
                    _record(target.id, _numeric_value(candidate))

    return bindings


def _bare_name(node: ast.AST) -> Optional[str]:
    """Return the identifier of a ``Name`` or the attribute of an ``Attribute``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_mask_complement(node: ast.AST) -> bool:
    """Is this factor the complement of a keep-mask?

    Two shapes count: a literal ``1 - <anything>`` subtraction (the spelling all
    seven known defects use), and a bare name whose identifier announces itself
    as a complement.

    :param node: the candidate factor of a multiplication.
    :return: ``True`` if the factor denotes ``1 - mask``.
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        if _numeric_value(node.left) == 1.0:
            return True
    name = _bare_name(node)
    return bool(name) and any(
        fragment in name.lower() for fragment in _COMPLEMENT_NAME_FRAGMENTS
    )


def _is_large_magnitude_factor(
    node: ast.AST, bindings: Dict[str, float]
) -> Optional[str]:
    """Is this factor a sentinel-sized number, directly or by indirection?

    :param node: the candidate factor of a multiplication.
    :param bindings: the file-flat name table from :func:`_numeric_bindings`.
    :return: a short description of what matched, or ``None``.
    """
    value = _numeric_value(node)
    if value is not None:
        return f"literal {value:g}" if abs(value) >= _ADDITIVE_MAGNITUDE else None

    name = _bare_name(node)
    if name is not None:
        magnitude = bindings.get(name, 0.0)
        if magnitude >= _ADDITIVE_MAGNITUDE:
            return f"name {name!r} bound to magnitude {magnitude:g}"
    return None


class _SymbolTracker(ast.NodeVisitor):
    """Base visitor maintaining a dotted enclosing-symbol path."""

    def __init__(self) -> None:
        self._stack: List[str] = []

    @property
    def symbol(self) -> str:
        return ".".join(self._stack) or "<module>"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]


def find_additive_mask_bias(source: str, path: str = "<synthetic>") -> List[Site]:
    """Find every multiplication of a mask complement by a sentinel-sized number.

    This is the forbidden form. Detecting the multiplication -- rather than the
    surrounding ``+`` or ``+=`` -- catches all three spellings at once
    (``x * m + (1 - m) * K``, ``x + (1 - m) * K``, ``x + inv_mask * K``) and is
    insensitive to how the result is subsequently combined.

    :param source: Python source text.
    :param path: label used in the returned sites.
    :return: one :class:`Site` per matching multiplication, ordered by line.
    :raises SyntaxError: if ``source`` does not parse.
    """
    tree = ast.parse(source)
    bindings = _numeric_bindings(tree)
    found: List[Site] = []

    class _Visitor(_SymbolTracker):
        def visit_BinOp(self, node: ast.BinOp) -> None:
            if isinstance(node.op, ast.Mult):
                for complement, factor in (
                    (node.left, node.right),
                    (node.right, node.left),
                ):
                    if not _is_mask_complement(complement):
                        continue
                    detail = _is_large_magnitude_factor(factor, bindings)
                    if detail is not None:
                        found.append(
                            Site(
                                path,
                                node.lineno,
                                self.symbol,
                                f"additive mask bias: mask complement * {detail}",
                            )
                        )
                        break
            self.generic_visit(node)

    _Visitor().visit(tree)
    return sorted(set(found))


def find_large_negative_constants(
    source: str, path: str = "<synthetic>"
) -> List[Site]:
    """Find every negative numeric literal at or below ``-_SENTINEL_MAGNITUDE``.

    This is the population predicate: it measures how many places in the tree
    still hold a local sentinel-sized value, whatever form consumes it. Values
    computed at runtime (``numpy.finfo(dtype).min``) carry no literal and are
    invisible here by construction.

    :param source: Python source text.
    :param path: label used in the returned sites.
    :return: one :class:`Site` per matching literal, ordered by line.
    :raises SyntaxError: if ``source`` does not parse.
    """
    tree = ast.parse(source)
    found: List[Site] = []

    class _Visitor(_SymbolTracker):
        def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
            if isinstance(node.op, ast.USub):
                value = _numeric_value(node.operand)
                if value is not None and value >= _SENTINEL_MAGNITUDE:
                    found.append(
                        Site(
                            path,
                            node.lineno,
                            self.symbol,
                            f"large negative literal {-value:g}",
                        )
                    )
            self.generic_visit(node)

    _Visitor().visit(tree)
    return sorted(set(found))


def _iter_source_files() -> Iterator[Tuple[str, str]]:
    """Yield ``(repo-relative path, source text)`` for every module under the tree."""
    for path in sorted(SRC_ROOT.rglob("*.py")):
        yield path.relative_to(SRC_ROOT).as_posix(), path.read_text(encoding="utf-8")


def _census(finder) -> Tuple[List[Site], int]:
    """Run ``finder`` over the whole package tree.

    :param finder: one of the two module-level predicates.
    :return: ``(all sites found, number of files walked)``.
    """
    sites: List[Site] = []
    walked = 0
    for relative_path, source in _iter_source_files():
        walked += 1
        sites.extend(finder(source, relative_path))
    return sites, walked


def _render(sites: List[Site]) -> str:
    return "\n".join(f"  {site}" for site in sites)


# ---------------------------------------------------------------------------
# Anti-vacuity: prove the instrument can see, and can decline to see
# ---------------------------------------------------------------------------

_POSITIVE_SOURCE = """
import keras


def masked_pool(logits, mask, inv_mask, mask_expanded):
    NEG = -1e9
    a = logits * mask + (1 - mask) * -1e9
    b = logits + (1.0 - mask_expanded) * 1e9
    c = logits + inv_mask * NEG
    return a, b, c
"""

_CLEAN_SOURCE = """
import keras

from dl_techniques.utils.dtype_policy import mask_sentinel


def masked_pool(logits, keep):
    sentinel = keras.ops.cast(mask_sentinel(logits.dtype), logits.dtype)
    scaled = logits * 1.0
    return keras.ops.where(keep, scaled, sentinel)
"""


class TestTheCensusInstrumentIsNotVacuous:
    """A census that cannot see anything reports zero violations forever.

    This class is the reason the rest of the file means something. It is
    two-sided on purpose: an instrument that fires on everything is exactly as
    useless as one that fires on nothing, and only the pair of arms distinguishes
    a working detector from either failure.
    """

    def test_the_additive_predicate_fires_on_a_synthetic_positive(self) -> None:
        sites = find_additive_mask_bias(_POSITIVE_SOURCE, "<positive>")
        assert len(sites) >= 1, (
            "the additive predicate found nothing in a source written to "
            "contain the defect three times; it is blind and every other "
            "assertion in this file is vacuous"
        )
        # All three spellings, including the indirected one, must be seen.
        assert len(sites) == 3, (
            "expected all three forbidden spellings (literal, positive-mirror, "
            f"name-indirected); saw {len(sites)}:\n{_render(sites)}"
        )

    def test_the_additive_predicate_is_silent_on_a_synthetic_clean_source(
        self,
    ) -> None:
        sites = find_additive_mask_bias(_CLEAN_SOURCE, "<clean>")
        assert sites == [], (
            "the additive predicate fired on the ops.where form this plan is "
            f"migrating TO; it would reject every correct fix:\n{_render(sites)}"
        )

    def test_the_population_predicate_fires_on_a_synthetic_positive(self) -> None:
        sites = find_large_negative_constants(_POSITIVE_SOURCE, "<positive>")
        assert len(sites) >= 1, (
            "the population predicate found no large negative literal in a "
            "source containing two of them"
        )

    def test_the_population_predicate_is_silent_on_a_synthetic_clean_source(
        self,
    ) -> None:
        sites = find_large_negative_constants(_CLEAN_SOURCE, "<clean>")
        assert sites == [], _render(sites)

    def test_the_predicates_ignore_prose(self) -> None:
        """A docstring and a comment spelling the defect out must not count.

        This file's own docstring does exactly that, as do five ``# DECISION``
        anchors in the tree. A text scan would rank them as the worst offenders.
        """
        prose = '''
"""x = logits * mask + (1 - mask) * -1e9 is forbidden."""
# Do not write logits + (1.0 - mask) * -1e9 here.
VALUE = 1.0
'''
        assert find_additive_mask_bias(prose, "<prose>") == []
        assert find_large_negative_constants(prose, "<prose>") == []

    @pytest.mark.parametrize(
        "label,snippet",
        [
            (
                "heads/nlp attention pooling",
                "attention_weights = attention_weights * mask + (1 - mask) * -1e9",
            ),
            (
                "heads/nlp QA start logits",
                "start_logits = start_logits * mask + (1 - mask) * -1e9",
            ),
            (
                "sequence_pooling max strategy",
                "masked_inputs = inputs + (1.0 - mask_expanded) * (-1e9)",
            ),
            (
                "sequence_pooling min strategy (POSITIVE sentinel)",
                "masked_inputs = inputs + (1.0 - mask_expanded) * 1e9",
            ),
            (
                "tree_transformer / nam dtype-branched indirection",
                'neg_inf = -1e4 if dt == "float16" else -1e9\n'
                "scores = scores + (1.0 - token_mask_float) * neg_inf",
            ),
        ],
    )
    def test_the_predicate_still_recognizes_each_real_defect_spelling(
        self, label: str, snippet: str
    ) -> None:
        """Pin the five source spellings the seven base-commit sites used.

        These snippets are copied verbatim from the tree at the plan's base
        commit. They outlive the fixes: once the real sites are converted the
        whole-tree census can only stay green, and this parametrization is then
        the only thing standing between a "simplified" predicate and a guard
        that silently sees nothing.
        """
        sites = find_additive_mask_bias(snippet, "<real-spelling>")
        assert len(sites) >= 1, f"predicate no longer recognizes: {label}"

    def test_the_walk_visits_a_realistic_number_of_files(self) -> None:
        _, walked = _census(find_additive_mask_bias)
        assert walked >= MINIMUM_FILES_WALKED, (
            f"walked only {walked} files under {SRC_ROOT}; the tree measured "
            f"803 at the plan's base commit. A short walk makes every "
            f"'no violations' assertion below vacuous"
        )


# ---------------------------------------------------------------------------
# The census proper
# ---------------------------------------------------------------------------


class TestTheAdditiveMaskBiasFormIsExtinct:
    """No allowlist. The additive form has no legitimate use anywhere.

    RED at the plan's base commit with seven sites; green once step 6 lands.
    """

    def test_no_module_multiplies_a_mask_complement_by_a_sentinel(self) -> None:
        sites, walked = _census(find_additive_mask_bias)
        assert sites == [], (
            f"{len(sites)} additive mask-bias site(s) across {walked} files "
            f"under {SRC_ROOT}.\n"
            f"Under mixed_float16 the sentinel overflows to -inf and 0 * -inf "
            f"= NaN poisons every KEPT position, not just the masked ones.\n"
            f"Rewrite as keras.ops.where(keep, x, sentinel) using "
            f"dl_techniques.utils.dtype_policy.mask_sentinel:\n"
            f"{_render(sites)}"
        )


class TestTheSentinelPopulationIsClosed:
    """Every local sentinel-sized literal is one somebody wrote a reason for.

    New ones fail here. That is the point: the allowlist is a permission list,
    and adding to it forces an author to state, in prose, why their site needs a
    local value instead of ``utils.dtype_policy.mask_sentinel``.
    """

    def test_every_large_negative_literal_is_allowlisted(self) -> None:
        sites, walked = _census(find_large_negative_constants)
        unlisted = [s for s in sites if (s.path, s.symbol) not in SENTINEL_ALLOWLIST]
        assert unlisted == [], (
            f"{len(unlisted)} un-allowlisted large-negative literal(s) across "
            f"{walked} files.\nIf this is a mask sentinel, route it through "
            f"dl_techniques.utils.dtype_policy.mask_sentinel and consume it with "
            f"keras.ops.where.\nIf it legitimately needs a local value, add "
            f"(path, enclosing symbol) to SENTINEL_ALLOWLIST WITH A WRITTEN "
            f"REASON:\n{_render(unlisted)}"
        )


class TestTheAllowlistItselfIsWellFormed:
    """An allowlist nobody validates is how a guard goes blind politely."""

    def test_every_entry_carries_a_written_reason(self) -> None:
        empty = [key for key, reason in SENTINEL_ALLOWLIST.items() if not reason.strip()]
        assert empty == [], f"allowlist entries with no reason: {empty}"

    def test_every_reason_is_a_sentence_not_a_shrug(self) -> None:
        """A reason has to carry information, not just be non-empty."""
        too_short = {
            key: reason
            for key, reason in SENTINEL_ALLOWLIST.items()
            if len(reason.split()) < 8
        }
        assert too_short == {}, (
            f"allowlist reasons too short to explain anything: {list(too_short)}"
        )

    def test_every_allowlisted_file_still_exists(self) -> None:
        """Rot check: a rename must surface as a failure, not as silence."""
        missing = sorted(
            {path for path, _ in SENTINEL_ALLOWLIST if not (SRC_ROOT / path).is_file()}
        )
        assert missing == [], (
            f"allowlisted files that no longer exist under {SRC_ROOT}: {missing}. "
            f"A moved file takes its exemption with it and the census goes "
            f"blind to whatever replaced it"
        )

    def test_every_allowlisted_symbol_still_exists_in_its_file(self) -> None:
        """A renamed class or method must force its reason to be re-read."""
        unknown: List[str] = []
        for path, symbol in sorted(SENTINEL_ALLOWLIST):
            source_path = SRC_ROOT / path
            if not source_path.is_file():
                continue  # already reported by the file-level rot check
            if symbol == "<module>":
                continue
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            names = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
            if not all(part in names for part in symbol.split(".")):
                unknown.append(f"{path}::{symbol}")
        assert unknown == [], (
            f"allowlisted symbols not found in their files: {unknown}"
        )
