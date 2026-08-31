"""Repo-wide census: the additive mask-bias form is extinct, and every local
large-magnitude mask sentinel that remains is one somebody wrote a reason for.

.. note::

   **History.** This file was written BEFORE the fixes it guards, deliberately,
   so that the real unfixed tree was its RED proof. At the plan's base commit
   ``043a59a52`` the additive-form census reported exactly seven violations::

       layers/heads/nlp/factory.py:461                    BaseNLPHead._pool_sequence
       layers/heads/nlp/factory.py:1255                   QuestionAnsweringHead.call
       layers/heads/nlp/factory.py:1256                   QuestionAnsweringHead.call
       layers/sequence_pooling/sequence_pooling.py:588    SequencePooling._apply_single_strategy
       layers/sequence_pooling/sequence_pooling.py:596    SequencePooling._apply_single_strategy
       models/language/tree_transformer/components.py:508 TreeMHA.call
       models/neural_computer/nam/cell.py:457             NAMCell.call

   All seven are converted. That list is history, not an expectation: the
   assertion is ``== 0``, and the file is NOT marked ``xfail`` or ``skip``. A
   guard that cannot go red is worthless, and a guard permanently excused from
   going green is worse.

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
above live outside that directory. This file walks the whole of ``src/``. It
follows the house convention for repo-wide closure guards
(``tests/test_the_shared_initializer_population_cannot_grow.py``).

What the walk covers, and why it is all of ``src/``
---------------------------------------------------
The first version of this file walked ``src/dl_techniques/`` only, because
``SRC_ROOT`` was derived from ``dl_techniques.__file__``. The attack-before-release
review measured what that hid: **six live additive sites in** ``src/train/nam/``,
in the trainer for the very model (``models/neural_computer/nam/``) whose cell
this plan converted for "defect-class closure". A census whose scope stops one
directory short of the defect it is named after manufactures confidence.

The walk is now ``src/`` entire -- ``dl_techniques/``, ``train/`` and
``applications/`` -- and the paths in every key below are ``src/``-relative. The
two populations are held to DIFFERENT standards, on purpose:

- ``src/dl_techniques/`` (the library) has **no allowlist** for the additive
  form. It is a library: any dtype policy may be set around it by any caller,
  so the form is never defensible there.
- everything else in ``src/`` is held to a **frozen inventory** rather than an
  allowlist: :data:`NON_LIBRARY_ADDITIVE_INVENTORY` is compared by EXACT
  EQUALITY, counts included. A new additive site reddens the guard, and so does
  fixing an inventoried one without updating the inventory. That is strictly
  stronger than a skip-list, which can only ever grow silently.

Why AST and never a text grep
-----------------------------
Every ``# DECISION`` anchor and module docstring in this tree that WARNS about
the additive form spells the additive form out verbatim, as does this docstring.
A textual scan counts those warnings as defects and this file as the worst
offender in the repository. The census walks ``ast`` nodes, so a docstring, a
comment and a type annotation are all invisible to it by construction.

The spellings the additive predicate recognizes
-----------------------------------------------
The review ran 21 hand-written spellings of the defect through the original
predicate and **13 escaped**, four of them invisible to BOTH predicates at
once -- including ``pad = 1.0 - mask`` on one line and ``x + pad * -1e9`` on the
next, which is the most natural way anybody would actually write it. The
predicate now recognizes, and :class:`TestTheCensusInstrumentIsNotVacuous` pins
one positive fixture per row:

- the complement written either way round (``1 - m`` **and** ``m - 1``), negated
  (``-(m - 1)``), or bound to a name on an earlier statement (the two-statement
  split -- a per-scope name-binding pass, described below);
- the complement produced by a call: ``ops.subtract(1.0, m)``,
  ``ops.logical_not(m)``, ``~m``, and any of those wrapped in ``ops.cast``;
- the multiplication written as a call (``ops.multiply(1 - m, -1e9)``) as well as
  with the ``*`` operator, and either operand order;
- the sentinel written as a literal, negated, built by arithmetic (``-10 ** 9``),
  parsed from a string (``float("-1e9")``), bound to a plain name, to a
  ``self.``/class attribute, through a tuple unpack (``NEG, POS = -1e9, 1e9``),
  through a dtype-branching ternary, or returned by a zero-argument helper.

Deliberate blind spots are ENUMERATED, not silent: see :data:`KNOWN_UNCAUGHT`,
each entry carrying its reason, and
:meth:`TestTheCensusInstrumentIsNotVacuous.test_each_known_blind_spot_is_still_blind`
which fails if one is quietly fixed -- so the list can only shrink on purpose.

Name resolution is deliberately file-flat, with one scoped exception
--------------------------------------------------------------------
An indirected SENTINEL (``NEG = -1e9`` at module scope, or
``neg_inf = -1e4 if ... else -1e9`` inside a method) is resolved by collecting
every numeric binding in the file into one flat table, with no regard for scope.
That over-approximates: a name bound to a large constant ANYWHERE in the file
counts wherever it appears. The bias is toward false positives, which are loud
and are fixed by allowlisting with a reason, and never toward blindness, which
is silent.

An indirected COMPLEMENT (``pad = 1.0 - mask``) is resolved per SCOPE instead --
the enclosing function's own assignments plus the module's. A short local name
like ``pad``, ``inv`` or ``m2`` is far likelier to be reused for something else
in a different method of the same file than a shouty module constant is, and a
file-flat complement table would start reporting unrelated multiplications. The
asymmetry is the point: over-approximate the rare shape, scope the common one.

Two magnitude thresholds, and the ground for each
-------------------------------------------------
``_ADDITIVE_MAGNITUDE = 1e4``. The additive predicate fires on a factor whose
magnitude is at least ``1e4``. That is the smallest value any masking site in
this tree uses for the purpose (``layers/moe/gating.py`` and the pooling family
converged on ``-1e4`` independently), and it is four orders of magnitude above
any ordinary activation scale, so an arithmetic expression reaching it is a
sentinel and not a coefficient. Sign is NOT part of this test: the magnitude is
taken absolute, because ``sequence_pooling.py`` masked a ``keras.ops.min``
reduction with ``+1e9`` -- the same hazard mirrored, and a negative-only
predicate would be blind to it, as it was to ``x + (mask - 1.0) * 1e9``.

``_SENTINEL_MAGNITUDE = 1e3``. The population predicate fires on a negative
numeric literal at or below ``-1e3``. The live values in this tree span
``-1024.0``, ``-1e4``, ``-9984``, ``-60000``, ``-1e8`` and ``-1e9``; the floor
is set just under the smallest of them (sam2's ``NO_OBJ_SCORE = -1024.0``) so
none escapes. Values derived from ``numpy.finfo(dtype).min`` at runtime carry no
literal and are invisible to this predicate; the two such sites are allowlisted
anyway, so a rename still trips the allowlist rot check.

The population predicate stays NEGATIVE-only, against the review's
recommendation to make it symmetric. Measured: symmetry would report every
positive literal at or above ``1e3`` in ``src/`` -- vocabulary sizes, buffer
lengths, pixel counts, timeouts, ``stderr[-2000:]`` slice bounds -- hundreds of
hits with nothing to do with masking, and an allowlist that large is a silencer,
not a permission list. The one place sign genuinely matters is the additive
predicate, which is already symmetric.

Non-finite values are excluded from both predicates. ``-float('inf')`` is not a
magnitude that a narrow dtype may fail to represent: it is ``-inf`` in every
float dtype by construction, which is the opposite of the hazard here, and it is
a deliberate idiom (``layers/attention/ring_attention.py`` seeds an online-softmax
running max with it). Folding it in would report that identity as a sentinel.
"""

import ast
import pathlib
from typing import Dict, FrozenSet, Iterator, List, NamedTuple, Optional, Set, Tuple

import math

import pytest

import dl_techniques

# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

#: The library package. Derived from the imported package rather than from this
#: file's own location, so it follows `pyproject.toml`'s `pythonpath = ["src"]`
#: to whichever tree is actually under test.
PACKAGE_ROOT = pathlib.Path(dl_techniques.__file__).resolve().parent

#: Everything the census walks: `src/`, i.e. the library plus `train/` and
#: `applications/`. Every path key in this file is relative to THIS root, so the
#: library's own paths carry a `dl_techniques/` prefix.
SRC_ROOT = PACKAGE_ROOT.parent

#: The library subtree, as a `SRC_ROOT`-relative prefix.
LIBRARY_PREFIX = PACKAGE_ROOT.name + "/"

#: Measured 1052 files under `src/` (803 library, 240 train, 9 applications) at
#: completion fix 3.1. The floor is a vacuity guard: a walk that visits zero
#: files satisfies every "no violations" assertion in this file.
MINIMUM_FILES_WALKED = 950

#: Measured 803 at the plan's base commit; the library half of the walk carries
#: its own floor so that a broken `dl_techniques` import cannot be masked by a
#: healthy `train/` walk.
MINIMUM_LIBRARY_FILES_WALKED = 700

_ADDITIVE_MAGNITUDE = 1e4
_SENTINEL_MAGNITUDE = 1e3

#: Guard on constant folding: refuse to materialize an absurd `10 ** 5000`.
_MAX_FOLD_MAGNITUDE = 1e300

#: Names that, on their own, denote the complement of a keep-mask. A factor
#: bearing one of these is treated as a `(1 - mask)` even when the subtraction
#: happened on an earlier line and in another scope.
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

#: Calls whose value is the numeric value of their first argument (or of the
#: string they parse). `float("-1e9")` is the shape the review found invisible.
_NUMERIC_CALLS = frozenset(
    {
        "float",
        "int",
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "array",
        "asarray",
        "convert_to_tensor",
        "constant",
        "cast",
    }
)

#: Calls that PRODUCE a mask complement.
_COMPLEMENT_CALLS = frozenset({"subtract", "logical_not", "bitwise_not"})

#: Calls that pass their first argument through unchanged for the purpose of
#: "is this a mask complement?" -- `ops.cast(ops.logical_not(m), "float32")`.
_PASSTHROUGH_CALLS = frozenset(
    {
        "cast",
        "convert_to_tensor",
        "constant",
        "array",
        "asarray",
        "expand_dims",
        "squeeze",
        "reshape",
        "stop_gradient",
    }
)

#: Calls that ARE a multiplication. `ops.multiply(1 - m, -1e9)` never reaches a
#: `visit_BinOp`, which is why the review's spelling 17 escaped entirely.
_MULTIPLY_CALLS = frozenset({"multiply", "mul"})


# ---------------------------------------------------------------------------
# Enumerated blind spots
# ---------------------------------------------------------------------------
#
# An honest, enumerated blind spot is acceptable; a silent one is not. Each row
# is a spelling of the defect this predicate deliberately does NOT catch, with
# the reason. `test_each_known_blind_spot_is_still_blind` asserts each is still
# uncaught, so a future hardening that closes one MUST delete its row here --
# the list can shrink, but only on purpose.

KNOWN_UNCAUGHT: Dict[str, Tuple[str, str]] = {
    "sentinel imported from another module": (
        'from . import consts\nx = logits + (1.0 - mask) * consts.NEG',
        "the name table is built from ONE parsed file; resolving `consts.NEG` "
        "would mean a whole-program import graph and an execution model for "
        "`__init__.py` re-exports. Partially mitigated: the DEFINITION "
        "`NEG = -1e9` is itself a large negative literal, so the population "
        "predicate reports it in the module that owns it, where an author must "
        "allowlist it with a reason.",
    ),
    "sentinel derived from finfo at runtime": (
        'x = logits + (1.0 - mask) * (-np.finfo("float32").max)',
        "there is no literal to see, and this shape is the SAFE one: a "
        "finfo-derived magnitude is representable in its own dtype by "
        "construction, which is precisely what `mask_sentinel` does. Catching "
        "it would mean reporting the recommended fix as the defect. The two "
        "in-tree finfo.min sites are allowlisted anyway so a rename still trips "
        "the rot check.",
    ),
    "complement produced by a project-local helper call": (
        "x = logits + complement_of(mask) * -1e9",
        "deciding that an arbitrary single-argument call returns `1 - mask` "
        "needs interprocedural analysis. A zero-argument helper returning a "
        "SENTINEL is caught (its return value is folded into the name table); a "
        "helper returning a MASK is not. Mitigated by the name-fragment list: "
        "`inv_mask`, `not_mask`, `one_minus` and friends are recognized "
        "wherever they appear, and by the population predicate seeing the "
        "`-1e9`.",
    ),
}


# ---------------------------------------------------------------------------
# The allowlist (population predicate only)
# ---------------------------------------------------------------------------
#
# Keyed by (`src/`-relative path, enclosing symbol), never by line number. A
# line-keyed allowlist is wrong within a week: this repository measured 78% of
# line-cited anchors drifting after two steps, while an enclosing symbol
# survives every edit that does not rename it -- and a rename is exactly the
# event that SHOULD force a re-read of the reason.
#
# Every entry is a site that legitimately holds a LOCAL large-magnitude value
# and must NOT be converted. Each carries one written reason, asserted non-empty
# by `TestTheAllowlistItselfIsWellFormed`.

SENTINEL_ALLOWLIST: Dict[Tuple[str, str], str] = {
    # -- Verified safe: the value is chosen per dtype, or the form is selection.
    ("dl_techniques/layers/moe/gating.py", "_mask_neg_inf"): (
        "dtype-branched helper: returns -1e4 under float16/bfloat16 and -1e9 "
        "otherwise, so the value is always finite in the dtype it is "
        "materialized in. Correct as written and deliberately not edited -- "
        "editing it would pull tests/test_layers/test_moe/ (451 tests, real "
        "multi-epoch model.fit loops) into a defect-scoped plan."
    ),
    ("dl_techniques/layers/attention/lighthouse_attention.py", "<module>"): (
        "_MASK_SENTINEL is a per-dtype table; every entry is finite in its own "
        "dtype (float16 -> -6.0e4 < 65504). Its bfloat16 entry (-1e9) "
        "disagrees with gating.py's (-1e4), but bfloat16 carries float32's "
        "exponent range so neither overflows; the claim that lighthouse is 'the "
        "numerically correct one' is UNSUPPORTED by any measurement in this "
        "tree (decisions.md D-002)."
    ),
    ("dl_techniques/layers/attention/lighthouse_attention.py", "_mask_value"): (
        "the lookup function reading _MASK_SENTINEL, with -1e9 as its float32 "
        "fallback. Same reason as the table above."
    ),
    ("dl_techniques/layers/attention/ideogram4_attention.py", "<module>"): (
        "_MASK_NEG = -1e9 is consumed only through keras.ops.where with an "
        "explicit cast, never multiplied, so an fp16 -inf reaches only the "
        "positions it was meant to. The file's own anchor records the measured "
        "fp16 softmax that proves it."
    ),
    (
        "dl_techniques/layers/attention/attention_routing_capsule.py",
        "AttentionRoutingCapsule._apply_top_k_mask",
    ): (
        "where-form top-k mask. At least k entries are always kept, so a fully "
        "-inf row cannot occur and the fp16 overflow is provably harmless here. "
        "The site carries its own WHAT-NOT-TO-DO anchor against the additive "
        "rewrite."
    ),
    (
        "dl_techniques/layers/attention/progressive_focused_attention.py",
        "ProgressiveFocusedAttention._apply_sparsity",
    ): (
        "where-form, inside the documented-inert 'threshold' sparsity branch. "
        "Selection, not multiplication, so the additive hazard cannot arise."
    ),
    ("dl_techniques/layers/attention/common.py", "<module>"): (
        "MASK_BIAS_VALUE is THE canonical constant, and it is contracted to be "
        "used only inside a mask_dtype(...) chain (>= float32), never in the raw "
        "compute dtype. Its own anchor forbids the (1 - keep) * MASK_BIAS_VALUE "
        "form explicitly. Since step 10 of plan-2026-08-31T134711-6271592d the "
        "right-hand side is mask_sentinel('float32') -- still exactly -1e9, "
        "bit-for-bit -- so no literal remains and this census no longer reports "
        "the site. The entry is KEPT so that reintroducing a literal here lands "
        "on a written reason rather than on a fresh decision."
    ),
    ("dl_techniques/layers/blt_blocks.py", "PatchPooling._max_pooling"): (
        "where-form (ops.where(mask, hiddens, -1e9)) so the fp16 -inf is never "
        "multiplied and cannot NaN a kept position. NOT in the plan's findings "
        "inventory -- discovered by this census at step 3. FLAGGED, NOT FIXED: "
        "under real float16 an all-masked patch yields max = -inf, finite "
        "garbage rather than NaN, which is a separate and lesser hazard than "
        "the one this plan closes. A future plan should route it through "
        "utils.dtype_policy.mask_sentinel."
    ),
    ("dl_techniques/models/vision_language/sam/sam2/model.py", "<module>"): (
        "NO_OBJ_SCORE = -1024.0 is a TRAINED constant, not a maskable sentinel: "
        "the memory encoder's sigmoid(x) * 20 - 10 saturates against it and the "
        "loss side pins the same number. It is deliberately representable in "
        "float16, so the suppression survives mixed_float16 unchanged. "
        "Replacing it would change what the model was trained against."
    ),
    (
        "dl_techniques/models/vision_language/sam/sam1/training_model.py",
        "<module>",
    ): (
        "OUTSIDE_REGION_LOGIT = -1e9 feeds keras.random.categorical, where -inf "
        "is the WRONG value: an all--inf row returns NaN-driven garbage, while "
        "an all-LOW row degenerates to a uniform draw that EMPTY_REGION_LABEL "
        "then neutralizes. A dtype-safe sentinel would not preserve that."
    ),
    ("dl_techniques/models/vision_language/sd3_mmdit/text_encoders.py", "<module>"): (
        "derives its sentinel from np.finfo(dtype).min at call time, so it is "
        "dtype-safe by construction and carries no literal for this census to "
        "see. Allowlisted so a rename still trips the rot check. "
        "Mechanistically DIFFERENT from non_local_attention.py despite the "
        "shared finfo.min origin: this one is a where-form selection."
    ),
    ("dl_techniques/layers/attention/non_local_attention.py", "<module>"): (
        "also finfo(dtype).min-derived, but the other mechanism: an additive "
        "ops.maximum clamp at finfo.min / 2.0, not a selection. Both are "
        "dtype-safe by construction; they are listed separately because "
        "collapsing them into one 'finfo.min site' would lose that distinction."
    ),
    ("dl_techniques/optimization/sled_supervision.py", "sled_builder"): (
        "a config-default fallback (inactive_logit_value, -1e9) threaded into a "
        "builder, not a live tensor bias. Nothing multiplies it here."
    ),
    ("dl_techniques/models/common/power_sampling/sampler.py", "PowerSampler._sample_token"): (
        "plain Python: working_logits is a list and the assignment is a list "
        "item mutation during sampling. No Keras tensor, no compute dtype, so "
        "no dtype policy applies."
    ),
    ("dl_techniques/utils/alignment/metrics.py", "_compute_nearest_neighbors"): (
        "-1e8 diagonal exclusion in a CKA/kNN metric. FLAGGED, NOT FIXED: -1e8 "
        "DOES overflow to -inf under real float16 and this site is genuinely "
        "unsafe there. It sits outside this plan's named defect set (an "
        "analysis metric, not a forward path), and is recorded here rather than "
        "silently converted so the next plan finds the hazard already measured "
        "instead of re-deriving it."
    ),
    # -- Verified NOT a mask sentinel at all.
    ("dl_techniques/utils/tensors.py", "<module>"): (
        "-1259.1392167224028 is a Lanczos approximation coefficient in the "
        "gamma-function series, not a mask sentinel. It is caught only because "
        "the population predicate is a magnitude filter, and is listed so the "
        "filter can stay simple rather than grow a special case."
    ),
    # -- Verified safe: already the where-form with a float16-representable value.
    (
        "dl_techniques/layers/sequence_pooling/attention_pooling.py",
        "AttentionPooling.call",
    ): (
        "already the target idiom: ops.where with a cast -1e4, plus its own "
        "anchor forbidding a revert to the additive -1e9 form."
    ),
    (
        "dl_techniques/layers/sequence_pooling/weighted_pooling.py",
        "WeightedPooling.call",
    ): (
        "already the target idiom: ops.where with a cast -1e4 over a "
        "broadcast-to-mask-shape weight vector."
    ),
    (
        "dl_techniques/layers/sequence_pooling/sequence_pooling.py",
        "SequencePooling._select_top_k",
    ): (
        "already the target idiom: ops.where with a cast -1e4 on the top-k norm "
        "scores."
    ),
    ("dl_techniques/models/language/colbert/components.py", "<module>"): (
        "DEFAULT_MAXSIM_MASK_VALUE = -1e4 is exactly representable in float16 "
        "(binary16 spacing at 1e4 is 8, and 10000 / 8 = 1250), is consumed via "
        "where-form, and is covered by dedicated mixed_float16 and XLA tests."
    ),
    (
        "dl_techniques/models/language/tree_transformer/components.py",
        "GroupAttention.call",
    ): (
        "the where-form sibling of the TreeMHA.call defect: dtype-branched to "
        "-1e4 under float16 AND selection rather than multiplication. Safe on "
        "both counts."
    ),
    # -- Sites this plan fixed. The constants are gone from the ADDITIVE form at
    #    each of them, but the enclosing symbols still hold a where-form literal
    #    or a neighbouring one, so the entries stay and name their history.
    ("dl_techniques/layers/heads/nlp/factory.py", "BaseNLPHead._pool_sequence"): (
        "ADDITIVE-FORM DEFECT, converted by step 4 of "
        "plan-2026-08-31T134711-6271592d. Allowlisted for the population "
        "predicate only; TestTheAdditiveMaskBiasFormIsExtinct has no allowlist "
        "and failed on this site until the conversion landed."
    ),
    ("dl_techniques/layers/heads/nlp/factory.py", "QuestionAnsweringHead.call"): (
        "ADDITIVE-FORM DEFECT (start and end logits), converted by step 4. Same "
        "accounting as above. The head is NOT dead code: it is bound in the "
        "factory dispatch table for QUESTION_ANSWERING and SPAN_EXTRACTION."
    ),
    (
        "dl_techniques/layers/sequence_pooling/sequence_pooling.py",
        "SequencePooling._apply_single_strategy",
    ): (
        "ADDITIVE-FORM DEFECT in the 'max' and 'min' strategies, converted by "
        "step 5. The 'min' strategy needs the POSITIVE sentinel. The same "
        "symbol also holds an already-correct where-form -1e4 in the "
        "'top_k_max' branch, which is why this entry survives step 5."
    ),
    (
        "dl_techniques/models/language/tree_transformer/components.py",
        "TreeMHA.call",
    ): (
        "ADDITIVE-FORM DEFECT, converted by step 6. Was dtype-guarded to -1e4 "
        "under float16 and therefore safe in practice; converted for "
        "defect-class closure, not because it was broken."
    ),
    ("dl_techniques/models/neural_computer/nam/cell.py", "NAMCell.call"): (
        "ADDITIVE-FORM DEFECT, converted by step 6. Same dtype guard and same "
        "reasoning as TreeMHA.call above."
    ),
    # -- Outside the library. Added by completion fix 3.1, when the walk widened
    #    from `src/dl_techniques/` to `src/`. None of these is a masked forward
    #    path under a mixed-precision policy; each says why.
    ("train/nam/train_dfsa.py", "DifferentiableFSA.reduce_step"): (
        "the five additive -1e9 masks inventoried in "
        "NON_LIBRARY_ADDITIVE_INVENTORY; see that entry for the full reason. "
        "Listed here too because the population predicate sees the same "
        "literals and the two predicates are independent by design."
    ),
    ("train/nam/train_dfsa_ste.py", "_make_ste_train_fn._step"): (
        "the STE trainer's consistency-loss mask, inventoried in "
        "NON_LIBRARY_ADDITIVE_INVENTORY; same reason, same file family."
    ),
    ("train/common/generation_probe.py", "GenerationProbeCallback._generate_token_ids"): (
        "logits is a numpy float32 array (`np.asarray(..., dtype='float32').copy()`) "
        "and the -1e9 writes are numpy ITEM ASSIGNMENTS suppressing EOT and "
        "special ids. No Keras tensor and no compute dtype, so no policy can "
        "narrow it; and the ordering is pinned against a test oracle "
        "(_reference_generate) that would have to change with it."
    ),
    ("train/common/timeseries.py", "WindowedTimeSeriesProcessor._safe_normalize"): (
        "np.clip(series, -1e6, 1e6) on the raw INPUT SERIES, in numpy, before "
        "any model sees it. It is a data-sanitation bound, not a mask sentinel: "
        "nothing multiplies it and it never reaches a softmax. Its exact value "
        "reproduces the four upstream processors this class merged and is "
        "pinned by tests/test_train/test_timeseries_base.py."
    ),
    (
        "train/time_series/adaptive_ema/train_adaptive_ema.py",
        "AdaptiveEMADataProcessor._safe_normalize",
    ): (
        "the same numpy np.clip(series, -1e6, 1e6) input bound as "
        "train/common/timeseries.py, in the AdaptiveEMA-specific override of "
        "that method. Same reason: data sanitation in numpy, never a mask."
    ),
    ("train/logic/multiseed_sweep.py", "run_one"): (
        "-2000 is a SLICE BOUND: `cp.stderr[-2000:]` keeps the last 2000 "
        "characters of a failed subprocess's stderr for the log. Not a number "
        "in a tensor at all. Caught because the predicate is a pure magnitude "
        "filter on negative literals, which is a shape it is worth keeping "
        "simple."
    ),
    ("train/rms_variants_train/sweep.py", "run_one"): (
        "-1500 is the same shape as the entry above: `f.read()[-1500:]`, the "
        "tail of a subprocess log file. Not a tensor value."
    ),
    ("train/sam/data.py", "<module>"): (
        "OUTSIDE_MASK_LOGIT = -1e9 is used by the numpy DATA PIPELINE when "
        "drawing a prompt point from a ground-truth mask -- it selects which "
        "pixel to sample, in numpy, before any model or dtype policy exists. "
        "The sibling model-side constant "
        "(models/vision_language/sam/sam1/training_model.py) is allowlisted "
        "separately and for a different reason."
    ),
}


# ---------------------------------------------------------------------------
# The frozen inventory (additive form, outside the library)
# ---------------------------------------------------------------------------
#
# Compared by EXACT EQUALITY including the count, never as a skip-list. Fixing
# one of these, or adding a new one, both redden the guard.

NON_LIBRARY_ADDITIVE_INVENTORY: Dict[Tuple[str, str], Tuple[int, str]] = {
    ("train/nam/train_dfsa.py", "DifferentiableFSA.reduce_step"): (
        5,
        "The DFSA research harness for the NAM model. FLAGGED, NOT CONVERTED, "
        "for three measured reasons. (1) No dtype policy is ever set in "
        "src/train/nam/ -- `grep -n 'mixed_\\|set_global_policy\\|dtype_policy'` "
        "over the directory returns nothing -- and every tensor in reduce_step "
        "is explicitly built in float32 (`ops.cast(..., \"float32\")`), so the "
        "fp16 overflow that makes this form a defect in the library cannot "
        "arise here. (2) The additive form is LOAD-BEARING in this function: "
        "the masks STACK (`scores + (1 - mask) * -1e9 + (1 - is_operator) * "
        "-1e9`) and a subsequent `+ (1 - any_op) * mask * -100.0` rescue adds "
        "to the result, so a where-conversion is not a local rewrite. (3) The "
        "file carries a documented bit-identity invariant against its iter-5 "
        "forward pass ('residual term is bit-zero in float32 -> forward "
        "bit-identical', plan_2026-05-12_995a621a/D-002), and converting to "
        "selection changes the masked values from `x - 1e9` to exactly `-1e9`, "
        "which is not bit-identical. This becomes a real defect the moment "
        "anyone sets a mixed policy around this trainer; that is what the "
        "inventory is for.",
    ),
    ("train/nam/train_dfsa_ste.py", "_make_ste_train_fn._step"): (
        1,
        "The STE trainer's consistency loss re-masks `residual_logits` with the "
        "SAME additive -1e9 that train_dfsa.py already applied to "
        "`pemdas_logits`, so that KL(P_pemdas || P_residual) is taken over the "
        "same support. Converting one side alone would silently change the "
        "support the KL is defined on; converting both is the train_dfsa.py "
        "change above, with the same three reasons against it. Pure float32 "
        "TensorFlow inside a tf.function train step, no policy set.",
    ),
}


# ---------------------------------------------------------------------------
# Census machinery
# ---------------------------------------------------------------------------


class Site(NamedTuple):
    """One census hit.

    :ivar path: ``src/``-relative path.
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


def _bare_name(node: ast.AST) -> Optional[str]:
    """Return the identifier of a ``Name`` or the attribute of an ``Attribute``.

    ``self.neg`` and ``Consts.NEG`` both reduce to their last component, which
    is what makes an attribute-bound sentinel visible to the name table.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _callee_name(node: ast.AST) -> Optional[str]:
    """Return the last component of a call's callee, e.g. ``"multiply"``."""
    return _bare_name(node.func) if isinstance(node, ast.Call) else None


def _fold(left: float, op: ast.AST, right: float) -> Optional[float]:
    """Constant-fold one binary arithmetic operator, or return ``None``.

    Only the five operators a sentinel is ever written with are folded, and the
    result must be finite and below :data:`_MAX_FOLD_MAGNITUDE`. ``**`` is
    additionally bounded so a parsed ``10 ** 100000`` cannot hang the walk.
    """
    try:
        if isinstance(op, ast.Add):
            value = left + right
        elif isinstance(op, ast.Sub):
            value = left - right
        elif isinstance(op, ast.Mult):
            value = left * right
        elif isinstance(op, ast.Div):
            value = left / right
        elif isinstance(op, ast.Pow):
            if abs(right) > 64.0 or abs(left) > 1e6:
                return None
            value = left ** right
        else:
            return None
    except (ArithmeticError, ValueError, OverflowError):
        return None
    value = float(value)
    if not math.isfinite(value) or abs(value) > _MAX_FOLD_MAGNITUDE:
        return None
    return value


def _numeric_value(
    node: ast.AST, bindings: Optional[Dict[str, float]] = None
) -> Optional[float]:
    """Return the constant numeric value of ``node``, or ``None``.

    Handles every spelling of a sentinel constant the review found escaping:
    a plain literal, a unary negation, an arithmetic expression over literals
    (``-10 ** 9``), a numeric string parsed by ``float("-1e9")``, and a numeric
    cast (``np.float32(-1e9)``, ``ops.cast(-1e9, dt)``). With ``bindings``
    supplied it also resolves a zero-argument helper call whose return value is
    a constant.

    ``bool`` is excluded: ``True`` is an ``int`` in Python and is never a
    sentinel. Non-finite results are excluded too -- see the module docstring on
    why ``-float('inf')`` is not this predicate's business.

    :param node: any AST node.
    :param bindings: optional name table from :func:`_numeric_bindings`.
    :return: the value, or ``None`` if the node is not a numeric constant.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return None
        if isinstance(node.value, (int, float)):
            value = float(node.value)
            return value if math.isfinite(value) else None
        return None

    if isinstance(node, ast.UnaryOp):
        inner = _numeric_value(node.operand, bindings)
        if inner is None:
            return None
        if isinstance(node.op, ast.USub):
            return -inner
        if isinstance(node.op, ast.UAdd):
            return inner
        return None

    if isinstance(node, ast.BinOp):
        left = _numeric_value(node.left, bindings)
        right = _numeric_value(node.right, bindings)
        if left is None or right is None:
            return None
        return _fold(left, node.op, right)

    if isinstance(node, ast.Call):
        name = _callee_name(node)
        if name in _NUMERIC_CALLS and node.args:
            argument = node.args[0]
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                try:
                    parsed = float(argument.value)
                except ValueError:
                    return None
                return parsed if math.isfinite(parsed) else None
            return _numeric_value(argument, bindings)
        if bindings is not None and name is not None and not node.args:
            return bindings.get(name)

    return None


def _binding_targets(target: ast.AST) -> List[str]:
    """Return every name an assignment target binds.

    Covers the three shapes the review found invisible: a plain ``Name``, an
    ``Attribute`` (``self.neg = -1e9``, recorded under ``"neg"``), and a tuple
    or list unpack (``NEG, POS = -1e9, 1e9``).
    """
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Attribute):
        return [target.attr]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: List[str] = []
        for element in target.elts:
            names.extend(_binding_targets(element))
        return names
    return []


def _assignment_candidates(value: ast.AST) -> List[ast.AST]:
    """Return every expression an assignment's right-hand side may evaluate to.

    A dtype-branching ternary (``-1e4 if fp16 else -1e9``) and a per-dtype dict
    literal each carry more than one candidate sentinel; both are live shapes in
    this tree.
    """
    if isinstance(value, ast.IfExp):
        return [value.body, value.orelse]
    if isinstance(value, ast.Dict):
        return list(value.values)
    return [value]


def _numeric_bindings(tree: ast.AST) -> Dict[str, float]:
    """Collect ``name -> largest bound magnitude`` over the whole module.

    Flat by design (see the module docstring): no scope tracking, so a name
    bound to a large constant anywhere in the file counts everywhere. Records
    plain assignments, annotated assignments, augmented assignments, attribute
    targets, tuple/list unpacking (element-wise when the shapes match), dict
    literal values, dtype-branching ternaries, and the constant return value of
    a zero-argument helper function.

    :param tree: a parsed module.
    :return: mapping from bound name to the largest magnitude it ever holds.
    """
    bindings: Dict[str, float] = {}

    def record(name: str, value: Optional[float]) -> None:
        if value is not None:
            bindings[name] = max(bindings.get(name, 0.0), abs(value))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # `def _neg(): return -1e9` then `(1 - m) * _neg()`.
            for inner in ast.walk(node):
                if isinstance(inner, ast.Return) and inner.value is not None:
                    for candidate in _assignment_candidates(inner.value):
                        record(node.name, _numeric_value(candidate))
            continue

        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        elif isinstance(node, ast.AugAssign):
            targets, value = [node.target], node.value
        else:
            continue

        for target in targets:
            if (
                isinstance(target, (ast.Tuple, ast.List))
                and isinstance(value, (ast.Tuple, ast.List))
                and len(target.elts) == len(value.elts)
            ):
                for element_target, element_value in zip(target.elts, value.elts):
                    for name in _binding_targets(element_target):
                        for candidate in _assignment_candidates(element_value):
                            record(name, _numeric_value(candidate))
                continue
            for name in _binding_targets(target):
                for candidate in _assignment_candidates(value):
                    record(name, _numeric_value(candidate))

    return bindings


def _is_mask_complement(
    node: ast.AST, complement_names: FrozenSet[str] = frozenset()
) -> bool:
    """Is this factor the complement of a keep-mask?

    Recognized shapes, each one a spelling the original predicate missed unless
    noted:

    - ``1 - m`` (the original shape) and ``m - 1`` (reversed subtraction);
    - ``-(m - 1)`` / ``+(1 - m)``, i.e. a unary wrapper around either;
    - ``~m`` and ``ops.logical_not(m)``, and ``ops.subtract(1.0, m)``;
    - any of the above inside a passthrough call (``ops.cast(..., "float32")``);
    - a bare name that either announces itself as a complement
      (``inv_mask``, ``one_minus_mask``, ...) or was BOUND to one of the shapes
      above earlier in the enclosing scope (the two-statement split).

    :param node: the candidate factor of a multiplication.
    :param complement_names: names bound to a complement in scope here.
    :return: ``True`` if the factor denotes ``1 - mask``.
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        if _numeric_value(node.left) == 1.0 or _numeric_value(node.right) == 1.0:
            return True

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Invert):
            return True
        if isinstance(node.op, (ast.USub, ast.UAdd)):
            return _is_mask_complement(node.operand, complement_names)

    if isinstance(node, ast.Call):
        name = _callee_name(node)
        if name in _COMPLEMENT_CALLS:
            if name == "subtract":
                return len(node.args) == 2 and (
                    _numeric_value(node.args[0]) == 1.0
                    or _numeric_value(node.args[1]) == 1.0
                )
            return True
        if name in _PASSTHROUGH_CALLS and node.args:
            return _is_mask_complement(node.args[0], complement_names)

    name = _bare_name(node)
    if name is None:
        return False
    if name in complement_names:
        return True
    return any(fragment in name.lower() for fragment in _COMPLEMENT_NAME_FRAGMENTS)


def _is_large_magnitude_factor(
    node: ast.AST, bindings: Dict[str, float]
) -> Optional[str]:
    """Is this factor a sentinel-sized number, directly or by indirection?

    :param node: the candidate factor of a multiplication.
    :param bindings: the file-flat name table from :func:`_numeric_bindings`.
    :return: a short description of what matched, or ``None``.
    """
    value = _numeric_value(node, bindings)
    if value is not None:
        return f"literal {value:g}" if abs(value) >= _ADDITIVE_MAGNITUDE else None

    name = _bare_name(node) or _callee_name(node)
    if name is not None:
        magnitude = bindings.get(name, 0.0)
        if magnitude >= _ADDITIVE_MAGNITUDE:
            return f"name {name!r} bound to magnitude {magnitude:g}"
    return None


def _complement_names_in(scope: ast.AST) -> Set[str]:
    """Return the names bound to a mask-complement expression inside ``scope``.

    This is the two-statement split, the spelling the review called "the single
    most likely real-world" one::

        pad = 1.0 - mask
        x = logits + pad * -1e9

    Only STRUCTURAL complements seed a binding (a subtraction, an inversion, a
    complement call) -- a name bound to another complement NAME does not chain,
    which keeps this a one-hop binding pass rather than a dataflow analysis.

    :param scope: a module, function or class node.
    :return: the set of names bound to a complement anywhere inside it.
    """
    names: Set[str] = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not _is_mask_complement(value):
            continue
        for target in targets:
            names.update(_binding_targets(target))
    return names


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

    This is the forbidden form. Detecting the MULTIPLICATION -- rather than the
    surrounding ``+`` or ``+=`` -- catches every way the product is subsequently
    combined at once. Both operand orders are tested, and the multiplication may
    be written with ``*`` or as ``keras.ops.multiply(a, b)``.

    The full list of recognized spellings is in the module docstring; the ones
    deliberately NOT recognized are in :data:`KNOWN_UNCAUGHT`.

    :param source: Python source text.
    :param path: label used in the returned sites.
    :return: one :class:`Site` per matching multiplication, ordered by line.
    :raises SyntaxError: if ``source`` does not parse.
    """
    tree = ast.parse(source)
    bindings = _numeric_bindings(tree)
    found: List[Site] = []

    class _Visitor(_SymbolTracker):
        def __init__(self) -> None:
            super().__init__()
            self._complement_stack: List[Set[str]] = [_complement_names_in(tree)]

        @property
        def _complements(self) -> FrozenSet[str]:
            names: Set[str] = set()
            for scope in self._complement_stack:
                names |= scope
            return frozenset(names)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._complement_stack.append(_complement_names_in(node))
            super().visit_FunctionDef(node)
            self._complement_stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

        def _record_if_defect(
            self, node: ast.AST, left: ast.AST, right: ast.AST
        ) -> None:
            complements = self._complements
            for complement, factor in ((left, right), (right, left)):
                if not _is_mask_complement(complement, complements):
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
                    return

        def visit_BinOp(self, node: ast.BinOp) -> None:
            if isinstance(node.op, ast.Mult):
                self._record_if_defect(node, node.left, node.right)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if _callee_name(node) in _MULTIPLY_CALLS and len(node.args) == 2:
                self._record_if_defect(node, node.args[0], node.args[1])
            self.generic_visit(node)

    _Visitor().visit(tree)
    return sorted(set(found))


def find_large_negative_constants(
    source: str, path: str = "<synthetic>"
) -> List[Site]:
    """Find every negative numeric constant at or below ``-_SENTINEL_MAGNITUDE``.

    This is the population predicate: it measures how many places in the tree
    still hold a local sentinel-sized value, whatever form consumes it. Values
    computed at runtime (``numpy.finfo(dtype).min``) carry no constant and are
    invisible here by construction, as is ``-float('inf')``. See the module
    docstring for why this one stays negative-only while the additive predicate
    is sign-symmetric.

    :param source: Python source text.
    :param path: label used in the returned sites.
    :return: one :class:`Site` per matching constant, ordered by line.
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


def _iter_source_files(root: pathlib.Path) -> Iterator[Tuple[str, str]]:
    """Yield ``(SRC_ROOT-relative path, source text)`` for every module under ``root``."""
    for path in sorted(root.rglob("*.py")):
        yield path.relative_to(SRC_ROOT).as_posix(), path.read_text(encoding="utf-8")


def _census(finder, root: Optional[pathlib.Path] = None) -> Tuple[List[Site], int]:
    """Run ``finder`` over a subtree of ``src/``.

    :param finder: one of the two module-level predicates.
    :param root: the subtree to walk; defaults to all of :data:`SRC_ROOT`.
    :return: ``(all sites found, number of files walked)``.
    """
    sites: List[Site] = []
    walked = 0
    for relative_path, source in _iter_source_files(root or SRC_ROOT):
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

#: The correct fixes for the newly-caught spellings. The widened predicate must
#: decline every one of them: a detector that fires on the migration target
#: rejects every correct fix, which is the mirror failure of blindness and just
#: as fatal.
_CLEAN_WIDENED_SOURCE = """
import keras

from dl_techniques.utils.dtype_policy import mask_sentinel


def selection_forms(logits, mask, scale):
    keep = keras.ops.cast(mask, "bool")
    sentinel = keras.ops.cast(mask_sentinel(logits.dtype), logits.dtype)
    a = keras.ops.where(keep, logits, sentinel)
    b = keras.ops.where(keras.ops.logical_not(keep), sentinel, logits)
    # A genuine complement times an ORDINARY coefficient is not a sentinel.
    pad = 1.0 - mask
    c = logits + pad * 0.5
    d = keras.ops.multiply(1.0 - mask, scale)
    # A large magnitude that multiplies something that is not a complement.
    e = logits * 1e9
    # -inf as an online-softmax identity, not a sentinel.
    f = keras.ops.full((2, 2), -float("inf"))
    return a, b, c, d, e, f
"""

#: ``(label, snippet)`` for every spelling the predicate is contracted to catch.
#: Rows 01-08 were caught by the original predicate; rows 09-22 are the review's
#: escapes. One arm per row, so a "simplification" of the predicate reddens here
#: rather than silently narrowing what the whole-tree census can see.
_CAUGHT_SPELLINGS: List[Tuple[str, str]] = [
    ("01 classic product-plus-complement", "x = logits * mask + (1 - mask) * -1e9"),
    ("02 bare complement times literal", "x = logits + (1.0 - mask) * -1e9"),
    ("03 parenthesised negative sentinel", "x = logits + (1.0 - mask) * (-1e9)"),
    ("04 sentinel as the LEFT operand", "x = logits + -1e9 * (1.0 - mask)"),
    ("05 positive mirror (a min reduction)", "x = logits + (1.0 - mask) * 1e9"),
    ("06 module-level named sentinel", "NEG = -1e9\nx = logits + inv_mask * NEG"),
    (
        "07 dtype-branching ternary sentinel",
        'neg = -1e4 if dt == "float16" else -1e9\nx = logits + (1.0 - m) * neg',
    ),
    ("08 attribute mask operand", "x = logits + (1.0 - self.mask) * -1e9"),
    ("09 reversed subtraction (m - 1)", "x = logits + (mask - 1.0) * 1e9"),
    ("10 negated complement -(m - 1)", "x = logits + -(mask - 1.0) * -1e9"),
    ("11 sentinel built by arithmetic", "x = logits + (1.0 - mask) * (-10 ** 9)"),
    ("12 sentinel parsed from a string", 'x = logits + (1.0 - mask) * float("-1e9")'),
    (
        "13 two-statement split (the natural spelling)",
        "def f(logits, mask):\n    pad = 1.0 - mask\n    return logits + pad * -1e9",
    ),
    (
        "14 sentinel on a self attribute",
        "class C:\n"
        "    def __init__(self):\n"
        "        self.neg = -1e9\n"
        "    def call(self, x, m):\n"
        "        return x + (1.0 - m) * self.neg",
    ),
    (
        "15 sentinel on a class attribute",
        "class C:\n    NEG = -1e9\nx = logits + (1.0 - m) * C.NEG",
    ),
    (
        "16 complement via ops.subtract",
        "x = logits + keras.ops.subtract(1.0, mask) * -1e9",
    ),
    (
        "17 multiplication via ops.multiply",
        "x = logits + keras.ops.multiply(1.0 - mask, -1e9)",
    ),
    (
        "18 complement via ops.logical_not",
        'x = logits + keras.ops.cast(keras.ops.logical_not(mask), "float32") * -1e9',
    ),
    (
        "19 complement via bitwise invert",
        'x = logits + keras.ops.cast(~mask, "float32") * -1e9',
    ),
    (
        "20 tuple-unpacked sentinel",
        "NEG, POS = -1e9, 1e9\nx = logits + (1.0 - m) * NEG",
    ),
    (
        "21 sentinel returned by a helper",
        "def _neg():\n    return -1e9\nx = logits + (1.0 - m) * _neg()",
    ),
    ("22 augmented assignment", "x += (1.0 - mask) * -1e9"),
]


class TestTheCensusInstrumentIsNotVacuous:
    """A census that cannot see anything reports zero violations forever.

    This class is the reason the rest of the file means something. It is
    two-sided on purpose: an instrument that fires on everything is exactly as
    useless as one that fires on nothing, and only the pair of arms distinguishes
    a working detector from either failure. Since completion fix 3.1 it is also
    three-way honest: :data:`KNOWN_UNCAUGHT` is asserted to be still uncaught, so
    the blind spots are measured rather than assumed.
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

    def test_the_widened_predicate_is_silent_on_every_correct_fix(self) -> None:
        """The second half of the two-sided proof, for the widened predicate.

        Every shape in ``_CLEAN_WIDENED_SOURCE`` is something the hardening
        brought within reach of the predicate -- a call-form multiply, a
        name-bound complement, a passthrough cast, a folded constant -- and every
        one of them is CORRECT code. Firing here would mean the hardening had
        turned the guard into an obstacle to fixing the defect.
        """
        sites = find_additive_mask_bias(_CLEAN_WIDENED_SOURCE, "<clean-widened>")
        assert sites == [], (
            "the hardened additive predicate fired on correct selection-form "
            f"code:\n{_render(sites)}"
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

    def test_the_population_predicate_ignores_negative_infinity(self) -> None:
        """``-float('inf')`` is an identity element, not a sentinel.

        ``layers/attention/ring_attention.py`` seeds an online-softmax running
        max with it. It is ``-inf`` in every float dtype by construction, so it
        cannot be the overflow hazard this census measures, and reporting it
        would push a correct idiom onto the allowlist.
        """
        source = "m = keras.ops.full((2, 2), -float('inf'))"
        assert find_large_negative_constants(source, "<inf>") == []
        assert find_additive_mask_bias(
            "x = logits + (1.0 - mask) * -float('inf')", "<inf>"
        ) == []

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

    @pytest.mark.parametrize("label,snippet", _CAUGHT_SPELLINGS)
    def test_the_predicate_recognizes_each_defect_spelling(
        self, label: str, snippet: str
    ) -> None:
        """One arm per spelling the predicate is contracted to catch.

        Rows 01-08 are the spellings the seven base-commit sites actually used,
        copied verbatim from the tree. Rows 09-22 are the escapes the
        attack-before-release review measured against the FIRST version of this
        predicate: 13 of its 21 candidates went unseen, four of them invisible to
        both predicates at once.

        These outlive the fixes. Once the real sites are converted the whole-tree
        census can only stay green, and this parametrization is then the only
        thing standing between a "simplified" predicate and a guard that silently
        sees nothing.
        """
        sites = find_additive_mask_bias(snippet, "<spelling>")
        assert len(sites) >= 1, f"predicate no longer recognizes: {label}"

    def test_every_caught_spelling_row_is_distinct(self) -> None:
        """A duplicated row would inflate the coverage claim for free."""
        snippets = [snippet for _, snippet in _CAUGHT_SPELLINGS]
        assert len(set(snippets)) == len(snippets), "duplicate spelling fixtures"
        assert len(_CAUGHT_SPELLINGS) >= 22, (
            "the review measured 21 candidate spellings; this list must cover at "
            "least the 22 rows recorded at completion fix 3.1"
        )

    @pytest.mark.parametrize("label", sorted(KNOWN_UNCAUGHT))
    def test_each_known_blind_spot_is_still_blind(self, label: str) -> None:
        """An enumerated blind spot must be real, and must shrink on purpose.

        If a future hardening starts catching one of these, this test fails and
        the author must DELETE the row -- so the list cannot quietly grow stale
        and keep claiming a weakness that no longer exists. The reverse failure
        (a row that was never blind) is caught the same way.
        """
        snippet, reason = KNOWN_UNCAUGHT[label]
        assert reason.strip(), f"blind spot {label!r} carries no reason"
        assert len(reason.split()) >= 12, (
            f"blind spot {label!r} needs a reason, not a shrug"
        )
        sites = find_additive_mask_bias(snippet, "<known-uncaught>")
        assert sites == [], (
            f"{label!r} is listed in KNOWN_UNCAUGHT but the predicate now "
            f"CATCHES it. Delete the row:\n{_render(sites)}"
        )

    def test_the_walk_visits_a_realistic_number_of_files(self) -> None:
        _, walked = _census(find_additive_mask_bias)
        assert walked >= MINIMUM_FILES_WALKED, (
            f"walked only {walked} files under {SRC_ROOT}; the tree measured "
            f"1052 at completion fix 3.1. A short walk makes every "
            f"'no violations' assertion below vacuous"
        )

    def test_the_library_half_of_the_walk_is_realistic_on_its_own(self) -> None:
        """A healthy ``train/`` walk must not mask a broken library walk."""
        _, walked = _census(find_additive_mask_bias, PACKAGE_ROOT)
        assert walked >= MINIMUM_LIBRARY_FILES_WALKED, (
            f"walked only {walked} files under {PACKAGE_ROOT}; the library "
            f"measured 803 at the plan's base commit"
        )


# ---------------------------------------------------------------------------
# The census proper
# ---------------------------------------------------------------------------


class TestTheAdditiveMaskBiasFormIsExtinct:
    """No allowlist, over the whole library. The additive form has no legitimate
    use anywhere in ``src/dl_techniques/``.

    RED at the plan's base commit with seven sites; green since step 6 landed.
    """

    def test_no_library_module_multiplies_a_mask_complement_by_a_sentinel(
        self,
    ) -> None:
        sites, walked = _census(find_additive_mask_bias, PACKAGE_ROOT)
        assert sites == [], (
            f"{len(sites)} additive mask-bias site(s) across {walked} files "
            f"under {PACKAGE_ROOT}.\n"
            f"Under mixed_float16 the sentinel overflows to -inf and 0 * -inf "
            f"= NaN poisons every KEPT position, not just the masked ones.\n"
            f"Rewrite as keras.ops.where(keep, x, sentinel) using "
            f"dl_techniques.utils.dtype_policy.mask_sentinel:\n"
            f"{_render(sites)}"
        )


class TestTheAdditiveFormOutsideTheLibraryIsInventoried:
    """``src/train/`` and ``src/applications/`` are held to a frozen inventory.

    Not an allowlist: the comparison is EXACT EQUALITY on ``(path, symbol) ->
    count``. A new additive site reddens this, and so does removing an
    inventoried one without updating :data:`NON_LIBRARY_ADDITIVE_INVENTORY`. A
    skip-list could only ever grow, silently, which is the failure mode that let
    six sites in ``src/train/nam/`` sit outside the census entirely until the
    attack-before-release review walked them by hand.
    """

    @staticmethod
    def _measured() -> Dict[Tuple[str, str], int]:
        counts: Dict[Tuple[str, str], int] = {}
        for subtree in sorted(SRC_ROOT.iterdir()):
            if not subtree.is_dir() or subtree == PACKAGE_ROOT:
                continue
            sites, _ = _census(find_additive_mask_bias, subtree)
            for site in sites:
                counts[(site.path, site.symbol)] = (
                    counts.get((site.path, site.symbol), 0) + 1
                )
        return counts

    def test_the_measured_sites_match_the_inventory_exactly(self) -> None:
        measured = self._measured()
        expected = {key: count for key, (count, _) in NON_LIBRARY_ADDITIVE_INVENTORY.items()}
        assert measured == expected, (
            "the additive form outside src/dl_techniques/ no longer matches the "
            "frozen inventory.\n"
            f"measured: {measured}\nexpected: {expected}\n"
            "If you ADDED a site: convert it to keras.ops.where. If you FIXED "
            "one: delete or decrement its NON_LIBRARY_ADDITIVE_INVENTORY entry "
            "in the same commit."
        )

    def test_the_inventory_is_not_empty(self) -> None:
        """Vacuity guard: an empty inventory would make the equality trivial.

        Delete this test on the day the inventory legitimately empties -- and
        the deletion is exactly the moment somebody must confirm the six sites
        were fixed rather than moved out of the walk.
        """
        assert NON_LIBRARY_ADDITIVE_INVENTORY, (
            "the inventory is empty; either every non-library site was fixed "
            "(delete this test and say so) or the walk stopped seeing them"
        )

    def test_every_inventory_entry_carries_a_real_reason(self) -> None:
        thin = {
            key: reason
            for key, (_, reason) in NON_LIBRARY_ADDITIVE_INVENTORY.items()
            if len(reason.split()) < 20
        }
        assert thin == {}, (
            f"an inventoried additive defect needs a written justification, not "
            f"a label: {list(thin)}"
        )

    def test_every_inventoried_file_and_symbol_still_exists(self) -> None:
        """Rot check, same shape as the allowlist's."""
        broken: List[str] = []
        for path, symbol in sorted(NON_LIBRARY_ADDITIVE_INVENTORY):
            source_path = SRC_ROOT / path
            if not source_path.is_file():
                broken.append(f"{path} (missing file)")
                continue
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            names = {
                node.name
                for node in ast.walk(tree)
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
            }
            if not all(part in names for part in symbol.split(".")):
                broken.append(f"{path}::{symbol} (missing symbol)")
        assert broken == [], f"inventory rot: {broken}"


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

    def test_the_library_prefix_actually_matches_the_walk(self) -> None:
        """The `src/`-relative keys must resolve, or every rot check goes quiet.

        A key written package-relative (`layers/moe/gating.py` rather than
        `dl_techniques/layers/moe/gating.py`) still parses fine as a tuple; it
        just never matches a site again. This asserts the prefix convention
        holds so that mistake surfaces here rather than as a silently permissive
        allowlist.
        """
        assert (SRC_ROOT / LIBRARY_PREFIX).is_dir()
        library_keys = [
            path for path, _ in SENTINEL_ALLOWLIST if path.startswith(LIBRARY_PREFIX)
        ]
        assert len(library_keys) >= 20, (
            f"only {len(library_keys)} allowlist keys are library-relative; the "
            f"census walks src/ and every library key needs the "
            f"{LIBRARY_PREFIX!r} prefix"
        )
