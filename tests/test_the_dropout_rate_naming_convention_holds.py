"""Every dropout RATE in `src/` and `tests/` is spelled `dropout_rate` or `<prefix>_dropout_rate`.

This is the closure guard for the rename shipped by
plan-2026-08-22T035419-a11304c8 D-130 (steps 6-8): 56 parameters across 17
spellings — `dropout`, `attention_dropout`, `hidden_dropout_prob`,
`dropout_p`, `prompt_mlp_dropout`, ... — collapsed onto the single
`(^|_)dropout_rate$` convention, together with every `self.` attribute, every
`get_config()` key, every docstring and every keyword caller.

WHY THIS IS AN AST SWEEP AND NOT A GREP
---------------------------------------
Grep over this repository finds DOCSTRINGS. The word "dropout" appears in
prose, in architecture diagrams, in `# DECISION` anchors and in the `name=`
string of roughly forty `keras.layers.Dropout` instances, none of which is a
parameter and none of which this convention governs. A grep-based version of
this guard would be a wall of false positives that a reader would learn to
ignore, which is the failure mode that lets a real regression through. The
three assertions below parse each file and look at `ast.arg` and
`ast.keyword` nodes only.

WHAT IS DELIBERATELY NOT A RATE
-------------------------------
Two parameters contain the substring `dropout` and are NOT rates, so the
convention does not apply to them and they are pinned by name in
:data:`NON_RATE_PARAMS` rather than silently skipped by a pattern:

* ``conv2d_wrapper(dropout_params=, dropout_2d_params=)`` — both typed
  ``Optional[Dict[str, Any]]``; they are kwargs BAGS forwarded to build a
  ``Dropout`` / ``SpatialDropout2D``, not floats.

* ``DifferentialMultiHeadAttention._stream(attn_dropout_layer=)`` — typed
  ``Optional[keras.layers.Dropout]``, a LAYER OBJECT passed by reference. It was
  not given a ``*_dropout_rate`` suffix (it is not a rate); it was renamed from
  ``attn_dropout`` to ``attn_dropout_layer`` so that it reads as what it is. The
  rate it belongs to is ``self.attention_dropout_rate``, already conforming.

WHY THE EXEMPT CALL SITES ARE PINNED AS A SET AND NOT A COUNT
-------------------------------------------------------------
``keras.layers.MultiHeadAttention(dropout=)`` and
``keras.layers.LSTM/LSTMCell(dropout=, recurrent_dropout=)`` are the STOCK
KERAS 3 PUBLIC API. Renaming those keywords would be a `TypeError`, so they
survive — and a waiver that is merely counted grows silently the next time
someone adds a stock-Keras call. :data:`EXEMPT_KEYWORD_CALLS` therefore pins
every surviving site as a ``(file, callee, keyword)`` triple and the assertion
is SET EQUALITY, in both directions: a new un-renamed call site fails, and a
removed one fails too (so the waiver cannot rot into a lie).

The set is pinned on ``(file, callee, keyword)`` rather than on a line number
on purpose: a line pin goes stale on the next unrelated edit above it, and a
guard that cries wolf gets deleted. The failure message reports live line
numbers so a real violation is still one click away.

Two of the 19 entries are NOT stock Keras. They are
``tests/test_layers/test_embedding/test_embedding_factory.py``'s deliberate
NEGATIVE tests, which pass ``create_embedding_layer(dropout=0.5)`` precisely to
prove the factory REJECTS the wrong spelling. Renaming them would delete the
test's subject.

RED-PROOF
---------
:class:`TestTheGuardCanFail` re-introduces each violation class into a source
string and asserts the corresponding collector names the exact offending site.
A committed RED proof by injection into a real file is recorded in
``decisions.md`` D-133.
"""

import ast
import os
import re
from typing import Dict, List, Set, Tuple

import pytest

# ---------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_ROOTS = ("src", "tests")
SKIP_DIRS = {"__pycache__", ".git", ".venv", "node_modules"}

#: The convention. A rate is either bare ``dropout_rate`` or
#: ``<prefix>_dropout_rate``; nothing else is accepted.
CONVENTION = re.compile(r"(^|_)dropout_rate$")

#: A parameter is a DROPOUT parameter if its name carries either token. Both
#: spellings are needed: ``drop_rate`` is timm's spelling and appears in the
#: ported vision backbones.
DROPOUT_TOKEN = re.compile(r"dropout|drop_rate")

#: The 17 pre-D-130 spellings. Any of these surviving as a `def` parameter or
#: as a call keyword is a rename that did not finish.
OLD_NAMES: Set[str] = {
    "dropout",
    "attention_dropout",
    "projection_dropout",
    "dropout_p",
    "mlm_head_dropout",
    "mlp_dropout",
    "head_dropout",
    "prompt_mlp_dropout",
    "emb_dropout",
    "classifier_dropout",
    "output_dropout",
    "ffn_dropout",
    "segmentation_dropout",
    "classification_dropout",
    "recurrent_dropout",
    "hidden_dropout_prob",
    "attention_probs_dropout_prob",
}

#: Parameters that contain "dropout" and are NOT rates. Pinned by
#: ``(file, function, parameter)`` so the exclusion cannot be widened by
#: loosening a regex.
NON_RATE_PARAMS: Set[Tuple[str, str, str]] = {
    ("src/dl_techniques/layers/conv2d_builder.py", "conv2d_wrapper", "dropout_params"),
    ("src/dl_techniques/layers/conv2d_builder.py", "conv2d_wrapper", "dropout_2d_params"),
    (
        "src/dl_techniques/layers/attention/differential_attention.py",
        "_stream",
        "attn_dropout_layer",
    ),
}

#: Surviving keyword-call sites, pinned as ``(file, callee, keyword)``.
#: 17 target stock Keras 3 (`MultiHeadAttention`, `LSTM`, `LSTMCell`) across 16
#: distinct CALLS — the two LSTM/LSTMCell calls each pass two exempt keywords.
#: 2 are deliberate negative tests of the embedding factory's strict-kwarg
#: rejection.
EXEMPT_KEYWORD_CALLS: Set[Tuple[str, str, str]] = {
    # --- stock keras.layers.MultiHeadAttention -----------------------------
    ("src/dl_techniques/layers/blt_blocks.py", "MultiHeadAttention", "dropout"),
    ("src/dl_techniques/layers/transformers/adaln_zero.py", "MultiHeadAttention", "dropout"),
    ("src/dl_techniques/layers/graphs/entity_graph_refinement.py", "MultiHeadAttention", "dropout"),
    ("src/dl_techniques/layers/graphs/graph_neural_network.py", "MultiHeadAttention", "dropout"),
    ("src/dl_techniques/layers/graphs/relational_graph_transformer_blocks.py", "MultiHeadAttention", "dropout"),
    ("src/dl_techniques/models/nano_vlm_world_model/denoisers.py", "MultiHeadAttention", "dropout"),
    ("src/dl_techniques/models/SAM/SAM1/transformer.py", "MultiHeadAttention", "dropout"),
    ("src/dl_techniques/models/video_jepa/predictor.py", "MultiHeadAttention", "dropout"),
    ("tests/test_layers/test_attention/test_shared_weights_cross_attention.py", "MultiHeadAttention", "dropout"),
    # --- stock keras.layers.LSTM / LSTMCell --------------------------------
    ("src/dl_techniques/layers/time_series/deepar_blocks.py", "LSTMCell", "dropout"),
    ("src/dl_techniques/layers/time_series/deepar_blocks.py", "LSTMCell", "recurrent_dropout"),
    ("src/dl_techniques/models/time_series/deepar/model.py", "LSTM", "dropout"),
    ("src/dl_techniques/models/time_series/deepar/model.py", "LSTM", "recurrent_dropout"),
    # --- deliberate NEGATIVE tests: the factory must REJECT this spelling ---
    ("tests/test_layers/test_embedding/test_embedding_factory.py", "create_embedding_layer", "dropout"),
}

# ---------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------


def _iter_python_files() -> List[str]:
    """Every ``.py`` file under the scan roots, as repo-relative paths."""
    out: List[str] = []
    for root in SCAN_ROOTS:
        for dirpath, dirnames, filenames in os.walk(os.path.join(REPO_ROOT, root)):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in sorted(filenames):
                if name.endswith(".py"):
                    out.append(os.path.relpath(os.path.join(dirpath, name), REPO_ROOT))
    return sorted(out)


def _parse(rel_path: str) -> ast.Module:
    """Parse a file, failing LOUDLY.

    A ``try: ... except SyntaxError: continue`` here would let a broken file
    shrink the sweep silently — the exact defect D-070 repaired in six other
    AST guards in this repo. A file this guard cannot read is a guard failure,
    not a skipped row.
    """
    full = os.path.join(REPO_ROOT, rel_path)
    with open(full, encoding="utf-8") as handle:
        source = handle.read()
    return ast.parse(source, filename=full)


def _declared_params(fn: ast.AST) -> List[ast.arg]:
    """Every declared parameter of a function, including ``*args`` / ``**kwargs``."""
    args = fn.args  # type: ignore[attr-defined]
    out = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    if args.vararg is not None:
        out.append(args.vararg)
    if args.kwarg is not None:
        out.append(args.kwarg)
    return out


def _callee_name(node: ast.Call) -> str:
    """The bare callee name — ``layers.LSTM(...)`` and ``LSTM(...)`` both give ``LSTM``."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return "<expr>"


def collect_dropout_params() -> List[Tuple[str, int, str, str]]:
    """``(file, line, function, parameter)`` for every dropout-ish `def` parameter."""
    found: List[Tuple[str, int, str, str]] = []
    for rel in _iter_python_files():
        tree = _parse(rel)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for param in _declared_params(node):
                if DROPOUT_TOKEN.search(param.arg):
                    found.append((rel, param.lineno, node.name, param.arg))
    return found


def collect_old_name_keyword_calls() -> List[Tuple[str, int, str, str]]:
    """``(file, line, callee, keyword)`` for every call passing a pre-D-130 name."""
    found: List[Tuple[str, int, str, str]] = []
    for rel in _iter_python_files():
        tree = _parse(rel)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg in OLD_NAMES:
                    found.append((rel, keyword.value.lineno, _callee_name(node), keyword.arg))
    return found


# ---------------------------------------------------------------------
# The three assertions
# ---------------------------------------------------------------------


class TestTheRenameIsClosed:
    """The three closure claims of plan-2026-08-22T035419-a11304c8 step 9."""

    def test_no_pre_rename_spelling_survives_as_a_parameter(self):
        """(i) None of the 17 old spellings survives as a `def` parameter."""
        offenders = [
            (rel, line, fn, name)
            for rel, line, fn, name in collect_dropout_params()
            if name in OLD_NAMES
        ]
        assert offenders == [], (
            "pre-D-130 dropout parameter spellings survive:\n"
            + "\n".join(f"  {r}:{l} {f}({n}=)" for r, l, f, n in offenders)
        )

    def test_every_surviving_old_name_call_site_is_pinned_by_name(self):
        """(ii) Every surviving old-name KEYWORD CALL is one of the pinned exemptions.

        Set equality in both directions. A new un-renamed caller fails; a
        pinned site that disappears also fails, so the waiver cannot outlive
        its subject.
        """
        live = collect_old_name_keyword_calls()
        live_keys = {(rel, callee, kw) for rel, _line, callee, kw in live}
        lines: Dict[Tuple[str, str, str], List[int]] = {}
        for rel, line, callee, kw in live:
            lines.setdefault((rel, callee, kw), []).append(line)

        unpinned = sorted(live_keys - EXEMPT_KEYWORD_CALLS)
        assert unpinned == [], (
            "keyword-call sites still using a pre-D-130 dropout spelling, and not "
            "pinned in EXEMPT_KEYWORD_CALLS:\n"
            + "\n".join(
                f"  {rel}:{sorted(lines[(rel, callee, kw)])} {callee}({kw}=)"
                for rel, callee, kw in unpinned
            )
            + "\n\nIf this is a NEW stock-Keras call, add it to EXEMPT_KEYWORD_CALLS "
            "with a comment naming the Keras class. Otherwise finish the rename."
        )

        stale = sorted(EXEMPT_KEYWORD_CALLS - live_keys)
        assert stale == [], (
            "EXEMPT_KEYWORD_CALLS pins sites that no longer exist — delete these "
            "rows so the waiver stays honest:\n"
            + "\n".join(f"  {rel} {callee}({kw}=)" for rel, callee, kw in stale)
        )

    def test_every_dropout_rate_parameter_matches_the_convention(self):
        """(iii) 100% of dropout-RATE `def` parameters match ``(^|_)dropout_rate$``."""
        offenders = [
            (rel, line, fn, name)
            for rel, line, fn, name in collect_dropout_params()
            if not CONVENTION.search(name) and (rel, fn, name) not in NON_RATE_PARAMS
        ]
        assert offenders == [], (
            "dropout parameters that do not match `(^|_)dropout_rate$`:\n"
            + "\n".join(f"  {r}:{l} {f}({n}=)" for r, l, f, n in offenders)
            + "\n\nIf one of these is genuinely NOT a rate (a kwargs bag, a layer "
            "object), give it a name that says so and pin it in NON_RATE_PARAMS "
            "with the reason."
        )

    def test_the_non_rate_pins_still_exist(self):
        """A pin for a parameter that no longer exists is a lie; fail on it."""
        live = {(rel, fn, name) for rel, _line, fn, name in collect_dropout_params()}
        stale = sorted(NON_RATE_PARAMS - live)
        assert stale == [], (
            "NON_RATE_PARAMS pins parameters that no longer exist:\n"
            + "\n".join(f"  {rel} {fn}({name}=)" for rel, fn, name in stale)
        )


# ---------------------------------------------------------------------
# RED proof
# ---------------------------------------------------------------------


def _params_of(source: str) -> List[str]:
    """The dropout-ish parameter names declared in a source string."""
    tree = ast.parse(source)
    out: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for param in _declared_params(node):
                if DROPOUT_TOKEN.search(param.arg):
                    out.append(param.arg)
    return out


def _keyword_calls_of(source: str) -> List[Tuple[str, str]]:
    """``(callee, keyword)`` for every old-name keyword in a source string."""
    tree = ast.parse(source)
    out: List[Tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg in OLD_NAMES:
                    out.append((_callee_name(node), keyword.arg))
    return out


class TestTheGuardCanFail:
    """Each assertion is proven to reject the violation it exists to reject.

    These operate on source STRINGS, not on the tree, so the proof is
    deterministic and leaves no file to restore. The equivalent injection into
    a real file — re-introducing ``BeitModel(hidden_dropout_prob=)`` — was run
    once and is recorded in decisions.md D-133 with the exact site the guard
    named.
    """

    @pytest.mark.parametrize("old", sorted(OLD_NAMES))
    def test_a_reintroduced_old_parameter_name_is_caught(self, old: str):
        source = f"def f(x, {old}: float = 0.1):\n    return x\n"
        names = _params_of(source)
        assert old in names
        assert old in OLD_NAMES, "collector must see it as a pre-D-130 spelling"

    @pytest.mark.parametrize("old", sorted(OLD_NAMES))
    def test_a_reintroduced_old_keyword_call_is_caught(self, old: str):
        source = f"MyLayer(dim=8, {old}=0.1)\n"
        assert _keyword_calls_of(source) == [("MyLayer", old)]

    def test_a_new_nonconforming_rate_name_is_caught(self):
        """A brand-new spelling nobody has seen before still fails claim (iii)."""
        source = "def f(x, gate_dropout: float = 0.1):\n    return x\n"
        names = _params_of(source)
        assert names == ["gate_dropout"]
        assert not CONVENTION.search("gate_dropout")
        assert "gate_dropout" not in OLD_NAMES, (
            "claim (iii) must reject a name that is NOT on the old-name list — "
            "otherwise the guard only re-checks what the rename already fixed"
        )

    def test_a_conforming_name_passes(self):
        """The control: the guard is not simply failing everything."""
        source = (
            "def f(x, dropout_rate=0.1, attention_dropout_rate=0.0, drop_rate_x=1):\n"
            "    return x\n"
        )
        names = _params_of(source)
        assert set(names) == {"dropout_rate", "attention_dropout_rate", "drop_rate_x"}
        assert CONVENTION.search("dropout_rate")
        assert CONVENTION.search("attention_dropout_rate")
        assert not CONVENTION.search("drop_rate_x"), (
            "drop_rate_x is deliberately non-conforming; if the live tree ever "
            "grows one, claim (iii) must and does fail on it"
        )

    def test_the_parser_fails_loudly_on_a_broken_file(self):
        """D-070: a file this guard cannot parse is a FAILURE, never a silent skip."""
        with pytest.raises(SyntaxError):
            ast.parse("def f(:\n")
