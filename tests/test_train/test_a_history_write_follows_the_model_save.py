"""Every ``save_training_history_json`` call in ``src/`` runs after the model is on disk.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 3 (D-174). The shared
writer raises on failure (``train.common.run_io``), so a failed history write ends
the run. Two things make that safe, and this AST census checks both at every call
site it finds:

- A model save comes before the write and none comes after it on the same path:
  either an explicit ``<model>.save(...)`` / ``.save_weights(...)`` earlier in the
  function, or a ``fit(..., callbacks=...)`` in a module that installs a checkpoint
  writer (``ModelCheckpoint``, one of the shared ``create_*callbacks`` helpers, or a
  ``*Checkpoint*`` callback), or a call to a same-module helper that does either.
  A save reachable after the write (in the function, or in an in-module caller after
  it calls the function) is a violation: a failed write would cost those weights.
  A write in a block that ends in ``raise`` or ``return`` does not reach the saves
  after that block.
- No ``try`` with an ``except`` encloses the write, so the failure is not swallowed.

At 2026-10-20 (before the step) the census found 53 sites: 13 wrote the history
before a later save and 1 sat inside ``try/except Exception``.
"""

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
WRITER = "save_training_history_json"
MODEL_SAVE_ATTRS = ("save", "save_weights")
NOT_A_MODEL = ("np", "numpy")
CHECKPOINT_WRITER = re.compile(
    r"\bModelCheckpoint\b|\bcreate_(?:common_|nlp_)?callbacks\b|\b\w*Checkpoint\w*\(")


@dataclass(frozen=True)
class Site:
    """One writer call: where it is and what the census found."""

    path: str
    line: int
    function: str
    save_before: bool
    saves_after: Tuple[int, ...]
    enclosing_excepts: Tuple[int, ...]


def _call_name(call: ast.Call) -> Optional[str]:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_model_save(node: ast.AST) -> bool:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return False
    if node.func.attr not in MODEL_SAVE_ATTRS:
        return False
    receiver = node.func.value
    return not (isinstance(receiver, ast.Name) and receiver.id in NOT_A_MODEL)


def _own_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """Walk ``node`` without entering nested def or class bodies (a lambda is entered: a
    ``LambdaCallback`` that saves during ``fit`` is a save)."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        for child in ast.iter_child_nodes(current):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                stack.append(child)


class _Module:
    def __init__(self, source: str):
        self.tree = ast.parse(source)
        self.has_checkpoint_writer = bool(CHECKPOINT_WRITER.search(source))
        self.parents: Dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(self.tree):
            for child in ast.iter_child_nodes(parent):
                self.parents[child] = parent
        self.functions: Dict[str, List[ast.AST]] = {}
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions.setdefault(node.name, []).append(node)

    def saves_directly(self, stmts: Sequence[ast.AST]) -> bool:
        for stmt in stmts:
            for node in _own_nodes(stmt):
                if _is_model_save(node):
                    return True
                if (isinstance(node, ast.Call) and _call_name(node) == "fit"
                        and any(k.arg == "callbacks" for k in node.keywords)
                        and self.has_checkpoint_writer):
                    return True
        return False

    def saves(self, stmts: Sequence[ast.AST]) -> bool:
        """A model save in ``stmts``, or in a same-module function they call (one level)."""
        if self.saves_directly(stmts):
            return True
        for stmt in stmts:
            for node in _own_nodes(stmt):
                if isinstance(node, ast.Call):
                    for helper in self.functions.get(_call_name(node) or "", []):
                        if self.saves_directly(helper.body):
                            return True
        return False

    def explicit_saves(self, stmts: Sequence[ast.AST]) -> List[int]:
        """Lines of explicit model saves in ``stmts`` or in same-module helpers they call."""
        lines = []
        for stmt in stmts:
            for node in _own_nodes(stmt):
                if _is_model_save(node):
                    lines.append(node.lineno)
                elif isinstance(node, ast.Call):
                    for helper in self.functions.get(_call_name(node) or "", []):
                        if any(_is_model_save(n) for s in helper.body for n in _own_nodes(s)):
                            lines.append(node.lineno)
        return lines

    def path_to(self, node: ast.AST, function: ast.AST) -> List[Tuple[List[ast.AST], int]]:
        """(statement list, index) pairs from the function body down to ``node``'s statement."""
        chain = []
        current = node
        while current is not function:
            parent = self.parents[current]
            for field in ("body", "orelse", "finalbody", "handlers"):
                block = getattr(parent, field, None)
                if isinstance(block, list) and current in block:
                    chain.append((block, block.index(current)))
                    break
            current = parent
        return chain[::-1]

    def enclosing_function(self, node: ast.AST) -> Optional[ast.AST]:
        current = node
        while current in self.parents:
            current = self.parents[current]
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return current
        return None

    def reachable_saves_after(self, node: ast.AST, function: ast.AST) -> List[int]:
        lines = []
        for block, index in reversed(self.path_to(node, function)):
            later = block[index + 1:]
            stop = next((i for i, s in enumerate(later) if isinstance(s, (ast.Raise, ast.Return))), None)
            lines += self.explicit_saves(later if stop is None else later[:stop])
            if stop is not None:
                return lines
        return lines

    def enclosing_excepts(self, node: ast.AST) -> List[int]:
        lines = []
        current = node
        while current in self.parents:
            parent = self.parents[current]
            if isinstance(parent, ast.Try) and parent.handlers and current in parent.body:
                lines.append(parent.lineno)
            current = parent
        return lines


def census(source: str, path: str = "<string>") -> List[Site]:
    """Every call of the writer in ``source``, with what precedes, follows and encloses it."""
    module = _Module(source)
    sites = []
    for node in ast.walk(module.tree):
        if not (isinstance(node, ast.Call) and _call_name(node) == WRITER):
            continue
        function = module.enclosing_function(node)
        if function is None:
            sites.append(Site(path, node.lineno, "<module>", False, (), tuple(module.enclosing_excepts(node))))
            continue
        chain = module.path_to(node, function)
        before = [s for block, index in chain for s in block[:index]]
        after = module.reachable_saves_after(node, function)
        # One level up: an in-module caller that saves after calling this function.
        for caller_call in ast.walk(module.tree):
            if isinstance(caller_call, ast.Call) and _call_name(caller_call) == function.name:
                caller = module.enclosing_function(caller_call)
                if caller is not None and caller is not function:
                    after += module.reachable_saves_after(caller_call, caller)
        sites.append(Site(
            path, node.lineno, function.name, module.saves(before),
            tuple(sorted(set(after))), tuple(module.enclosing_excepts(node))))
    return sites


def _repo_sites() -> List[Site]:
    sites = []
    for path in sorted(SRC.rglob("*.py")):
        source = path.read_text()
        if WRITER in source:
            sites += census(source, str(path.relative_to(SRC.parent)))
    return [s for s in sites if not s.path.endswith("train/common/run_io.py")]


SITES = _repo_sites()


def test_the_census_sees_every_call():
    """No call escapes the AST walk (an alias or a wrapper would hide a site)."""
    text_calls = 0
    for path in SRC.rglob("*.py"):
        if path.name == "run_io.py" and path.parent.name == "common":
            continue
        text_calls += len(re.findall(rf"\b{WRITER}\(", path.read_text()))
    assert len(SITES) == text_calls
    assert len(SITES) >= 50


@pytest.mark.parametrize("site", SITES, ids=lambda s: f"{s.path}:{s.line}")
def test_a_model_save_precedes_the_write_and_none_follows_it(site):
    assert site.save_before, f"{site.path}:{site.line} ({site.function}) writes before any model save"
    assert not site.saves_after, (
        f"{site.path}:{site.line} ({site.function}) writes the history before the model "
        f"save at line(s) {site.saves_after}; a failed write would cost those weights")


@pytest.mark.parametrize("site", SITES, ids=lambda s: f"{s.path}:{s.line}")
def test_no_except_encloses_the_write(site):
    assert not site.enclosing_excepts, (
        f"{site.path}:{site.line} ({site.function}) is inside try/except at line(s) "
        f"{site.enclosing_excepts}; the failure would be swallowed")


_GOOD = '''
import keras
def train(model, data):
    history = model.fit(data, callbacks=[keras.callbacks.ModelCheckpoint("b.keras")])
    model.save("final.keras")
    save_training_history_json(history, ".")
'''

_DIVERGED = '''
def train(model, data, cbs):
    history = model.fit(data, callbacks=cbs)  # ModelCheckpoint in cbs
    if bad(history):
        save_training_history_json(history, ".")
        raise RuntimeError("diverged")
    model.save("final.keras")
    save_training_history_json(history, ".")
'''

_BAD = {
    "write before the final save": ('''
def train(model, data):
    history = model.fit(data, callbacks=[ModelCheckpoint("b.keras")])
    save_training_history_json(history, ".")
    model.save("final.keras")
''', "after"),
    "write inside try/except": ('''
def train(model, data):
    model.save("final.keras")
    try:
        save_training_history_json(model.fit(data), ".")
    except Exception:
        pass
''', "except"),
    "no save at all": ('''
def train(model, data):
    history = model.fit(data, callbacks=[])
    save_training_history_json(history, ".")
''', "before"),
    "the caller saves after": ('''
def train(model, data):
    model.save_weights("w.h5")
    save_training_history_json(model.fit(data), ".")
def main():
    model = make()
    train(model, None)
    model.save("final.keras")
''', "after"),
    "a helper saves after": ('''
def finish(model):
    model.save("final.keras")
def train(model, data):
    history = model.fit(data, callbacks=[ModelCheckpoint("b.keras")])
    save_training_history_json(history, ".")
    finish(model)
''', "after"),
}


def _violations(site: Site) -> List[str]:
    return ([] if site.save_before else ["before"]) + (["after"] if site.saves_after else []) + (
        ["except"] if site.enclosing_excepts else [])


def test_positive_control_good_sources_pass():
    for source in (_GOOD, _DIVERGED):
        sites = census(source)
        assert sites and all(not _violations(s) for s in sites), [(s, _violations(s)) for s in sites]


@pytest.mark.parametrize("name", sorted(_BAD))
def test_positive_control_a_planted_bad_site_is_caught(name):
    source, expected = _BAD[name]
    (site,) = census(source)
    assert expected in _violations(site), (name, site)
