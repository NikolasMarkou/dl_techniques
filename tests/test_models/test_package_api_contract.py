"""Package-level API contract for every subpackage of ``dl_techniques.models``.

These tests are cheap (no model is built) and cover the class of defect that
per-model suites structurally cannot see: a package's *public surface*. They are
parameterized over the real directory listing, so a new model package is covered
the moment it is added — nobody has to remember to extend a hand-written list.

Motivating defect: ``models/convnext/__init__.py`` declared

    __all__ = [ConvNeXtV1, ConvNeXtV2, create_convnext_v1, create_convnext_v2]

with the *objects* rather than their names. Every convnext test passed — none of
them imported the package that way — while ``from dl_techniques.models.convnext
import *`` raised ``TypeError: Item in __all__ must be str, not type``. An AST
scan showed convnext was the only package affected; ``test_all_entries_are_strings``
is what keeps it that way.

See ``src/dl_techniques/models/CLAUDE.md`` § "House Model Module Shape" for the
convention these tests enforce.
"""

import ast
import importlib
import inspect
import re
import warnings
from pathlib import Path
from typing import Any, List, Tuple

import pytest

MODELS_DIR = Path(__file__).resolve().parents[2] / "src" / "dl_techniques" / "models"


def _package_names() -> List[str]:
    """Every top-level model subpackage, from the directory listing."""
    return sorted(
        p.name
        for p in MODELS_DIR.iterdir()
        if p.is_dir() and p.name != "__pycache__" and (p / "__init__.py").exists()
    )


PACKAGES = _package_names()


def _pretrained_factories() -> List[Tuple[str, Any]]:
    """Every public model factory that (a) takes ``pretrained`` and (b) is callable
    with no other argument.

    Discovered from the real package listing, so a new model package is covered the
    moment it is added. The "no other required argument" filter is what keeps this
    test cheap and generic: it needs no hand-written table of variant names, and the
    contract under test (``pretrained=True`` raises) fires before anything is built.
    """
    out: List[Tuple[str, Any]] = []
    seen = set()
    for pkg in PACKAGES:
        try:
            module = importlib.import_module(f"dl_techniques.models.{pkg}")
        except Exception:  # covered by test_package_imports
            continue
        for name in dir(module):
            if name.startswith("_"):
                continue
            obj = getattr(module, name)
            candidates = []
            if inspect.isfunction(obj):
                candidates.append((f"{pkg}.{name}", obj))
            elif inspect.isclass(obj) and callable(getattr(obj, "from_variant", None)):
                candidates.append((f"{pkg}.{name}.from_variant", obj.from_variant))
            for label, fn in candidates:
                try:
                    sig = inspect.signature(fn)
                except (TypeError, ValueError):
                    continue
                params = sig.parameters
                if "pretrained" not in params:
                    continue
                required = [
                    n
                    for n, p in params.items()
                    if n != "pretrained"
                    and p.default is inspect.Parameter.empty
                    and p.kind
                    not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                ]
                if required:
                    continue
                key = (fn.__module__, getattr(fn, "__qualname__", label))
                if key in seen:
                    continue
                seen.add(key)
                out.append((label, fn))
    return sorted(out, key=lambda t: t[0])


PRETRAINED_FACTORIES = _pretrained_factories()


def _all_node(init_path: Path):
    """Return the ``__all__`` assignment node, or None if the file has none."""
    tree = ast.parse(init_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == "__all__" for t in node.targets
        ):
            return node.value
    return None


class TestPackageDiscovery:
    """The parameterization itself must not silently collapse to nothing."""

    def test_packages_were_found(self):
        assert MODELS_DIR.is_dir(), f"models dir not found at {MODELS_DIR}"
        assert len(PACKAGES) > 50, (
            f"expected the full model package set, found {len(PACKAGES)}: {PACKAGES}"
        )

    def test_parent_init_is_not_a_public_api(self):
        """``dl_techniques.models`` itself exports nothing; import from the subpackage.

        Pinned because the parent init being empty is a documented convention
        (``models/CLAUDE.md``), not an oversight, and a well-meaning "fix" that
        re-exports all 73 packages there would make importing any single model
        pull in every model in the library.
        """
        parent = importlib.import_module("dl_techniques.models")
        assert not getattr(parent, "__all__", []), (
            "dl_techniques.models must stay empty; import from the subpackage"
        )


@pytest.mark.parametrize("pkg", PACKAGES)
class TestAllDeclaration:
    """``__all__``, where declared, must be well-formed and honest."""

    def test_all_entries_are_strings(self, pkg: str):
        """``__all__`` must hold NAMES, not the objects themselves.

        A list of objects passes every ordinary import but breaks ``import *``
        with ``TypeError: Item in __all__ must be str``.
        """
        node = _all_node(MODELS_DIR / pkg / "__init__.py")
        if node is None:
            pytest.skip(f"{pkg} declares no __all__")
        assert isinstance(node, (ast.List, ast.Tuple)), (
            f"{pkg}: __all__ must be a list or tuple literal"
        )
        offenders = [
            ast.dump(e) for e in node.elts if not isinstance(e, ast.Constant)
        ]
        assert not offenders, (
            f"{pkg}: __all__ must contain string names, not objects. "
            f"Offending entries: {offenders}"
        )

    def test_all_entries_resolve(self, pkg: str):
        """Every name in ``__all__`` must actually be bound by the package."""
        module = importlib.import_module(f"dl_techniques.models.{pkg}")
        declared = getattr(module, "__all__", None)
        if not declared:
            pytest.skip(f"{pkg} declares no __all__")
        missing = [name for name in declared if not hasattr(module, name)]
        assert not missing, f"{pkg}: __all__ names not bound by the package: {missing}"

    def test_no_duplicate_entries(self, pkg: str):
        module = importlib.import_module(f"dl_techniques.models.{pkg}")
        declared = getattr(module, "__all__", None)
        if not declared:
            pytest.skip(f"{pkg} declares no __all__")
        duplicates = sorted({n for n in declared if declared.count(n) > 1})
        assert not duplicates, f"{pkg}: duplicate __all__ entries: {duplicates}"


@pytest.mark.parametrize("pkg", PACKAGES)
class TestPackageImports:
    """A package must import cleanly, and its submodules must not be dead."""

    def test_package_imports(self, pkg: str):
        importlib.import_module(f"dl_techniques.models.{pkg}")

    def test_star_import_succeeds(self, pkg: str):
        """``from dl_techniques.models.<pkg> import *`` must not raise.

        This is the exact call that the convnext ``__all__`` defect broke while
        the package's own 73-test suite stayed green.
        """
        namespace: dict = {}
        exec(f"from dl_techniques.models.{pkg} import *", namespace)  # noqa: S102


class TestDecisionAnchorsIntact:
    """``# DECISION <plan-id>/D-NNN`` comments are a tracked, append-only record.

    They resolve through ``plans/ANCHORS.md`` and are the only thing keeping a
    non-obvious code choice explicable after its plan directory is gone. A
    comment-tidying sweep that removes one destroys that link silently, so the
    count is pinned here rather than left to reviewer attention.

    This is deliberately a floor, not an equality: adding new anchors is normal
    and must not fail the suite; losing them is what must fail.

    The pattern must match the anchor FORM (``# DECISION <plan-id>/D-NNN``), not
    the bare phrase. A plain ``grep "# DECISION"`` reports 284 because
    ``bias_free_denoisers/bfconvunext.py`` mentions ``# DECISION`` inside a
    docstring while pointing at the real anchor below it. Counting that mention
    as an anchor is the same false positive this repo has hit repeatedly with
    mechanical scans, and it would make the floor un-holdable the moment the
    docstring were reworded.
    """

    #: True anchors under src/dl_techniques/models/, measured 2026-08-14 and
    #: confirmed identical at commit 4300b2f19 (pre-work) and at HEAD.
    MINIMUM_ANCHOR_COUNT = 283

    ANCHOR_RE = re.compile(r"# DECISION [A-Za-z0-9_.-]+/D-\d+")

    def test_anchor_count_has_not_regressed(self):
        total = 0
        for path in MODELS_DIR.rglob("*.py"):
            total += len(self.ANCHOR_RE.findall(path.read_text()))
        assert total >= self.MINIMUM_ANCHOR_COUNT, (
            f"DECISION anchor count fell to {total}, below the pinned floor of "
            f"{self.MINIMUM_ANCHOR_COUNT}. A comment cleanup has removed tracked "
            f"provenance anchors; restore them and see plans/ANCHORS.md."
        )


class TestNoPlaceholderWeightURLs:
    """No model may ship a pretrained-weights table of unreachable URLs.

    Until 2026-08-14, 83 ``https://example.com/...`` URLs sat across 12 files in
    ``PRETRAINED_WEIGHTS`` tables. Each was paired with a ``try/except`` in
    ``from_variant`` that logged a warning and continued, so ``pretrained=True``
    returned a randomly-initialized model and the caller was never told. The
    house contract is now that ``_download_weights`` raises
    ``NotImplementedError``; this test stops the placeholder pattern coming back.
    """

    def test_no_example_com_urls(self):
        offenders = []
        for path in MODELS_DIR.rglob("*.py"):
            for i, line in enumerate(path.read_text().splitlines(), start=1):
                if "example.com" in line:
                    offenders.append(f"{path.relative_to(MODELS_DIR)}:{i}")
        assert not offenders, (
            "placeholder weight URLs are forbidden; make _download_weights raise "
            f"NotImplementedError instead. Found: {offenders}"
        )


def _log_only(body: List[ast.stmt]) -> bool:
    """True if every statement in ``body`` is a bare ``logger.*(...)`` call."""
    if not body:
        return False
    for stmt in body:
        if not isinstance(stmt, ast.Expr):
            return False
        call = stmt.value
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "logger"
        ):
            return False
    return True


class TestPretrainedNeverSilentlyRandom:
    """``pretrained`` must never resolve to "log something and hand back random weights".

    ``TestNoPlaceholderWeightURLs`` above only forbids the literal ``example.com``.
    That is the *symptom* the 2026-08-14 sweep happened to leave behind, not the
    contract: nine factories -- ``dino_v2``, ``dino_v3``, ``swin_transformer``,
    ``mobilenet_v1``-``v4``, ``mobile_clip_v1``, ``mobile_clip_v2`` -- had no URL
    table at all, wrote ``if pretrained: logger.warning(...)``, and passed the
    ``example.com`` test while returning an untrained model to a caller who asked
    for a trained one. ``mobile_clip_v2`` is recent code, so the contract was not
    reaching new work either.

    Two arms, deliberately:

    * a **static** arm that reads every ``models/`` module, so packages whose
      factories need variant names / vocab sizes are still covered;
    * a **behavioural** arm over the factories that are callable with no other
      argument, so the assertion is about what the code *does*, not what it looks
      like. Both are derived from the real directory listing.
    """

    def test_no_pretrained_branch_only_logs(self):
        """No ``if pretrained:`` branch may consist solely of logging.

        Matching on the AST shape rather than on the warning text is deliberate:
        the nine sites shared the string "Pretrained weights are not yet
        implemented", and a guard keyed to that string would be defeated by
        rewording it. A branch that only logs cannot be doing anything else --
        it must raise, or load something.
        """
        offenders = []
        for path in sorted(MODELS_DIR.rglob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.If):
                    continue
                if not any(
                    isinstance(n, ast.Name) and n.id == "pretrained"
                    for n in ast.walk(node.test)
                ):
                    continue
                if _log_only(node.body):
                    offenders.append(f"{path.relative_to(MODELS_DIR)}:{node.lineno}")
        assert not offenders, (
            "a `pretrained` branch that only logs returns a randomly initialized "
            "model to a caller who asked for a trained one. Raise "
            "NotImplementedError instead (see models/CLAUDE.md Axis 3 and "
            f"resnet/model.py). Found: {offenders}"
        )

    def test_the_behavioural_arm_found_factories(self):
        """The parameterization must not silently collapse to nothing."""
        assert len(PRETRAINED_FACTORIES) >= 20, (
            "expected at least 20 no-required-argument factories taking "
            f"`pretrained`; discovery found {len(PRETRAINED_FACTORIES)}: "
            f"{[label for label, _ in PRETRAINED_FACTORIES]}"
        )

    @pytest.mark.parametrize(
        "label,factory",
        PRETRAINED_FACTORIES,
        ids=[label for label, _ in PRETRAINED_FACTORIES],
    )
    def test_pretrained_true_raises(self, label: str, factory):
        """``pretrained=True`` must raise; no public weights ship with this repo."""
        with pytest.raises(NotImplementedError):
            factory(pretrained=True)


def _docstring_line_numbers(path: Path) -> set:
    """1-based line numbers covered by any module/class/function docstring."""
    lines: set = set()
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:  # covered by the import tests
        return lines
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, holders):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(
            first.value.value, str
        ):
            lines.update(range(first.lineno, first.end_lineno + 1))
    return lines


class TestNoKeras2Residues:
    """The forward path must use Keras 3 spellings.

    ``keras.backend.GradientTape`` does not exist in Keras 3 at all: it sat in
    ``latent_gmm_registration.train_step`` and made the model untrainable while
    its suite stayed green, because every test was forward-pass only.
    """

    def test_no_keras_backend_calls(self):
        offenders = []
        for path in MODELS_DIR.rglob("*.py"):
            docstring_lines = _docstring_line_numbers(path)
            for i, line in enumerate(path.read_text().splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                # Prose is not code. Before this exclusion, a module docstring
                # explaining *why* `keras.backend.` must not be used failed the
                # suite -- the guard could not be documented in the tree it guards.
                if i in docstring_lines:
                    continue
                if "keras.backend." in line:
                    offenders.append(f"{path.relative_to(MODELS_DIR)}:{i} {stripped}")
        assert not offenders, (
            "use keras.config.floatx()/epsilon() and tf.GradientTape; "
            f"keras.backend.* found at: {offenders}"
        )


# ---------------------------------------------------------------------------
# Silent-kwarg-drop guard (the inverse of the factories' strict-drop raise).
# ---------------------------------------------------------------------------

LAYERS_DIR = Path(__file__).resolve().parents[2] / "src" / "dl_techniques" / "layers"

#: Paths in every sweep below are reported relative to this, so a waiver key is
#: stable no matter which root the file was reached through.
SRC_ROOT = MODELS_DIR.parent

#: Root set of the registry sweep. Unchanged since the guard shipped: the three
#: registry-backed factories are called from ``models/`` and from the shared
#: transformer blocks, and widening it has never been measured to add a site.
_REGISTRY_SWEEP_ROOTS = (MODELS_DIR, LAYERS_DIR / "transformers")

#: Root set of the two NORM sweeps, deliberately WIDER than the registry sweep's.
#: ``grep -rn "TransformerLayer(" src/dl_techniques/`` (2026-08-19) found three
#: construction sites outside ``layers/transformers/`` -- ``layers/blt_blocks.py``,
#: ``layers/graphs/relational_graph_transformer_blocks.py`` and
#: ``layers/reasoning/hrm_reasoning_module.py`` -- so the narrower root set would
#: have been silently blind to them. Measured cost of widening: 0 extra hits, and
#: the sweep still runs in well under a second.
_NORM_SWEEP_ROOTS = (MODELS_DIR, LAYERS_DIR)

#: The three factories whose registries declare a per-type parameter set. Maps
#: the factory's function name to (its type-selecting parameter, its registry).
#: ``norms``/``activations``/``sampling`` are deliberately absent: they have no
#: ``required_params``/``optional_params`` registry, so there is no declared-param
#: ground truth to diff a call site against.
_REGISTRY_FACTORIES = {
    "create_attention_layer": "attention_type",
    "create_ffn_layer": "ffn_type",
    "create_embedding_layer": "embedding_type",
}

#: Attributes every Keras layer/model carries. A call site is never "demonstrably
#: had the value" because ``keras.layers.Layer`` set one of these.
_KERAS_BASE_ATTRS = frozenset(
    {
        "built",
        "name",
        "trainable",
        "dtype",
        "dtype_policy",
        "supports_masking",
        "activity_regularizer",
        "input_spec",
    }
)

#: Sites that are SCHEDULED WORK, not accepted exceptions.
#:
#: **EMPTY as of 2026-08-18.** All 13 entries were fixed at their call sites; the
#: constant stays so the guard's shape is unchanged and so the next measured
#: instance has somewhere to be waived while its fix lands. It must never be used
#: to park an omission that is actually deliberate -- that is what
#: ``_NAME_COLLISIONS`` below is for, and it carries the read that clears it.
#:
#: Twelve of the thirteen were forwarded verbatim. The thirteenth pair -- both
#: Qwen3 wrappers' ``positional_learned`` ``dropout_rate`` -- was a REFUTATION:
#: the classes already applied that rate through a second standalone ``Dropout``
#: on the same tensor, so forwarding alone would have stacked two dropouts
#: (effective ``1-(1-p)^2``). They are cleared by forwarding the kwarg AND
#: deleting the redundant layer; see
#: ``tests/test_models/test_qwen/test_embedding_dropout_applied_once.py``.
#:
#: Key is ``(path relative to src/dl_techniques, class, factory type, param)``.
#: Deliberately NOT keyed by line number: the fixes themselves move lines, and a
#: waiver that silently stops matching is a waiver that hides a live defect.
_SCHEDULED_FIXES: set = set()

#: Name collisions this static predicate cannot resolve, cleared by manual read.
#:
#: Unlike ``_SCHEDULED_FIXES`` these are permanent: the enclosing class stores an
#: attribute that merely *shares a name* with a declared factory parameter.
#:
#: Five of the six are the ViT-family ``scale``, which is the variant-size string
#: (``"base"``, ``"large"``, ...; e.g. ``vit/model.py`` ``self.scale = str(scale)``)
#: and has nothing to do with ``positional_learned``'s embedding-scale parameter.
#: No AST predicate can tell the two apart -- both are ``self.scale`` assigned
#: from a same-named ``__init__`` argument -- so the discrimination is recorded
#: here, with its evidence, rather than pretended at.
#:
#: The sixth is ``ViT.activation``, added 2026-08-18 when the four REAL drops at
#: the same call site were fixed. ``patch_2d`` declares an ``activation``
#: (default ``'linear'``) and ``ViT`` stores ``self.activation``, but ViT's is the
#: FFN activation -- documented as such in ``vit/model.py`` and passed to every
#: ``TransformerLayer`` -- and forwarding its ``'gelu'`` default into the patch
#: projection would make the stem nonlinear, which no ViT is. See the
#: ``D-022`` anchor at that call site.
_NAME_COLLISIONS = {
    ("models/vit/model.py", "ViT", "patch_2d", "activation"),
    ("models/vit/model.py", "ViT", "positional_learned", "scale"),
    ("models/vit_hmlp/model.py", "ViTHMLP", "positional_learned", "scale"),
    ("models/vit_siglip/model.py", "SigLIPVisionTransformer", "positional_learned", "scale"),
    ("models/beit/model.py", "BeitModel", "positional_learned", "scale"),
    ("models/energy_transformer/model.py", "EnergyTransformerBackbone", "positional_learned", "scale"),
}


def _declared_params(registry: dict, type_name: str):
    """Every parameter a registry entry declares, or None if the type is unknown."""
    entry = registry.get(type_name)
    if entry is None:
        return None
    return set(entry.get("required_params") or []) | set(entry.get("optional_params") or {})


def _init_stored_attrs(cls: ast.ClassDef) -> set:
    """``self.<name>`` targets assigned anywhere in the class's ``__init__``."""
    init = next(
        (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
        None,
    )
    if init is None:
        return set()
    stored = set()
    for node in ast.walk(init):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                stored.add(target.attr)
    return stored - _KERAS_BASE_ATTRS


def _callee_name(node: ast.Call) -> str:
    """The bare name a call node invokes, for ``f(...)`` and ``mod.f(...)`` alike.

    Contract: takes an ``ast.Call``; returns the callee's last name component, or
    ``""`` when the callee is neither a ``Name`` nor an ``Attribute`` (a call on a
    subscript or a call result, which no sweep in this file matches). Shared by all
    three sweeps below -- do not re-derive it inline.
    """
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _iter_classes(roots, src_root):
    """Yield ``(relpath, ast.ClassDef)`` for every class under ``roots``.

    Contract: ``roots`` is any iterable of directories to ``rglob("*.py")``;
    ``src_root`` is the directory paths are reported relative to (it must be a
    parent of every root, or ``relative_to`` raises). Files that do not parse are
    skipped silently -- ``test_every_package_imports`` is what catches those.
    Shared by the three call-site sweeps so they cannot drift in what they walk.
    """
    for root in roots:
        for path in sorted(Path(root).rglob("*.py")):
            rel = path.relative_to(src_root).as_posix()
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:  # covered by the import tests
                continue
            for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
                yield rel, cls


def _sweep_factory_call_sites() -> Tuple[List[tuple], List[str]]:
    """Return ``(dropped, dynamic)`` for every registry-backed factory call site.

    ``dropped`` holds ``(relpath, lineno, class, type, param)`` for each call that
    omits a parameter the registry declares *and* the enclosing class stores --
    the demonstrable "the caller had the value and did not forward it" signature.
    ``dynamic`` holds ``"path:line"`` for call sites whose type is a variable,
    which cannot be resolved statically.
    """
    from dl_techniques.layers.attention.factory import ATTENTION_REGISTRY
    from dl_techniques.layers.embedding.factory import EMBEDDING_REGISTRY
    from dl_techniques.layers.ffn.factory import FFN_REGISTRY

    registries = {
        "create_attention_layer": (_REGISTRY_FACTORIES["create_attention_layer"], ATTENTION_REGISTRY),
        "create_ffn_layer": (_REGISTRY_FACTORIES["create_ffn_layer"], FFN_REGISTRY),
        "create_embedding_layer": (_REGISTRY_FACTORIES["create_embedding_layer"], EMBEDDING_REGISTRY),
    }
    dropped: List[tuple] = []
    dynamic: List[str] = []

    for rel, cls in _iter_classes(_REGISTRY_SWEEP_ROOTS, SRC_ROOT):
        stored = _init_stored_attrs(cls)
        for node in ast.walk(cls):
            if not isinstance(node, ast.Call):
                continue
            fname = _callee_name(node)
            if fname not in registries:
                continue
            type_kw, registry = registries[fname]
            type_node = node.args[0] if node.args else None
            for kw in node.keywords:
                if kw.arg == type_kw:
                    type_node = kw.value
            if not (
                isinstance(type_node, ast.Constant)
                and isinstance(type_node.value, str)
            ):
                dynamic.append(f"{rel}:{node.lineno}")
                continue
            declared = _declared_params(registry, type_node.value)
            if declared is None:
                dynamic.append(f"{rel}:{node.lineno} (unknown type {type_node.value!r})")
                continue
            # A `**config` unpack can carry anything; only literal kwargs
            # are provably present, so an unpack clears the whole call.
            if any(kw.arg is None for kw in node.keywords):
                continue
            passed = {kw.arg for kw in node.keywords}
            for param in sorted((declared - passed) & stored):
                dropped.append(
                    (rel, node.lineno, cls.name, type_node.value, param)
                )
    return dropped, dynamic


class TestFactoryKnobsAreForwarded:
    """A parameter the caller stores and the factory declares must be passed.

    ``attention``/``ffn``/``embedding`` factories are STRICT about the *undeclared*
    direction -- passing a key the type does not accept raises
    ``STRICT_DROPPED_KEY_MARKER``. This is the inverse and until now unguarded
    direction: the call site hand-writes its kwarg list, the enclosing class holds
    the value in ``self.<name>``, ``get_config()`` faithfully serializes it, and
    the factory never sees it. Nothing raises, nothing warns, and every existing
    test passes because the built layer is *valid* -- just not the one that was
    asked for. Thirteen live instances across five call sites were found the day
    this guard was written, including a Swin/ViT patch embedding that trains at
    glorot no matter which initializer the caller supplies.

    The predicate is deliberately narrow, and each narrowing is a place it can
    miss a real defect:

    * only **literal** ``type=`` strings (a variable type has no static declared-
      param set; those sites are reported non-fatally by
      ``test_dynamic_call_sites_are_reported``);
    * only calls with **no** ``**unpack`` (an unpack may well carry the param);
    * only the three registry-backed factories -- ``create_normalization_layer``
      has no ``required_params``/``optional_params`` registry, so the norm-epsilon
      family of this same defect is out of reach here. **Superseded 2026-08-19**:
      the norm family is now covered by ``TestNormalizationKnobsAreForwarded`` and
      ``TestTransformerLayerNormArgsAreForwarded`` below, which reach it through
      ``inspect.signature`` and through the ``TransformerLayer(...)`` indirection
      respectively -- but THIS predicate still does not, and must not be widened
      to try. This is also why the two by-design omissions catalogued in the sweep
      that motivated this guard (``adaln_zero.py`` and ``text_encoder.py``'s
      ``**norm_config`` unpack) need no waiver here: both are
      ``create_normalization_layer`` calls and this predicate never reaches them.
      Waiving them anyway would have been a waiver guarding nothing;
    * only params the enclosing class stores in ``__init__``, which is what makes
      a hit *demonstrable* rather than merely suspicious.
    """

    def test_no_declared_and_stored_param_is_dropped(self):
        # DECISION plan-2026-08-18T140459-7991552f/D-015
        # Guards the SILENT-DROP direction of the factory contract. The opposite
        # direction -- a caller passing a key the type does not declare -- is
        # already guarded at runtime by the strict raise added in D-011/D-023
        # (`STRICT_DROPPED_KEY_MARKER`). That raise cannot see this direction at
        # all: an omitted kwarg is indistinguishable from a caller who wanted the
        # default. Do NOT "simplify" this into a runtime check inside the
        # factories -- the information needed (does the CALLER hold this value?)
        # exists only at the call site, statically. Do NOT widen it by dropping
        # the "enclosing class stores it" clause either: without that clause every
        # deliberate use of a factory default becomes a failure. See
        # plans/.../findings/dead-knob-systematic-sweep.md Part D, which rejected
        # a per-call-site test (a) and a `strict_forward` helper (b) in favour of
        # this AST guard (c).
        dropped, _ = _sweep_factory_call_sites()
        waived = _SCHEDULED_FIXES | _NAME_COLLISIONS
        offenders = [
            f"{rel}:{line} {cls}: create_*_layer({typ!r}) drops {param!r} "
            f"(self.{param} is stored in __init__)"
            for rel, line, cls, typ, param in dropped
            if (rel, cls, typ, param) not in waived
        ]
        assert not offenders, (
            "a factory call site drops a parameter its enclosing class stores. "
            "Forward it, or -- if the omission is deliberate -- add it to "
            "_NAME_COLLISIONS with the read that clears it. Found:\n  "
            + "\n  ".join(offenders)
        )

    def test_waivers_still_match_a_real_site(self):
        """A waiver that no longer matches anything is a waiver hiding nothing.

        Both lists are keyed by (path, class, type, param) precisely so they keep
        matching as the fixes move lines. A stale entry means either the fix
        landed (delete the entry) or the code moved (re-key it) -- and a stale
        entry left in place would silently start waiving nothing while looking
        like it still guards something.
        """
        dropped, _ = _sweep_factory_call_sites()
        live = {(rel, cls, typ, param) for rel, _, cls, typ, param in dropped}
        stale = sorted((_SCHEDULED_FIXES | _NAME_COLLISIONS) - live)
        assert not stale, (
            "waiver entries no longer match any call site; delete them if the "
            f"fix landed, re-key them if the code moved: {stale}"
        )

    def test_the_sweep_found_call_sites(self):
        """The parameterization must not silently collapse to nothing."""
        dropped, dynamic = _sweep_factory_call_sites()
        assert len(dropped) + len(dynamic) >= 20, (
            "expected the factory call-site sweep to reach dozens of sites; it "
            f"found {len(dropped)} dropped-param hits and {len(dynamic)} dynamic "
            "sites, which means the AST walk stopped seeing the tree"
        )

    def test_dynamic_call_sites_are_reported(self):
        """Non-fatal: list the call sites this guard structurally cannot check.

        A variable ``type=`` has no static declared-param set. These are not
        unguarded in practice -- the factories' strict raise fires the moment such
        a path is actually built -- but they are invisible to the check above, and
        an inventory that shrinks silently is how a guard's coverage rots.
        """
        _, dynamic = _sweep_factory_call_sites()
        if dynamic:
            warnings.warn(
                "factory call sites with a non-literal type (not statically "
                f"checkable, {len(dynamic)}): {dynamic}",
                UserWarning,
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# Norm-forwarding guards: the same silent-drop defect, one indirection away.
# ---------------------------------------------------------------------------

#: Stored-attribute names that mean "this class holds a normalization epsilon".
#:
#: A plain ``"eps" in name`` substring test matches ``mask_annealing_steps``
#: (``layers/transformers/eomt_transformer.py``) -- st-EPS. The token form below
#: matches ``eps``, ``epsilon``, ``norm_eps``, ``layer_norm_eps``,
#: ``norm_epsilon`` and misses ``steps``/``keeps``/``epsilon_decay``-style
#: near-misses. Measured 2026-08-19: the substring form found 8 candidate classes,
#: this form finds 7, and the one it drops is the ``steps`` collision.
_EPS_ATTR_RE = re.compile(r"(?:^|_)(?:eps|epsilon)(?:$|_)")

#: The two ``TransformerLayer`` parameters that carry a caller's norm config into
#: every in-block norm. ``TransformerLayer._create_normalization_layer`` calls
#: ``create_normalization_layer(..., **custom_args)`` where ``custom_args`` is one
#: of these (default ``{}``), so omitting both silently pins every in-block norm
#: to the FACTORY's ``epsilon`` default regardless of the model's own knob.
_TRANSFORMER_NORM_ARG_KWARGS = ("attention_norm_args", "ffn_norm_args")

#: Name collisions for the DIRECT ``create_normalization_layer`` predicate,
#: cleared by manual read. Sibling of ``_NAME_COLLISIONS``, deliberately a
#: separate constant: that one is keyed by a registry factory's declared params,
#: this one by ``_accepted_params``' ``inspect.signature``-derived set, and
#: conflating the two would hide which predicate a waiver actually clears.
#:
#: The single entry is the ViT-family ``scale`` shape already catalogued in
#: ``_NAME_COLLISIONS``, reached here through a different factory:
#: ``SigLIPVisionTransformer`` stores ``self.scale = str(scale)`` (the variant
#: size, ``vit_siglip/model.py:357``) while ``LayerNormalization.__init__``
#: declares ``scale: bool``. Same two words, unrelated meanings.
#:
#: Key is ``(path relative to src/dl_techniques, class, normalization type,
#: param)`` -- never a line number, for the reason ``_SCHEDULED_FIXES`` gives.
_NORM_NAME_COLLISIONS = {
    ("models/vit_siglip/model.py", "SigLIPVisionTransformer", "layer_norm", "scale"),
}


def _stored_eps_attrs(stored: set) -> set:
    """The subset of ``stored`` that names a normalization epsilon."""
    return {attr for attr in stored if _EPS_ATTR_RE.search(attr.lower())}


def _sweep_transformer_layer_norm_args(roots=None, src_root=None):
    """Find ``TransformerLayer(...)`` sites that cannot pass their own epsilon on.

    Contract: returns ``(hits, n_constructions, n_candidates)``.

    * ``hits`` -- ``(relpath, lineno, class, sorted eps attrs, sorted missing
      kwargs)`` for every construction inside a class that stores an epsilon
      attribute and passes NEITHER ``attention_norm_args`` NOR ``ffn_norm_args``
      (a call passing one of the two is not flagged: the caller demonstrably knows
      the channel exists, and which of the two norms a knob belongs to is a design
      choice this predicate has no standing to make).
    * ``n_constructions`` -- every ``TransformerLayer(...)`` seen, for the vacuity
      assertion.
    * ``n_candidates`` -- those whose enclosing class stores an epsilon attribute.

    ``roots``/``src_root`` default to the real tree; they exist so the predicate
    can be pointed at a synthetic fixture and proven to fire. A ``**unpack`` clears
    a call, exactly as in ``_sweep_factory_call_sites``.
    """
    roots = _NORM_SWEEP_ROOTS if roots is None else roots
    src_root = SRC_ROOT if src_root is None else src_root
    hits: List[tuple] = []
    n_constructions = 0
    n_candidates = 0

    for rel, cls in _iter_classes(roots, src_root):
        eps_attrs = _stored_eps_attrs(_init_stored_attrs(cls))
        for node in ast.walk(cls):
            if not isinstance(node, ast.Call):
                continue
            if _callee_name(node) != "TransformerLayer":
                continue
            n_constructions += 1
            if not eps_attrs:
                continue
            n_candidates += 1
            if any(kw.arg is None for kw in node.keywords):
                continue
            passed = {kw.arg for kw in node.keywords}
            missing = [k for k in _TRANSFORMER_NORM_ARG_KWARGS if k not in passed]
            if len(missing) == len(_TRANSFORMER_NORM_ARG_KWARGS):
                hits.append((rel, node.lineno, cls.name, sorted(eps_attrs), missing))
    return hits, n_constructions, n_candidates


def _sweep_norm_factory_call_sites(roots=None, src_root=None):
    """The registry-shaped predicate for ``create_normalization_layer``.

    Contract: returns ``(dropped, dynamic, n_literal)`` with the same element
    shapes as ``_sweep_factory_call_sites``' ``(dropped, dynamic)``, plus the
    literal-type call count for the vacuity assertion.

    ``norms/factory.py`` has no ``required_params``/``optional_params`` registry,
    so the declared-param ground truth comes from ``_accepted_params(type)``, which
    derives it from ``inspect.signature`` of ``_TYPE_TO_CLASS[type].__init__``. Do
    NOT hand-maintain a second accepted-param list here: that list has drifted
    twice already, which is why the factory replaced it with the signature.
    """
    from dl_techniques.layers.norms.factory import _TYPE_TO_CLASS, _accepted_params

    roots = _NORM_SWEEP_ROOTS if roots is None else roots
    src_root = SRC_ROOT if src_root is None else src_root
    dropped: List[tuple] = []
    dynamic: List[str] = []
    n_literal = 0

    for rel, cls in _iter_classes(roots, src_root):
        stored = _init_stored_attrs(cls)
        for node in ast.walk(cls):
            if not isinstance(node, ast.Call):
                continue
            if _callee_name(node) != "create_normalization_layer":
                continue
            type_node = node.args[0] if node.args else None
            for kw in node.keywords:
                if kw.arg in ("normalization_type", "type"):
                    type_node = kw.value
            if not (
                isinstance(type_node, ast.Constant)
                and isinstance(type_node.value, str)
            ):
                dynamic.append(f"{rel}:{node.lineno}")
                continue
            norm_type = type_node.value
            if norm_type not in _TYPE_TO_CLASS:
                dynamic.append(f"{rel}:{node.lineno} (unknown type {norm_type!r})")
                continue
            n_literal += 1
            if any(kw.arg is None for kw in node.keywords):
                continue
            declared = _accepted_params(norm_type)
            passed = {kw.arg for kw in node.keywords}
            for param in sorted((declared - passed) & stored):
                dropped.append((rel, node.lineno, cls.name, norm_type, param))
    return dropped, dynamic, n_literal


#: A synthetic module, never imported -- only parsed. Both predicates are proven
#: to fire on it, and proven NOT to fire on its fixed twin below, so neither can
#: pass by finding nothing. Kept as source text rather than as a real defect in
#: the tree for the obvious reason: the tree is supposed to be clean.
_INJECTED_DEFECT_SRC = '''
class InjectedTransformerUser(keras.layers.Layer):
    def __init__(self, layer_norm_eps=1e-12, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm_eps = layer_norm_eps
        self.blocks = [
            TransformerLayer(hidden_size=8, num_heads=2, intermediate_size=16)
            for _ in range(2)
        ]


class InjectedNormUser(keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.norm = create_normalization_layer('layer_norm', name='n')
'''

#: The same two classes with the omission repaired. A predicate that flags this
#: too is shape-matching, not measuring.
_INJECTED_FIXED_SRC = '''
class InjectedTransformerUser(keras.layers.Layer):
    def __init__(self, layer_norm_eps=1e-12, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm_eps = layer_norm_eps
        norm_args = {'epsilon': layer_norm_eps}
        self.blocks = [
            TransformerLayer(
                hidden_size=8,
                num_heads=2,
                intermediate_size=16,
                attention_norm_args=dict(norm_args),
                ffn_norm_args=dict(norm_args),
            )
            for _ in range(2)
        ]


class InjectedNormUser(keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.norm = create_normalization_layer(
            'layer_norm', epsilon=epsilon, name='n'
        )
'''


def _write_fixture(tmp_path: Path, source: str) -> Tuple[tuple, Path]:
    """Write ``source`` as a fake package file; return ``(roots, src_root)``."""
    pkg = tmp_path / "models" / "injected"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "model.py").write_text(source)
    return (tmp_path / "models",), tmp_path


class TestTransformerLayerNormArgsAreForwarded:
    """A model's epsilon knob must have a route into its own in-block norms.

    This is the predicate that finds real defects. The direct-call one below is
    the shape the norm factory's own registry gap suggests, but MEASURED against
    this tree it finds zero genuine drops (44 literal-type calls, one hit, and
    that hit is the ``vit_siglip`` ``scale`` name collision). The defect family
    actually lives one indirection out: a model stores ``self.layer_norm_eps``,
    forwards it to its embedding norm, and then builds its encoder stack with
    ``TransformerLayer(...)`` passing neither ``attention_norm_args`` nor
    ``ffn_norm_args`` -- so all ``2*num_layers`` in-block norms run at the
    factory's default instead. Five models did exactly that until 2026-08-19
    (BERT/DistilBERT/ModernBERT at ``1e-6`` instead of their own ``1e-12``, a
    ~1e5x mismatch against the embedding norm they sit beside).

    Narrowings, each a place this can miss a real defect:

    * only classes storing an epsilon-named attribute. ``MobileClipTextEncoder``
      (``models/mobile_clip/components.py``) is invisible to this predicate for
      that reason -- its epsilon is a module-level ``REFERENCE_NORM_EPSILON``
      constant, not a stored attribute -- and it is a *correct* site, so the
      narrowing costs nothing there today. Dropping the clause is not the fix:
      without it every ``TransformerLayer(...)`` that legitimately wants the
      default becomes a failure (14 such sites at the time of writing);
    * only calls with no ``**unpack``;
    * only ``TransformerLayer``. ``TransformerDecoderLayer`` takes the same two
      dicts and is not swept -- no measured instance, and adding it unmeasured
      would be a guard nobody has seen fire;
    * nothing here reaches a knob that has no channel at all. ``GroupAttention``
      (``models/tree_transformer/components.py``) had no ``layer_norm_eps``
      parameter until it was given one as a PRODUCT fix; no call-site predicate
      can see a parameter that does not exist. It is still out of reach today for
      a second reason -- it calls ``create_normalization_layer`` with a dynamic
      type -- and is therefore reported by ``test_dynamic_call_sites_are_reported``
      rather than checked.
    """

    def test_no_eps_storing_class_omits_both_norm_arg_dicts(self):
        # DECISION plan-2026-08-19T070627-a616f581/D-008
        # This guard is written against the TransformerLayer INDIRECTION, not
        # against `create_normalization_layer(...)` call sites, because the
        # direct-call form guards ZERO: measured over models/ + layers/, the
        # signature-derived direct predicate returns exactly one hit and it is a
        # known name collision. The epsilon a model stores never reaches a
        # `create_normalization_layer` call it writes itself -- it reaches (or
        # fails to reach) one that `TransformerLayer._create_normalization_layer`
        # writes, through the `attention_norm_args`/`ffn_norm_args` dicts.
        # WHAT NOT TO DO: do not "simplify" this into the direct-call predicate
        # below, and do not delete this class as redundant with it -- they have
        # disjoint reach, and this is the one with a demonstrated defect family
        # (5 models, fixed the same day this shipped). See decisions.md D-008.
        hits, _, _ = _sweep_transformer_layer_norm_args()
        offenders = [
            f"{rel}:{line} {cls}: TransformerLayer(...) passes neither "
            f"{' nor '.join(missing)} while the class stores {eps} -- every "
            "in-block norm runs at the factory's epsilon default"
            for rel, line, cls, eps, missing in hits
        ]
        assert not offenders, (
            "a model's epsilon knob has no route into its own in-block norms. "
            "Build a `{'epsilon': self.<knob>}` dict and pass it as BOTH "
            "attention_norm_args and ffn_norm_args. Found:\n  "
            + "\n  ".join(offenders)
        )

    def test_the_sweep_found_call_sites(self):
        """The AST walk must not silently collapse to nothing.

        A guard that reports zero offenders because it looked in the wrong place
        is indistinguishable from a clean tree -- which is the entire failure mode
        of writing this guard the day after the last offender was fixed. Floors
        are set well under the 2026-08-19 measurement (28 constructions, 7
        candidates) so ordinary churn does not trip them.
        """
        _, n_constructions, n_candidates = _sweep_transformer_layer_norm_args()
        assert n_constructions >= 20, (
            f"expected dozens of TransformerLayer(...) sites, found "
            f"{n_constructions}: the AST walk stopped seeing the tree"
        )
        assert n_candidates >= 5, (
            f"expected several eps-storing classes to build TransformerLayers, "
            f"found {n_candidates}: the stored-attribute predicate stopped matching"
        )

    def test_predicate_fires_on_an_injected_defect(self, tmp_path):
        """Dead-component probe: the predicate must go RED on a real omission."""
        roots, src_root = _write_fixture(tmp_path, _INJECTED_DEFECT_SRC)
        hits, n_constructions, n_candidates = _sweep_transformer_layer_norm_args(
            roots, src_root
        )
        assert n_constructions == 1 and n_candidates == 1
        assert len(hits) == 1, hits
        rel, _, cls, eps, missing = hits[0]
        assert cls == "InjectedTransformerUser"
        assert eps == ["layer_norm_eps"]
        assert missing == ["attention_norm_args", "ffn_norm_args"]

    def test_predicate_is_silent_on_the_fixed_twin(self, tmp_path):
        """...and must NOT fire once the same site forwards the dicts."""
        roots, src_root = _write_fixture(tmp_path, _INJECTED_FIXED_SRC)
        hits, _, n_candidates = _sweep_transformer_layer_norm_args(roots, src_root)
        assert n_candidates == 1, "the fixture must still be reached"
        assert hits == [], hits


class TestNormalizationKnobsAreForwarded:
    """``create_normalization_layer`` call sites, checked against the signature.

    The registry-shaped sibling of ``TestFactoryKnobsAreForwarded``, for the one
    factory that has no registry: the declared-param set comes from
    ``norms/factory.py::_accepted_params``, i.e. from ``inspect.signature`` of the
    class each type instantiates.

    Its measured yield on this tree is zero genuine drops, and that is expected --
    it is here for the regression direction, not because it found anything. The
    reason is structural and worth stating so nobody "fixes" it by loosening the
    name match: an epsilon knob is almost always stored under a *different* name
    than the parameter that carries it (``self.norm_eps`` -> ``epsilon=``,
    ``self.norm_epsilon`` -> ``epsilon=``), so the stored-attribute clause that
    makes a hit demonstrable is also what makes this predicate quiet. Matching on
    value rather than name would need dataflow, not AST shape.

    Same four narrowings as ``TestFactoryKnobsAreForwarded``, and the first one
    bites much harder here: 138 of ~167 call sites pass a dynamic
    ``normalization_type`` (it is nearly always ``self.normalization_type``), so
    this predicate sees under a fifth of the tree's calls.
    """

    def test_no_declared_and_stored_param_is_dropped(self):
        dropped, _, _ = _sweep_norm_factory_call_sites()
        offenders = [
            f"{rel}:{line} {cls}: create_normalization_layer({typ!r}) drops "
            f"{param!r} (self.{param} is stored in __init__)"
            for rel, line, cls, typ, param in dropped
            if (rel, cls, typ, param) not in _NORM_NAME_COLLISIONS
        ]
        assert not offenders, (
            "a create_normalization_layer call site drops a parameter its "
            "enclosing class stores. Forward it, or -- if the names merely "
            "collide -- add it to _NORM_NAME_COLLISIONS with the read that "
            "clears it. Found:\n  " + "\n  ".join(offenders)
        )

    def test_waivers_still_match_a_real_site(self):
        """A waiver matching nothing is a waiver hiding nothing (see the sibling)."""
        dropped, _, _ = _sweep_norm_factory_call_sites()
        live = {(rel, cls, typ, param) for rel, _, cls, typ, param in dropped}
        stale = sorted(_NORM_NAME_COLLISIONS - live)
        assert not stale, (
            "norm waiver entries no longer match any call site; delete them if "
            f"the collision is gone, re-key them if the code moved: {stale}"
        )

    def test_the_sweep_found_call_sites(self):
        """The walk must reach a plausible share of the tree's call sites."""
        dropped, dynamic, n_literal = _sweep_norm_factory_call_sites()
        assert n_literal >= 15, (
            f"expected dozens of literal-type create_normalization_layer calls, "
            f"found {n_literal} (2026-08-19: 29)"
        )
        assert len(dynamic) >= 50, (
            f"expected the dynamic-type inventory to stay large, found "
            f"{len(dynamic)} (2026-08-19: 138); a collapse here means the AST "
            "walk stopped seeing the tree, not that the tree got more static"
        )
        assert len(dropped) + n_literal >= 15

    def test_predicate_fires_on_an_injected_defect(self, tmp_path):
        """Dead-component probe: a stored, declared, unpassed param must be seen."""
        roots, src_root = _write_fixture(tmp_path, _INJECTED_DEFECT_SRC)
        dropped, _, n_literal = _sweep_norm_factory_call_sites(roots, src_root)
        assert n_literal == 1
        assert len(dropped) == 1, dropped
        _, _, cls, typ, param = dropped[0]
        assert (cls, typ, param) == ("InjectedNormUser", "layer_norm", "epsilon")

    def test_predicate_is_silent_on_the_fixed_twin(self, tmp_path):
        """...and must NOT fire once the same call forwards the parameter."""
        roots, src_root = _write_fixture(tmp_path, _INJECTED_FIXED_SRC)
        dropped, _, n_literal = _sweep_norm_factory_call_sites(roots, src_root)
        assert n_literal == 1, "the fixture must still be reached"
        assert dropped == [], dropped

    def test_dynamic_call_sites_are_reported(self):
        """Non-fatal inventory of what this predicate structurally cannot check.

        ``GroupAttention``'s own norm lives in here, not in the checked set.
        """
        _, dynamic, _ = _sweep_norm_factory_call_sites()
        if dynamic:
            warnings.warn(
                "create_normalization_layer call sites with a non-literal type "
                f"(not statically checkable, {len(dynamic)}): {dynamic}",
                UserWarning,
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# MODEL_VARIANTS: the house rule's variant registry, previously unguarded.
# ---------------------------------------------------------------------------

#: Legacy spellings of the variant registry, per ``models/CLAUDE.md`` Axis 2:
#: "Packages that predate this spec also use ``VARIANT_CONFIGS``, ``NAM_VARIANTS``,
#: ``NTM_VARIANTS`` or ``MCI_VARIANTS`` for that same role; where one of those is
#: the package's *only* variant table, add ``MODEL_VARIANTS`` as a class-level
#: alias to the same dict."
#:
#: ``SCALE_CONFIGS`` is deliberately NOT here: the same section says in bold that
#: it "is NOT a stale spelling of MODEL_VARIANTS, and the two must not be merged
#: where both appear" -- it is the architecture table, MODEL_VARIANTS is the
#: public-name registry, and ``beit``/``vit``/``energy_transformer`` carry both.
_LEGACY_VARIANT_TABLE_RE = re.compile(r"^(?:[A-Z0-9]+_VARIANTS|VARIANT_CONFIGS)$")

#: Deliberate exceptions, keyed by ``(relpath, symbol, kind)`` -- never by line
#: number, for the reason ``_SCHEDULED_FIXES`` gives. Each carries the read that
#: clears it, and ``test_waivers_still_match_a_real_site`` fails if one stops
#: matching, so a waiver cannot outlive the thing it waives.
_MODEL_VARIANTS_WAIVERS = {
    # SD3VAE is not a keras.Model: it is a plain Python holder pairing an
    # ideogram4 ``AutoEncoder`` with SD3's latent-norm helpers (its own docstring
    # says so). Its ``from_variant`` delegates to ``create_sd3_vae`` ->
    # ``config.get_sd3_config(variant)``, and the sd3_mmdit family's variant
    # registry has ONE home there (``config.PRESETS``, shared by the transformer,
    # the VAE and the pipeline). Restating it as ``SD3VAE.MODEL_VARIANTS`` would
    # create the second home the house rule exists to prevent. models/CLAUDE.md
    # § "When the shape does not apply": multi-model families apply the shape per
    # inner architecture, and the inner architecture here is ``AutoEncoder``.
    (
        "models/sd3_mmdit/vae.py",
        "SD3VAE",
        "from_variant-without-table",
    ),
}


def _module_level_names(tree: ast.Module) -> set:
    """Names assigned at module scope (``Assign`` and ``AnnAssign`` alike)."""
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _class_body_names(cls: ast.ClassDef) -> set:
    """Names assigned in a class BODY (class attributes), not in its methods."""
    names = set()
    for node in cls.body:
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _is_variant_table_literal(node: ast.expr) -> bool:
    """A ``{'name': {...}, ...}`` literal: string keys, dict values, non-empty."""
    if not isinstance(node, ast.Dict) or not node.keys:
        return False
    if not all(
        isinstance(k, ast.Constant) and isinstance(k.value, str) for k in node.keys
    ):
        return False
    return all(isinstance(v, ast.Dict) for v in node.values)


def _iter_modules(roots, src_root):
    """Yield ``(relpath, ast.Module)`` for every parseable file under ``roots``.

    Contract: same arguments and same silent-skip-on-SyntaxError policy as
    ``_iter_classes``; this one keeps the MODULE node because the MODEL_VARIANTS
    rule is satisfiable at module scope (``beit``/``energy_transformer`` define
    theirs there, deliberately) and because one of its three predicates walks
    module-level FUNCTIONS, which ``_iter_classes`` cannot reach.
    """
    for root in roots:
        for path in sorted(Path(root).rglob("*.py")):
            rel = path.relative_to(src_root).as_posix()
            try:
                yield rel, ast.parse(path.read_text())
            except SyntaxError:  # covered by the import tests
                continue


def _sweep_model_variants(roots=None, src_root=None):
    """Find named-variant registries that are not reachable as ``MODEL_VARIANTS``.

    Contract: returns ``(hits, counts)``.

    * ``hits`` -- ``(relpath, lineno, symbol, kind, detail)`` where ``kind`` is one
      of three predicates, each transcribed from a different sentence of
      ``models/CLAUDE.md`` § Axis 2:

      - ``"from_variant-without-table"``: *"``from_variant(cls, variant, ...)``
        looks the name up in ``MODEL_VARIANTS``"*. A class defining
        ``from_variant`` whose module and class body both lack ``MODEL_VARIANTS``
        **and** whose ``from_variant`` body never reads the name at all. The last
        clause is what clears ``DINOv2``, whose ``from_variant`` deliberately
        reads ``DINOv2VisionTransformer.MODEL_VARIANTS`` -- one table, one home,
        exactly what the rule wants.
      - ``"legacy-table-without-alias"``: *"where one of those is the package's
        only variant table, add ``MODEL_VARIANTS`` as a class-level alias"*.
      - ``"function-local-table"``: a function taking a ``variant`` parameter that
        builds its variant table as a LOCAL dict literal. Nothing external can
        reach it -- ``getattr(cls, "MODEL_VARIANTS")`` raises, which is the exact
        failure mode that got ``fastvit`` fixed on 2026-08-19.

    * ``counts`` -- vacuity denominators: ``n_classes``, ``n_from_variant``,
      ``n_legacy_tables``, ``n_variant_functions``.

    ``roots``/``src_root`` default to the real tree; they exist so the predicates
    can be pointed at a synthetic fixture and proven to fire, exactly as in
    ``_sweep_transformer_layer_norm_args``.
    """
    roots = (MODELS_DIR,) if roots is None else roots
    src_root = SRC_ROOT if src_root is None else src_root
    hits: List[tuple] = []
    counts = dict(
        n_classes=0, n_from_variant=0, n_legacy_tables=0, n_variant_functions=0
    )

    for rel, tree in _iter_modules(roots, src_root):
        module_names = _module_level_names(tree)
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            counts["n_classes"] += 1
            body_names = _class_body_names(cls)
            has_alias = "MODEL_VARIANTS" in body_names or "MODEL_VARIANTS" in module_names
            legacy = sorted(
                n
                for n in body_names
                if _LEGACY_VARIANT_TABLE_RE.match(n) and n != "MODEL_VARIANTS"
            )
            from_variant = next(
                (
                    n
                    for n in cls.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name == "from_variant"
                ),
                None,
            )
            if from_variant is not None:
                counts["n_from_variant"] += 1
                reads_table = any(
                    (isinstance(n, ast.Attribute) and n.attr == "MODEL_VARIANTS")
                    or (isinstance(n, ast.Name) and n.id == "MODEL_VARIANTS")
                    for n in ast.walk(from_variant)
                )
                if not (has_alias or reads_table):
                    hits.append(
                        (
                            rel,
                            from_variant.lineno,
                            cls.name,
                            "from_variant-without-table",
                            "from_variant resolves no MODEL_VARIANTS table",
                        )
                    )
            if legacy:
                counts["n_legacy_tables"] += 1
                if "MODEL_VARIANTS" not in body_names:
                    hits.append(
                        (
                            rel,
                            cls.lineno,
                            cls.name,
                            "legacy-table-without-alias",
                            f"only table is {'/'.join(legacy)}",
                        )
                    )

        for fn in [
            n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]:
            args = fn.args
            names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
            if "variant" not in names:
                continue
            counts["n_variant_functions"] += 1
            for node in ast.walk(fn):
                if isinstance(node, ast.Assign) and _is_variant_table_literal(node.value):
                    local = [t.id for t in node.targets if isinstance(t, ast.Name)]
                    if not local:
                        continue
                    hits.append(
                        (
                            rel,
                            node.lineno,
                            fn.name,
                            "function-local-table",
                            f"local `{local[0]}` holds {len(node.value.keys)} named "
                            "variants that nothing can introspect",
                        )
                    )
    return hits, counts


#: Synthetic package-shaped source, never imported -- only parsed. All three
#: predicates must fire on it and be silent on its twin below.
_INJECTED_VARIANTS_DEFECT_SRC = '''
class InjectedFromVariantUser(keras.Model):
    def __init__(self, width=8, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_variant(cls, variant, **kwargs):
        if variant == "small":
            return cls(width=8, **kwargs)
        return cls(width=16, **kwargs)


class InjectedLegacyTableUser(keras.Model):
    NAM_VARIANTS = {"small": {"width": 8}, "large": {"width": 16}}

    def __init__(self, width=8, **kwargs):
        super().__init__(**kwargs)


def create_injected(variant="small", **kwargs):
    configs = {"small": {"width": 8}, "large": {"width": 16}}
    return InjectedFromVariantUser(**configs[variant], **kwargs)
'''

#: The same three sites, repaired the way the house rule asks.
_INJECTED_VARIANTS_FIXED_SRC = '''
class InjectedFromVariantUser(keras.Model):
    MODEL_VARIANTS = {"small": {"width": 8}, "large": {"width": 16}}

    def __init__(self, width=8, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_variant(cls, variant, **kwargs):
        return cls(**cls.MODEL_VARIANTS[variant], **kwargs)


class InjectedLegacyTableUser(keras.Model):
    NAM_VARIANTS = {"small": {"width": 8}, "large": {"width": 16}}
    MODEL_VARIANTS = NAM_VARIANTS

    def __init__(self, width=8, **kwargs):
        super().__init__(**kwargs)


def create_injected(variant="small", **kwargs):
    return InjectedFromVariantUser.from_variant(variant, **kwargs)
'''


class TestModelVariantsArePresent:
    """Named variants must be reachable as ``MODEL_VARIANTS``, not just callable.

    ``models/CLAUDE.md`` § Axis 2 makes ``MODEL_VARIANTS`` the canonical name for
    the registry of publicly named variants, tells packages carrying a legacy
    spelling to add it as a class-level alias, and defines ``from_variant`` as the
    method that "looks the name up in ``MODEL_VARIANTS``". Until this class
    shipped **nothing enforced any of that** -- the convention was documented and
    unguarded, which is how ``fastvit`` reached 2026-08-19 with
    ``getattr(FastVitImageEncoder, "MODEL_VARIANTS")`` raising ``AttributeError``
    while this very file asserted the opposite in prose.

    Narrowings, each a place this can miss a real defect:

    * a package whose named sizes are exposed under a DIFFERENT parameter name is
      invisible here. ``depth_anything`` takes ``encoder_type='vit_s'|'vit_b'|
      'vit_l'`` -- three genuine published sizes -- and no predicate below fires
      on it, because none of them can tell that knob apart from any other
      validated string argument. It was given a ``MODEL_VARIANTS`` table anyway
      (see decisions.md D-009); the guard simply cannot be what keeps it;
    * the local-table predicate needs a ``variant``-named parameter AND a literal
      ``{str: dict}`` assignment. A table built by a loop, a comprehension, or
      ``dict(...)`` is not seen;
    * ``SCALE_CONFIGS`` is out of scope by explicit instruction of the rule.
    """

    def test_every_from_variant_class_resolves_a_model_variants_table(self):
        # DECISION plan-2026-08-19T070627-a616f581/D-009
        # The trigger for this guard is EVIDENCE OF NAMED VARIANTS (a
        # `from_variant`, a legacy table, or a hidden local table) -- NOT "is a
        # keras.Model". WHAT NOT TO DO: do not "strengthen" this into "every
        # model class must declare MODEL_VARIANTS". models/CLAUDE.md
        # § "When the shape does not apply" says in terms: "Do not invent a
        # MODEL_VARIANTS table to satisfy the template" -- a package with no
        # genuine named variants is compliant WITHOUT one, and the strengthened
        # form would demand fabricated tables from ~20 packages. See D-009.
        hits, _ = _sweep_model_variants()
        offenders = [
            f"{rel}:{line} {sym}: {detail}"
            for rel, line, sym, kind, detail in hits
            if kind == "from_variant-without-table"
            and (rel, sym, kind) not in _MODEL_VARIANTS_WAIVERS
        ]
        assert not offenders, (
            "a from_variant classmethod resolves no MODEL_VARIANTS table, so its "
            "variants exist only inside its own body. Hoist them to a class-level "
            "MODEL_VARIANTS dict (models/CLAUDE.md Axis 2). Found:\n  "
            + "\n  ".join(offenders)
        )

    def test_every_legacy_variant_table_has_a_model_variants_alias(self):
        hits, _ = _sweep_model_variants()
        offenders = [
            f"{rel}:{line} {sym}: {detail}"
            for rel, line, sym, kind, detail in hits
            if kind == "legacy-table-without-alias"
            and (rel, sym, kind) not in _MODEL_VARIANTS_WAIVERS
        ]
        assert not offenders, (
            "a class's only variant table uses a legacy spelling with no "
            "MODEL_VARIANTS alias. Add `MODEL_VARIANTS = <existing name>` in the "
            "class body -- an ALIAS, never a rename: src/train/ and the test "
            "suites reference the old spelling. Found:\n  " + "\n  ".join(offenders)
        )

    def test_no_variant_table_hides_inside_a_factory_body(self):
        hits, _ = _sweep_model_variants()
        offenders = [
            f"{rel}:{line} {sym}(): {detail}"
            for rel, line, sym, kind, detail in hits
            if kind == "function-local-table"
            and (rel, sym, kind) not in _MODEL_VARIANTS_WAIVERS
        ]
        assert not offenders, (
            "a factory's variant table is a local variable, so no caller can "
            "enumerate the variants it accepts. Hoist it to the model class as "
            "MODEL_VARIANTS and read it from the factory. Found:\n  "
            + "\n  ".join(offenders)
        )

    def test_the_sweep_found_variant_sites(self):
        """The AST walk must not silently collapse to nothing.

        Floors are set well under the 2026-08-19 measurement (252 classes, 64
        ``from_variant`` classes, 2 legacy tables, 150 functions taking
        ``variant``) so ordinary churn does not trip them. The legacy-table
        denominator is deliberately NOT floored above 1: there are only two such
        classes left in the tree, and fixing them by renaming is forbidden, so it
        should stay at two.
        """
        _, counts = _sweep_model_variants()
        assert counts["n_classes"] >= 150, counts
        assert counts["n_from_variant"] >= 40, counts
        assert counts["n_variant_functions"] >= 100, counts
        assert counts["n_legacy_tables"] >= 2, counts

    def test_predicate_fires_on_an_injected_defect(self, tmp_path):
        """Dead-component probe: all three predicates must go RED on a real gap."""
        roots, src_root = _write_fixture(tmp_path, _INJECTED_VARIANTS_DEFECT_SRC)
        hits, counts = _sweep_model_variants(roots, src_root)
        assert counts["n_from_variant"] == 1 and counts["n_legacy_tables"] == 1
        # 2, not 1: `from_variant(cls, variant, **kwargs)` takes a `variant`
        # parameter too, so it is counted alongside `create_injected`.
        assert counts["n_variant_functions"] == 2
        by_kind = {kind: (sym, detail) for _, _, sym, kind, detail in hits}
        assert set(by_kind) == {
            "from_variant-without-table",
            "legacy-table-without-alias",
            "function-local-table",
        }, hits
        assert by_kind["from_variant-without-table"][0] == "InjectedFromVariantUser"
        assert by_kind["legacy-table-without-alias"][0] == "InjectedLegacyTableUser"
        assert by_kind["function-local-table"][0] == "create_injected"

    def test_predicate_is_silent_on_the_fixed_twin(self, tmp_path):
        """...and must NOT fire once the same three sites are repaired."""
        roots, src_root = _write_fixture(tmp_path, _INJECTED_VARIANTS_FIXED_SRC)
        hits, counts = _sweep_model_variants(roots, src_root)
        assert counts["n_from_variant"] == 1, "the fixture must still be reached"
        assert counts["n_legacy_tables"] == 1, "the fixture must still be reached"
        assert counts["n_variant_functions"] == 2, "the fixture must still be reached"
        assert hits == [], hits

    @pytest.mark.parametrize(
        "module_path,class_name,expected_keys",
        [
            ("dl_techniques.models.SAM.SAM1.model", "SAM", ["vit_b", "vit_h", "vit_l"]),
            ("dl_techniques.models.kan.model", "KAN",
             ["large", "medium", "micro", "small", "xlarge"]),
            ("dl_techniques.models.ntm.model", "NTMModel", ["base", "large", "tiny"]),
            ("dl_techniques.models.nano_vlm.model", "NanoVLM",
             ["base", "large", "mini"]),
            ("dl_techniques.models.nano_vlm_world_model.model", "ScoreBasedNanoVLM",
             ["base", "large", "mini"]),
            ("dl_techniques.models.pft_sr.model", "PFTSR",
             ["base", "large", "light"]),
            ("dl_techniques.models.depth_anything.model", "DepthAnything",
             ["vit_b", "vit_l", "vit_s"]),
        ],
    )
    def test_repaired_packages_expose_their_variants_at_runtime(
        self, module_path, class_name, expected_keys
    ):
        """The seven classes repaired on 2026-08-19, pinned by RESOLVED value.

        The three sweeps above are static: they prove a class BODY assigns the
        name. This one proves ``getattr(cls, "MODEL_VARIANTS")`` actually
        resolves on the imported class and enumerates the variants the package
        already supported -- which is the property that was missing (the fastvit
        report was an ``AttributeError`` at runtime, not an AST observation), and
        the property an alias or a class-attribute hoist could get wrong.

        The key lists are DERIVED, not chosen: every one of them is the key set
        the package's own factory or ``from_variant`` already accepted before the
        repair. No variant was invented.
        """
        cls = getattr(importlib.import_module(module_path), class_name)
        table = getattr(cls, "MODEL_VARIANTS", None)
        assert isinstance(table, dict), (
            f"{class_name}.MODEL_VARIANTS must resolve to a dict, got {table!r}"
        )
        assert sorted(table) == sorted(expected_keys), (
            f"{class_name}.MODEL_VARIANTS keys changed: {sorted(table)} != "
            f"{sorted(expected_keys)}"
        )
        assert all(isinstance(v, dict) for v in table.values()), table

    def test_legacy_aliases_are_the_same_object_not_a_copy(self):
        """``MODEL_VARIANTS = <legacy>`` must ALIAS, so one edit reaches both.

        A copy would satisfy every static predicate above and still let the two
        spellings drift -- which is the whole reason the house rule says "alias to
        the same dict" and "prefer an alias over renaming in place".
        """
        from dl_techniques.models.kan.model import KAN
        from dl_techniques.models.ntm.model import NTMModel

        assert KAN.MODEL_VARIANTS is KAN.VARIANT_CONFIGS
        assert NTMModel.MODEL_VARIANTS is NTMModel.NTM_VARIANTS

    def test_waivers_still_match_a_real_site(self):
        """A waiver matching nothing is a waiver hiding nothing (see the siblings)."""
        hits, _ = _sweep_model_variants()
        live = {(rel, sym, kind) for rel, _, sym, kind, _ in hits}
        stale = sorted(_MODEL_VARIANTS_WAIVERS - live)
        assert not stale, (
            "MODEL_VARIANTS waiver entries no longer match any site; delete them "
            f"if the exception is gone, re-key them if the code moved: {stale}"
        )
