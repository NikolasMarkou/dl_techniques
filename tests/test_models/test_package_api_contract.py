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
    src_root = MODELS_DIR.parent
    dropped: List[tuple] = []
    dynamic: List[str] = []

    for root in (MODELS_DIR, LAYERS_DIR / "transformers"):
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(src_root).as_posix()
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:  # covered by the import tests
                continue
            for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
                stored = _init_stored_attrs(cls)
                for node in ast.walk(cls):
                    if not isinstance(node, ast.Call):
                        continue
                    func = node.func
                    fname = (
                        func.id
                        if isinstance(func, ast.Name)
                        else func.attr
                        if isinstance(func, ast.Attribute)
                        else None
                    )
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
      family of this same defect is out of reach here. This is also why the two
      by-design omissions catalogued in the sweep that motivated this guard
      (``adaln_zero.py`` and ``text_encoder.py``'s ``**norm_config`` unpack) need
      no waiver: both are ``create_normalization_layer`` calls and the predicate
      never reaches them. Waiving them anyway would have been a waiver guarding
      nothing;
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
