"""The vision models share one house shape, read from their source text alone.

``resnet/model.py`` is the reference. Every other classifier, and the models
that are not classifiers, is held to the same module layout, class member order,
constructor surface, ``build`` and factory shape by pure AST rules. Nothing is
imported or built, so the census costs a second and can run before a model
exists. The runtime half of the same contract (what a built model does) lives in
``test_the_vision_classifiers_share_one_runtime_contract.py``.

The rules
---------
``h1_module_header``
    A model module opens on a docstring with a ``References:`` section and has
    exactly one ``# local imports`` banner. A helper or layer module needs only
    a docstring.
``h1_factories_follow_the_classes``
    No module-level function is defined between the first and the last class,
    and no ``create_*`` factory before the first class: every factory follows
    the last class.
``no_from_keras_import``
    ``import keras`` and qualify (``keras.ops.softmax``); never ``from keras
    import ...``.
``h2_member_order``
    Class constants come before the first method, ``MODEL_VARIANTS`` first among
    them; then ``__init__``, the ``_build_*`` methods, ``build``, ``call``,
    ``compute_output_shape``, ``load_pretrained_weights``, ``from_variant``,
    ``get_config``, ``from_config``, ``summary``. A method that is not on that
    list may sit anywhere.
``h3_one_init_log``, ``h3_headless_attribute``, ``h3_builders_assign``
    ``__init__`` writes exactly one ``logger.info``; it assigns the head
    attribute ``None`` itself, so a headless model still has the attribute; the
    ``_build_*`` methods assign ``self.<part>`` and return nothing.
``h4_no_logging_outside_init``
    No ``logger`` call in ``call``, ``build``, ``compute_output_shape``,
    ``summary`` or a ``_build_*`` method.
``h5_build_materializes``
    ``build`` is ``if self.built: return``, ``materialize_sublayers(self,
    input_shape)``, then ``super().build(input_shape)`` last.
``h9_factory_is_one_delegating_return``
    Every ``create_*`` factory has the signature ``(variant, num_classes=1000,
    input_shape=None, pretrained=False, **kwargs)`` and is one ``return
    Class.from_variant(..., **kwargs)`` that forwards every parameter the
    factory takes, ``**kwargs`` included.
``h10_*``
    ``num_classes`` is the first constructor parameter and ``**kwargs`` the
    last; ``include_top``, ``input_shape`` and ``head_config_overrides`` come in
    that order; a stored attribute carries its argument's name (``input_shape``
    is kept as ``input_shape_config`` and ``head_config_overrides`` as
    ``_head_config_overrides``); one name per concept (``kernel_regularizer``,
    never ``weight_decay``; ``variant``, never ``variant_config``).
``h11_get_config_covers_the_constructor``, ``h11_from_config_does_real_work``
    ``get_config`` starts from ``super().get_config()`` and writes a key for
    every constructor argument; a ``from_config`` that only copies the dict and
    calls ``cls(**config)`` is deleted.
``h11_get_config_has_the_reference_form``
    There is one way to build the dict: ``config = {...}``, then ``base_config =
    super().get_config()`` and ``return {**base_config, **config}``.
``h2_call_takes_inputs``
    The forward method names its input argument ``inputs``.
``h10_block_container_names``
    A list of blocks built in ``__init__`` or a ``_build_*`` method is
    ``self.blocks`` (a flat sequence) or ``self.stages`` (one entry per stage);
    ``blocks_list``, ``stage_blocks`` and the like are one concept under several
    names.
``h15_initializer_storage``
    A model stores its initializers as specs (``initializer_spec(...)``, a
    string or a dict literal), never ``keras.initializers.get(...)`` or the
    caller's raw argument, because a stored instance reaches several layers and
    replays one draw. A model may be exempt by naming a decision
    (:data:`EXEMPTIONS`, ``D-039``).
``anchors``
    Every ``# DECISION <plan>/D-NNN`` token that exists today is still in its
    file; for the two families that move code between files it is still in the
    file set.

Each rule is a function of source text, so :data:`PLANTED` hands it a module
that breaks the rule (it must report) and a clean twin (it must not). The same
plants are moved into a base class to prove a leaf is read together with its base.

Exemptions, each with the reason it is allowed
----------------------------------------------
See :data:`EXEMPTIONS`. A subject or module names the exemption that waives a
rule and no row is generated for a waived pair; an exemption no waiver names is
tied to a check that the situation it describes exists in the source
(:data:`EXEMPTION_WITNESSES`), and a meta test keeps both ends honest.

Rows that pass because there is nothing to read
-----------------------------------------------
A rule whose precondition is absent (``from_config`` not defined, no ``_build_*``
method, not a classifier) returns no violation by construction.
:data:`VACUOUS` pins those ``(rule, subject)`` pairs; a subject that gains the
missing member changes the computed list and fails the run until the pin is
decided.

The ledger
----------
:data:`LEDGER` lists every row that fails today as a strict ``xfail`` carrying
the number measured by running this file, the migration that closes it and a
``signature``: the pieces of the failure text that name which violations the row
has. A strict ``xfail`` that starts passing fails the run, and a ledgered row that
fails with text lacking its signature is a plain failure (the assertion type alone
cannot tell one violation from another), so the migration that fixes or changes a
row edits its ledger line in the same commit. An empty ledger means the house
shape is complete.
"""

import ast
import functools
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "dl_techniques"
VISION = "models/vision"
#: The one spelling of the ConvNeXt base file's path: the module row, the leaves'
#: ``base`` and the anchor set all read it from here.
CONVNEXT_COMMON = f"{VISION}/convnext/common.py"

#: What every exemption says about itself. A waiver names one of these keys and
#: ``test_every_waiver_names_a_written_exemption`` checks both ends.
EXEMPTIONS: Dict[str, str] = {
    "E-1": (
        "CliffordNet and HierarchicalCliffordNet: ``include_top`` is the last "
        "named parameter and must be a real bool (a second positional slot "
        "once bound it silently); there is no constructor ``input_shape``, so "
        "``from_variant(input_shape=...)`` builds the model; ``num_classes`` "
        "defaults to 1000."
    ),
    "E-2": (
        "SqueezeNet V1 and V2 are functional models: no ``build``, ``call`` or "
        "``compute_output_shape`` override, and their ``_build_*`` methods take "
        "and return tensors; their classifier is the paper's ``conv10`` layer, "
        "not a ``classification_head`` attribute. VAE builds its encoder and "
        "decoder the same way (``_build_*`` take and return tensors)."
    ),
    "E-4": (
        "Models without named variants (RBFProtoNet, CliffordRBFProtoNet, "
        "UcaEncoder, ConvUNextStem, CapsNet, CapsNetV2): no ``MODEL_VARIANTS``, "
        "so ``from_variant`` and the delegating factory do not apply. The RBF "
        "factories keep a ``pretrained`` argument and refuse it."
    ),
    "E-5": (
        "``VAE.from_variant`` keeps its ``(variant, input_shape, latent_dim)`` "
        "signature and gains ``pretrained``; there is no ``num_classes``, so "
        "``create_vae`` is held to a delegating return with ``pretrained`` and "
        "``**kwargs`` rather than to the canonical signature."
    ),
    "E-6": "RBF ``num_classes`` defaults to 100: it is a prototype count.",
    "E-7": (
        "CapsNet V1 and V2: the factories compile and return the model, V1 "
        "keeps its custom ``train_step`` and ``test_step``, ``num_classes`` has "
        "no default, and ``save_model`` / ``load_model`` stay."
    ),
    "D-039": (
        "MobileNetV1, V2 and V3 keep the caller's ``Initializer`` and hand each site a "
        "clone (``clone_initializer``); each computes ``initializer_spec`` once, only to "
        "refuse a bad initializer at construction and to feed ``get_config``. A stored "
        "spec would re-draw every seeded kernel (V1: 0 of 28 small kernels; V2: 0 of 53 "
        "small; V3: 0 of 54 small and 0 of 64 large would stay equal to the base "
        "revision's)."
    ),
    "not-a-traced-model": (
        "Rule h5 describes a subclassed model whose ``call`` is traced by "
        "``materialize_sublayers``. ConvUNextStem builds its one ConvStem, "
        "UcaEncoder builds an explicit shape chain, and VAE, like SqueezeNet, "
        "is a functional graph built in ``__init__``: none of them is in that "
        "rule's roster."
    ),
    "layer-not-model": (
        "ConvUNextStem is a layer, and a layer does not log its construction; "
        "the one-``logger.info`` rule is for models."
    ),
}


@dataclass(frozen=True)
class Module:
    """One source file under the census."""

    key: str
    path: str  # relative to ``src/dl_techniques``
    classes: Tuple[str, ...] = ()  # the subject classes the module defines
    model_module: bool = True  # needs ``References:`` and the import banner
    factory_signature: Optional[Tuple[str, ...]] = None  # rule h9; None = no factory rule
    factory_exact: bool = True
    waived: Dict[str, str] = field(default_factory=dict)
    #: ``(path, class)`` of the base class that holds ``from_variant`` when the module's own
    #: classes do not, so rule h9 can check a positional factory call against its slots.
    base: Optional[Tuple[str, str]] = None


@dataclass(frozen=True)
class Subject:
    """One class whose members the class rules read."""

    key: str
    module: str  # ``Module.key``
    cls: str
    rules: FrozenSet[str]
    head_attribute: Optional[str] = None
    required_params: Tuple[str, ...] = ()
    classifier: bool = False
    include_top_last: bool = False
    base: Optional[Tuple[str, str]] = None  # (module path, class) that holds inherited members
    waived: Dict[str, str] = field(default_factory=dict)


CANONICAL_FACTORY = ("variant", "num_classes", "input_shape", "pretrained")

CLASS_RULES = (
    "h2_member_order",
    "h3_one_init_log",
    "h3_headless_attribute",
    "h3_builders_assign",
    "h4_no_logging_outside_init",
    "h5_build_materializes",
    "h10_num_classes_first_kwargs_last",
    "h10_trailing_parameter_order",
    "h10_stored_attribute_names",
    "h10_one_name_per_concept",
    "h11_get_config_covers_the_constructor",
    "h11_from_config_does_real_work",
    "h11_get_config_has_the_reference_form",
    "h2_call_takes_inputs",
    "h10_block_container_names",
    "h15_initializer_storage",
)
_ALL = frozenset(CLASS_RULES)
_NO_HEAD_ATTRIBUTE = _ALL - {"h3_headless_attribute"}

_MOBILENET = "models/vision/mobilenet"
_CONVNEXT_BASE = (CONVNEXT_COMMON, "_ConvNeXtBase")
_E4_FACTORY = {"h9_factory_is_one_delegating_return": "E-4"}
_E7_FACTORY = {"h9_factory_is_one_delegating_return": "E-7"}

MODULES: Tuple[Module, ...] = (
    Module("resnet", f"{VISION}/resnet/model.py", ("ResNet",),
           factory_signature=CANONICAL_FACTORY),
    Module("cliffordnet", f"{VISION}/cliffordnet/model.py", ("CliffordNet",),
           factory_signature=CANONICAL_FACTORY),
    Module("hierarchical", f"{VISION}/cliffordnet/hierarchical.py",
           ("HierarchicalCliffordNet",), factory_signature=CANONICAL_FACTORY),
    Module("mobilenet_v1", f"{_MOBILENET}/mobilenet_v1.py", ("MobileNetV1",),
           factory_signature=CANONICAL_FACTORY),
    Module("mobilenet_v2", f"{_MOBILENET}/mobilenet_v2.py", ("MobileNetV2",),
           factory_signature=CANONICAL_FACTORY),
    Module("mobilenet_v3", f"{_MOBILENET}/mobilenet_v3.py", ("MobileNetV3",),
           factory_signature=CANONICAL_FACTORY),
    Module("mobilenet_v4", f"{_MOBILENET}/mobilenet_v4.py", ("MobileNetV4",),
           factory_signature=CANONICAL_FACTORY),
    Module("mobilenet_common", f"{_MOBILENET}/common.py", model_module=False),
    Module("convnext_v1", f"{VISION}/convnext/convnext_v1.py", ("ConvNeXtV1",),
           factory_signature=CANONICAL_FACTORY, base=_CONVNEXT_BASE),
    Module("convnext_v2", f"{VISION}/convnext/convnext_v2.py", ("ConvNeXtV2",),
           factory_signature=CANONICAL_FACTORY, base=_CONVNEXT_BASE),
    Module("convnext_common", CONVNEXT_COMMON, model_module=False),
    Module("rad_convnet", f"{VISION}/rad_convnet/model.py", ("RADConvNet",),
           factory_signature=CANONICAL_FACTORY),
    Module("squeezenet_v1", f"{VISION}/squeezenet/squeezenet_v1.py", ("SqueezeNetV1",),
           factory_signature=CANONICAL_FACTORY),
    Module("squeezenet_v2", f"{VISION}/squeezenet/squeezenet_v2.py",
           ("SqueezeNoduleNetV2",), factory_signature=CANONICAL_FACTORY),
    Module("convunext", f"{VISION}/convunext/model.py", ("ConvUNextStem",), waived=_E4_FACTORY),
    Module("vae", f"{VISION}/vae/model.py", ("VAE",),
           factory_signature=("input_shape", "pretrained"), factory_exact=False),
    Module("rbf_protonet", f"{VISION}/rbf_protonet/model.py",
           ("RBFProtoNet", "CliffordRBFProtoNet"), waived=_E4_FACTORY),
    Module("uca_encoder", f"{VISION}/mambalct/uca_encoder.py", ("UcaEncoder",), waived=_E4_FACTORY),
    Module("capsnet", f"{VISION}/capsnet/model.py", ("CapsNet",), waived=_E7_FACTORY),
    Module("capsnet_v2", f"{VISION}/capsnet/model_v2.py", ("CapsNetV2",), waived=_E7_FACTORY),
    Module("stem", "layers/conv_blocks/stem.py", ("ConvStem",), model_module=False),
)

_HEAD = "classification_head"
_INIT_AND_REG = ("kernel_initializer", "kernel_regularizer")
_E2 = {"h3_builders_assign": "E-2", "h3_headless_attribute": "E-2",
       "h5_build_materializes": "E-2"}
_NOT_TRACED = {"h5_build_materializes": "not-a-traced-model"}


def _subject(key, module, cls, rules=_ALL, **kw) -> Subject:
    return Subject(key=key, module=module, cls=cls, rules=frozenset(rules), **kw)


SUBJECTS: Tuple[Subject, ...] = (
    _subject("resnet", "resnet", "ResNet", head_attribute=_HEAD, classifier=True,
             required_params=_INIT_AND_REG),
    _subject("cliffordnet", "cliffordnet", "CliffordNet", head_attribute=_HEAD,
             classifier=True, include_top_last=True, required_params=_INIT_AND_REG),
    _subject("hierarchical", "hierarchical", "HierarchicalCliffordNet",
             head_attribute=_HEAD, classifier=True, include_top_last=True,
             required_params=_INIT_AND_REG),
    _subject("mobilenet_v1", "mobilenet_v1", "MobileNetV1", head_attribute=_HEAD,
             classifier=True, required_params=_INIT_AND_REG,
             waived={"h15_initializer_storage": "D-039"},
             rules=_ALL - {"h15_initializer_storage"}),
    _subject("mobilenet_v2", "mobilenet_v2", "MobileNetV2", head_attribute=_HEAD,
             classifier=True, required_params=_INIT_AND_REG,
             waived={"h15_initializer_storage": "D-039"},
             rules=_ALL - {"h15_initializer_storage"}),
    _subject("mobilenet_v3", "mobilenet_v3", "MobileNetV3", head_attribute=_HEAD,
             classifier=True, required_params=_INIT_AND_REG,
             waived={"h15_initializer_storage": "D-039"},
             rules=_ALL - {"h15_initializer_storage"}),
    _subject("mobilenet_v4", "mobilenet_v4", "MobileNetV4", head_attribute=_HEAD,
             classifier=True, required_params=_INIT_AND_REG),
    _subject("convnext_v1", "convnext_v1", "ConvNeXtV1", head_attribute=_HEAD,
             classifier=True, required_params=("kernel_regularizer",), base=_CONVNEXT_BASE),
    _subject("convnext_v2", "convnext_v2", "ConvNeXtV2", head_attribute=_HEAD,
             classifier=True, required_params=("kernel_regularizer",), base=_CONVNEXT_BASE),
    _subject("rad_convnet", "rad_convnet", "RADConvNet", head_attribute=_HEAD,
             classifier=True),
    _subject("squeezenet_v1", "squeezenet_v1", "SqueezeNetV1", classifier=True,
             required_params=_INIT_AND_REG, waived=_E2, rules=_ALL - set(_E2)),
    _subject("squeezenet_v2", "squeezenet_v2", "SqueezeNoduleNetV2", classifier=True,
             required_params=_INIT_AND_REG, waived=_E2, rules=_ALL - set(_E2)),
    _subject("convunext_stem", "convunext", "ConvUNextStem",
             rules=_NO_HEAD_ATTRIBUTE - {"h5_build_materializes", "h3_one_init_log"},
             required_params=_INIT_AND_REG,
             waived={**_NOT_TRACED, "h3_one_init_log": "layer-not-model"}),
    _subject("vae", "vae", "VAE",
             rules=_NO_HEAD_ATTRIBUTE - {"h5_build_materializes", "h3_builders_assign"},
             required_params=_INIT_AND_REG,
             waived={**_NOT_TRACED, "h3_builders_assign": "E-2"}),
    _subject("rbf_protonet", "rbf_protonet", "RBFProtoNet", rules=_NO_HEAD_ATTRIBUTE,
             required_params=("kernel_regularizer",)),
    _subject("clifford_rbf_protonet", "rbf_protonet", "CliffordRBFProtoNet",
             rules=_NO_HEAD_ATTRIBUTE, required_params=("kernel_regularizer",)),
    _subject("uca_encoder", "uca_encoder", "UcaEncoder",
             rules=_NO_HEAD_ATTRIBUTE - {"h5_build_materializes"}, waived=_NOT_TRACED),
    _subject("capsnet", "capsnet", "CapsNet", rules=_NO_HEAD_ATTRIBUTE,
             required_params=_INIT_AND_REG),
    _subject("capsnet_v2", "capsnet_v2", "CapsNetV2", rules=_NO_HEAD_ATTRIBUTE,
             required_params=_INIT_AND_REG),
    _subject("stem", "stem", "ConvStem", rules=frozenset({"h4_no_logging_outside_init"})),
)


def module_rules(module: Module) -> Tuple[str, ...]:
    """The module rules that apply to ``module``."""
    rules = ["h1_module_header", "no_from_keras_import"]
    if module.model_module:
        rules.append("h1_factories_follow_the_classes")
        if module.factory_signature is not None:
            rules.append("h9_factory_is_one_delegating_return")
    return tuple(rules)


#: Families whose ``# DECISION`` tokens are compared over a file SET, because the
#: family moves code between its files: ConvNeXt V1 and V2 become one base class
#: (a token both leaves carry may be merged into one copy), and the Clifford pair
#: shares builders (the stem's BatchNorm constants live in ``model.py`` while the
#: epsilon note stays in ``hierarchical.py``, beside the downsampler norm that
#: still uses the bare Keras epsilon).
ANCHOR_SETS: Dict[str, Tuple[str, ...]] = {
    "convnext": (f"src/dl_techniques/{VISION}/convnext/convnext_v1.py",
                 f"src/dl_techniques/{VISION}/convnext/convnext_v2.py",
                 f"src/dl_techniques/{CONVNEXT_COMMON}"),
    "cliffordnet": (f"src/dl_techniques/{VISION}/cliffordnet/model.py",
                    f"src/dl_techniques/{VISION}/cliffordnet/hierarchical.py"),
}

# ---------------------------------------------------------------------------
# Source helpers
# ---------------------------------------------------------------------------

Violations = List[str]
Member = Tuple[str, str, ast.AST]  # (kind: "def" | "const", name, node)


def _members(cls: ast.ClassDef) -> List[Member]:
    out: List[Member] = []
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(("def", node.name, node))
        elif isinstance(node, ast.Assign):
            out.extend(("const", t.id, node) for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.append(("const", node.target.id, node))
    return out


def _find_class(tree: ast.Module, name: str) -> Optional[ast.ClassDef]:
    return next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name), None)


def _effective_members(
        tree: ast.Module, cls_name: str, base_tree: Optional[ast.Module] = None,
        base_name: Optional[str] = None) -> List[Member]:
    """The members a reader of ``cls_name`` meets: its own, plus the base class's.

    Without a base this is the class body. With one, the class's constants come
    first, then the base's constants and methods in the base's order (a method the
    class defines replacing the base's), then the class's remaining methods.
    """
    cls = _find_class(tree, cls_name)
    own = _members(cls) if cls is not None else []
    base_cls = _find_class(base_tree, base_name) if base_tree is not None else None
    if base_cls is None:
        return own
    own_names = {name for _, name, _ in own}
    own_defs = {name: member for member in own if member[0] == "def" for name in [member[1]]}
    base = _members(base_cls)
    merged = [m for m in own if m[0] == "const"]
    merged += [m for m in base if m[0] == "const" and m[1] not in own_names]
    placed = set()
    for kind, name, node in base:
        if kind == "def":
            merged.append(own_defs.get(name, (kind, name, node)))
            placed.add(name)
    merged += [m for m in own if m[0] == "def" and m[1] not in placed]
    return merged


def _def(members: Sequence[Member], name: str) -> Optional[ast.FunctionDef]:
    return next((n for kind, member, n in members if kind == "def" and member == name), None)


def _body_without_docstring(fn: ast.FunctionDef) -> List[ast.stmt]:
    body = list(fn.body)
    if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return body


def _walk_own(fn: ast.AST):
    """The nodes of ``fn``'s own body, not those of a function defined inside it."""
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(node))


def _logger_calls(fn: ast.AST, level: Optional[str] = None) -> List[ast.Call]:
    return [n for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "logger"
            and (level is None or n.func.attr == level)]


def _params(fn: ast.FunctionDef) -> List[str]:
    return [a.arg for a in fn.args.args[1:]] + [a.arg for a in fn.args.kwonlyargs]


# ---------------------------------------------------------------------------
# Module rules: ``rule(source, module) -> violations``
# ---------------------------------------------------------------------------


def h1_module_header(source: str, module: Module) -> Violations:
    doc = ast.get_docstring(ast.parse(source))
    out = [] if doc else ["the module has no docstring"]
    if module.model_module:
        if doc and not re.search(r"^References:\s*$", doc, re.M):
            out.append("the module docstring has no `References:` section")
        banners = len(re.findall(r"^# local imports\s*$", source, re.M))
        if banners != 1:
            out.append(f"{banners} `# local imports` banners, expected 1")
    return out


def h1_factories_follow_the_classes(source: str, module: Module) -> Violations:
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if not classes:
        return []
    first, last = classes[0].lineno, classes[-1].lineno
    out = []
    for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
        if first < fn.lineno < last:
            out.append(f"`def {fn.name}` (line {fn.lineno}) sits between classes")
        elif fn.name.startswith("create_") and fn.lineno < first:
            out.append(f"`def {fn.name}` (line {fn.lineno}) comes before the first class")
    return out


def no_from_keras_import(source: str, module: Module) -> Violations:
    return [f"line {n.lineno}: from {n.module} import ..."
            for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.ImportFrom) and not n.level
            and (n.module == "keras" or (n.module or "").startswith("keras."))]


def _is_delegating_return(fn: ast.FunctionDef, classes: Sequence[str]) -> bool:
    body = _body_without_docstring(fn)
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    call = body[0].value
    return (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and call.func.attr == "from_variant" and isinstance(call.func.value, ast.Name)
            and call.func.value.id in classes and any(k.arg is None for k in call.keywords))


def _from_variant_slots(tree: ast.Module, classes: Sequence[str]) -> Optional[List[str]]:
    """The positional parameter names of ``from_variant`` after ``cls`` in ``classes`` of ``tree``,
    or ``None`` when they define none (the method lives in a base class)."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in classes:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == "from_variant":
                    return [a.arg for a in member.args.args[1:]]
    return None


def _routes_to_its_own_slot(call: ast.Call, name: str, slots: Optional[List[str]]) -> bool:
    """True when ``name`` reaches the SAME-NAMED keyword, or the positional slot of that name."""
    by_keyword = any(k.arg == name and isinstance(k.value, ast.Name) and k.value.id == name
                     for k in call.keywords)
    by_position = slots is not None and any(
        isinstance(arg, ast.Name) and arg.id == name and i < len(slots) and slots[i] == name
        for i, arg in enumerate(call.args))
    return by_keyword or by_position


def h9_factory_is_one_delegating_return(source: str, module: Module) -> Violations:
    factories = [n for n in ast.parse(source).body
                 if isinstance(n, ast.FunctionDef) and n.name.startswith("create_")]
    if not factories:
        return ["the module defines no `create_*` factory"]
    expected = module.factory_signature or CANONICAL_FACTORY
    out = []
    for fn in factories:
        names = [a.arg for a in fn.args.args]
        defaults = dict(zip(names[len(names) - len(fn.args.defaults):], fn.args.defaults))
        if module.factory_exact and names != list(expected):
            out.append(f"{fn.name}{tuple(names)} is not {expected}")
        if not module.factory_exact:
            out += [f"{fn.name} has no `{p}` parameter" for p in expected if p not in names]
        if fn.args.kwarg is None or fn.args.kwarg.arg != "kwargs":
            out.append(f"{fn.name} does not end in **kwargs")
        if module.factory_exact:
            for name, value in {"num_classes": 1000, "input_shape": None, "pretrained": False}.items():
                node = defaults.get(name)
                if not (isinstance(node, ast.Constant) and node.value == value
                        and type(node.value) is type(value)):
                    out.append(f"{fn.name}: `{name}` does not default to {value!r}")
        if not _is_delegating_return(fn, module.classes):
            out.append(f"{fn.name} is not one `return Class.from_variant(..., **kwargs)`")
        else:
            forwarded = {n.id for n in ast.walk(_body_without_docstring(fn)[0].value)
                         if isinstance(n, ast.Name)}
            taken = names + ([fn.args.kwarg.arg] if fn.args.kwarg else [])
            unsent = [p for p in taken if p not in forwarded]
            if unsent:
                out.append(f"{fn.name} does not forward {unsent} to `from_variant`")
            slots = _from_variant_slots(ast.parse(source), module.classes)
            if slots is None and module.base is not None:
                base_text = _read(module.base[0])
                if base_text is not None:
                    slots = _from_variant_slots(ast.parse(base_text), (module.base[1],))
            misrouted = [p for p in names if p in forwarded
                         and not _routes_to_its_own_slot(_body_without_docstring(fn)[0].value, p, slots)]
            if misrouted:
                out.append(f"{fn.name} sends {misrouted} to a `from_variant` slot of another name")
    return out


MODULE_RULE_FUNCTIONS: Dict[str, Callable[[str, Module], Violations]] = {
    "h1_module_header": h1_module_header,
    "h1_factories_follow_the_classes": h1_factories_follow_the_classes,
    "no_from_keras_import": no_from_keras_import,
    "h9_factory_is_one_delegating_return": h9_factory_is_one_delegating_return,
}

# ---------------------------------------------------------------------------
# Class rules: ``rule(members, subject) -> violations``
# ---------------------------------------------------------------------------

_RANK = {"__init__": 1, "build": 3, "call": 4, "compute_output_shape": 5,
         "load_pretrained_weights": 6, "from_variant": 7, "get_config": 8,
         "from_config": 9, "summary": 10}


def _rank(name: str) -> Optional[int]:
    return 2 if name.startswith("_build_") else _RANK.get(name)


def h2_member_order(members: Sequence[Member], subject: Subject) -> Violations:
    kinds = [kind for kind, _, _ in members]
    first_def = kinds.index("def") if "def" in kinds else len(kinds)
    consts = [name for kind, name, _ in members if kind == "const"]
    out = []
    late = [name for i, (kind, name, _) in enumerate(members) if kind == "const" and i > first_def]
    if late:
        out.append(f"class constants after the first method: {late}")
    if "MODEL_VARIANTS" in consts and consts[0] != "MODEL_VARIANTS":
        out.append(f"MODEL_VARIANTS is not the first class constant ({consts[0]} is)")
    ranked = [(name, _rank(name)) for kind, name, _ in members
              if kind == "def" and _rank(name) is not None]
    out += [f"`{b}` comes after `{a}`" for (a, ra), (b, rb) in zip(ranked, ranked[1:]) if rb < ra]
    return out


def h3_one_init_log(members: Sequence[Member], subject: Subject) -> Violations:
    init = _def(members, "__init__")
    if init is None:
        return []
    count = len(_logger_calls(init, "info"))
    return [] if count == 1 else [f"__init__ writes {count} logger.info calls, expected 1"]


def h3_headless_attribute(members: Sequence[Member], subject: Subject) -> Violations:
    init = _def(members, "__init__")
    if init is None or subject.head_attribute is None:
        return []
    for node in ast.walk(init):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                and node.value.value is None):
            for t in node.targets:
                if (isinstance(t, ast.Attribute) and t.attr == subject.head_attribute
                        and isinstance(t.value, ast.Name) and t.value.id == "self"):
                    return []
    return [f"__init__ never assigns `self.{subject.head_attribute} = None`, so a "
            "headless model has no such attribute"]


def h3_builders_assign(members: Sequence[Member], subject: Subject) -> Violations:
    out = []
    for kind, name, node in members:
        if kind == "def" and name.startswith("_build_"):
            returns = [n for n in _walk_own(node) if isinstance(n, ast.Return) and n.value is not None]
            if returns:
                out.append(f"`{name}` returns a value (line {returns[0].lineno})")
    return out


_NO_LOGGING_IN = ("call", "build", "compute_output_shape", "summary")


def h4_no_logging_outside_init(members: Sequence[Member], subject: Subject) -> Violations:
    out = []
    for kind, name, node in members:
        if kind == "def" and (name in _NO_LOGGING_IN or name.startswith("_build_")):
            calls = _logger_calls(node)
            if calls:
                out.append(f"`{name}` writes {len(calls)} logger call(s)")
    return out


def _is_built_guard(node: ast.stmt) -> bool:
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Attribute)
            and node.test.attr == "built" and isinstance(node.test.value, ast.Name)
            and node.test.value.id == "self" and len(node.body) == 1
            and isinstance(node.body[0], ast.Return) and node.body[0].value is None)


def _is_super_build(node: ast.stmt) -> bool:
    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
        return False
    func = node.value.func
    return (isinstance(func, ast.Attribute) and func.attr == "build"
            and isinstance(func.value, ast.Call) and isinstance(func.value.func, ast.Name)
            and func.value.func.id == "super")


def h5_build_materializes(members: Sequence[Member], subject: Subject) -> Violations:
    build = _def(members, "build")
    if build is None:
        return ["the class defines no `build`"]
    body = _body_without_docstring(build)
    out = []
    if not any(_is_built_guard(s) for s in body):
        out.append("`build` lacks `if self.built: return`")
    if not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "materialize_sublayers" for n in ast.walk(build)):
        out.append("`build` does not call `materialize_sublayers(self, input_shape)`")
    if not body or not _is_super_build(body[-1]):
        out.append("`build` does not end in `super().build(input_shape)`")
    return out


def h10_num_classes_first_kwargs_last(members: Sequence[Member], subject: Subject) -> Violations:
    init = _def(members, "__init__")
    if init is None:
        return []
    names = _params(init)
    out = []
    if "num_classes" in names and names[0] != "num_classes":
        out.append(f"`num_classes` is parameter {names.index('num_classes') + 1}, not the first")
    if init.args.kwarg is None or init.args.kwarg.arg != "kwargs":
        out.append("`__init__` does not end in **kwargs")
    return out


def h10_trailing_parameter_order(members: Sequence[Member], subject: Subject) -> Violations:
    init = _def(members, "__init__")
    if init is None or not subject.classifier:
        return []
    names = _params(init)
    if subject.include_top_last:
        return [] if names[-1] == "include_top" else [
            f"`include_top` must be the last named parameter, `{names[-1]}` is"]
    present = [n for n in ("include_top", "input_shape", "head_config_overrides") if n in names]
    ordered = sorted(present, key=names.index)
    return [] if ordered == present else [f"{ordered} come in the order {present}"]


#: Call forms that only re-spell their one argument, so the attribute that holds
#: the result is "the argument, stored".
_WRAPPERS = {"get", "list", "tuple", "dict", "deepcopy", "initializer_spec", "bool",
             "deserialize_activation", "serialize_activation"}
#: The one stored name that differs from its argument's.
_STORED_AS = {"input_shape": "input_shape_config", "head_config_overrides": "_head_config_overrides"}


def _has_name(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Name) for n in ast.walk(node))


def _lone_param(expr: ast.AST, params: FrozenSet[str]) -> Optional[str]:
    """The one constructor argument ``expr`` only re-spells, or ``None``."""
    if isinstance(expr, ast.Name):
        return expr.id if expr.id in params else None
    if isinstance(expr, ast.Call):
        func = expr.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name in _WRAPPERS and len(expr.args) == 1 and not expr.keywords:
            return _lone_param(expr.args[0], params)
        return None
    if isinstance(expr, (ast.IfExp, ast.BoolOp)):
        operands = [expr.body, expr.orelse] if isinstance(expr, ast.IfExp) else expr.values
        named = [o for o in operands if _has_name(o)]
        return _lone_param(named[0], params) if len(named) == 1 else None
    return None


def h10_stored_attribute_names(members: Sequence[Member], subject: Subject) -> Violations:
    init = _def(members, "__init__")
    if init is None:
        return []
    params = frozenset(_params(init))
    assigned = set()
    respelled: Dict[str, List[str]] = {}
    for node in _walk_own(init):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                        and t.value.id == "self"):
                    assigned.add(t.attr)
                    param = _lone_param(node.value, params)
                    if param:
                        respelled.setdefault(param, []).append(t.attr)
    out = []
    for param, attrs in sorted(respelled.items()):
        want = _STORED_AS.get(param, param)
        if want not in attrs:
            out.append(f"argument `{param}` is stored as `self.{attrs[0]}`, expected `self.{want}`")
    for param, want in _STORED_AS.items():
        if param in params and want not in assigned and param not in respelled:
            out.append(f"argument `{param}` is never stored as `self.{want}`")
    return out


_RENAMED_CONCEPTS = {"weight_decay": "kernel_regularizer", "variant_config": "variant",
                     "stem": "stem_type"}


def h10_one_name_per_concept(members: Sequence[Member], subject: Subject) -> Violations:
    init = _def(members, "__init__")
    if init is None:
        return []
    names = _params(init)
    out = [f"constructor argument `{old}` should be `{new}`"
           for old, new in _RENAMED_CONCEPTS.items() if old in names]
    out += [f"constructor has no `{p}` argument" for p in subject.required_params if p not in names]
    return out


def _config_keys(fn: ast.FunctionDef) -> FrozenSet[str]:
    keys = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Dict):
            keys.update(k.value for k in node.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str))
        elif (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)
              and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
            keys.add(node.slice.value)
        elif isinstance(node, ast.Call):
            keys.update(k.arg for k in node.keywords if k.arg)
    return frozenset(keys)


def _calls_super_get_config(fn: ast.FunctionDef) -> bool:
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "get_config" and isinstance(n.func.value, ast.Call)
               and isinstance(n.func.value.func, ast.Name) and n.func.value.func.id == "super"
               for n in ast.walk(fn))


def h11_get_config_covers_the_constructor(members: Sequence[Member], subject: Subject) -> Violations:
    init, get_config = _def(members, "__init__"), _def(members, "get_config")
    if init is None:
        return []
    if get_config is None:
        return ["the class defines no `get_config`"]
    out = []
    if not _calls_super_get_config(get_config):
        out.append("`get_config` does not start from `super().get_config()`")
    keys = _config_keys(get_config)
    missing = [p for p in _params(init) if p not in keys and p != "name"]
    if missing:
        out.append(f"`get_config` writes no key for {missing}")
    return out


def _copies_the_dict(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Call):
        return False
    func = stmt.value.func
    return ((isinstance(func, ast.Name) and func.id == "dict")
            or (isinstance(func, ast.Attribute) and func.attr == "copy"))


def _returns_plain_cls_call(stmt: ast.stmt) -> bool:
    call = getattr(stmt, "value", None)
    return (isinstance(stmt, ast.Return) and isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name) and call.func.id == "cls"
            and not call.args and all(k.arg is None for k in call.keywords))


def h11_from_config_does_real_work(members: Sequence[Member], subject: Subject) -> Violations:
    fn = _def(members, "from_config")
    if fn is None:
        return []
    body = _body_without_docstring(fn)
    if body and all(_copies_the_dict(s) or _returns_plain_cls_call(s) for s in body):
        return ["`from_config` only copies the dict and calls `cls(**config)`; delete it"]
    return []


def _is_name_pair_dict(node: Optional[ast.AST]) -> bool:
    """``{**base_config, **config}``: a dict of exactly two splatted names."""
    return (isinstance(node, ast.Dict) and node.keys == [None, None]
            and [getattr(v, "id", None) for v in node.values] == ["base_config", "config"])


def h11_get_config_has_the_reference_form(members: Sequence[Member], subject: Subject) -> Violations:
    fn = _def(members, "get_config")
    if fn is None:
        return []
    body = _body_without_docstring(fn)
    out = []
    base_assigned = any(
        isinstance(s, ast.Assign) and any(getattr(t, "id", None) == "base_config" for t in s.targets)
        and _calls_super_get_config(s)
        for s in body)
    if not base_assigned:
        out.append("`get_config` does not bind `base_config = super().get_config()`")
    if not body or not (isinstance(body[-1], ast.Return) and _is_name_pair_dict(body[-1].value)):
        out.append("`get_config` does not end in `return {**base_config, **config}`")
    return out


def h2_call_takes_inputs(members: Sequence[Member], subject: Subject) -> Violations:
    call = _def(members, "call")
    if call is None or len(call.args.args) < 2:
        return []
    name = call.args.args[1].arg
    return [] if name == "inputs" else [f"`call` names its input `{name}`, expected `inputs`"]


_CONTAINER_CONCEPTS = (("block", "blocks"), ("stage", "stages"))


def _container_attributes(members: Sequence[Member]) -> List[str]:
    """Names of the ``self.<attr>`` lists a constructor or a ``_build_*`` method creates."""
    names = []
    for kind, name, fn in members:
        if kind != "def" or not (name == "__init__" or name.startswith("_build_")):
            continue
        for node in _walk_own(fn):
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(
                    node.value, (ast.List, ast.ListComp)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names += [t.attr for t in targets if isinstance(t, ast.Attribute)
                          and isinstance(t.value, ast.Name) and t.value.id == "self"]
    return names


def h10_block_container_names(members: Sequence[Member], subject: Subject) -> Violations:
    out = []
    for attr in _container_attributes(members):
        for word, canonical in _CONTAINER_CONCEPTS:
            if word in attr and attr != canonical:
                out.append(f"the {word} container `self.{attr}` should be `self.{canonical}`")
                break
    return out


def _is_a_spec(value: ast.AST) -> bool:
    """Whether ``value`` stores an initializer as a spec: ``initializer_spec(...)``, a string, a dict."""
    if isinstance(value, ast.Call):
        func = value.func
        return (func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")) == "initializer_spec"
    if isinstance(value, ast.Constant):
        return value.value is None or isinstance(value.value, str)
    if isinstance(value, ast.Dict):
        return True
    if isinstance(value, ast.IfExp):
        return _is_a_spec(value.body) and _is_a_spec(value.orelse)
    return False


def _initializer_assignments(init: ast.FunctionDef) -> List[Tuple[str, ast.AST]]:
    found = []
    for node in _walk_own(init):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            found += [(t.attr, node.value) for t in targets if isinstance(t, ast.Attribute)
                      and isinstance(t.value, ast.Name) and t.value.id == "self"
                      and "initializer" in t.attr]
    return found


def h15_initializer_storage(members: Sequence[Member], subject: Subject) -> Violations:
    init = _def(members, "__init__")
    if init is None:
        return []
    return [f"`self.{attr}` stores `{ast.unparse(value)}`, not an `initializer_spec(...)`"
            for attr, value in _initializer_assignments(init) if not _is_a_spec(value)]


_NO_CALL = "no call (or no input argument) to inspect"
_NO_CONTAINER = "no list of blocks or stages is built"
_NO_INITIALIZER = "__init__ stores no initializer attribute"

_NEEDS_INIT = frozenset({
    "h3_one_init_log", "h3_headless_attribute", "h10_num_classes_first_kwargs_last",
    "h10_trailing_parameter_order", "h10_stored_attribute_names", "h10_one_name_per_concept",
    "h11_get_config_covers_the_constructor", "h15_initializer_storage"})


def precondition_absent(members: Sequence[Member], subject: Subject, rule: str) -> Optional[str]:
    """Why ``rule`` can report nothing about ``subject``, or ``None`` when it can look.

    A rule that returns no violation because the thing it inspects is absent (a
    ``from_config`` that is not defined, a class that is not a classifier) passes by
    construction. Interface contract: ``members`` are the subject's
    :func:`subject_members`; returns a short reason, or ``None`` when the rule has
    something to read. Mirrors each rule's own early return, nothing else.
    """
    if rule in _NEEDS_INIT and _def(members, "__init__") is None:
        return "no __init__ to inspect"
    if rule == "h3_headless_attribute" and subject.head_attribute is None:
        return "the subject names no head attribute"
    if rule == "h10_trailing_parameter_order" and not subject.classifier:
        return "not a classifier: rule returns [] by construction"
    if rule == "h10_trailing_parameter_order" and not subject.include_top_last:
        init = _def(members, "__init__")
        ordered = ("include_top", "input_shape", "head_config_overrides")
        if init is not None and sum(n in _params(init) for n in ordered) < 2:
            return "fewer than two of include_top, input_shape, head_config_overrides to order"
    if rule == "h3_builders_assign" and not any(
            kind == "def" and name.startswith("_build_") for kind, name, _ in members):
        return "no _build_* method to inspect"
    if rule == "h4_no_logging_outside_init" and not any(
            kind == "def" and (name in _NO_LOGGING_IN or name.startswith("_build_"))
            for kind, name, _ in members):
        return "no method the rule reads"
    if rule == "h11_from_config_does_real_work" and _def(members, "from_config") is None:
        return "no from_config defined (rule only bites a defined one)"
    if rule == "h11_get_config_has_the_reference_form" and _def(members, "get_config") is None:
        return "no get_config defined"
    if rule == "h2_call_takes_inputs":
        call = _def(members, "call")
        if call is None or len(call.args.args) < 2:
            return _NO_CALL
    if rule == "h10_block_container_names" and not _container_attributes(members):
        return _NO_CONTAINER
    if rule == "h15_initializer_storage":
        init = _def(members, "__init__")
        if init is not None and not _initializer_assignments(init):
            return _NO_INITIALIZER
    return None


CLASS_RULE_FUNCTIONS: Dict[str, Callable[[Sequence[Member], Subject], Violations]] = {
    "h2_member_order": h2_member_order,
    "h3_one_init_log": h3_one_init_log,
    "h3_headless_attribute": h3_headless_attribute,
    "h3_builders_assign": h3_builders_assign,
    "h4_no_logging_outside_init": h4_no_logging_outside_init,
    "h5_build_materializes": h5_build_materializes,
    "h10_num_classes_first_kwargs_last": h10_num_classes_first_kwargs_last,
    "h10_trailing_parameter_order": h10_trailing_parameter_order,
    "h10_stored_attribute_names": h10_stored_attribute_names,
    "h10_one_name_per_concept": h10_one_name_per_concept,
    "h11_get_config_covers_the_constructor": h11_get_config_covers_the_constructor,
    "h11_from_config_does_real_work": h11_from_config_does_real_work,
    "h11_get_config_has_the_reference_form": h11_get_config_has_the_reference_form,
    "h2_call_takes_inputs": h2_call_takes_inputs,
    "h10_block_container_names": h10_block_container_names,
    "h15_initializer_storage": h15_initializer_storage,
}

# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------

_ANCHOR_TOKEN = re.compile(r"[A-Za-z0-9_.-]+/D-\d+")


def anchor_token_counts(source: str) -> Counter:
    """How many ``# DECISION`` blocks of ``source`` carry each ``<plan>/D-NNN`` token.

    A block is the line carrying ``# DECISION`` plus the comment-only lines that
    follow it, so an anchor naming a second decision on a continuation line counts
    both; a token repeated inside one block counts once.
    """
    lines = source.splitlines()
    counts: Counter = Counter()
    i = 0
    while i < len(lines):
        if "# DECISION" in lines[i]:
            block = [lines[i].split("# DECISION", 1)[1]]
            j = i + 1
            while j < len(lines) and lines[j].lstrip().startswith("#") and "# DECISION" not in lines[j]:
                block.append(lines[j])
                j += 1
            counts.update(set(_ANCHOR_TOKEN.findall("\n".join(block))))
            i = j
        else:
            i += 1
    return counts


def lost_anchor_tokens(source: str, expected: Dict[str, int]) -> Violations:
    """The expected tokens ``source`` carries fewer times than before."""
    have = anchor_token_counts(source)
    return [f"{token} x{want} -> x{have[token]}" for token, want in sorted(expected.items())
            if have[token] < want]


def lost_anchor_tokens_over(sources: Sequence[str], expected: Dict[str, int]) -> Violations:
    """Tokens absent from every file of a family (a merged duplicate is not a loss)."""
    present = set().union(*(anchor_token_counts(s) for s in sources))
    return [token for token in sorted(expected) if token not in present]


#: The ``# DECISION <plan>/D-NNN`` tokens each touched file carries today, with how
#: many anchor blocks carry each (the live anchor census at the commit this file
#: was written). A token may move with its code inside the file or, for a family
#: in :data:`ANCHOR_SETS`, to another file of the family; it may not disappear.
ANCHOR_TOKENS: Dict[str, Dict[str, int]] = {
    "src/train/cliffordnet/train_downsampling_techniques.py": {
        "plan-2026-10-14T062616-81938e81/D-095": 1,
        "plan-2026-10-14T062616-81938e81/D-117": 1,
        "plan-2026-10-14T062616-81938e81/D-118": 1,
    },
    "src/train/rbf_protonet/train_rbf_protonet.py": {
        "plan-2026-09-16T052349-7dfede94/D-003": 1,
        "plan-2026-09-16T052349-7dfede94/D-008": 1,
    },
    "src/train/vae/train_vae.py": {
        "plan_2026-06-04_7ff8ea8b/D-002": 1,
        "plan_2026-06-04_7ff8ea8b/D-007": 1,
    },
    "src/dl_techniques/initializers/clone.py": {
        "plan-2026-10-26T125959-7af851e5/D-033": 1,
    },
    "src/dl_techniques/layers/attention/attention_routing_capsule.py": {
        "plan-2026-07-27T130643-38c5646a/D-013": 1,
    },
    "src/dl_techniques/layers/conv_blocks/basic_block.py": {
        "plan_2026-05-18_6776f8ba/D-003": 1,
    },
    "src/dl_techniques/layers/conv_blocks/bottleneck_block.py": {
        "plan-2026-09-14T165315-47f9d575/D-006": 1,
        "plan_2026-05-18_6776f8ba/D-003": 1,
    },
    "src/dl_techniques/layers/conv_blocks/stem.py": {
        "plan-2026-10-26T125959-7af851e5/D-005": 1,
        "plan-2026-10-26T125959-7af851e5/D-020": 2,
        "plan-2026-10-26T125959-7af851e5/D-030": 1,
        "plan-2026-10-26T125959-7af851e5/D-031": 1,
    },
    "src/dl_techniques/layers/heads/vision/classification.py": {
        "plan-2026-10-14T062616-81938e81/D-001": 1,
        "plan-2026-10-14T062616-81938e81/D-013": 1,
    },
    "src/dl_techniques/models/vision/capsnet/model.py": {
        "plan-2026-08-14T233721-d4f9beb2/D-057": 1,
        "plan-2026-08-17T183311-79c63e38/D-021": 2,
        "plan-2026-08-18T073231-52a93f8c/D-006": 1,
        "plan-2026-08-18T073231-52a93f8c/D-007": 2,
        "plan-2026-08-19T163559-499b6f0e/D-011": 2,
        "plan-2026-08-19T163559-499b6f0e/D-085": 1,
        "plan-2026-08-19T163559-499b6f0e/D-089": 1,
        "plan-2026-08-22T035419-a11304c8/D-053": 1,
    },
    "src/dl_techniques/models/vision/cliffordnet/__init__.py": {
        "plan-2026-08-10T130454-3649c19e/D-006": 1,
        "plan-2026-10-23T132722-77331e7a/D-002": 1,
    },
    "src/dl_techniques/models/vision/cliffordnet/hierarchical.py": {
        "plan-2026-10-26T125959-7af851e5/D-007": 1,
        "plan-2026-10-26T125959-7af851e5/D-031": 1,
    },
    "src/dl_techniques/models/vision/cliffordnet/model.py": {
        "plan-2026-08-23T091307-9a110062/D-480": 1,
        "plan-2026-10-26T125959-7af851e5/D-007": 1,
        "plan-2026-10-26T125959-7af851e5/D-023": 3,
        "plan-2026-10-26T125959-7af851e5/D-031": 1,
        "plan-2026-10-26T125959-7af851e5/D-041": 1,
        "plan_2026-05-11_0090b0b8/D-001": 2,
    },
    "src/dl_techniques/models/vision/convnext/convnext_v1.py": {
        "plan-2026-08-14T233721-d4f9beb2/D-067": 1,
        "plan-2026-08-19T163559-499b6f0e/D-069": 1,
        "plan-2026-08-19T163559-499b6f0e/D-125": 1,
        "plan-2026-10-14T062616-81938e81/D-013": 1,
        "plan-2026-10-14T062616-81938e81/D-076": 1,
        "plan_2026-06-15_e6a0391c/D-003": 1,
    },
    "src/dl_techniques/models/vision/convnext/convnext_v2.py": {
        "plan-2026-08-14T233721-d4f9beb2/D-067": 1,
        "plan-2026-08-19T163559-499b6f0e/D-125": 1,
        "plan-2026-10-14T062616-81938e81/D-013": 1,
        "plan-2026-10-14T062616-81938e81/D-076": 1,
        "plan_2026-06-15_e6a0391c/D-003": 1,
    },
    "src/dl_techniques/models/vision/convunext/model.py": {
        "D-002/D-005": 1,
        "D-006/D-012": 1,
        "plan-2026-08-11T201945-91938f65/D-002": 1,
        "plan-2026-08-14T092357-0e3d792d/D-012": 2,
        "plan-2026-09-14T075146-6b083728/D-003": 1,
        "plan-2026-09-15T034909-a7edc8da/D-009": 1,
        "plan-2026-09-15T094955-31fbe3db/D-006": 1,
        "plan_2026-06-21_eb7fd829/D-002": 1,
        "plan_2026-06-21_eb7fd829/D-005": 1,
        "plan_2026-06-21_eb7fd829/D-006": 1,
        "plan_2026-06-26_0ec1a304/D-001": 1,
        "plan_2026-07-01_8054f023/D-001": 1,
        "plan_2026-07-10_be906be8/D-002": 1,
        "plan_2026-07-11_bb4b38b5/D-001": 1,
        "plan_2026-07-11_bb4b38b5/D-002": 1,
    },
    "src/dl_techniques/models/vision/depth_anything/model.py": {
        "plan-2026-08-14T233721-d4f9beb2/D-014": 1,
        "plan-2026-08-17T183311-79c63e38/D-028": 1,
        "plan-2026-08-17T183311-79c63e38/D-033": 3,
        "plan-2026-08-17T183311-79c63e38/D-034": 1,
        "plan-2026-08-19T070627-a616f581/D-009": 1,
        "plan-2026-08-22T035419-a11304c8/D-009": 1,
        "plan-2026-10-14T062616-81938e81/D-031": 1,
        "plan-2026-10-14T062616-81938e81/D-113": 1,
        "plan-2026-10-14T062616-81938e81/D-128": 1,
        "plan-2026-10-14T062616-81938e81/D-131": 1,
        "plan-2026-10-14T062616-81938e81/D-136": 1,
    },
    "src/dl_techniques/models/vision/mobilenet/common.py": {
        "plan-2026-08-22T035419-a11304c8/D-111": 1,
    },
    "src/dl_techniques/models/vision/mobilenet/mobilenet_v1.py": {
        "plan-2026-08-14T233721-d4f9beb2/D-065": 1,
        "plan-2026-08-14T233721-d4f9beb2/D-069": 1,
        "plan-2026-08-22T035419-a11304c8/D-203": 1,
        "plan-2026-10-26T125959-7af851e5/D-039": 1,
    },
    "src/dl_techniques/models/vision/mobilenet/mobilenet_v2.py": {
        "plan-2026-08-14T233721-d4f9beb2/D-065": 1,
        "plan-2026-08-14T233721-d4f9beb2/D-069": 1,
        "plan-2026-08-22T035419-a11304c8/D-203": 1,
        "plan-2026-10-26T125959-7af851e5/D-039": 1,
    },
    "src/dl_techniques/models/vision/mobilenet/mobilenet_v3.py": {
        "plan-2026-08-14T233721-d4f9beb2/D-065": 1,
        "plan-2026-08-14T233721-d4f9beb2/D-069": 1,
        "plan-2026-08-22T035419-a11304c8/D-203": 1,
        "plan-2026-10-14T062616-81938e81/D-232": 1,
        "plan-2026-10-26T125959-7af851e5/D-039": 1,
    },
    "src/dl_techniques/models/vision/mobilenet/mobilenet_v4.py": {
        "plan-2026-08-14T183218-f4c612aa/D-011": 1,
        "plan-2026-08-14T233721-d4f9beb2/D-065": 1,
        "plan-2026-08-14T233721-d4f9beb2/D-069": 1,
        "plan-2026-08-19T163559-499b6f0e/D-085": 1,
        "plan-2026-08-22T035419-a11304c8/D-203": 1,
    },
    "src/dl_techniques/models/vision/rbf_protonet/model.py": {
        "plan-2026-09-16T052349-7dfede94/D-002": 1,
        "plan-2026-09-16T052349-7dfede94/D-006": 1,
        "plan-2026-09-16T064227-b83f88ad/D-002": 1,
        "plan-2026-09-16T064227-b83f88ad/D-005": 1,
    },
    "src/dl_techniques/models/vision/resnet/model.py": {
        "plan-2026-08-17T183311-79c63e38/D-019": 1,
        "plan-2026-08-19T163559-499b6f0e/D-041": 1,
        "plan-2026-08-19T163559-499b6f0e/D-127": 1,
        "plan-2026-08-23T091307-9a110062/D-480": 1,
        "plan-2026-08-23T203721-009b7ccf/D-019": 1,
        "plan-2026-10-26T125959-7af851e5/D-019": 1,
        "plan-2026-10-26T125959-7af851e5/D-020": 1,
        "plan_2026-05-18_6776f8ba/D-003": 1,
    },
    "src/dl_techniques/models/vision/squeezenet/squeezenet_v1.py": {
        "plan-2026-08-17T183311-79c63e38/D-020": 2,
        "plan-2026-08-19T163559-499b6f0e/D-129": 1,
        "plan-2026-08-23T091307-9a110062/D-481": 1,
        "plan-2026-10-14T062616-81938e81/D-017": 1,
    },
    "src/dl_techniques/models/vision/squeezenet/squeezenet_v2.py": {
        "plan-2026-08-17T183311-79c63e38/D-020": 1,
        "plan-2026-08-19T163559-499b6f0e/D-129": 1,
        "plan-2026-08-23T091307-9a110062/D-481": 1,
        "plan-2026-10-14T062616-81938e81/D-017": 1,
        "plan-2026-10-26T125959-7af851e5/D-004": 1,
    },
    "src/dl_techniques/models/vision/vae/model.py": {
        "plan-2026-08-18T140459-7991552f/D-028": 1,
        "plan-2026-08-19T163559-499b6f0e/D-011": 5,
        "plan-2026-08-19T163559-499b6f0e/D-078": 1,
        "plan-2026-08-19T163559-499b6f0e/D-089": 2,
        "plan-2026-08-22T035419-a11304c8/D-014": 1,
        "plan_2026-06-04_6196678d/D-003": 1,
        "plan_2026-06-04_6196678d/D-005": 1,
        "plan_2026-06-04_6196678d/D-007": 1,
        "plan_2026-06-04_6196678d/D-009": 1,
        "plan_2026-06-04_7ff8ea8b/D-001": 1,
        "plan_2026-06-04_d4ef81f1/D-003": 1,
        "plan_2026-06-04_d4ef81f1/D-004": 1,
        "plan_2026-06-05_56b39171/D-004": 1,
    },
    "src/dl_techniques/models/vision/vq_vae_rotation/model.py": {
        "plan-2026-08-14T233721-d4f9beb2/D-040": 1,
        "plan-2026-08-18T140459-7991552f/D-026": 2,
        "plan-2026-08-19T163559-499b6f0e/D-011": 2,
        "plan-2026-08-19T163559-499b6f0e/D-089": 1,
    },
    "src/dl_techniques/models/vision_language/clifford_clip/model.py": {
        "plan-2026-07-15T114613-5add9baa/D-001": 2,
        "plan-2026-07-15T114613-5add9baa/D-004": 1,
        "plan-2026-07-15T140843-168d5bac/D-001": 1,
        "plan-2026-08-19T163559-499b6f0e/D-072": 1,
        "plan-2026-08-23T091307-9a110062/D-480": 1,
        "plan-2026-09-16T085926-41908265/D-001": 1,
        "plan-2026-10-14T062616-81938e81/D-047": 1,
        "plan-2026-10-14T062616-81938e81/D-070": 1,
        "plan-2026-10-14T062616-81938e81/D-225": 1,
        "plan_2026-05-31_76981d58/D-001": 1,
    },
    "src/dl_techniques/models/vision_language/sam/sam1/prompt_encoder.py": {
        "plan-2026-08-03T191222-1d751f81/D-013": 1,
        "plan-2026-08-03T191222-1d751f81/D-016": 1,
        "plan-2026-08-03T191222-1d751f81/D-025": 1,
    },
    "src/dl_techniques/utils/weight_transfer.py": {
        "plan-2026-08-14T233721-d4f9beb2/D-070": 1,
        "plan-2026-10-11T004932-b83935e1/D-027": 1,
        "plan-2026-10-11T004932-b83935e1/D-031": 2,
        "plan-2026-10-11T170958-dc4d2202/D-007": 1,
        "plan-2026-10-11T170958-dc4d2202/D-015": 1,
        "plan-2026-10-13T035756-5dd4d94d/D-007": 1,
        "plan-2026-10-26T125959-7af851e5/D-023": 1,
        "plan-2026-10-26T125959-7af851e5/D-040": 1,
        "plan-2026-10-26T125959-7af851e5/D-041": 2,
    },
}

# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Open:
    """A row that fails today: the migration that closes it and what was measured.

    ``signature`` is the pieces of the failure text that name which violations the
    row has, each holding no count or line number. A ledgered row that fails with a
    message lacking any piece is a plain failure, not an expected one: the rule
    still fails, but not the way the row says, so a partial closure shows up as red.
    """

    closed_by: str
    measured: str
    signature: Tuple[str, ...]


#: Every row that fails today, keyed ``(rule, subject-or-module key)``.
LEDGER: Dict[Tuple[str, str], Open] = {
    ("h9_factory_is_one_delegating_return", "squeezenet_v1"): Open(
        "the SqueezeNet migration",
        "3 violation(s): create_squeezenet_v1('variant', 'num_classes', 'input_shape', 'weights')"
        " is not ('variant', 'num_classes', 'input_shape', 'pretrained'); create_squeezenet_v1: "
        '`input_shape` does not default to None; create_squeezenet_v1: `pretrained` does not '
        'default to False',
        ("create_squeezenet_v1('variant', 'num_classes', 'input_shape', "
         "'weights') is not ('variant', 'num_classes', 'input_shape', "
         "'pretrained')",
         'create_squeezenet_v1: `input_shape` does not default to None',
         'create_squeezenet_v1: `pretrained` does not default to False')),
    ("h9_factory_is_one_delegating_return", "squeezenet_v2"): Open(
        "the SqueezeNet migration",
        "3 violation(s): create_squeezenodule_net_v2('variant', 'num_classes', 'input_shape', "
        "'weights') is not ('variant', 'num_classes', 'input_shape', 'pretrained'); "
        'create_squeezenodule_net_v2: `input_shape` does not default to None; '
        'create_squeezenodule_net_v2: `pretrained` does not default to False',
        ("create_squeezenodule_net_v2('variant', 'num_classes', "
         "'input_shape', 'weights') is not ('variant', 'num_classes', "
         "'input_shape', 'pretrained')",
         'create_squeezenodule_net_v2: `input_shape` does not default to None',
         'create_squeezenodule_net_v2: `pretrained` does not default to False')),
    ("h9_factory_is_one_delegating_return", "vae"): Open(
        "the ConvUNext and VAE migration",
        '6 violation(s): create_vae has no `pretrained` parameter; create_vae is not one `return '
        'Class.from_variant(..., **kwargs)`; create_vae_from_config has no `input_shape` '
        'parameter; create_vae_from_config has no `pretrained` parameter; create_vae_from_config '
        'does not end in **kwargs; create_vae_from_config is not one `return '
        'Class.from_variant(..., **kwargs)`',
        ('create_vae has no `pretrained` parameter',
         'create_vae is not one `return Class.from_variant(..., **kwargs)`',
         'create_vae_from_config has no `input_shape` parameter',
         'create_vae_from_config has no `pretrained` parameter',
         'create_vae_from_config does not end in **kwargs',
         'create_vae_from_config is not one `return Class.from_variant(..., '
         '**kwargs)`')),
    ("h1_factories_follow_the_classes", "rbf_protonet"): Open(
        "the RBFProtoNet and UcaEncoder migration",
        '1 violation(s): `def create_rbf_protonet` (line 487) sits between classes',
        ('`def create_rbf_protonet`',)),
    ("no_from_keras_import", "uca_encoder"): Open(
        "the RBFProtoNet and UcaEncoder migration",
        '1 violation(s): line 25: from keras import ...',
        ('from keras import ...',)),
    ("no_from_keras_import", "capsnet"): Open(
        "the CapsNet migration",
        '1 violation(s): line 25: from keras import ...',
        ('from keras import ...',)),
    ("h3_one_init_log", "squeezenet_v1"): Open(
        "the SqueezeNet migration",
        '1 violation(s): __init__ writes 0 logger.info calls, expected 1',
        ('__init__ writes 0 logger.info calls, expected 1',)),
    ("h10_stored_attribute_names", "squeezenet_v1"): Open(
        "the SqueezeNet migration",
        '1 violation(s): argument `input_shape` is stored as `self._input_shape`, expected '
        '`self.input_shape_config`',
        ('argument `input_shape` is stored as `self._input_shape`',)),
    ("h10_one_name_per_concept", "squeezenet_v1"): Open(
        "the SqueezeNet migration",
        '1 violation(s): constructor argument `variant_config` should be `variant`',
        ('constructor argument `variant_config` should be `variant`',)),
    ("h3_one_init_log", "squeezenet_v2"): Open(
        "the SqueezeNet migration",
        '1 violation(s): __init__ writes 0 logger.info calls, expected 1',
        ('__init__ writes 0 logger.info calls, expected 1',)),
    ("h10_stored_attribute_names", "squeezenet_v2"): Open(
        "the SqueezeNet migration",
        '1 violation(s): argument `input_shape` is stored as `self._input_shape`, expected '
        '`self.input_shape_config`',
        ('argument `input_shape` is stored as `self._input_shape`',)),
    ("h10_one_name_per_concept", "squeezenet_v2"): Open(
        "the SqueezeNet migration",
        '1 violation(s): constructor argument `variant_config` should be `variant`',
        ('constructor argument `variant_config` should be `variant`',)),
    ("h10_stored_attribute_names", "convunext_stem"): Open(
        "the ConvUNext and VAE migration",
        '1 violation(s): argument `activation` is stored as `self.activation_name`, expected '
        '`self.activation`',
        ('argument `activation` is stored as `self.activation_name`',)),
    ("h11_from_config_does_real_work", "convunext_stem"): Open(
        "the ConvUNext and VAE migration",
        '1 violation(s): `from_config` only copies the dict and calls `cls(**config)`; delete it',
        ('`from_config` only copies the dict and calls `cls(**config)`; delete it',)),
    ("h4_no_logging_outside_init", "vae"): Open(
        "the ConvUNext and VAE migration",
        '1 violation(s): `summary` writes 9 logger call(s)',
        ('`summary` writes',)),
    ("h10_stored_attribute_names", "vae"): Open(
        "the ConvUNext and VAE migration",
        '1 violation(s): argument `input_shape` is stored as `self._input_shape`, expected '
        '`self.input_shape_config`',
        ('argument `input_shape` is stored as `self._input_shape`',)),
    ("h10_num_classes_first_kwargs_last", "rbf_protonet"): Open(
        "the RBFProtoNet and UcaEncoder migration",
        '1 violation(s): `num_classes` is parameter 10, not the first',
        ('`num_classes` is parameter',)),
    ("h10_num_classes_first_kwargs_last", "clifford_rbf_protonet"): Open(
        "the RBFProtoNet and UcaEncoder migration",
        '1 violation(s): `num_classes` is parameter 2, not the first',
        ('`num_classes` is parameter',)),
    ("h2_member_order", "capsnet"): Open(
        "the CapsNet migration",
        '1 violation(s): `_build_sublayer_tree` comes after `build`',
        ('`_build_sublayer_tree` comes after `build`',)),
    ("h3_one_init_log", "capsnet"): Open(
        "the CapsNet migration",
        '1 violation(s): __init__ writes 0 logger.info calls, expected 1',
        ('__init__ writes 0 logger.info calls, expected 1',)),
    ("h4_no_logging_outside_init", "capsnet"): Open(
        "the CapsNet migration",
        '2 violation(s): `build` writes 1 logger call(s); `summary` writes 11 logger call(s)',
        ('`build` writes',
         '`summary` writes')),
    ("h5_build_materializes", "capsnet"): Open(
        "the CapsNet migration",
        '2 violation(s): `build` lacks `if self.built: return`; `build` does not call '
        '`materialize_sublayers(self, input_shape)`',
        ('`build` lacks `if self.built: return`',
         '`build` does not call `materialize_sublayers(self, input_shape)`')),
    ("h10_stored_attribute_names", "capsnet"): Open(
        "the CapsNet migration",
        '1 violation(s): argument `input_shape` is stored as `self._input_shape`, expected '
        '`self.input_shape_config`',
        ('argument `input_shape` is stored as `self._input_shape`',)),
    ("h3_one_init_log", "capsnet_v2"): Open(
        "the CapsNet migration",
        '1 violation(s): __init__ writes 0 logger.info calls, expected 1',
        ('__init__ writes 0 logger.info calls, expected 1',)),
    ("h5_build_materializes", "capsnet_v2"): Open(
        "the CapsNet migration",
        '1 violation(s): the class defines no `build`',
        ('the class defines no `build`',)),
    ("h10_stored_attribute_names", "capsnet_v2"): Open(
        "the CapsNet migration",
        '1 violation(s): argument `input_shape` is stored as `self._input_shape`, expected '
        '`self.input_shape_config`',
        ('argument `input_shape` is stored as `self._input_shape`',)),
    ("h10_one_name_per_concept", "capsnet_v2"): Open(
        "the CapsNet migration",
        '1 violation(s): constructor argument `stem` should be `stem_type`',
        ('constructor argument `stem` should be `stem_type`',)),
    ('h11_get_config_has_the_reference_form', 'squeezenet_v1'): Open(
        'the SqueezeNet migration',
        '2 violation(s): `get_config` does not bind `base_config = '
        'super().get_config()`; `get_config` does not end in `return {**base_config, '
        '**config}`',
        ('`get_config` does not bind `base_config = super().get_config()`',
         '`get_config` does not end in `return {**base_config, **config}`')),
    ('h15_initializer_storage', 'squeezenet_v1'): Open(
        'the SqueezeNet migration',
        '1 violation(s): `self.kernel_initializer` stores `kernel_initializer`, not '
        'an `initializer_spec(...)`',
        ('`self.kernel_initializer` stores `kernel_initializer`, not an '
         '`initializer_spec(...)`',)),
    ('h11_get_config_has_the_reference_form', 'squeezenet_v2'): Open(
        'the SqueezeNet migration',
        '2 violation(s): `get_config` does not bind `base_config = '
        'super().get_config()`; `get_config` does not end in `return {**base_config, '
        '**config}`',
        ('`get_config` does not bind `base_config = super().get_config()`',
         '`get_config` does not end in `return {**base_config, **config}`')),
    ('h15_initializer_storage', 'squeezenet_v2'): Open(
        'the SqueezeNet migration',
        '1 violation(s): `self.kernel_initializer` stores `kernel_initializer`, not '
        'an `initializer_spec(...)`',
        ('`self.kernel_initializer` stores `kernel_initializer`, not an '
         '`initializer_spec(...)`',)),
    ('h11_get_config_has_the_reference_form', 'convunext_stem'): Open(
        'the ConvUNext and VAE migration',
        '2 violation(s): `get_config` does not bind `base_config = '
        'super().get_config()`; `get_config` does not end in `return {**base_config, '
        '**config}`',
        ('`get_config` does not bind `base_config = super().get_config()`',
         '`get_config` does not end in `return {**base_config, **config}`')),
    ('h15_initializer_storage', 'convunext_stem'): Open(
        'the ConvUNext and VAE migration',
        '1 violation(s): `self.kernel_initializer` stores '
        '`keras.initializers.get(kernel_initializer)`, not an `initializer_spec(...)`',
        ('`self.kernel_initializer` stores '
         '`keras.initializers.get(kernel_initializer)`, not an `initializer_spec(...)`',)),
    ('h11_get_config_has_the_reference_form', 'vae'): Open(
        'the ConvUNext and VAE migration',
        '2 violation(s): `get_config` does not bind `base_config = '
        'super().get_config()`; `get_config` does not end in `return {**base_config, '
        '**config}`',
        ('`get_config` does not bind `base_config = super().get_config()`',
         '`get_config` does not end in `return {**base_config, **config}`')),
    ('h15_initializer_storage', 'vae'): Open(
        'the ConvUNext and VAE migration',
        '1 violation(s): `self.kernel_initializer` stores `kernel_initializer`, not '
        'an `initializer_spec(...)`',
        ('`self.kernel_initializer` stores `kernel_initializer`, not an '
         '`initializer_spec(...)`',)),
    ('h11_get_config_has_the_reference_form', 'uca_encoder'): Open(
        'the RBFProtoNet and UcaEncoder migration',
        '2 violation(s): `get_config` does not bind `base_config = '
        'super().get_config()`; `get_config` does not end in `return {**base_config, '
        '**config}`',
        ('`get_config` does not bind `base_config = super().get_config()`',
         '`get_config` does not end in `return {**base_config, **config}`')),
    ('h11_get_config_has_the_reference_form', 'capsnet'): Open(
        'the CapsNet migration',
        '2 violation(s): `get_config` does not bind `base_config = '
        'super().get_config()`; `get_config` does not end in `return {**base_config, '
        '**config}`',
        ('`get_config` does not bind `base_config = super().get_config()`',
         '`get_config` does not end in `return {**base_config, **config}`')),
    ('h15_initializer_storage', 'capsnet'): Open(
        'the CapsNet migration',
        '1 violation(s): `self.kernel_initializer` stores '
        '`keras.initializers.get(kernel_initializer)`, not an `initializer_spec(...)`',
        ('`self.kernel_initializer` stores '
         '`keras.initializers.get(kernel_initializer)`, not an `initializer_spec(...)`',)),
    ('h11_get_config_has_the_reference_form', 'capsnet_v2'): Open(
        'the CapsNet migration',
        '2 violation(s): `get_config` does not bind `base_config = '
        'super().get_config()`; `get_config` does not end in `return {**base_config, '
        '**config}`',
        ('`get_config` does not bind `base_config = super().get_config()`',
         '`get_config` does not end in `return {**base_config, **config}`')),
    ('h15_initializer_storage', 'capsnet_v2'): Open(
        'the CapsNet migration',
        '1 violation(s): `self.kernel_initializer` stores '
        '`keras.initializers.get(kernel_initializer)`, not an `initializer_spec(...)`',
        ('`self.kernel_initializer` stores '
         '`keras.initializers.get(kernel_initializer)`, not an `initializer_spec(...)`',)),
}


def _read(path: str) -> Optional[str]:
    full = SRC / path
    return full.read_text(encoding="utf-8") if full.is_file() else None


def module_violations(module: Module, rule: str, source: Optional[str] = None) -> Violations:
    text = source if source is not None else _read(module.path)
    assert text is not None, f"{module.path} does not exist yet"
    return MODULE_RULE_FUNCTIONS[rule](text, module)


def subject_members(source: str, subject: Subject, base_source: Optional[str] = None) -> List[Member]:
    """The members a reader of ``subject.cls`` meets in ``source`` (and ``base_source``)."""
    tree = ast.parse(source)
    base_tree = ast.parse(base_source) if base_source is not None else None
    assert _find_class(tree, subject.cls) is not None, f"{subject.cls} is not in the source"
    return _effective_members(tree, subject.cls, base_tree, subject.base[1] if subject.base else None)


def run_class_rule(source: str, subject: Subject, rule: str,
                   base_source: Optional[str] = None) -> Violations:
    """Run one class rule on ``source``; ``base_source`` holds the base class, if any."""
    return CLASS_RULE_FUNCTIONS[rule](subject_members(source, subject, base_source), subject)


def base_source(subject: Subject) -> Optional[str]:
    """The text of the file that holds ``subject``'s base class: the one accessor every row uses.

    Interface contract: returns ``None`` when the subject has no base or its file does
    not exist yet (the leaf is then read alone, and the rows of the base module itself
    stay ledgered). Once the file exists it is read, and the base class must be in it:
    raises ``AssertionError`` otherwise, so a renamed base cannot silently drop the
    inherited members out of every rule.
    """
    if subject.base is None:
        return None
    path, name = subject.base
    text = _read(path)
    if text is not None:
        assert _find_class(ast.parse(text), name) is not None, f"{name} is not in {path}"
    return text


def subject_source_members(subject: Subject) -> List[Member]:
    """:func:`subject_members` for the files the subject really lives in."""
    module = next(m for m in MODULES if m.key == subject.module)
    text = _read(module.path)
    assert text is not None, f"{module.path} does not exist yet"
    return subject_members(text, subject, base_source(subject))


def class_violations(subject: Subject, rule: str) -> Violations:
    """Run one class rule on the file the subject lives in (and its base's file)."""
    return CLASS_RULE_FUNCTIONS[rule](subject_source_members(subject), subject)


class LedgerSignatureMismatch(RuntimeError):
    """A ledgered row failed, but not the way its ledger line says it fails.

    Not an ``AssertionError``, so the strict ``xfail`` (``raises=AssertionError``)
    does not absorb it: the row is red.
    """


def failure_text(error: BaseException) -> str:
    """The text a ledger signature is matched against: ``"<Type>: <message>"``."""
    return f"{type(error).__name__}: {error}"


def _ledger_key(kwargs: Dict[str, Any]) -> Tuple[str, str]:
    """The ledger key of the row a test was called for (``rule`` and ``key``, or an anchor ``target``)."""
    return (kwargs["rule"], kwargs["key"]) if "rule" in kwargs else ("anchors", kwargs["target"])


def with_signature_check(test: Callable, ledger: Dict[Tuple[str, str], Open]) -> Callable:
    """``test`` wrapped so that a ledgered row's failure must carry every piece of its signature.

    Interface contract: the test is called by keyword with ``key`` and ``rule`` (a
    module or class row) or with ``target`` (an anchor row). A row with no ledger line
    is untouched. A ledgered row whose failure text lacks a piece of its ``signature``
    raises :class:`LedgerSignatureMismatch`, chained to the original error; any other
    failure propagates unchanged.
    """

    @functools.wraps(test)
    def run(*args: Any, **kwargs: Any) -> Any:
        try:
            return test(*args, **kwargs)
        except Exception as error:
            row = ledger.get(_ledger_key(kwargs))
            lacking = [] if row is None else [p for p in row.signature if p not in failure_text(error)]
            if lacking:
                raise LedgerSignatureMismatch(
                    f"{_ledger_key(kwargs)} failed without {lacking}: {failure_text(error)}") from error
            raise

    run.checks_ledger_signatures = True
    return run


def signature_checked(test: Callable) -> Callable:
    """Decorator form of :func:`with_signature_check` over the module's :data:`LEDGER`."""
    return with_signature_check(test, LEDGER)


def _marks(rule: str, key: str) -> list:
    row = LEDGER.get((rule, key))
    if row is None:
        return []
    return [pytest.mark.xfail(strict=True, raises=AssertionError,
                              reason=f"closed by {row.closed_by}; measured: {row.measured}")]


def _module_rows():
    for module in MODULES:
        for rule in module_rules(module):
            yield pytest.param(module.key, rule, id=f"{rule}[{module.key}]",
                               marks=_marks(rule, module.key))


def _class_rows():
    for subject in SUBJECTS:
        for rule in CLASS_RULES:
            if rule in subject.rules:
                yield pytest.param(subject.key, rule, id=f"{rule}[{subject.key}]",
                                   marks=_marks(rule, subject.key))


def _anchor_family(path: str) -> Optional[str]:
    return next((family for family, paths in ANCHOR_SETS.items() if path in paths), None)


def _anchor_rows():
    families = set()
    for path in sorted(ANCHOR_TOKENS):
        family = _anchor_family(path)
        if family is None:
            yield pytest.param(path, id=f"anchors[{path.removeprefix('src/dl_techniques/')}]",
                               marks=_marks("anchors", path))
        elif family not in families:
            families.add(family)
            yield pytest.param(family, id=f"anchors[{family} family]",
                               marks=_marks("anchors", family))


@pytest.mark.parametrize(("key", "rule"), list(_module_rows()))
@signature_checked
def test_a_module_follows_the_house_shape(key: str, rule: str) -> None:
    module = next(m for m in MODULES if m.key == key)
    violations = module_violations(module, rule)
    assert not violations, f"{module.path}: {violations}"


@pytest.mark.parametrize(("key", "rule"), list(_class_rows()))
@signature_checked
def test_a_class_follows_the_house_shape(key: str, rule: str) -> None:
    subject = next(s for s in SUBJECTS if s.key == key)
    violations = class_violations(subject, rule)
    assert not violations, f"{subject.cls}: {violations}"


@pytest.mark.parametrize("target", list(_anchor_rows()))
@signature_checked
def test_every_decision_anchor_is_still_in_its_file(target: str) -> None:
    root = SRC.parents[1]
    if target in ANCHOR_SETS:
        paths = ANCHOR_SETS[target]
        expected: Dict[str, int] = {}
        for path in paths:
            expected.update(ANCHOR_TOKENS.get(path, {}))
        sources = [(root / p).read_text(encoding="utf-8") for p in paths if (root / p).is_file()]
        lost = lost_anchor_tokens_over(sources, expected)
    else:
        lost = lost_anchor_tokens((root / target).read_text(encoding="utf-8"), ANCHOR_TOKENS[target])
    assert not lost, f"{target} lost {lost}"


# ---------------------------------------------------------------------------
# The rules can fail: a planted violation per rule, and the clean twin
# ---------------------------------------------------------------------------

#: A module that satisfies every rule. Each planted violation below is this text
#: with one edit, so the twin differs from the violation by that edit alone.
TOY = '''"""Toy model.

References:
    - Somebody, 2020.
"""

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from toy.utils import logger


class Toy(keras.Model):
    """Toy."""

    MODEL_VARIANTS = {"a": {}}
    OTHER = 1

    def __init__(self, num_classes=1000, kernel_initializer="he_normal",
                 kernel_regularizer=None, include_top=True, input_shape=(8, 8, 3),
                 head_config_overrides=None, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.kernel_initializer = initializer_spec(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.include_top = include_top
        self.input_shape_config = input_shape
        self._head_config_overrides = head_config_overrides
        self.classification_head = None
        self.blocks = []
        self._build_stem()
        logger.info("built")

    def _build_stem(self):
        self.stem = keras.layers.Dense(1)

    def build(self, input_shape):
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        return self.stem(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    @classmethod
    def from_variant(cls, variant, **kwargs):
        return cls(**kwargs)

    def get_config(self):
        config = {
            "num_classes": self.num_classes,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "include_top": self.include_top,
            "input_shape": self.input_shape_config,
            "head_config_overrides": self._head_config_overrides,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["kernel_regularizer"] = keras.regularizers.deserialize(config["kernel_regularizer"])
        return cls(**config)


def create_toy(variant="a", num_classes=1000, input_shape=None, pretrained=False, **kwargs):
    """Toy factory."""
    return Toy.from_variant(variant, num_classes=num_classes, input_shape=input_shape,
                            pretrained=pretrained, **kwargs)
'''

_TOY_MODULE = Module("toy", "toy.py", ("Toy",), factory_signature=CANONICAL_FACTORY)
_TOY_SUBJECT = Subject(
    "toy", "toy", "Toy", _ALL, head_attribute="classification_head", classifier=True,
    required_params=("kernel_initializer", "kernel_regularizer"))
_CALL = "    def call(self, inputs, training=None):\n        return self.stem(inputs)\n\n"
_TOY_FACTORY = TOY[TOY.index("def create_toy"):]


def _plant(old: str, new: str, source: str = TOY) -> str:
    """``source`` with ``old`` replaced by ``new``; the edit must change it."""
    assert old in source, f"the plant's anchor text is not in the source: {old!r}"
    return source.replace(old, new, 1)


#: ``rule -> {what the plant does: violating source}``. Every rule has at least one.
PLANTED: Dict[str, Dict[str, str]] = {
    "h1_module_header": {
        "no References section": _plant("References:\n    - Somebody, 2020.\n", ""),
        "no local imports banner": _plant("# local imports\n", ""),
    },
    "h1_factories_follow_the_classes": {
        "a factory between two classes": TOY + "\n\nclass Other(keras.Model):\n    pass\n",
        "a factory before the class": _plant(
            "class Toy(keras.Model):", _TOY_FACTORY + "\n\nclass Toy(keras.Model):",
            TOY.replace(_TOY_FACTORY, "")),
    },
    "no_from_keras_import": {
        "from keras import ops": _plant("import keras\n", "from keras import ops\nimport keras\n"),
    },
    "h9_factory_is_one_delegating_return": {
        "a vestigial factory parameter": _plant(
            "pretrained=False, **kwargs):", "pretrained=False, cache_dir=None, **kwargs):"),
        "a factory that does its own work": _plant(
            "    return Toy.from_variant(variant,", "    model = Toy.from_variant(variant,").replace(
            "pretrained=pretrained, **kwargs)\n", "pretrained=pretrained, **kwargs)\n    return model\n"),
        "an input_shape default": _plant("input_shape=None, pretrained", "input_shape=(8, 8, 3), pretrained"),
        "a factory that swaps num_classes and input_shape": _plant(
            "num_classes=num_classes, input_shape=input_shape,",
            "num_classes=input_shape, input_shape=num_classes,"),
        "a factory that swaps two positional slots": _plant(
            "Toy.from_variant(variant, num_classes=num_classes, input_shape=input_shape,\n"
            "                            pretrained=pretrained, **kwargs)",
            "Toy.from_variant(variant, num_classes, input_shape=input_shape,\n"
            "                            pretrained=pretrained, **kwargs)"),
        "a factory that ignores its arguments": _plant(
            "    return Toy.from_variant(variant, num_classes=num_classes, input_shape=input_shape,\n"
            "                            pretrained=pretrained, **kwargs)\n",
            '    return Toy.from_variant("a", **kwargs)\n'),
    },
    "h2_member_order": {
        "MODEL_VARIANTS after another constant": _plant(
            '    MODEL_VARIANTS = {"a": {}}\n    OTHER = 1\n',
            '    OTHER = 1\n    MODEL_VARIANTS = {"a": {}}\n'),
        "call after from_config": _plant("    @classmethod\n    def from_config",
                                         _CALL + "    @classmethod\n    def from_config",
                                         _plant(_CALL, "")),
    },
    "h3_one_init_log": {
        "no logger.info": _plant('        logger.info("built")\n', ""),
        "two logger.info": _plant('        logger.info("built")\n',
                                  '        logger.info("built")\n        logger.info("again")\n'),
    },
    "h3_headless_attribute": {
        "no headless attribute": _plant("        self.classification_head = None\n", ""),
    },
    "h3_builders_assign": {
        "a builder that returns": _plant("        self.stem = keras.layers.Dense(1)\n",
                                         "        self.stem = keras.layers.Dense(1)\n        return self.stem\n"),
    },
    "h4_no_logging_outside_init": {
        "a logger call in call": _plant("        return self.stem(inputs)\n",
                                        '        logger.info("x")\n        return self.stem(inputs)\n'),
    },
    "h5_build_materializes": {
        "no built guard": _plant("        if self.built:\n            return\n", ""),
        "no materialize_sublayers": _plant("        materialize_sublayers(self, input_shape)\n", ""),
        "no super().build": _plant("        super().build(input_shape)\n", ""),
    },
    "h10_num_classes_first_kwargs_last": {
        "num_classes second": _plant(
            '(self, num_classes=1000, kernel_initializer="he_normal",',
            '(self, kernel_initializer="he_normal", num_classes=1000,'),
        "no **kwargs": _plant("head_config_overrides=None, **kwargs):", "head_config_overrides=None):")
        .replace("super().__init__(**kwargs)", "super().__init__()"),
    },
    "h10_trailing_parameter_order": {
        "input_shape before include_top": _plant(
            "include_top=True, input_shape=(8, 8, 3),", "input_shape=(8, 8, 3), include_top=True,"),
    },
    "h10_stored_attribute_names": {
        "input_shape kept as _input_shape": _plant("self.input_shape_config = input_shape",
                                                   "self._input_shape = input_shape"),
        "an argument stored under another name": _plant(
            "self.include_top = include_top", "self.with_head = include_top"),
    },
    "h10_one_name_per_concept": {
        "weight_decay for kernel_regularizer": _plant(
            "kernel_regularizer=None, include_top", "weight_decay=None, include_top").replace(
            "keras.regularizers.get(kernel_regularizer)", "keras.regularizers.get(weight_decay)"),
    },
    "h11_get_config_covers_the_constructor": {
        "an argument with no config key": _plant(
            '            "head_config_overrides": self._head_config_overrides,\n', ""),
        "no super().get_config()": _plant("base_config = super().get_config()", "base_config = {}"),
    },
    "h11_from_config_does_real_work": {
        "a from_config that only calls cls(**config)": _plant(
            '        config["kernel_regularizer"] = keras.regularizers.deserialize(config["kernel_regularizer"])\n',
            ""),
    },
    "h11_get_config_has_the_reference_form": {
        "a config built with update()": _plant(
            "        base_config = super().get_config()\n        return {**base_config, **config}\n",
            "        config.update(super().get_config())\n        return config\n"),
        "base_config merged the other way round": _plant(
            "{**base_config, **config}", "{**config, **base_config}"),
    },
    "h2_call_takes_inputs": {
        "call names its input x": _plant(
            "    def call(self, inputs, training=None):\n        return self.stem(inputs)\n",
            "    def call(self, x, training=None):\n        return self.stem(x)\n"),
    },
    "h10_block_container_names": {
        "blocks_list for blocks": _plant("self.blocks = []", "self.blocks_list = []"),
        "stage_blocks for stages": _plant("self.blocks = []", "self.stage_blocks = []"),
    },
    "h15_initializer_storage": {
        "keras.initializers.get stored": _plant(
            "initializer_spec(kernel_initializer)", "keras.initializers.get(kernel_initializer)"),
        "the raw argument stored": _plant(
            "initializer_spec(kernel_initializer)", "kernel_initializer"),
    },
}


def _rule_cases():
    for rule, plants in PLANTED.items():
        for label in plants:
            yield pytest.param(rule, label, id=f"{rule}: {label}")


def _run_planted(rule: str, source: str) -> Violations:
    if rule in MODULE_RULE_FUNCTIONS:
        return MODULE_RULE_FUNCTIONS[rule](source, _TOY_MODULE)
    return run_class_rule(source, _TOY_SUBJECT, rule)


def test_the_planted_twin_breaks_no_rule() -> None:
    for rule in list(MODULE_RULE_FUNCTIONS) + list(CLASS_RULE_FUNCTIONS):
        assert _run_planted(rule, TOY) == [], rule


@pytest.mark.parametrize(("rule", "label"), list(_rule_cases()))
def test_a_rule_reports_the_violation_planted_in_the_twin(rule: str, label: str) -> None:
    assert _run_planted(rule, PLANTED[rule][label]), f"{rule} saw nothing wrong with: {label}"


def test_every_rule_has_a_planted_violation() -> None:
    rules = set(MODULE_RULE_FUNCTIONS) | set(CLASS_RULE_FUNCTIONS)
    assert rules == set(PLANTED)


_LEAF = 'class Toy(_Base):\n    """Leaf."""\n'
_LEAF_SUBJECT = Subject("toy", "toy", "Toy", _ALL, head_attribute="classification_head",
                        classifier=True, base=("toy_base.py", "_Base"))


def _as_base(source: str) -> str:
    return _plant("class Toy(keras.Model):", "class _Base(keras.Model):", source)


def test_a_leaf_class_is_read_together_with_its_base() -> None:
    """After the ConvNeXt base lands, the leaves hold almost nothing; the rules must still bite."""
    base = _as_base(TOY)
    for rule in CLASS_RULE_FUNCTIONS:
        assert run_class_rule(_LEAF, _LEAF_SUBJECT, rule, base) == [], rule


def _class_plants():
    for rule, plants in PLANTED.items():
        if rule in CLASS_RULE_FUNCTIONS:
            for label in plants:
                yield pytest.param(rule, label, id=f"{rule}: {label}")


@pytest.mark.parametrize(("rule", "label"), list(_class_plants()))
def test_a_violation_placed_in_the_base_class_is_caught_through_the_leaf(rule: str, label: str) -> None:
    """Every planted class violation, moved into the base, is reported for the leaf."""
    found = run_class_rule(_LEAF, _LEAF_SUBJECT, rule, _as_base(PLANTED[rule][label]))
    assert found, f"{rule} saw nothing wrong with `{label}` once it sat in the base class"


def test_the_rows_read_the_base_file_through_one_accessor(tmp_path: Path, monkeypatch) -> None:
    """The base file is read when it exists, ignored while it does not, and must hold its class."""
    this = sys.modules[__name__]
    module = Module("toy_leaf", "toy_leaf.py", ("Toy",))
    subject = Subject("toy", "toy_leaf", "Toy", frozenset({"h5_build_materializes"}),
                      base=("toy_base.py", "_Base"))
    monkeypatch.setattr(this, "SRC", tmp_path)
    monkeypatch.setattr(this, "MODULES", (*MODULES, module))
    (tmp_path / "toy_leaf.py").write_text(_LEAF + "\n    def build(self, input_shape):\n        pass\n")

    assert base_source(subject) is None
    # No base file yet: the leaf is read alone, and its own `build` is what is judged.
    assert class_violations(subject, "h5_build_materializes")

    (tmp_path / "toy_base.py").write_text(_as_base(TOY))
    assert base_source(subject) == _as_base(TOY)
    leaf_without_build = _LEAF + "    pass\n"
    (tmp_path / "toy_leaf.py").write_text(leaf_without_build)
    assert class_violations(subject, "h5_build_materializes") == []

    (tmp_path / "toy_base.py").write_text(_as_base(
        _plant("        materialize_sublayers(self, input_shape)\n", "")))
    assert class_violations(subject, "h5_build_materializes"), "a violation in the base file was not read"

    (tmp_path / "toy_base.py").write_text(_plant("class Toy(keras.Model):", "class Renamed(keras.Model):"))
    with pytest.raises(AssertionError, match="_Base is not in toy_base.py"):
        base_source(subject)


def test_the_factory_rule_reads_from_variant_in_the_base_class(tmp_path: Path, monkeypatch) -> None:
    """A leaf module whose ``from_variant`` lives in its base is checked against the base's slots."""
    this = sys.modules[__name__]
    monkeypatch.setattr(this, "SRC", tmp_path)
    (tmp_path / "toy_base.py").write_text(
        "class _Base:\n"
        "    @classmethod\n"
        "    def from_variant(cls, variant, num_classes=1000, input_shape=None, pretrained=False, **kwargs):\n"
        "        pass\n")
    module = Module("toy_leaf", "toy_leaf.py", ("Leaf",), factory_signature=CANONICAL_FACTORY,
                    base=("toy_base.py", "_Base"))
    clean = (
        "class Leaf(_Base):\n    pass\n\n\n"
        "def create_leaf(variant='a', num_classes=1000, input_shape=None, pretrained=False, **kwargs):\n"
        "    return Leaf.from_variant(variant, num_classes, input_shape, pretrained, **kwargs)\n")
    assert h9_factory_is_one_delegating_return(clean, module) == []
    swapped = clean.replace("variant, num_classes, input_shape,", "variant, input_shape, num_classes,")
    assert swapped != clean
    found = h9_factory_is_one_delegating_return(swapped, module)
    assert any("another name" in v for v in found), found
    # Without a base the same positional call cannot be checked, and says so.
    alone = Module("toy_leaf", "toy_leaf.py", ("Leaf",), factory_signature=CANONICAL_FACTORY)
    assert h9_factory_is_one_delegating_return(clean, alone), "a positional call needs the method's own signature"


def test_a_subject_without_a_base_has_no_base_source() -> None:
    assert base_source(next(s for s in SUBJECTS if s.key == "resnet")) is None


def test_the_anchor_rule_sees_a_dropped_and_a_moved_token() -> None:
    two = "# DECISION p/D-001: a\n# continued\nx = 1\n# DECISION p/D-002: b\n"
    one = "# DECISION p/D-001: a\n# continued\nx = 1\n"
    expected = dict(anchor_token_counts(two))
    assert expected == {"p/D-001": 1, "p/D-002": 1}
    assert lost_anchor_tokens(two, expected) == []
    assert lost_anchor_tokens(one, expected) == ["p/D-002 x1 -> x0"]
    assert lost_anchor_tokens_over([one, "# DECISION p/D-002: moved\n"], expected) == []
    assert lost_anchor_tokens_over([one, "x = 2\n"], expected) == ["p/D-002"]
    # An anchor naming a second decision on its continuation line counts both.
    assert set(anchor_token_counts("# DECISION p/D-001 + p/D-009: a\n# also p/D-003\n")) == {
        "p/D-001", "p/D-009", "p/D-003"}


def test_a_ledgered_failure_without_its_signature_is_a_plain_failure() -> None:
    ledger = {("toy_rule", "toy"): Open("x", "m", ("kernel_initializer", "weight_decay")),
              ("anchors", "toy.py"): Open("x", "m", ("lost",))}

    def failing(message: str) -> Callable:
        def test(key: str, rule: str) -> None:
            raise AssertionError(message)
        return with_signature_check(test, ledger)

    def lost(target: str) -> None:
        raise AssertionError("toy.py lost ['p/D-001']")

    with pytest.raises(AssertionError) as expected:
        failing("Toy: ['has no kernel_initializer', 'weight_decay is used']")(key="toy", rule="toy_rule")
    assert not isinstance(expected.value, LedgerSignatureMismatch)
    # One violation of the same rule is gone: the row no longer fails the way it says.
    with pytest.raises(LedgerSignatureMismatch, match="weight_decay") as changed:
        failing("Toy: ['has no kernel_initializer']")(key="toy", rule="toy_rule")
    assert isinstance(changed.value.__cause__, AssertionError)
    # A different violation of the same rule is not the ledgered one either.
    with pytest.raises(LedgerSignatureMismatch):
        failing("Toy: ['num_classes comes second']")(key="toy", rule="toy_rule")
    with pytest.raises(AssertionError) as unledgered:
        failing("Toy: ['num_classes comes second']")(key="other", rule="toy_rule")
    assert not isinstance(unledgered.value, LedgerSignatureMismatch)
    with pytest.raises(AssertionError) as anchored:
        with_signature_check(lost, ledger)(target="toy.py")
    assert not isinstance(anchored.value, LedgerSignatureMismatch)
    assert with_signature_check(lambda key, rule: 7, ledger)(key="toy", rule="toy_rule") == 7


def test_a_signature_mismatch_is_not_absorbed_by_the_strict_xfail() -> None:
    assert not issubclass(LedgerSignatureMismatch, AssertionError)


def test_every_row_test_checks_its_ledger_signatures() -> None:
    checked = [fn for fn in globals().values() if getattr(fn, "checks_ledger_signatures", False)]
    assert {fn.__name__ for fn in checked} == {
        "test_a_module_follows_the_house_shape", "test_a_class_follows_the_house_shape",
        "test_every_decision_anchor_is_still_in_its_file"}


# ---------------------------------------------------------------------------
# The roster, the exemptions and the ledger keep each other honest
# ---------------------------------------------------------------------------


def test_the_roster_names_the_nineteen_subject_classes() -> None:
    classes = [c for m in MODULES if m.key != "stem" for c in m.classes]
    assert len(classes) == len(set(classes)) == 19
    assert {s.cls for s in SUBJECTS if s.key != "stem"} == set(classes)


def test_every_subject_class_is_in_its_module() -> None:
    for subject in SUBJECTS:
        module = next(m for m in MODULES if m.key == subject.module)
        text = _read(module.path)
        assert text is not None and _find_class(ast.parse(text), subject.cls), (
            f"{subject.cls} is not in {module.path}")


def test_every_waiver_names_a_written_exemption() -> None:
    for owner in (*MODULES, *SUBJECTS):
        for rule, label in owner.waived.items():
            assert label in EXEMPTIONS, f"{owner.key}: {rule} waived by unknown {label!r}"
    for owner in MODULES:
        for rule in owner.waived:
            assert rule not in module_rules(owner), f"{owner.key}: {rule} is waived and still has a row"
    for owner in SUBJECTS:
        for rule in owner.waived:
            assert rule in CLASS_RULES and rule not in owner.rules, (
                f"{owner.key}: {rule} is waived and still has a row")


def _init_facts(source: str, cls: str, method: str = "__init__") -> Tuple[List[str], Dict[str, Any]]:
    """``(parameter names, {parameter: constant default})`` of ``cls.method`` in ``source``."""
    fn = _def(_members(_find_class(ast.parse(source), cls)), method)
    assert fn is not None, f"{cls} has no {method}"
    names = [a.arg for a in fn.args.args]
    defaults = {n: d.value for n, d in zip(names[len(names) - len(fn.args.defaults):], fn.args.defaults)
                if isinstance(d, ast.Constant)}
    return _params(fn), defaults


def e1_problems(source: str, cls: str) -> Violations:
    """E-1 holds of ``cls``: ``include_top`` last, no ``input_shape``, ``num_classes`` 1000."""
    names, defaults = _init_facts(source, cls)
    out = [] if names[-1] == "include_top" else [f"{cls}: the last parameter is `{names[-1]}`"]
    out += [f"{cls} takes an `input_shape`"] if "input_shape" in names else []
    out += [] if defaults.get("num_classes") == 1000 else [f"{cls}: num_classes does not default to 1000"]
    return out


def e5_problems(source: str, cls: str) -> Violations:
    """E-5 holds of ``cls``: ``from_variant`` takes variant, input_shape, latent_dim, no num_classes."""
    names, _ = _init_facts(source, cls, "from_variant")
    out = [f"{cls}.from_variant has no `{n}`" for n in ("variant", "input_shape", "latent_dim")
           if n not in names]
    return out + ([f"{cls}.from_variant takes `num_classes`"] if "num_classes" in names else [])


def e6_problems(source: str, cls: str) -> Violations:
    """E-6 holds of ``cls``: ``num_classes`` defaults to 100, a prototype count."""
    _, defaults = _init_facts(source, cls)
    return [] if defaults.get("num_classes") == 100 else [f"{cls}: num_classes does not default to 100"]


def _facts_of_the_waived_situation_e1() -> Violations:
    flagged = {s.key for s in SUBJECTS if s.include_top_last}
    out = [] if flagged == {"cliffordnet", "hierarchical"} else [f"include_top_last is set on {sorted(flagged)}"]
    for module, cls in (("cliffordnet", "CliffordNet"), ("hierarchical", "HierarchicalCliffordNet")):
        out += e1_problems(_read(next(m.path for m in MODULES if m.key == module)), cls)
    return out


def _facts_of_the_waived_situation_e5() -> Violations:
    inexact = {m.key for m in MODULES if m.model_module and m.factory_signature and not m.factory_exact}
    out = [] if inexact == {"vae"} else [f"factory_exact is off for {sorted(inexact)}"]
    return out + e5_problems(_read(next(m.path for m in MODULES if m.key == "vae")), "VAE")


def _facts_of_the_waived_situation_e6() -> Violations:
    path = next(m.path for m in MODULES if m.key == "rbf_protonet")
    return e6_problems(_read(path), "RBFProtoNet") + e6_problems(_read(path), "CliffordRBFProtoNet")


#: The exemptions no ``waived`` entry names. Their effect is on a rule's data (a flag,
#: a signature, a default) rather than on a missing row, so each is tied to a check
#: that the situation it describes exists in the source today.
EXEMPTION_WITNESSES: Dict[str, Callable[[], Violations]] = {
    "E-1": _facts_of_the_waived_situation_e1,
    "E-5": _facts_of_the_waived_situation_e5,
    "E-6": _facts_of_the_waived_situation_e6,
}


def test_every_exemption_is_used() -> None:
    used = {label for owner in (*MODULES, *SUBJECTS) for label in owner.waived.values()}
    assert used | set(EXEMPTION_WITNESSES) == set(EXEMPTIONS)
    assert not used & set(EXEMPTION_WITNESSES), "a waived exemption needs no witness"


@pytest.mark.parametrize("label", sorted(EXEMPTION_WITNESSES))
def test_an_unwaived_exemption_describes_a_situation_that_exists(label: str) -> None:
    assert EXEMPTION_WITNESSES[label]() == []


def test_the_exemption_witnesses_can_fail() -> None:
    """The twin has none of the three situations; a source that has each one passes."""
    assert e1_problems(TOY, "Toy"), "E-1 saw nothing wrong with a constructor that takes input_shape"
    assert e6_problems(TOY, "Toy"), "E-6 saw nothing wrong with num_classes=1000"
    assert e5_problems(TOY, "Toy"), "E-5 saw nothing wrong with a from_variant that has no latent_dim"
    assert e1_problems(
        "class A:\n    def __init__(self, num_classes=1000, include_top=True, **kwargs):\n        pass\n",
        "A") == []
    assert e6_problems("class A:\n    def __init__(self, num_classes=100):\n        pass\n", "A") == []
    assert e5_problems(
        "class A:\n    @classmethod\n    def from_variant(cls, variant, input_shape, latent_dim=None):\n"
        "        pass\n", "A") == []


def test_the_ledger_names_only_rows_that_exist() -> None:
    rows = {(rule, m.key) for m in MODULES for rule in module_rules(m)}
    rows |= {(rule, s.key) for s in SUBJECTS for rule in CLASS_RULES if rule in s.rules}
    rows |= {("anchors", p) for p in ANCHOR_TOKENS} | {("anchors", f) for f in ANCHOR_SETS}
    assert set(LEDGER) <= rows, sorted(set(LEDGER) - rows)
    for row in LEDGER.values():
        assert row.closed_by and row.measured
        assert row.signature and all(row.signature), "a ledger row without a signature pins nothing"


# ---------------------------------------------------------------------------
# A rule that passes because its precondition is absent is pinned, not silent
# ---------------------------------------------------------------------------

_NOT_A_CLASSIFIER = "not a classifier: rule returns [] by construction"
_NO_FROM_CONFIG = "no from_config defined (rule only bites a defined one)"
_NO_BUILDERS = "no _build_* method to inspect"

#: The ``(rule, subject)`` rows that pass because there is nothing for the rule to
#: read. A subject that gains a ``from_config`` or a ``_build_*`` method (or a
#: classifier flag) stops being vacuous for that rule and
#: ``test_the_vacuous_rows_are_exactly_the_pinned_ones`` fails: the author decides
#: whether the new method obeys the rule (delete the line) or why it does not.
VACUOUS: Dict[Tuple[str, str], str] = {
    **{("h10_trailing_parameter_order", key): _NOT_A_CLASSIFIER for key in (
        "convunext_stem", "vae", "rbf_protonet", "clifford_rbf_protonet", "uca_encoder",
        "capsnet", "capsnet_v2")},
    **{("h11_from_config_does_real_work", key): _NO_FROM_CONFIG for key in (
        "resnet", "cliffordnet", "hierarchical", "mobilenet_v1", "mobilenet_v2", "mobilenet_v3",
        "mobilenet_v4", "convnext_v1", "convnext_v2", "rad_convnet", "uca_encoder")},
    **{("h3_builders_assign", key): _NO_BUILDERS for key in ("convunext_stem", "uca_encoder")},
    **{("h2_call_takes_inputs", key): _NO_CALL for key in ("squeezenet_v1", "squeezenet_v2", "vae")},
    **{("h10_block_container_names", key): _NO_CONTAINER for key in ("convunext_stem", "vae")},
    **{("h15_initializer_storage", key): _NO_INITIALIZER for key in (
        "convnext_v1", "convnext_v2", "rad_convnet", "rbf_protonet", "clifford_rbf_protonet",
        "uca_encoder")},
}


def test_the_vacuous_rows_are_exactly_the_pinned_ones() -> None:
    found = {}
    for subject in SUBJECTS:
        members = subject_source_members(subject)
        for rule in CLASS_RULES:
            if rule in subject.rules:
                reason = precondition_absent(members, subject, rule)
                if reason:
                    found[(rule, subject.key)] = reason
    assert found == VACUOUS, (
        f"became readable: {sorted(set(VACUOUS) - set(found))}; "
        f"became vacuous: {sorted(set(found) - set(VACUOUS))}")


def test_the_precondition_detector_sees_each_absent_thing() -> None:
    def reasons(source: str, subject: Subject) -> Dict[str, str]:
        members = subject_members(source, subject)
        return {rule: reason for rule in CLASS_RULES
                if (reason := precondition_absent(members, subject, rule))}

    assert reasons(TOY, _TOY_SUBJECT) == {}
    no_from_config = _plant(TOY[TOY.index("    @classmethod\n    def from_config"):TOY.index("def create_toy")], "", TOY)
    assert reasons(no_from_config, _TOY_SUBJECT) == {"h11_from_config_does_real_work": _NO_FROM_CONFIG}
    no_builder = _plant("        self._build_stem()\n", "", _plant(
        "    def _build_stem(self):\n        self.stem = keras.layers.Dense(1)\n\n", "", TOY))
    assert reasons(no_builder, _TOY_SUBJECT) == {"h3_builders_assign": _NO_BUILDERS}
    assert reasons(_plant(_CALL, ""), _TOY_SUBJECT) == {"h2_call_takes_inputs": _NO_CALL}
    assert reasons(_plant("        self.blocks = []\n", ""), _TOY_SUBJECT) == {
        "h10_block_container_names": _NO_CONTAINER}
    no_initializer = _plant(
        "        self.kernel_initializer = initializer_spec(kernel_initializer)\n", "")
    assert reasons(no_initializer, _TOY_SUBJECT) == {"h15_initializer_storage": _NO_INITIALIZER}
    not_a_classifier = Subject("toy", "toy", "Toy", _ALL, head_attribute="classification_head",
                               classifier=False)
    assert reasons(TOY, not_a_classifier) == {"h10_trailing_parameter_order": _NOT_A_CLASSIFIER}
    no_init = "class Toy:\n    def call(self, x):\n        return x\n"
    assert set(reasons(no_init, _TOY_SUBJECT)) >= {"h3_one_init_log", "h10_one_name_per_concept"}


def test_every_pinned_vacuous_row_is_a_row_that_exists() -> None:
    rows = {(rule, s.key) for s in SUBJECTS for rule in CLASS_RULES if rule in s.rules}
    assert set(VACUOUS) <= rows, sorted(set(VACUOUS) - rows)



def test_a_factory_argument_must_reach_the_slot_of_its_own_name() -> None:
    def route(call: str, slots: Optional[List[str]]) -> List[str]:
        node = ast.parse(call).body[0].value
        return [p for p in ("variant", "num_classes", "input_shape")
                if not _routes_to_its_own_slot(node, p, slots)]

    slots = ["variant", "num_classes", "input_shape", "pretrained"]
    assert route("C.from_variant(variant, num_classes=num_classes, input_shape=input_shape)", slots) == []
    assert route("C.from_variant(variant, num_classes, input_shape)", slots) == []
    assert route("C.from_variant(variant, num_classes, input_shape)", None) == [
        "variant", "num_classes", "input_shape"], "a positional slot needs the method's own signature"
    assert route("C.from_variant(variant, num_classes=num_classes, input_shape=num_classes)", slots) == [
        "input_shape"]
    assert route("C.from_variant(input_shape, num_classes, variant)", slots) == [
        "variant", "input_shape"]
