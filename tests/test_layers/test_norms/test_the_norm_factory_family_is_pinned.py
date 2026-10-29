"""Exact census + raise contract for ``create_normalization_layer``'s call-site family.

Why this file exists
====================

Every call site of ``create_normalization_layer`` under ``src/`` is pinned, per
file, as ``(dynamic, literal)``: a site whose ``normalization_type`` is a
variable (an ``Attribute`` such as ``self.normalization_type`` or a bare
``Name``) cannot be checked by any static instrument, because no sweep can know
which registry key it will ask for. Measured at Step 15 of
``plan-2026-10-14T062616-81938e81`` (D-253): **200 sites in 100 files, 153
dynamic, 47 literal.** Plan-2026-10-23T190557-972a9ac5 step 10 (D-030) moved the
TopoLM block's two norms (now one ``_make_norm`` call site) and the model's final
norm from literal ``'layer_norm'`` to the ``normalization_type`` knob: **200 sites
in 101 files, 155 dynamic, 45 literal.** Plan-2026-10-26T125959-7af851e5 step 1
re-derived it after two moves that had left the pins stale: the vision-head
refactor moved ``heads/vision/factory.py``'s site into ``heads/vision/classification.py``,
and the stems of ConvUNext, ResNet and UcaEncoder are now built by ``ConvStem``
(``conv_blocks/stem.py``, one dynamic site in ``_make_norm``, shared by both of the
stem's norms), so ``convunext`` and ``resnet`` lose their site and ``uca_encoder``
loses ``stem_norm``: **198 sites in 100 files, 153 dynamic, 45 literal.** The split of
``heads/nlp/factory.py`` (2026-10-27) then moved its one dynamic site into ``heads/nlp/base.py`` (same
count), and the split of ``heads/vlm/factory.py`` turned that file's ``(1, 3)`` entry into two files,
``image_text_matching.py`` ``(1, 0)`` and ``image_captioning.py`` ``(0, 3)``:
**198 sites in 101 files, 153 dynamic, 45 literal.** The auto-built
encoder of ``vq_vae_rotation`` builds its down-steps with ``ConvStem`` (plan-2026-10-26T125959-7af851e5
step 19), so its ``(4, 0)`` became ``(3, 0)``: **197 sites in 101 files, 152 dynamic, 45 literal.**
The mask stack of SAM1's ``PromptEncoder`` is built by ``ConvStem`` as well, which removes that
file's ``(2, 0)`` entry: **195 sites in 100 files, 150 dynamic, 45 literal.** (Each step is read from
the table of the commit that made it: files 100, 101, 100, 101, 101, 100 in this order; sites only fall.)

The population was first pinned by two ``<=`` ceilings (D-201 of
``plan-2026-08-22T035419-a11304c8``: 151 dynamic, 180 total, 2026-09-04). A
ceiling leaves slack after a removal, and it went red at 157 / 200 without
saying which site was new (D-120). The exact census names the file. A new or
removed site fails here until :data:`NORM_CALL_SITES` is edited in the same
commit; a new DYNAMIC site also needs a reason: a literal ``normalization_type=``
or a ruling in ``decisions.md``. D-253 of ``plan-2026-10-14T062616-81938e81``
re-opened D-201 for six dynamic sites, each a documented, stored and serialized
constructor knob (:data:`RULED_DYNAMIC_KNOBS`).

The residual risk of a dynamic site is a key nobody exercised, not a silent
fallback: the factory *raises* on an unknown key, so each dynamic site is
self-guarding the moment its branch runs.
``test_an_unknown_normalization_type_raises_and_names_the_valid_ones`` pins
that raise.
"""

import ast
import os
from typing import Dict, List, Tuple

import keras
import pytest

from tests.norm_epsilon_oracle import _epsilon_of

from dl_techniques.layers.norms.factory import (
    _TYPE_TO_CLASS,
    create_normalization_layer,
)

_TESTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_ROOT = os.path.join(os.path.dirname(_TESTS_DIR), "src")

#: Exact census of ``create_normalization_layer`` call sites under ``src/``
#: (``dl_techniques`` + ``applications`` + ``train``), by an AST walk (a grep is
#: wrong in both directions here, D-140 of plan a11304c8):
#: ``{path relative to src/: (dynamic sites, literal sites)}``. Re-derived at
#: Step 15 of plan-2026-10-14T062616-81938e81 (D-253): 200 sites, 100 files,
#: 153 dynamic, 47 literal; then D-030 of plan-2026-10-23T190557-972a9ac5 (TopoLM's
#: ``normalization_type`` knob): 200 sites, 101 files, 155 dynamic, 45 literal;
#: then step 1 of plan-2026-10-26T125959-7af851e5 (``ConvStem``, the vision-head
#: move): 198 sites, 100 files, 153 dynamic, 45 literal; then the heads nlp/vlm split
#: (2026-10-27): 198 sites, 101 files, 153 dynamic, 45 literal; then step 19 of the same plan
#: (``vq_vae_rotation``'s down-steps are ``ConvStem``): 197 sites, 101 files, 152 dynamic, 45 literal;
#: then SAM1's mask stack (``sam1/prompt_encoder.py`` leaves the table): 195 sites, 100 files, 150
#: dynamic, 45 literal; then ``layers/fastvit/`` dissolved (its three files' sites move unchanged to
#: ``conv_blocks/reparam_large_kernel_conv.py`` and ``models/vision/fastvit/components.py``): 195 sites,
#: 99 files, 150 dynamic, 45 literal; then the ``layers/time_series`` de-dup (78bba4525) left
#: ``mixed_sequential_block.py`` with one site instead of three: 193 sites, 99 files, 148 dynamic, 45 literal.
#: Edit it in the commit that adds or removes a site.
NORM_CALL_SITES: Dict[str, Tuple[int, int]] = {
    "dl_techniques/layers/attention/area_attention.py": (2, 0),
    "dl_techniques/layers/attention/capsule_routing_attention.py": (2, 0),
    "dl_techniques/layers/attention/differential_attention.py": (4, 0),
    "dl_techniques/layers/attention/gated_attention.py": (2, 1),
    "dl_techniques/layers/attention/group_query_attention.py": (2, 0),
    "dl_techniques/layers/attention/hopfield_attention.py": (2, 0),
    "dl_techniques/layers/attention/lighthouse_attention.py": (2, 0),
    "dl_techniques/layers/attention/multi_head_cross_attention.py": (2, 0),
    "dl_techniques/layers/attention/multi_head_latent_attention.py": (2, 0),
    "dl_techniques/layers/attention/non_local_attention.py": (3, 0),
    "dl_techniques/layers/attention/ring_attention.py": (2, 0),
    "dl_techniques/layers/attention/rpc_attention.py": (2, 0),
    "dl_techniques/layers/attention/shared_weights_cross_attention.py": (2, 0),
    "dl_techniques/layers/attention/single_window_attention.py": (2, 0),
    "dl_techniques/layers/conv_blocks/basic_block.py": (3, 0),
    "dl_techniques/layers/conv_blocks/bottleneck_block.py": (4, 0),
    "dl_techniques/layers/conv_blocks/conv_block.py": (1, 0),
    "dl_techniques/layers/conv_blocks/depthwise_separable_block.py": (2, 0),
    "dl_techniques/layers/conv_blocks/gabor_depthwise_separable_block.py": (1, 0),
    "dl_techniques/layers/conv_blocks/reparam_large_kernel_conv.py": (0, 1),
    "dl_techniques/layers/conv_blocks/stem.py": (1, 0),
    "dl_techniques/layers/conv_blocks/universal_inverted_bottleneck.py": (1, 0),
    "dl_techniques/layers/dense_block.py": (1, 0),
    "dl_techniques/layers/eomt_mask.py": (2, 0),
    "dl_techniques/layers/fnet_encoder_block.py": (2, 0),
    "dl_techniques/layers/fusion/multimodal_fusion.py": (5, 0),
    "dl_techniques/layers/geometric/clifford_block.py": (2, 0),
    "dl_techniques/layers/geometric/clifford_rnn.py": (2, 0),
    "dl_techniques/layers/graphs/graph_neural_network.py": (0, 2),
    "dl_techniques/layers/graphs/relational_graph_transformer_blocks.py": (2, 0),
    "dl_techniques/layers/heads/nlp/base.py": (1, 0),
    "dl_techniques/layers/heads/vision/classification.py": (1, 0),
    "dl_techniques/layers/heads/vlm/image_captioning.py": (0, 3),
    "dl_techniques/layers/heads/vlm/image_text_matching.py": (1, 0),
    "dl_techniques/layers/moe/experts.py": (2, 0),
    "dl_techniques/layers/moe/gating.py": (1, 0),
    "dl_techniques/layers/norms/factory.py": (1, 0),
    "dl_techniques/layers/residual_dense_block.py": (1, 0),
    "dl_techniques/layers/ssm/context_mamba.py": (1, 0),
    "dl_techniques/layers/ssm/selective_ssm.py": (0, 1),
    "dl_techniques/layers/time_series/mixed_sequential_block.py": (1, 0),
    "dl_techniques/layers/time_series/xlstm_blocks.py": (2, 0),
    "dl_techniques/layers/transformers/adaln_zero.py": (2, 0),
    "dl_techniques/layers/transformers/free_transformer.py": (2, 0),
    "dl_techniques/layers/transformers/gated_linear_attention_block.py": (1, 0),
    "dl_techniques/layers/transformers/gemma3_transformer.py": (0, 4),
    "dl_techniques/layers/transformers/text_decoder.py": (2, 0),
    "dl_techniques/layers/transformers/text_encoder.py": (2, 0),
    "dl_techniques/layers/transformers/transformer.py": (1, 0),
    "dl_techniques/layers/transformers/transformer_decoder.py": (3, 0),
    "dl_techniques/layers/transformers/vision_encoder.py": (6, 0),
    "dl_techniques/models/language/fnet/model.py": (1, 0),
    "dl_techniques/models/language/gemma/gemma3.py": (0, 1),
    "dl_techniques/models/language/hnet/components.py": (0, 3),
    "dl_techniques/models/language/qwen/components.py": (2, 0),
    "dl_techniques/models/language/qwen/qwen3.py": (1, 0),
    "dl_techniques/models/language/qwen/qwen3_embeddings.py": (2, 0),
    "dl_techniques/models/language/qwen/qwen3_next.py": (1, 0),
    "dl_techniques/models/language/topolm/components.py": (1, 0),
    "dl_techniques/models/language/topolm/model.py": (1, 0),
    "dl_techniques/models/language/tree_transformer/components.py": (3, 0),
    "dl_techniques/models/language/tree_transformer/model.py": (1, 0),
    "dl_techniques/models/neural_computer/nam/cell.py": (2, 0),
    "dl_techniques/models/neural_computer/nam/model.py": (1, 0),
    "dl_techniques/models/probabilistic/neural_hmm/encoders.py": (1, 0),
    "dl_techniques/models/time_series/tirex/model.py": (0, 1),
    "dl_techniques/models/time_series/xlstm/forecaster.py": (1, 0),
    "dl_techniques/models/time_series/xlstm/model.py": (1, 0),
    "dl_techniques/models/vision/alexnet/model.py": (0, 1),
    "dl_techniques/models/vision/dasiamrpn/model.py": (0, 1),
    "dl_techniques/models/vision/deepsort/appearance.py": (0, 3),
    "dl_techniques/models/vision/dino/dino_v1.py": (3, 0),
    "dl_techniques/models/vision/dino/dino_v2.py": (3, 0),
    "dl_techniques/models/vision/dino/dino_v3.py": (1, 0),
    "dl_techniques/models/vision/fastvit/components.py": (1, 1),
    "dl_techniques/models/vision/fftnet/model.py": (3, 0),
    "dl_techniques/models/vision/image_restoration/darkir/model.py": (0, 4),
    "dl_techniques/models/vision/image_restoration/pw_fnet/model.py": (2, 0),
    "dl_techniques/models/vision/mambalct/model.py": (0, 1),
    "dl_techniques/models/vision/mambalct/uca_encoder.py": (2, 0),
    "dl_techniques/models/vision/siamfc/model.py": (0, 1),
    "dl_techniques/models/vision/super_resolution/pft_sr/blocks.py": (0, 2),
    "dl_techniques/models/vision/super_resolution/pft_sr/model.py": (0, 2),
    "dl_techniques/models/vision/vit/model.py": (1, 0),
    "dl_techniques/models/vision/vit_hmlp/model.py": (1, 0),
    "dl_techniques/models/vision/vit_siglip/model.py": (1, 1),
    "dl_techniques/models/vision/vq_vae_rotation/model.py": (3, 0),
    "dl_techniques/models/vision_language/sam/sam1/image_encoder.py": (4, 0),
    "dl_techniques/models/vision_language/sam/sam1/mask_decoder.py": (1, 0),
    "dl_techniques/models/vision_language/sam/sam1/transformer.py": (5, 0),
    "dl_techniques/models/vision_language/sam/sam2/mask_decoder.py": (1, 0),
    "dl_techniques/models/vision_language/sam/sam3/decoder.py": (0, 6),
    "dl_techniques/models/vision_language/sam/sam3/maskformer_segmentation.py": (0, 1),
    "dl_techniques/models/vision_language/sam/sam3/vitdet.py": (0, 4),
    "train/rms_variants_train/experiments/e3_tinytransformer_imdb.py": (1, 0),
    "train/rms_variants_train/experiments/e4_deep_residual_reg.py": (2, 0),
    "train/rms_variants_train/experiments/e5_norm_layer_microbench.py": (1, 0),
    "train/rms_variants_train/experiments/e6_clm_wiki.py": (1, 0),
    "train/rms_variants_train/norm_overhead_bench.py": (1, 0),
}

#: The six dynamic sites D-253 ruled real constructor knobs (D-122 item 4), and the
#: two D-030 of plan-2026-10-23T190557-972a9ac5 added: ``TopoLMBlock._make_norm``,
#: the one call that builds both block norms (so the name it passes is the
#: ``name`` parameter), and ``TopoLM.final_norm``; and ``ConvStem._make_norm`` of
#: plan-2026-10-26T125959-7af851e5 step 1, which builds ``norm1`` and ``norm2`` in
#: one call (the name it passes is built from the ``suffix`` parameter). The
#: ``UcaEncoder`` ``stem_norm`` row is gone: that encoder's stem is now a ``ConvStem``. Its
#: downsampler norms are named ``downsample_{i}_norm`` (they were ``merge_norm_{s}``).
#: ``(file relative to src/, class, the attribute the norm is assigned to)``.
#: Each class documents ``normalization_type`` (``:param``), takes it in
#: ``__init__`` and returns it from ``get_config``; a literal type there would
#: silently ignore the argument.
RULED_DYNAMIC_KNOBS: Tuple[Tuple[str, str, str], ...] = (
    ("dl_techniques/layers/conv_blocks/gabor_depthwise_separable_block.py",
     "GaborDepthwiseSeparableBlock", "gabor_norm"),
    ("dl_techniques/layers/ssm/context_mamba.py", "ContextMambaLayer", "norm"),
    ("dl_techniques/models/probabilistic/neural_hmm/encoders.py",
     "NeuralHMMEncoder", "output_norm"),
    ("dl_techniques/models/vision/mambalct/uca_encoder.py", "UcaEncoder", "final_norm"),
    ("dl_techniques/models/vision/mambalct/uca_encoder.py", "UcaEncoder", "downsample_"),
    ("dl_techniques/models/language/topolm/components.py", "TopoLMBlock", "name"),
    ("dl_techniques/models/language/topolm/model.py", "TopoLM", "final_norm"),
    ("dl_techniques/layers/conv_blocks/stem.py", "ConvStem", "suffix"),
)


def _sweep_norm_call_sites(src_root: str = SRC_ROOT) -> Dict[str, Tuple[int, int]]:
    """Count ``create_normalization_layer`` call sites per file, by AST.

    :param src_root: The tree to sweep (``src/``; the planted-site control
        passes a temporary one).
    :return: ``{path relative to src_root: (dynamic, literal)}`` for every file
        with at least one site. A site is literal when its
        ``normalization_type`` (first positional or keyword) is a string
        constant, and dynamic otherwise.
    """
    per_file: Dict[str, List[int]] = {}
    for dirpath, dirnames, filenames in os.walk(src_root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            rel = os.path.relpath(path, src_root).replace(os.sep, "/")
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            # Deliberately no `except SyntaxError: continue` -- a silent skip
            # would silently shrink every number in this file (D-070).
            tree = ast.parse(source, filename=path)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                callee = getattr(node.func, "attr", getattr(node.func, "id", None))
                if callee != "create_normalization_layer":
                    continue
                arg = node.args[0] if node.args else None
                for keyword in node.keywords:
                    if keyword.arg == "normalization_type":
                        arg = keyword.value
                literal = isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                per_file.setdefault(rel, [0, 0])[1 if literal else 0] += 1
    return {path: (pair[0], pair[1]) for path, pair in sorted(per_file.items())}


@pytest.fixture(scope="module")
def norm_call_sites() -> Dict[str, Tuple[int, int]]:
    return _sweep_norm_call_sites()


class TestTheNormFactoryCallSitePopulationIsPinned:
    """Every site is named by its file; a new or removed one fails until the census says so."""

    def test_the_census_is_exact(self, norm_call_sites):
        if norm_call_sites == NORM_CALL_SITES:
            return
        added = sorted(set(norm_call_sites) - set(NORM_CALL_SITES))
        removed = sorted(set(NORM_CALL_SITES) - set(norm_call_sites))
        changed = sorted(
            f"{path}: {NORM_CALL_SITES[path]} -> {norm_call_sites[path]}"
            for path in set(norm_call_sites) & set(NORM_CALL_SITES)
            if norm_call_sites[path] != NORM_CALL_SITES[path])
        pytest.fail(
            "create_normalization_layer call sites changed, as (dynamic, literal) "
            f"per file. New files: {[(p, norm_call_sites[p]) for p in added]}; "
            f"files gone: {[(p, NORM_CALL_SITES[p]) for p in removed]}; "
            f"changed: {changed}. Edit NORM_CALL_SITES in the same commit. A new "
            "DYNAMIC site needs a literal `normalization_type=` or a ruling in "
            "decisions.md (D-253 re-opened D-201 for six named knobs)."
        )

    def test_the_totals_match_the_census(self, norm_call_sites):
        dynamic = sum(pair[0] for pair in NORM_CALL_SITES.values())
        literal = sum(pair[1] for pair in NORM_CALL_SITES.values())
        assert (dynamic, literal, len(NORM_CALL_SITES)) == (148, 45, 99), (
            "the census's own totals moved; re-derive the module docstring's numbers"
        )

    @pytest.mark.parametrize("path, class_name, attribute", RULED_DYNAMIC_KNOBS)
    def test_each_ruled_dynamic_site_is_a_serialized_knob(self, path, class_name, attribute):
        """The D-253 premise: documented, taken in ``__init__``, returned by ``get_config``."""
        with open(os.path.join(SRC_ROOT, path), encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
        assert classes, f"{class_name} is not defined in {path}"
        cls = classes[0]
        methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        init_args = [a.arg for a in methods["__init__"].args.args + methods["__init__"].args.kwonlyargs]
        assert "normalization_type" in init_args, f"{class_name}.__init__ takes no normalization_type"
        assert ":param normalization_type:" in (ast.get_docstring(cls) or ""), (
            f"{class_name} does not document normalization_type")
        config_keys = {
            node.value for node in ast.walk(methods["get_config"])
            if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        assert "normalization_type" in config_keys, f"{class_name}.get_config omits normalization_type"
        dynamic_targets = []
        for node in ast.walk(cls):
            if not isinstance(node, ast.Call):
                continue
            callee = getattr(node.func, "attr", getattr(node.func, "id", None))
            if callee != "create_normalization_layer":
                continue
            arg = node.args[0] if node.args else None
            for keyword in node.keywords:
                if keyword.arg == "normalization_type":
                    arg = keyword.value
            if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                names = [k.value for k in node.keywords if k.arg == "name"]
                dynamic_targets.append(ast.unparse(names[0]) if names else "")
        assert any(attribute in target for target in dynamic_targets), (
            f"no dynamic create_normalization_layer call named like {attribute!r} in "
            f"{class_name}: {dynamic_targets}")

    def test_the_census_sees_a_planted_site(self, tmp_path):
        """The instrument's own control: one planted dynamic and one literal site."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "planted.py").write_text(
            "create_normalization_layer(self.kind)\n"
            "create_normalization_layer(normalization_type='rms_norm')\n"
            "f = 'create_normalization_layer(x)'  # a string is not a site\n")
        assert _sweep_norm_call_sites(str(tmp_path)) == {"pkg/planted.py": (1, 1)}


class TestTheNormFactoryRaisesRatherThanFallingBack:
    """The reason a population ceiling is a sufficient ruling for this family."""

    def test_an_unknown_normalization_type_raises_and_names_the_valid_ones(self):
        with pytest.raises(ValueError) as excinfo:
            create_normalization_layer("not_a_real_normalization")
        message = str(excinfo.value)
        assert "not_a_real_normalization" in message, (
            "the raise must echo the offending key; a config-driven caller has "
            "no other way to find which of its dynamic sites misfired"
        )
        for valid_key in sorted(_TYPE_TO_CLASS):
            assert valid_key in message, (
                f"the ValueError omits the supported type {valid_key!r}. The "
                "error text is the only inventory a caller sees at the moment "
                "it fails."
            )

    def test_the_raise_is_not_a_silent_fallback(self):
        """A returned layer -- of ANY class -- would defeat every dynamic site."""
        try:
            result = create_normalization_layer("definitely_not_a_norm_type")
        except ValueError:
            return
        pytest.fail(
            "create_normalization_layer returned "
            f"{type(result).__name__} for an unknown key instead of raising. "
            "That single behaviour change would turn every statically "
            "uncheckable call site from self-guarding into silently wrong, and "
            "the census above would no longer be an adequate ruling."
        )

    @pytest.mark.parametrize("valid_key", sorted(_TYPE_TO_CLASS))
    def test_every_registry_key_actually_constructs(self, valid_key):
        layer = create_normalization_layer(valid_key)
        assert isinstance(layer, type(layer))
        assert layer is not None


# ---------------------------------------------------------------------------
# N-11: the factory is NOT a drop-in for constructing the layer directly.
# ---------------------------------------------------------------------------

#: What each registry key's epsilon is when the caller says nothing, MEASURED
#: 2026-08-23 on keras 3.8.0: ``(factory default, the class's OWN default)``.
#: ``None`` means the layer has no epsilon at all (``dynamic_tanh`` -- the factory
#: pops the key). Entries whose two values differ are the trap: a caller who
#: "just routes through the factory like resnet does" silently moves epsilon.
#:
#: **11 of the 16 bare-constructible types diverge.** The 1000x ``batch_norm`` /
#: ``layer_norm`` rows are
#: the ones that nearly shipped: step 18's brief proposed routing mobilenet and
#: cbam (189 BatchNormalization layers) through this factory, which would have
#: divided their epsilon by 1000 with no test noticing.
_EPSILON_DEFAULTS = {
    "adaptive_band_rms": (1e-06, 1e-07),
    "band_logit_norm": (1e-06, 1e-07),
    "band_rms": (1e-06, 1e-07),
    "batch_norm": (1e-06, 1e-03),
    "bias_free_batch_norm": (1e-06, 1e-06),
    "decoupled_max_logit": (1e-06, 1e-07),
    "dynamic_tanh": (None, None),
    "energy_layer_norm": (1e-06, 1e-05),
    "global_response_norm": (1e-06, 1e-06),
    "layer_norm": (1e-06, 1e-03),
    "logit_norm": (1e-06, 1e-07),
    "max_logit_norm": (1e-06, 1e-07),
    "rms_norm": (1e-06, 1e-06),
    "zero_centered_adaptive_band_rms_norm": (1e-06, 1e-07),
    "zero_centered_band_rms_norm": (1e-06, 1e-07),
    "zero_centered_rms_norm": (1e-06, 1e-06),
}

#: ``dml_plus_focal`` / ``dml_plus_center`` cannot be constructed bare (their
#: classes take a required positional argument), so their class-own default is
#: not observable this way. Named rather than silently absent, because "the dict
#: happens not to mention it" and "it was checked and excluded" must not look the
#: same (LESSONS: an all-skip module reads as a pass).
_EPSILON_UNCONSTRUCTIBLE_BARE = ("dml_plus_focal", "dml_plus_center")


class TestTheFactoryEpsilonIsNotTheLayerDefault:
    """N-11. Make the 1000x trap impossible to fall into a second time."""

    def test_the_keras_batchnorm_default_is_still_a_thousand_times_the_factorys(self):
        keras_default = keras.layers.BatchNormalization().epsilon
        factory_default = _epsilon_of(create_normalization_layer("batch_norm"))
        assert keras_default == pytest.approx(1e-3), (
            f"keras.layers.BatchNormalization default epsilon moved to "
            f"{keras_default}. Every claim in create_normalization_layer's "
            "not-a-drop-in warning is calibrated against 1e-3; re-measure the "
            "whole table in D-202 before touching this number."
        )
        assert factory_default == pytest.approx(1e-6), (
            f"create_normalization_layer's epsilon default moved to "
            f"{factory_default}. That silently re-tunes every call site in "
            "NORM_CALL_SITES that does not pass epsilon explicitly."
        )
        assert keras_default / factory_default == pytest.approx(1000.0), (
            "the documented 1000x divergence between "
            "keras.layers.BatchNormalization() and "
            "create_normalization_layer('batch_norm') no longer holds. Either "
            "the trap is gone (update the factory warning and D-202) or one "
            "default drifted (find out which)."
        )

    @pytest.mark.parametrize("norm_type", sorted(_EPSILON_DEFAULTS))
    def test_the_factory_and_class_defaults_are_both_pinned(self, norm_type):
        expected_factory, expected_own = _EPSILON_DEFAULTS[norm_type]
        observed_factory = _epsilon_of(create_normalization_layer(norm_type))
        observed_own = _epsilon_of(_TYPE_TO_CLASS[norm_type]())
        matches_factory = (
            observed_factory is None
            if expected_factory is None
            else observed_factory == pytest.approx(expected_factory)
        )
        matches_own = (
            observed_own is None
            if expected_own is None
            else observed_own == pytest.approx(expected_own)
        )
        assert matches_factory, (
            f"{norm_type}: factory epsilon default moved from "
            f"{expected_factory} to {observed_factory}"
        )
        assert matches_own, (
            f"{norm_type}: the CLASS's own epsilon default moved from "
            f"{expected_own} to {observed_own}. The divergence table in "
            "create_normalization_layer's warning is now wrong."
        )

    def test_the_divergent_majority_is_still_the_majority(self):
        diverging = sorted(
            key
            for key, (factory, own) in _EPSILON_DEFAULTS.items()
            if factory != own
        )
        assert len(diverging) == 11, (
            f"the number of registry types whose factory epsilon differs from "
            f"their own class default changed from 11 to {len(diverging)}: "
            f"{diverging}. That is the headline number in the factory's "
            "not-a-drop-in warning."
        )

    @pytest.mark.parametrize("norm_type", _EPSILON_UNCONSTRUCTIBLE_BARE)
    def test_the_excluded_types_are_excluded_for_the_stated_reason(self, norm_type):
        with pytest.raises(TypeError):
            _TYPE_TO_CLASS[norm_type]()

    def test_the_factory_warns_in_its_own_docstring(self):
        doc = create_normalization_layer.__doc__ or ""
        assert "NOT A DROP-IN REPLACEMENT" in doc, (
            "create_normalization_layer's docstring no longer carries the "
            "not-a-drop-in warning. That warning is the only thing standing "
            "between a reviewer and the 1000x epsilon change that step 18's "
            "brief proposed for 189 mobilenet/cbam layers."
        )
        assert "1000x" in doc and "batch_norm" in doc
