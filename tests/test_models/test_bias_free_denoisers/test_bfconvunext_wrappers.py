"""Contract tests for the two THIN WRAPPERS in `models/bias_free_denoisers/bfconvunext.py`.

After the ConvUNext merge, `bfconvunext.py` is two things and nothing else: a pair of
`use_bias=False` wrappers over `models/convunext/model.create_convunext`, and the Keras
REGISTRAR that `applications/bias_free_denoiser/denoiser_prior.py` and the two bfunet eval
tools import for its side effect. This file pins exactly those two contracts.

It is deliberately a SEPARATE file from `test_bfconvunext_denoiser.py` /
`test_bfconvunext_gabor.py`: plan invariant I-1 freezes those two suites at 78 assertions
with exactly one sanctioned edit, so new coverage may not be appended to them.

The three pinned facts (plan assumption A-1, decisions.md D-003 / D-014):

(i)   `bfconvunext.create_convunext_variant('tiny', ...)` builds with BATCHNORM blocks.
(ii)  A bare `bfconvunext.create_convunext_denoiser(...)` still builds with LAYERNORM.
(iii) `models.convunext.model.create_convunext_variant('tiny', ...)` — the bias-ON path —
      still builds with LAYERNORM.

(iii) is the one a careless implementation gets wrong, by putting `block_normalization`
into the shared `CONVUNEXT_CONFIGS` dict instead of into the bias-free wrapper.
"""

import subprocess
import sys

import keras
import pytest

from dl_techniques.models.bias_free_denoisers.bfconvunext import (
    create_convunext_denoiser,
    create_convunext_variant as bf_create_convunext_variant,
)
from dl_techniques.models.convunext.model import (
    create_convunext_variant as std_create_convunext_variant,
)

# Small everywhere: these tests measure WIRING, not capacity. `blocks_per_level=1`
# and `initial_filters=8` keep every build under a second on CPU.
SMALL = dict(initial_filters=8, blocks_per_level=1, enable_deep_supervision=False)
INPUT_SHAPE = (32, 32, 1)


def _norm_type_counts(model: keras.Model) -> dict:
    """Count every normalization-ish sub-layer by class name.

    Asserted on the built TYPE rather than on the `block_normalization` string the
    caller passed, so the test cannot pass by echoing its own input back.
    """
    counts: dict = {}
    for layer in model._flatten_layers():
        name = type(layer).__name__
        if 'orm' in name:
            counts[name] = counts.get(name, 0) + 1
    return counts


def _bias_offenders(model: keras.Model) -> list:
    """The repo's own bias-compliance instrument (SC5/SC7 in the frozen bf suites)."""
    offenders = []
    for layer in model._flatten_layers():
        if getattr(layer, "use_bias", False):
            offenders.append(layer.name)
        if isinstance(layer, keras.layers.LayerNormalization) and getattr(
            layer, "center", False
        ):
            offenders.append(f"{layer.name} (LN center)")
    return offenders


class TestDenoiserWrapperPinsBiasFree:
    """The `create_convunext_denoiser` wrapper must pin `use_bias=False`."""

    def test_denoiser_wrapper_pins_use_bias_false(self) -> None:
        model = create_convunext_denoiser(
            input_shape=INPUT_SHAPE, depth=3, **SMALL
        )
        offenders = _bias_offenders(model)
        assert not offenders, (
            "the create_convunext_denoiser wrapper did not pin use_bias=False; "
            f"offenders: {offenders}"
        )

    def test_variant_wrapper_pins_use_bias_false(self) -> None:
        # Second entry point, same contract. Without this, an injection at the
        # variant wrapper alone would leave the file green.
        model = bf_create_convunext_variant('tiny', INPUT_SHAPE, **SMALL)
        offenders = _bias_offenders(model)
        assert not offenders, (
            "the create_convunext_variant wrapper did not pin use_bias=False; "
            f"offenders: {offenders}"
        )


class TestVariantBlockNormalization:
    """Plan assumption A-1: only the BIAS-FREE VARIANT path flips to batchnorm."""

    def test_variant_defaults_to_batchnorm_under_bias_free(self) -> None:
        # (i)
        counts = _norm_type_counts(
            bf_create_convunext_variant('tiny', INPUT_SHAPE, **SMALL)
        )
        assert counts.get('BiasFreeBatchNorm', 0) > 0, (
            f"bf create_convunext_variant did not select batchnorm blocks: {counts}"
        )
        assert counts.get('LayerNormalization', 0) == 0, (
            f"bf create_convunext_variant still has LayerNormalization blocks: {counts}"
        )

    def test_bare_denoiser_still_builds_with_layernorm(self) -> None:
        # (ii) The batchnorm setdefault must NOT leak into the raw builder call.
        # This is what keeps the byte-identity tripwire in test_bfconvunext_denoiser.py
        # green and `utils/multiplicative_miyasawa.py`'s omitted-kwarg call unchanged.
        counts = _norm_type_counts(
            create_convunext_denoiser(input_shape=INPUT_SHAPE, depth=3, **SMALL)
        )
        assert counts.get('LayerNormalization', 0) > 0, (
            f"bare create_convunext_denoiser lost its layernorm blocks: {counts}"
        )
        assert counts.get('BiasFreeBatchNorm', 0) == 0, (
            "the batchnorm setdefault leaked out of create_convunext_variant into "
            f"the raw builder call: {counts}"
        )

    def test_bias_on_variant_still_builds_with_layernorm(self) -> None:
        # (iii) The CONTROL that isolates the contract: the shared CONVUNEXT_CONFIGS
        # dict must NOT carry `block_normalization`, or the bias-ON variants flip too.
        counts = _norm_type_counts(
            std_create_convunext_variant('tiny', INPUT_SHAPE, **SMALL)
        )
        assert counts.get('LayerNormalization', 0) > 0, (
            f"the bias-ON variant lost its layernorm blocks: {counts}"
        )
        assert counts.get('BiasFreeBatchNorm', 0) == 0, (
            "block_normalization was put in the SHARED variant dict — the bias-ON "
            f"variants flipped to batchnorm too: {counts}"
        )

    def test_variant_respects_an_explicit_block_normalization(self) -> None:
        # `setdefault`, not an unconditional assignment: a caller-supplied value wins.
        counts = _norm_type_counts(
            bf_create_convunext_variant(
                'tiny', INPUT_SHAPE, block_normalization='layernorm', **SMALL
            )
        )
        assert counts.get('LayerNormalization', 0) > 0, (
            "a caller-supplied block_normalization='layernorm' did not survive the "
            f"wrapper's setdefault: {counts}"
        )
        assert counts.get('BiasFreeBatchNorm', 0) == 0, (
            "the wrapper overwrote the caller's block_normalization instead of "
            f"defaulting it: {counts}"
        )


# The registrar probe runs in a FRESH interpreter on purpose. Run inside this process it
# would prove nothing: a sibling test has already imported half the library, so every key
# would resolve no matter what `bfconvunext.py` does.
_REGISTRAR_PROBE = r'''
import json
import keras
import dl_techniques.models.bias_free_denoisers.bfconvunext as m  # the ONLY import

resolved = {}
for key, obj in keras.saving.get_custom_objects().items():
    resolved.setdefault(getattr(obj, "__name__", ""), []).append(key)
# Reported as DATA, never asserted inside the probe: a missing attribute must surface
# as a named test failure, not as a subprocess crash that reds the whole class.
attrs = sorted(n for n in dir(m) if not n.startswith('_'))
print("RESULT " + json.dumps({"resolved": resolved, "module_attrs": attrs}))
'''

# The five classes `applications/bias_free_denoiser/denoiser_prior.py` names in its
# loading contract, plus the two a merged bias-free graph now also contains.
REGISTRAR_CLASSES = [
    'ConvUNextStem',
    'ConvNextV1Block',
    'GlobalResponseNormalization',
    'MatchChannels',
    'GaborFiltersInitializer',
    'DownsampleAndSkip',
    'SpatialLinearAttention',
]

STEM_REGISTRY_KEY = 'dl_techniques.bias_free_denoisers>ConvUNextStem'

# The names `bfconvunext` must keep bound as module ATTRIBUTES. This is a DIFFERENT
# contract from registry presence, and the difference is measured, not assumed: deleting
# the `from ...convunext.model import ConvUNextStem` re-export does NOT de-register the
# class, because `bfconvunext` imports `create_convunext` from that same module and the
# decorator runs at module-exec time. What the re-export line actually buys is this name
# binding, which `test_bfconvunext_denoiser.py` and `train/bfunet` import by name.
RE_EXPORTED_NAMES = [
    'ConvUNextStem',
    'SpatialLinearAttention',
    'ConvNextV1Block',
    'ConvNextV2Block',
    'GlobalResponseNormalization',
    'MatchChannels',
    'DownsampleAndSkip',
    'StochasticDepth',
    'CONVUNEXT_CONFIGS',
    'create_convunext_denoiser',
    'create_convunext_variant',
]


@pytest.fixture(scope="module")
def registrar_probe_result() -> dict:
    """Import ONLY `bfconvunext` in a fresh interpreter; return the registry contents."""
    import json
    import os

    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = '-1'
    env['MPLBACKEND'] = 'Agg'
    completed = subprocess.run(
        [sys.executable, '-c', _REGISTRAR_PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert completed.returncode == 0, (
        f"registrar probe failed (rc={completed.returncode}):\n{completed.stderr[-4000:]}"
    )
    for line in completed.stdout.splitlines():
        if line.startswith('RESULT '):
            return json.loads(line[len('RESULT '):])
    raise AssertionError(f"probe produced no RESULT line:\n{completed.stdout[-4000:]}")


class TestRegistrarContract:
    """Importing `bfconvunext` alone must register every class a saved graph names."""

    def test_registrar_import_registers_convunext_stem(
        self, registrar_probe_result: dict
    ) -> None:
        keys = registrar_probe_result['resolved'].get('ConvUNextStem', [])
        assert keys == [STEM_REGISTRY_KEY], (
            f"ConvUNextStem registry keys after importing only bfconvunext: {keys}; "
            f"expected exactly [{STEM_REGISTRY_KEY!r}]"
        )

    @pytest.mark.parametrize("name", RE_EXPORTED_NAMES)
    def test_bfconvunext_re_exports_the_name(
        self, registrar_probe_result: dict, name: str
    ) -> None:
        # The name-binding half of the registrar contract, and the ONLY assertion that
        # a deleted re-export line can turn red (registry presence survives it).
        assert name in registrar_probe_result['module_attrs'], (
            f"bfconvunext no longer binds the name {name!r}; a re-export line was "
            "removed and every caller importing it by name breaks at import time"
        )

    @pytest.mark.parametrize("class_name", REGISTRAR_CLASSES)
    def test_registrar_import_resolves_every_saved_graph_class(
        self, registrar_probe_result: dict, class_name: str
    ) -> None:
        keys = registrar_probe_result['resolved'].get(class_name, [])
        assert keys, (
            f"{class_name} is NOT in the Keras registry after importing only "
            "dl_techniques.models.bias_free_denoisers.bfconvunext; "
            "keras.models.load_model would fail on a saved bias-free ConvUNext"
        )
