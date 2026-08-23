"""Pin every model variant that impersonates a published config to the published numbers.

A variant named after an upstream checkpoint makes a claim. When a package presents
itself as a faithful port and names its variant with the upstream's own identity
(matching dims, the upstream's variant name), the numbers in that row are not a
tunable of this repo -- they are quoted data, and quoted data that drifts is a false
citation. Five packages were caught drifting by fetching the upstream sources; this
module is the instrument that keeps them from drifting again.

Scope, stated because the absence of a row here is meaningful: only variants with a
real upstream counterpart appear below. Repo-invented size tiers (HRM's
micro/tiny/base/large/xlarge, sd3_mmdit's "tiny", ...) are deliberately NOT pinned --
there is nothing to pin them to, and asserting them here would manufacture the very
false-citation impression this file exists to prevent.

Each row carries the URL its values were fetched from, so a future reader can
re-verify a number without re-doing the search that found it.

References:
    - HRM: https://raw.githubusercontent.com/sapientinc/HRM/main/config/arch/hrm_v1.yaml
    - SD3 MMDiT: https://huggingface.co/v2ray/stable-diffusion-3-medium-diffusers/raw/main/transformer/config.json
      (community re-upload; the canonical `stabilityai/stable-diffusion-3-medium-diffusers`
      is gated and returns 401 unauthenticated)
    - NTM: https://arxiv.org/abs/1410.5401 (Graves et al. 2014), Tables 1 and 2
"""

from typing import Any, Callable, Dict

import pytest


def _hrm_variants() -> Dict[str, Dict[str, Any]]:
    from dl_techniques.models.hierarchical_reasoning_model.model import (
        HierarchicalReasoningModel,
    )

    return HierarchicalReasoningModel.MODEL_VARIANTS


def _sd3_variants() -> Dict[str, Dict[str, Any]]:
    from dl_techniques.models.sd3_mmdit.config import PRESETS

    return {k: v["config"] for k, v in PRESETS.items()}


def _ntm_variants() -> Dict[str, Dict[str, Any]]:
    from dl_techniques.models.ntm.model import NTMModel

    return NTMModel.NTM_VARIANTS


# package -> zero-arg loader for that package's variant table.
# Loaders are lazy so one package failing to import cannot mask the others.
VARIANT_TABLES: Dict[str, Callable[[], Dict[str, Dict[str, Any]]]] = {
    "hierarchical_reasoning_model": _hrm_variants,
    "sd3_mmdit": _sd3_variants,
    "ntm": _ntm_variants,
}

_NTM_URL = "https://arxiv.org/abs/1410.5401"  # Graves et al. 2014, Tables 1 & 2

_SD3_URL = (
    "https://huggingface.co/v2ray/stable-diffusion-3-medium-diffusers/raw/main/"
    "transformer/config.json"
)

# (package, variant, field, expected_value, source_url)
# Every tuple is a quoted upstream number. Changing one requires changing the source.
UPSTREAM_PINS = [
    # sapientinc/HRM config/arch/hrm_v1.yaml -- the ONLY official HRM architecture
    # config. hidden_size -> embed_dim, H_* -> h_*, L_* -> l_*.
    ("hierarchical_reasoning_model", "small", "embed_dim", 512,
     "https://raw.githubusercontent.com/sapientinc/HRM/main/config/arch/hrm_v1.yaml"),
    ("hierarchical_reasoning_model", "small", "num_heads", 8,
     "https://raw.githubusercontent.com/sapientinc/HRM/main/config/arch/hrm_v1.yaml"),
    ("hierarchical_reasoning_model", "small", "h_layers", 4,
     "https://raw.githubusercontent.com/sapientinc/HRM/main/config/arch/hrm_v1.yaml"),
    ("hierarchical_reasoning_model", "small", "l_layers", 4,
     "https://raw.githubusercontent.com/sapientinc/HRM/main/config/arch/hrm_v1.yaml"),
    ("hierarchical_reasoning_model", "small", "h_cycles", 2,
     "https://raw.githubusercontent.com/sapientinc/HRM/main/config/arch/hrm_v1.yaml"),
    ("hierarchical_reasoning_model", "small", "l_cycles", 2,
     "https://raw.githubusercontent.com/sapientinc/HRM/main/config/arch/hrm_v1.yaml"),
    ("hierarchical_reasoning_model", "small", "halt_max_steps", 16,
     "https://raw.githubusercontent.com/sapientinc/HRM/main/config/arch/hrm_v1.yaml"),

    # stabilityai/stable-diffusion-3-medium-diffusers transformer/config.json.
    # Name mapping: caption_projection_dim -> embedding_size,
    # num_attention_heads -> num_heads, num_layers -> depth. The "tiny" preset is a
    # deliberately untethered smoke-training size and is NOT pinned.
    ("sd3_mmdit", "full", "patch_size", 2, _SD3_URL),
    ("sd3_mmdit", "full", "in_channels", 16, _SD3_URL),
    ("sd3_mmdit", "full", "out_channels", 16, _SD3_URL),
    ("sd3_mmdit", "full", "embedding_size", 1536, _SD3_URL),
    ("sd3_mmdit", "full", "num_heads", 24, _SD3_URL),
    ("sd3_mmdit", "full", "depth", 24, _SD3_URL),
    ("sd3_mmdit", "full", "joint_attention_dim", 4096, _SD3_URL),
    ("sd3_mmdit", "full", "pooled_projection_dim", 2048, _SD3_URL),
    ("sd3_mmdit", "full", "pos_embed_max_size", 192, _SD3_URL),
    ("sd3_mmdit", "full", "sample_size", 128, _SD3_URL),

    # Graves et al. 2014, Tables 1 and 2: EVERY experiment row uses a 128 x 20
    # memory and the paper never varies N or M. num_read/write_heads=1 matches 4 of
    # the 5 LSTM-controller rows in Table 2. controller_dim is deliberately NOT
    # pinned: no LSTM row uses 256, so that field is a repo choice, not a quote.
    # 'tiny' and 'large' have no published counterpart and are not pinned.
    ("ntm", "base", "memory_size", 128, _NTM_URL),
    ("ntm", "base", "memory_dim", 20, _NTM_URL),
    ("ntm", "base", "num_read_heads", 1, _NTM_URL),
    ("ntm", "base", "num_write_heads", 1, _NTM_URL),
]


@pytest.mark.parametrize(
    "package,variant,field,expected,url",
    UPSTREAM_PINS,
    ids=[f"{p}-{v}-{f}" for p, v, f, _e, _u in UPSTREAM_PINS],
)
def test_variant_field_matches_upstream(
    package: str, variant: str, field: str, expected: Any, url: str
) -> None:
    """The named field of the named variant must equal the value published upstream."""
    table = VARIANT_TABLES[package]()
    assert variant in table, (
        f"{package}: variant '{variant}' vanished from the variant table; it is the row "
        f"that quotes {url} and cannot be renamed away silently"
    )
    actual = table[variant][field]
    assert actual == expected, (
        f"{package}['{variant}']['{field}'] is {actual!r} but upstream publishes "
        f"{expected!r}. Source: {url}. Either restore the upstream value or stop "
        f"naming this variant after the upstream config."
    )
