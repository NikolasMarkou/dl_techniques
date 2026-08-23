"""The two advisories this repo emits on PURPOSE, pinned so they cannot vanish.

Why this file exists
--------------------
``pyproject.toml``'s ``[tool.pytest.ini_options] filterwarnings`` turns every
``UserWarning`` into an error, and the two advisories below are then suppressed
**module by module** at the sites that provoke them (a ``pytestmark`` carrying
``pytest.mark.filterwarnings("ignore:...")`` and a comment naming the family).
That arrangement has one failure mode and this file is its only defence: if the
advisory ever stops being emitted -- deleted, its message reworded, its
triggering branch inverted -- every one of those ``ignore`` entries silently
becomes a no-op and NOTHING goes red. A suppression with no paired positive
assertion is a claim nobody checks.

So each advisory gets two arms here:

* a **positive** arm asserting the warning IS raised, with a ``match=`` on the
  exact prefix the ``ignore`` filters key on -- so a reword breaks this file
  before it silently widens the filters; and
* a **control** arm asserting the non-provoking configuration does NOT warn, so
  the positive arm cannot pass by the advisory having become unconditional.

See ``plans/plan-2026-08-22T035419-a11304c8/decisions.md`` D-252 (R-038 closure).
"""

import warnings

import numpy as np
import pytest

from dl_techniques.initializers.hypersphere_orthogonal_initializer import (
    OrthogonalHypersphereInitializer,
)
from dl_techniques.layers.transformers.transformer import TransformerLayer


# ---------------------------------------------------------------------
# W-03 -- the orthogonality fallback
# `src/dl_techniques/initializers/hypersphere_orthogonal_initializer.py`
# ---------------------------------------------------------------------


class TestTheOrthogonalityFallbackAdvisory:
    """``num_vectors > latent_dim`` is mathematically impossible; we say so."""

    def test_an_infeasible_request_warns(self):
        init = OrthogonalHypersphereInitializer()
        with pytest.warns(
            UserWarning, match=r"Orthogonality constraint violation"
        ):
            out = init(shape=(8, 4))
        # The fallback must still produce the requested geometry -- an advisory
        # that came with a broken tensor would be a defect, not an advisory.
        assert tuple(out.shape) == (8, 4)

    def test_a_feasible_request_does_not_warn(self):
        """The control: without it the positive arm cannot fail."""
        init = OrthogonalHypersphereInitializer()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = init(shape=(4, 8))
        assert tuple(out.shape) == (4, 8)

    def test_the_message_names_both_numbers(self):
        """The ``ignore`` filters key on the prefix; the numbers are the payload."""
        with pytest.warns(UserWarning) as rec:
            OrthogonalHypersphereInitializer()(shape=(9, 3))
        text = str(rec[0].message)
        assert "requesting 9 orthogonal vectors" in text, text
        assert "3-dimensional space" in text, text


# ---------------------------------------------------------------------
# W-14 -- MoE supersedes ffn_type / ffn_args
# `src/dl_techniques/layers/transformers/transformer.py`
# ---------------------------------------------------------------------


def _moe_config():
    return {
        "num_experts": 2,
        "expert_config": {"ffn_config": {"type": "mlp", "hidden_dim": 8}},
    }


class TestTheMoeSupersessionAdvisory:
    """``moe_config`` wins over ``ffn_type``/``ffn_args``, and says so."""

    def test_a_conflicting_ffn_type_warns(self):
        with pytest.warns(UserWarning, match=r"moe_config is provided"):
            TransformerLayer(
                hidden_size=8, num_heads=2, intermediate_size=8,
                ffn_type="swiglu", moe_config=_moe_config(),
            )

    def test_the_default_ffn_type_does_not_warn(self):
        """The control: ``ffn_type='mlp'`` with no ``ffn_args`` is not a conflict."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            TransformerLayer(
                hidden_size=8, num_heads=2, intermediate_size=8,
                moe_config=_moe_config(),
            )

    def test_no_moe_config_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            TransformerLayer(
                hidden_size=8, num_heads=2, intermediate_size=8,
                ffn_type="swiglu",
            )
