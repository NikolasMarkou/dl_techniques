"""Drift guard for ``generation_metrics.GENERATION_METRIC_REGISTRY``.

The registry's key set, its aliases and each entry's DECLARED PARAMETERS are
public surface. Once callers pass a metric name from a configuration file, the
exact spelling stops being an implementation detail: a rename or a dropped key
breaks those configs with a message that names the new name rather than the one
they asked for.

``dl_techniques.metrics.generation_metrics`` states this in its own module
docstring, and this test is the enforcement. It pins:

* the exact key set, so removing or renaming an entry is a deliberate act that
  has to be made here;
* each entry's required and optional parameters, so a parameter cannot silently
  stop being configurable;
* that no entry declares a parameter the underlying function does not accept --
  which would leave the raise-on-undeclared-key rule guarding a key that ends in
  a ``TypeError`` instead.

Every expected value below was read off the registry, so a change here is a
change a reviewer sees in the diff rather than a failure discovered at runtime.
"""

from __future__ import annotations

import inspect
from typing import Dict, Tuple

from dl_techniques.metrics.generation_metrics import (
    DEFAULT_BLEU_MAX_ORDER,
    DEFAULT_CHRF_BETA,
    DEFAULT_CHRF_CHAR_ORDER,
    DEFAULT_CHRF_WORD_ORDER,
    DEFAULT_CIDER_SIGMA,
    DEFAULT_ROUGE_L_BETA,
    GENERATION_METRIC_REGISTRY,
)

#: The public key set, exactly. Adding a key is fine; removing or renaming one
#: is a breaking change and has to be made in both places deliberately.
EXPECTED_KEYS: Tuple[str, ...] = (
    "bleu",
    "chrf",
    "cider",
    "rouge_l",
    "rouge_lsum",
    "rouge_n",
)

#: ``name -> (required, optional)``, sorted so the comparison is order-free.
EXPECTED_PARAMS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "bleu": ((), ("max_order", "smooth", "tokenizer")),
    "chrf": ((), ("beta", "char_order", "remove_whitespace", "tokenizer", "word_order")),
    "cider": ((), ("n", "sigma", "tokenizer", "variant")),
    "rouge_l": ((), ("beta", "tokenizer")),
    "rouge_lsum": ((), ("beta", "tokenizer")),
    "rouge_n": (("n",), ("score", "tokenizer")),
}


def test_the_registry_has_exactly_the_expected_keys() -> None:
    assert len(GENERATION_METRIC_REGISTRY) > 0, (
        "the registry is empty, so every test in this module passes vacuously"
    )
    assert tuple(sorted(GENERATION_METRIC_REGISTRY)) == EXPECTED_KEYS, (
        f"registry keys drifted: {sorted(GENERATION_METRIC_REGISTRY)} != "
        f"{list(EXPECTED_KEYS)}. Adding a metric is fine; removing or renaming "
        f"one breaks callers that pass the name from configuration."
    )


def test_every_entry_declares_exactly_the_expected_parameters() -> None:
    for name, (required, optional) in EXPECTED_PARAMS.items():
        assert name in GENERATION_METRIC_REGISTRY, f"{name} disappeared from the registry"
        entry = GENERATION_METRIC_REGISTRY[name]
        assert tuple(sorted(entry["required_params"])) == required, (
            f"{name} required params drifted: "
            f"{sorted(entry['required_params'])} != {list(required)}"
        )
        assert tuple(sorted(entry["optional_params"])) == optional, (
            f"{name} optional params drifted: "
            f"{sorted(entry['optional_params'])} != {list(optional)}"
        )


def test_no_entry_declares_a_parameter_its_function_does_not_accept() -> None:
    """Otherwise the raise-on-undeclared-key rule protects a key that reaches a
    ``TypeError`` instead -- still an error, but the wrong one, and one that
    would read as a bug in the caller."""
    for name, entry in GENERATION_METRIC_REGISTRY.items():
        accepted = set(inspect.signature(entry["function"]).parameters)
        declared = set(entry["required_params"]) | set(entry["optional_params"])
        missing = declared - accepted
        assert not missing, (
            f"{name} declares {sorted(missing)}, which its function does not "
            f"accept; accepted: {sorted(accepted)}"
        )


def test_optional_defaults_are_the_module_constants() -> None:
    """The registry's defaults must BE the documented defaults, not copies that
    drift. A registry default of ``char_order=6`` written as a literal while
    :data:`DEFAULT_CHRF_CHAR_ORDER` moves would make the two disagree with no
    test able to see it."""
    expected = {
        "bleu": {"max_order": DEFAULT_BLEU_MAX_ORDER, "smooth": "exp", "tokenizer": None},
        "chrf": {
            "char_order": DEFAULT_CHRF_CHAR_ORDER,
            "word_order": DEFAULT_CHRF_WORD_ORDER,
            "beta": DEFAULT_CHRF_BETA,
            "remove_whitespace": True,
            "tokenizer": None,
        },
        "cider": {
            "n": DEFAULT_BLEU_MAX_ORDER,
            "sigma": DEFAULT_CIDER_SIGMA,
            "variant": "d",
            "tokenizer": None,
        },
        "rouge_l": {"beta": DEFAULT_ROUGE_L_BETA, "tokenizer": None},
        "rouge_lsum": {"beta": DEFAULT_ROUGE_L_BETA, "tokenizer": None},
        "rouge_n": {"score": "fmeasure", "tokenizer": None},
    }
    for name, defaults in expected.items():
        assert GENERATION_METRIC_REGISTRY[name]["optional_params"] == defaults, (
            f"{name} optional DEFAULTS drifted: "
            f"{GENERATION_METRIC_REGISTRY[name]['optional_params']} != {defaults}"
        )