"""`PTRMTrainer` builds the variant its config names.

Rationale
---------
`main()` puts a preset's ``variant`` key into ``config["model"]`` for every
``--preset``. `PTRMTrainer._create_model` used to call the `PTRM` constructor
with ``.get`` defaults and never read that key, so a preset's architecture came
from the trainer's defaults (which happened to equal ``att_base``), not from the
variant. A config that names a variant now goes through ``create_ptrm`` with the
config's other keys as explicit overrides; a config without one is a fully
explicit architecture for the constructor.

``att_small`` is the probe because its width (384) and head count (6) differ
from the old defaults (512, 8), so a dropped variant cannot pass.

See decisions.md D-019 (plan-2026-10-13T035756-5dd4d94d).
"""

import keras
import pytest

from dl_techniques.models.language.ptrm import PTRM, create_ptrm
from train.ptrm.train_ptrm import PTRMTrainer, get_quick_test_config

# ---------------------------------------------------------------------

_VARIANT_MODEL = {
    "variant": "att_small",
    "vocab_size": 16,
    "seq_len": 8,
    "puzzle_emb_len": 4,
    # Explicit overrides laid over the variant (keeps the build small).
    "h_layers": 1,
    "l_layers": 1,
    # Read by main() for the dataset, not a model argument.
    "batch_size": 2,
}


def _config(model_section):
    config = get_quick_test_config()
    config["model"] = dict(model_section)
    return config


def _architecture(model: PTRM) -> dict:
    config = model.get_config()
    config.pop("name", None)
    config.pop("trainable", None)
    config.pop("dtype", None)
    return config


def test_a_named_variant_reaches_the_model():
    trainer = PTRMTrainer(config=_config(_VARIANT_MODEL), train_dataset=None)
    model = trainer.model
    variant = PTRM.MODEL_VARIANTS["att_small"]
    assert model.built
    assert model.hidden_size == variant["hidden_size"] == 384
    assert model.num_heads == variant["num_heads"] == 6
    assert model.halt_max_steps == variant["halt_max_steps"]
    # The explicit overrides win over the variant.
    assert (model.h_layers, model.l_layers) == (1, 1)


def test_the_trainer_builds_what_create_ptrm_builds():
    trainer = PTRMTrainer(config=_config(_VARIANT_MODEL), train_dataset=None)
    expected = create_ptrm(
        "att_small", vocab_size=16, seq_len=8, puzzle_emb_len=4,
        h_layers=1, l_layers=1,
    )
    assert _architecture(trainer.model) == _architecture(expected)
    assert len(trainer.model.weights) == len(expected.weights)


def test_an_unknown_variant_is_refused():
    with pytest.raises(ValueError, match="Unknown variant 'mlp_base'"):
        PTRMTrainer(
            config=_config(dict(_VARIANT_MODEL, variant="mlp_base")),
            train_dataset=None,
        )


def test_a_config_without_a_variant_is_the_constructor():
    """The quick preset states every hyperparameter and names no variant."""
    config = get_quick_test_config()
    assert "variant" not in config["model"]
    trainer = PTRMTrainer(config=config, train_dataset=None)
    expected = PTRM(**config["model"])
    expected.build()
    assert _architecture(trainer.model) == _architecture(expected)
    keras.backend.clear_session()
