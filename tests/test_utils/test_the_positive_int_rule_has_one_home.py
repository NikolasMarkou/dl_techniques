"""One positive-integer rule, imported by the adapters, the MoE config, the heads and the poolers.

Guard for plan-2026-10-14T062616-81938e81 step 12 Part F item 11 (D-061,
D-088). Measured at d6995a932, three private copies refused different sets:
``layers/adapters/factory.py`` and ``layers/heads/validation.py`` accepted
``2**31`` (a width no int32 tensor dimension holds), ``layers/moe/config.py``
refused it, and only the MoE copy named ``np.bool_``. Each package now binds
``dl_techniques.utils.validation.positive_int`` and the refusal table below
holds through each package's own entry point. The sequence poolers joined in
step 5.4 (D-203); their entry points are pinned by
``tests/test_layers/test_the_pooling_layers_refuse_bad_arguments_early.py``.
"""

import numpy as np
import pytest

from dl_techniques.layers.adapters import factory as adapters_factory
from dl_techniques.layers.heads import validation as heads_validation
from dl_techniques.layers.heads.vlm.task_types import VLMTaskConfig, VLMTaskType
from dl_techniques.layers.moe import config as moe_config
from dl_techniques.layers.sequence_pooling import (
    attention_pooling,
    factory as pooling_factory,
    sequence_pooling,
    weighted_pooling,
)
from dl_techniques.utils import validation

ACCEPTED = [(1, 1), (7, 7), (np.int64(4), 4), (np.uint8(3), 3),
            (2 ** 31 - 1, 2 ** 31 - 1)]
REFUSED = [True, False, np.bool_(True), 0, -3, 2.0, 2.5, "4", None, 2 ** 31,
           np.int64(2 ** 31)]


@pytest.mark.parametrize(
    "module",
    [adapters_factory, heads_validation, moe_config, attention_pooling,
     weighted_pooling, sequence_pooling, pooling_factory],
    ids=["adapters", "heads", "moe", "attention_pooling", "weighted_pooling",
         "sequence_pooling", "pooling_factory"],
)
def test_each_package_binds_the_one_helper(module):
    assert module.positive_int is validation.positive_int


@pytest.mark.parametrize("module", [adapters_factory, moe_config], ids=["adapters", "moe"])
def test_no_private_copy_is_left(module):
    assert not hasattr(module, "_validate_positive_int")


@pytest.mark.parametrize("value, expected", ACCEPTED)
def test_accepted_values_come_back_as_python_ints(value, expected):
    got = validation.positive_int(value, "n", "Owner")
    assert got == expected and type(got) is int


@pytest.mark.parametrize("value", REFUSED, ids=repr)
def test_refused_values_name_the_owner_and_the_argument(value):
    with pytest.raises(ValueError, match=r"^Owner: n must be"):
        validation.positive_int(value, "n", "Owner")


def _adapter(value):
    adapters_factory.validate_adapter_config(
        "lora", output_dim=value, rank=2, num_adapters=1, alpha=1.0)


def _moe(value):
    moe_config.GatingConfig(top_k=value)


def _vlm(value):
    VLMTaskConfig(name="t", task_type=VLMTaskType.VISUAL_GROUNDING, hidden_size=value)


@pytest.mark.parametrize("entry", [_adapter, _moe, _vlm], ids=["adapters", "moe", "vlm"])
@pytest.mark.parametrize("value", REFUSED, ids=repr)
def test_every_entry_point_refuses_the_same_values(entry, value):
    with pytest.raises(ValueError, match="must be"):
        entry(value)


@pytest.mark.parametrize("entry", [_adapter, _moe, _vlm], ids=["adapters", "moe", "vlm"])
@pytest.mark.parametrize("value", [1, np.int64(4)], ids=repr)
def test_every_entry_point_accepts_the_same_values(entry, value):
    entry(value)
