"""``MODEL_VARIANTS`` is the four released checkpoints.

``data/released_configs.json`` holds the ``config.json`` of each Hugging Face repo
(``mistralai/Timestral-{5M,28M,123M,209M}``, fetched 2026-10-19), keyed by variant.
Until 2026-10-19 (plan-2026-10-14T062616-81938e81 D-168) the table matched none of them:
its 5M was 256 wide with 8 heads, an alternating schedule and a quantile width factor of
2.0, where the release is 192 wide with 3 heads, ``(time, time, time, time_variate) x 2``
and 3.0.
"""

import json
from pathlib import Path

import pytest

from dl_techniques.models.time_series.timestral import TimestralModel

RELEASED = json.loads((Path(__file__).parent / "data" / "released_configs.json").read_text())


def test_the_table_names_exactly_the_released_variants():
    assert sorted(TimestralModel.MODEL_VARIANTS) == sorted(RELEASED)


@pytest.mark.parametrize("variant", sorted(RELEASED))
def test_every_variant_is_its_released_checkpoint(variant):
    config = TimestralModel.from_variant(variant).config.to_dict()
    assert config.pop("prediction_length") == 96  # the port's own field, not in the release
    assert config == RELEASED[variant]
