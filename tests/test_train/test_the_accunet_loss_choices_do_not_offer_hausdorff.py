"""``train_accunet --loss`` must not offer ``hausdorff`` (plan-2026-11-11T131349-13b28b37 D-014).

``SegmentationLosses.hausdorff_distance_loss`` is a 3-step max-pool heuristic, not a
Hausdorff distance. Measured on a 2-class 16x16 square: it is 1.0 for a perfect, a
shifted and an inverted prediction (it saturates), and 0.0 for an all-zeros
prediction, so gradient descent is steered to an empty mask. A trainer flag that
steers a user there is refused; the function stays in the package, labelled honestly.
``boundary`` (heuristic, but it ranks perfect below all-zeros) stays offered.
"""

import sys

import pytest

from train.accunet import train_accunet


def _parse(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["train_accunet", *argv])
    return train_accunet._parse_args()


# DECISION plan-2026-11-11T131349-13b28b37/D-014
def test_hausdorff_is_refused_by_the_parser(monkeypatch, capsys):
    with pytest.raises(SystemExit) as info:
        _parse(monkeypatch, "--loss", "hausdorff")
    assert info.value.code == 2
    assert "hausdorff" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["combo", "dice", "focal", "cross_entropy",
                                  "focal_tversky", "tversky", "lovasz", "boundary"])
def test_every_other_loss_choice_still_parses(monkeypatch, name):
    assert _parse(monkeypatch, "--loss", name).loss_name == name
