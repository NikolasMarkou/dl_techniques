"""``create_bridge_sde`` -- the four-key registry factory, which had ZERO tests.

Split out of ``test_the_package_keeps_the_repo_contracts.py`` deliberately. That
file's static scans must keep running when a name in this package is renamed;
if they shared a module with these runtime imports, renaming ``SDE_TYPES`` would
turn the whole module into a collection ImportError and the scans would report
nothing rather than firing. Measured: that is exactly what happened on the first
RED proof of this step.
"""

import pytest

from dl_techniques.models.vision_language.bit_diffusion import (
    SDE_TYPES,
    BridgeSDE,
    create_bridge_sde,
)




@pytest.mark.parametrize("key", sorted(SDE_TYPES))
def test_the_factory_builds_every_registered_type(key):
    """Every advertised key constructs, and returns the class it maps to.

    An advertised-but-never-constructed branch stays dead under a green suite;
    parametrizing over the registry's OWN keys means a new entry is exercised
    the moment it is added rather than the moment someone remembers.
    """
    sde = create_bridge_sde(key)
    assert isinstance(sde, BridgeSDE)
    assert type(sde) is SDE_TYPES[key]


def test_the_factory_forwards_kwargs_to_the_constructor():
    sde = create_bridge_sde("uniform", A=2.5, K=0.75)
    assert sde.A == pytest.approx(2.5)
    assert sde.K == pytest.approx(0.75)


def test_the_factory_raises_on_an_unregistered_type():
    with pytest.raises(ValueError, match="Unknown SDE type"):
        create_bridge_sde("not_a_real_sde")


def test_the_factory_parameter_is_positional_under_its_new_name():
    """The rename is a contract: `sde_type=` must work, `variant=` must not.

    Without this arm the rename could be silently reverted by a merge and only
    the repo-wide suite would notice.
    """
    assert isinstance(create_bridge_sde(sde_type="periodic"), BridgeSDE)
    with pytest.raises(TypeError):
        create_bridge_sde(variant="periodic")
