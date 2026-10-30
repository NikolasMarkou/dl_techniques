"""THERA's heat-field init defaults are defined once, in the layer that owns them.

``DEFAULT_COMPONENTS_INIT_SCALE`` (16.0) and ``DEFAULT_K_INIT`` lived as separate
literals in ``layers/thera_heat_field.py``, ``models/vision/thera/model.py``,
``models/vision/thera/hypernetwork.py`` (a bare ``16.0`` default) and
``train/thera/train_thera.py``. A copy that drifts changes the init of a model
built through one entry point and not another, with no error. Each entry point
now imports the layer's constant; this pins that the names are the layer's
object and that every default a user can reach resolves to it.
"""

import dataclasses
import inspect

import pytest

from dl_techniques.layers import thera_heat_field as heat_field
from dl_techniques.models.vision import thera as thera_package
from dl_techniques.models.vision.thera import hypernetwork, model
from train.thera import train_thera

CANONICAL = {
    "DEFAULT_COMPONENTS_INIT_SCALE": heat_field.DEFAULT_COMPONENTS_INIT_SCALE,
    "DEFAULT_K_INIT": heat_field.DEFAULT_K_INIT,
}


def test_the_canonical_values_are_the_reference_ones():
    assert heat_field.DEFAULT_COMPONENTS_INIT_SCALE == 16.0
    assert heat_field.DEFAULT_K_INIT == pytest.approx(0.0596482885568, rel=1e-9)


@pytest.mark.parametrize("name", ["DEFAULT_COMPONENTS_INIT_SCALE", "DEFAULT_K_INIT"])
@pytest.mark.parametrize("owner", [model, hypernetwork, train_thera],
                         ids=["model", "hypernetwork", "train_thera"])
def test_every_module_that_names_a_default_names_the_layers_object(owner, name):
    assert getattr(owner, name) is CANONICAL[name]


def test_the_package_re_exports_the_same_components_scale():
    assert (thera_package.DEFAULT_COMPONENTS_INIT_SCALE
            is heat_field.DEFAULT_COMPONENTS_INIT_SCALE)
    assert "DEFAULT_COMPONENTS_INIT_SCALE" in thera_package.__all__


def test_every_reachable_default_resolves_to_the_canonical_value():
    signature = inspect.signature(hypernetwork.TheraHypernetwork.__init__)
    assert (signature.parameters["components_init_scale"].default
            is heat_field.DEFAULT_COMPONENTS_INIT_SCALE)
    assert signature.parameters["k_init"].default is heat_field.DEFAULT_K_INIT
    assert (hypernetwork.TheraHypernetwork(hidden_dim=8).components_init_scale
            == heat_field.DEFAULT_COMPONENTS_INIT_SCALE)
    fields = {f.name: f.default for f in dataclasses.fields(train_thera.TheraConfig)}
    assert fields["components_init_scale"] is heat_field.DEFAULT_COMPONENTS_INIT_SCALE
    assert fields["k_init"] is heat_field.DEFAULT_K_INIT
    built = model.build_thera(size="air")
    assert built.components_init_scale == heat_field.DEFAULT_COMPONENTS_INIT_SCALE
    assert built.k_init == heat_field.DEFAULT_K_INIT
