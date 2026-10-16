"""Every functional ``keras.Model`` subclass under ``models/`` refuses an unknown kwarg.

Keras 3.8's functional ``Model.__init__`` drops a keyword it does not know, so a
misspelled argument (``dropout=0.3`` for ``dropout_rate``) built the default
model with no error. A SUBCLASSED ``keras.Model`` raises ``ValueError:
Unrecognized keyword arguments`` for the same call. ``refuse_unknown_model_kwargs``
(``utils/model_build.py``) gives the functional classes the subclassed refusal.

MEASURED at fb905f29b (plan-2026-10-14T062616-81938e81 step 12, D-077):
``bogus_kw=1`` was accepted silently by 8 of the 13 classes below (CoShNet,
DINOv1, DINOv2VisionTransformer, DINOv2 through its backbone, DINOv3,
YOLOv12MultiTask, VAE, KAN); FractalNet, SwinTransformer, SqueezeNetV1,
SqueezeNoduleNetV2 and FastVLM already refused (step 4 and step 5).

The row table is checked against an AST census of the tree, so a new
functional class with no row fails :func:`test_every_functional_model_has_a_row`.
"""

import ast
import os
from typing import Any, Callable, Dict, Set, Tuple

import keras
import pytest

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src", "dl_techniques", "models",
)


def _coshnet(**kw):
    from dl_techniques.models.vision.coshnet.model import CoShNet
    return CoShNet(num_classes=4, input_shape=(32, 32, 3), conv_filters=[4, 8],
                   dense_units=[8, 4], **kw)


def _fractalnet(**kw):
    from dl_techniques.models.vision.fractalnet.model import FractalNet
    return FractalNet(num_classes=4, depths=[1], filters=[8], strides=[1],
                      input_shape=(32, 32, 3), **kw)


def _dino_v1(**kw):
    from dl_techniques.models.vision.dino.dino_v1 import DINOv1
    return DINOv1(embed_dim=16, depth=1, num_heads=2, patch_size=14,
                  image_size=28, num_classes=4, **kw)


def _dino_v2_backbone(**kw):
    from dl_techniques.models.vision.dino.dino_v2 import DINOv2VisionTransformer
    return DINOv2VisionTransformer(image_size=28, patch_size=14, embed_dim=16,
                                   depth=1, num_heads=2, **kw)


def _dino_v2(**kw):
    # DINOv2 forwards every non-base keyword to its backbone, which refuses it.
    from dl_techniques.models.vision.dino.dino_v2 import DINOv2
    return DINOv2(image_size=28, patch_size=14, num_classes=4, embed_dim=16,
                  depth=1, num_heads=2, **kw)


def _dino_v3(**kw):
    from dl_techniques.models.vision.dino.dino_v3 import DINOv3
    return DINOv3(image_size=28, patch_size=14, num_classes=4, embed_dim=16,
                  depth=1, num_heads=2, **kw)


def _yolo12(**kw):
    from dl_techniques.models.vision.yolo12.multitask import YOLOv12MultiTask
    return YOLOv12MultiTask(num_detection_classes=4, input_shape=(64, 64, 3),
                            scale="n", **kw)


def _swin(**kw):
    from dl_techniques.models.vision.swin_transformer.model import SwinTransformer
    return SwinTransformer(num_classes=4, input_shape=(32, 32, 3), embed_dim=16,
                           depths=[1, 1, 1, 1], num_heads=[2, 2, 2, 2],
                           window_size=2, patch_size=4, **kw)


def _vae(**kw):
    from dl_techniques.models.vision.vae.model import VAE
    return VAE(latent_dim=4, input_shape=(32, 32, 1), depths=1,
               steps_per_depth=1, filters=[8], **kw)


def _squeezenet_v1(**kw):
    from dl_techniques.models.vision.squeezenet.squeezenet_v1 import SqueezeNetV1
    return SqueezeNetV1.from_variant("1.1", num_classes=4,
                                     input_shape=(64, 64, 3), **kw)


def _squeezenet_v2(**kw):
    from dl_techniques.models.vision.squeezenet.squeezenet_v2 import SqueezeNoduleNetV2
    return SqueezeNoduleNetV2(num_classes=4, input_shape=(64, 64, 3), **kw)


def _kan(**kw):
    from dl_techniques.models.general_purpose.kan.model import KAN
    return KAN(layer_configs=[{"features": 8, "grid_size": 5, "activation": "swish"},
                              {"features": 4, "grid_size": 4, "activation": "linear"}],
               input_features=6, **kw)


def _fastvlm(**kw):
    from dl_techniques.models.vision_language.fastvlm.model import FastVLM
    return FastVLM(num_classes=4, **kw)


#: ``(relpath under models/, class name) -> builder(**kwargs)``. One row per
#: functional ``keras.Model`` subclass the census below finds.
ROWS: Dict[Tuple[str, str], Callable[..., Any]] = {
    ("vision/coshnet/model.py", "CoShNet"): _coshnet,
    ("vision/fractalnet/model.py", "FractalNet"): _fractalnet,
    ("vision/dino/dino_v1.py", "DINOv1"): _dino_v1,
    ("vision/dino/dino_v2.py", "DINOv2VisionTransformer"): _dino_v2_backbone,
    ("vision/dino/dino_v2.py", "DINOv2"): _dino_v2,
    ("vision/dino/dino_v3.py", "DINOv3"): _dino_v3,
    ("vision/yolo12/multitask.py", "YOLOv12MultiTask"): _yolo12,
    ("vision/swin_transformer/model.py", "SwinTransformer"): _swin,
    ("vision/vae/model.py", "VAE"): _vae,
    ("vision/squeezenet/squeezenet_v1.py", "SqueezeNetV1"): _squeezenet_v1,
    ("vision/squeezenet/squeezenet_v2.py", "SqueezeNoduleNetV2"): _squeezenet_v2,
    ("general_purpose/kan/model.py", "KAN"): _kan,
    ("vision_language/fastvlm/model.py", "FastVLM"): _fastvlm,
}


def _functional_model_classes() -> Set[Tuple[str, str]]:
    """Every class whose ``__init__`` calls ``super().__init__(inputs=..., outputs=...)``.

    :return: ``(relpath under models/, class name)`` pairs.
    """
    found: Set[Tuple[str, str]] = set()
    for dirpath, dirnames, filenames in os.walk(MODELS_DIR):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            with open(path, encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
            for cls in ast.walk(tree):
                if not isinstance(cls, ast.ClassDef):
                    continue
                for member in cls.body:
                    if not (isinstance(member, ast.FunctionDef)
                            and member.name == "__init__"):
                        continue
                    for call in ast.walk(member):
                        if not (isinstance(call, ast.Call)
                                and isinstance(call.func, ast.Attribute)
                                and call.func.attr == "__init__"
                                and isinstance(call.func.value, ast.Call)
                                and getattr(call.func.value.func, "id", "") == "super"):
                            continue
                        keywords = {k.arg for k in call.keywords}
                        if {"inputs", "outputs"} & keywords:
                            found.add((os.path.relpath(path, MODELS_DIR), cls.name))
    return found


def test_every_functional_model_has_a_row():
    census = _functional_model_classes()
    assert len(census) >= 13, f"the census found only {len(census)} classes"
    assert census == set(ROWS), (
        f"functional classes without a row: {sorted(census - set(ROWS))}; "
        f"rows naming no functional class: {sorted(set(ROWS) - census)}"
    )


@pytest.mark.parametrize("row", sorted(ROWS), ids=[name for _, name in sorted(ROWS)])
def test_an_unknown_kwarg_is_refused(row):
    keras.utils.set_random_seed(0)
    with pytest.raises(ValueError, match="Unrecognized keyword arguments.*bogus_kw"):
        ROWS[row](bogus_kw=1)


@pytest.mark.parametrize("row", sorted(ROWS), ids=[name for _, name in sorted(ROWS)])
def test_the_base_keywords_still_pass(row):
    """``name`` and ``trainable`` are applied, never refused."""
    keras.utils.set_random_seed(0)
    model = ROWS[row](name="refusal_probe", trainable=False)
    assert model.name == "refusal_probe"
    assert model.trainable is False
