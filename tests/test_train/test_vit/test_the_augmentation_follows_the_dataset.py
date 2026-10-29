"""``train_vit`` augments CIFAR with the shared ``cifar_light`` preset and ImageNet with the loader's crop.

Measured at HEAD: ``train_vit`` passed its CIFAR ``augment=`` map for every dataset, and the
shared loader refused a map for ImageNet, so ``--dataset imagenet`` with the default
``augment_data=True`` raised before the first step. The one ImageNet test passed
``augment_data=False`` and never saw it.
"""

import pytest

import train.vit.train_vit as train_vit
from dl_techniques.datasets.vision.augment_hook import is_seeded
from tests.imagenet_fake_tree import build_fake_imagenet, use_the_fake_tree_class_count


@pytest.mark.parametrize("dataset", ["cifar10", "cifar100"])
def test_cifar_gets_the_seeded_shared_preset(dataset):
    hook = train_vit._augment_hook(train_vit.TrainingConfig(dataset=dataset, num_classes=10))
    assert is_seeded(hook)


def test_no_augmentation_and_imagenet_get_no_hook(tmp_path):
    assert train_vit._augment_hook(
        train_vit.TrainingConfig(dataset="cifar10", num_classes=10, augment_data=False)) is None
    root = build_fake_imagenet(tmp_path / "imagenet2012")
    assert train_vit._augment_hook(train_vit.TrainingConfig(
        dataset="imagenet", imagenet_root=root, num_classes=1000)) is None


def test_imagenet_trains_with_the_default_augmentation(tmp_path, monkeypatch):
    root = build_fake_imagenet(tmp_path / "tree" / "imagenet2012")
    use_the_fake_tree_class_count(monkeypatch)
    monkeypatch.setattr(train_vit, "setup_gpu", lambda *a, **k: None)
    monkeypatch.setattr(train_vit, "create_callbacks", lambda *a, **k: ([], None))
    config = train_vit.TrainingConfig(
        dataset="imagenet", imagenet_root=root, image_size=32, patch_size=8, batch_size=4,
        model_variant="vit_pico", num_classes=3, epochs=1, warmup_epochs=0,
        augment_data=True, output_dir=str(tmp_path / "out"), experiment_name="aug",
        holdout_per_class=1, conformal_alpha=None)
    assert train_vit.train_vit(config)["total_epochs"] == 1
