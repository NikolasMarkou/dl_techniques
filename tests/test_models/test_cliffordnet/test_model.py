"""Lock-in tests for the iter-1 cliffordnet refactor.

See plans/plan_2026-05-11_0090b0b8/plan.md for context. These tests pin:
1. The new `create_cliffordnet` top-level factory returns a built model.
2. `pretrained=True` raises `NotImplementedError` (no silent fallback).
3. The trimmed 4-name public API of `dl_techniques.models.cliffordnet`.
"""

import pytest


class TestCliffordNetIter1Refactor:
    """Pin the iter-1 refactor contract."""

    def test_create_cliffordnet_factory_returns_instance(self):
        from dl_techniques.models.cliffordnet import (
            CliffordNet,
            create_cliffordnet,
        )

        model = create_cliffordnet(variant="nano", num_classes=10)
        assert isinstance(model, CliffordNet)

        # Build with a small input to verify the factory wires through.
        model.build((None, 32, 32, 3))
        assert model.built

    def test_pretrained_true_raises_not_implemented(self):
        from dl_techniques.models.cliffordnet import create_cliffordnet

        with pytest.raises(NotImplementedError):
            create_cliffordnet(
                variant="nano", num_classes=10, pretrained=True
            )

    def test_public_api_surface(self):
        import dl_techniques.models.cliffordnet as pkg

        # Surface expanded by plan_2026-05-12_632605aa to add the
        # CliffordNetEmbedding bidirectional U-Net embedding model + factories,
        # and by plan_2026-06-17_4b339fb7 to add the CliffordLaplacianUNet
        # Laplacian-pyramid autoencoder + factory.
        assert set(pkg.__all__) == {
            "CliffordNet",
            "create_cliffordnet",
            "CliffordCLIP",
            "CliffordNetLMRouting",
            "CliffordNetEmbedding",
            "create_cliffordnet_embedding",
            "create_cliffordnet_embedding_with_head",
            "CliffordLaplacianUNet",
            "create_clifford_laplacian_unet",
        }
