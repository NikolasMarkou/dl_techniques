"""Lock-in tests for the iter-1 cliffordnet refactor.

See plans/plan_2026-05-11_0090b0b8/plan.md for context. These tests pin:
1. The new `create_cliffordnet` top-level factory returns a built model.
2. `pretrained=True` raises `NotImplementedError` (no silent fallback).
3. The trimmed public API of `dl_techniques.models.cliffordnet` (2 names since
   plan-2026-08-10-3649c19e/iter-1/step-2 — see the comment on that test).
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

        # Surface REDUCED to 2 names by plan-2026-08-10-3649c19e
        # (iter-1/step-2, decisions.md D-005/D-006). Provenance of what used
        # to be here, and why each name left:
        #   - CliffordNetEmbedding / create_cliffordnet_embedding /
        #     create_cliffordnet_embedding_with_head — added by
        #     plan_2026-05-12_632605aa; REMOVED with embedding_unet.py, whose
        #     encoder is built out of CliffordNetBlockDSv2. The user declared
        #     the DSv2 block dead, so its whole consumer closure went with it.
        #   - CliffordLaplacianUNet / create_clifford_laplacian_unet — added by
        #     plan_2026-06-17_4b339fb7; the module backing them
        #     (cliffordnet/autoencoder.py) was already deleted before this plan
        #     started, so the name was already dangling.
        #   - CliffordNetLMRouting — lm_routing.py likewise already deleted.
        #   - CliffordCLIP is deliberately ABSENT: commit 6bc9b69b
        #     (plan-2026-07-15-776c737a/iter-1/step-1) git-mv'd it from
        #     cliffordnet/clip.py to clip/clifford_clip.py and dropped the
        #     re-export. Import it from dl_techniques.models.clip.clifford_clip,
        #     not from here.
        assert set(pkg.__all__) == {
            "CliffordNet",
            "create_cliffordnet",
        }
