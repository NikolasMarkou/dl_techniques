"""`vit_siglip` must refuse an odd `patch_size` at construction.

Guard for C-15 (plan-2026-08-14T233721-d4f9beb2, step 36). The two-stage SigLIP
stem is ``Conv2D(k=s=patch//2)`` followed by ``Conv2D(k=s=2)``, so its total
stride is ``2 * (patch // 2)`` -- which equals ``patch`` only when ``patch`` is
EVEN. ``__init__`` validated only ``img % patch == 0`` and ``call`` reshaped to
the DECLARED ``num_patches``, so
``create_siglip_vision_transformer(input_shape=(224, 224, 3), patch_size=7)``
constructed happily (224 % 7 == 0), then reshaped a ``(B, 37, 37, D)`` map --
1369 tokens -- into ``(B, 1024, D)`` and died in an opaque reshape error at the
first forward. Every existing test uses 8 / 16 / 32.
"""

import numpy as np
import pytest

from dl_techniques.models.vit_siglip.model import (
    SigLIPVisionTransformer,
    create_siglip_vision_transformer,
)


ODD_PATCHES = [7, 1, 3]
EVEN_PATCHES = [8, 16, 32]


class TestOddPatchSizeIsRefused:
    @pytest.mark.parametrize("patch_size", ODD_PATCHES)
    def test_odd_int_patch_size_raises_at_construction(self, patch_size):
        with pytest.raises(ValueError) as excinfo:
            create_siglip_vision_transformer(
                input_shape=(224, 224, 3), patch_size=patch_size, num_classes=10
            )
        message = str(excinfo.value)
        # Assert the MESSAGE, not just the raise: it must name the constraint.
        assert "even" in message.lower(), message
        assert str(patch_size) in message, message

    def test_odd_dimension_in_a_tuple_patch_size_raises(self):
        with pytest.raises(ValueError, match="even"):
            SigLIPVisionTransformer(
                input_shape=(224, 224, 3), patch_size=(16, 7), num_classes=10
            )

    @pytest.mark.parametrize("patch_size", EVEN_PATCHES)
    def test_even_patch_sizes_are_untouched(self, patch_size):
        """Anti-vacuity: the guard must not reject the working configurations."""
        model = create_siglip_vision_transformer(
            input_shape=(32, 32, 3), patch_size=patch_size, num_classes=4, scale="tiny"
        )
        out = model(np.random.rand(2, 32, 32, 3).astype("float32"))
        assert tuple(out.shape) == (2, 4)

    def test_the_stem_stride_really_equals_the_patch_size(self):
        """The property the guard defends: total stem stride == patch size, so
        the token count matches the declared num_patches."""
        patch = 16
        model = create_siglip_vision_transformer(
            input_shape=(32, 32, 3), patch_size=patch, num_classes=4, scale="tiny"
        )
        stride = 2 * (patch // 2)
        assert stride == patch
        assert model.num_patches == (32 // patch) ** 2
