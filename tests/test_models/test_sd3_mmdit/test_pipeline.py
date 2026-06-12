"""Tiny end-to-end integration test for the SD3 inference pipeline (step 10).

Exercises the orchestration of the four step-6..9 components through
``SD3Pipeline``: prompt-feature assembly, the Euler denoise loop, and the VAE
decode. Kept tiny (``"tiny"`` preset, B=1, 3 steps) so it runs in the scoped
suite; marked ``integration`` because it touches every SD3 component at once.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.sd3_mmdit.config import get_sd3_config
from dl_techniques.models.sd3_mmdit.pipeline import (
    SD3Pipeline,
    assemble_prompt_features,
    create_sd3_pipeline,
)
from dl_techniques.models.sd3_mmdit.text_encoders import (
    CLIPTextEncoder,
    OpenCLIPTextEncoder,
    T5Encoder,
)


# ---------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_pipeline() -> SD3Pipeline:
    """A tiny SD3 pipeline (built once per module)."""
    return create_sd3_pipeline("tiny", seed=0)


def _ids(batch: int, length: int, vocab: int) -> np.ndarray:
    """Random integer token ids in ``[0, vocab)`` of shape ``(batch, length)``."""
    rng = np.random.default_rng(1234)
    return rng.integers(0, vocab, size=(batch, length)).astype("int32")


# ---------------------------------------------------------------------
# (1) build + dim contract
# ---------------------------------------------------------------------


@pytest.mark.integration
class TestPipelineBuild:
    def test_create_tiny_builds(self, tiny_pipeline: SD3Pipeline) -> None:
        config, _ = get_sd3_config("tiny")
        # Dim contract holds (SD3Pipeline.__init__ would have raised otherwise).
        assert tiny_pipeline.t5.embed_dim == config.joint_attention_dim
        assert (
            tiny_pipeline.clip.embed_dim + tiny_pipeline.openclip.embed_dim
            == config.pooled_projection_dim
        )

    def test_dim_mismatch_raises(self) -> None:
        """A T5 whose embed_dim != joint_attention_dim must fail loud."""
        config, _ = get_sd3_config("tiny")
        from dl_techniques.models.sd3_mmdit.transformer import create_sd3_mmdit
        from dl_techniques.models.sd3_mmdit.vae import create_sd3_vae
        from dl_techniques.models.sd3_mmdit.scheduler import (
            FlowMatchEulerScheduler,
        )

        # T5 with the WRONG width (joint_attention_dim is 512 for tiny).
        bad_t5 = T5Encoder(
            vocab_size=512, embed_dim=256, num_layers=1, num_heads=4, ff_dim=128
        )
        clip = CLIPTextEncoder(
            vocab_size=512, embed_dim=128, num_layers=1, num_heads=4, max_seq_len=32
        )
        openclip = OpenCLIPTextEncoder(
            vocab_size=512, embed_dim=128, num_layers=1, num_heads=4, max_seq_len=32
        )
        with pytest.raises(ValueError, match="joint_attention_dim"):
            SD3Pipeline(
                transformer=create_sd3_mmdit("tiny"),
                vae=create_sd3_vae(variant="tiny"),
                clip=clip,
                openclip=openclip,
                t5=bad_t5,
                scheduler=FlowMatchEulerScheduler(),
            )

    def test_pooled_mismatch_raises(self) -> None:
        """CLIP+OpenCLIP pooled width != pooled_projection_dim must fail loud."""
        from dl_techniques.models.sd3_mmdit.transformer import create_sd3_mmdit
        from dl_techniques.models.sd3_mmdit.vae import create_sd3_vae
        from dl_techniques.models.sd3_mmdit.scheduler import (
            FlowMatchEulerScheduler,
        )

        t5 = T5Encoder(
            vocab_size=512, embed_dim=512, num_layers=1, num_heads=4, ff_dim=128
        )
        # 128 + 64 = 192 != pooled_projection_dim (256).
        clip = CLIPTextEncoder(
            vocab_size=512, embed_dim=128, num_layers=1, num_heads=4, max_seq_len=32
        )
        openclip = OpenCLIPTextEncoder(
            vocab_size=512, embed_dim=64, num_layers=1, num_heads=4, max_seq_len=32
        )
        with pytest.raises(ValueError, match="pooled_projection_dim"):
            SD3Pipeline(
                transformer=create_sd3_mmdit("tiny"),
                vae=create_sd3_vae(variant="tiny"),
                clip=clip,
                openclip=openclip,
                t5=t5,
                scheduler=FlowMatchEulerScheduler(),
            )


# ---------------------------------------------------------------------
# (3) assemble_prompt_features dims
# ---------------------------------------------------------------------


@pytest.mark.integration
class TestAssemblePromptFeatures:
    def test_feature_and_pooled_dims(self) -> None:
        config, _ = get_sd3_config("tiny")
        B, L_clip, L_t5 = 2, 7, 5
        clip_dim, openclip_dim, t5_dim = 128, 128, 512

        clip_out = {
            "pooled": ops.zeros((B, clip_dim)),
            "last_hidden": ops.zeros((B, L_clip, clip_dim)),
            "penultimate": ops.ones((B, L_clip, clip_dim)),
        }
        openclip_out = {
            "pooled": ops.zeros((B, openclip_dim)),
            "last_hidden": ops.zeros((B, L_clip, openclip_dim)),
            "penultimate": ops.ones((B, L_clip, openclip_dim)),
        }
        t5_out = ops.ones((B, L_t5, t5_dim))

        ehs, pooled = assemble_prompt_features(clip_out, openclip_out, t5_out)

        # encoder_hidden_states: (B, L_clip + L_t5, joint_attention_dim).
        assert tuple(ehs.shape) == (B, L_clip + L_t5, config.joint_attention_dim)
        assert ehs.shape[-1] == config.joint_attention_dim
        # pooled: (B, pooled_projection_dim).
        assert tuple(pooled.shape) == (B, config.pooled_projection_dim)
        assert pooled.shape[-1] == config.pooled_projection_dim

    def test_clip_context_zero_pad(self) -> None:
        """The CLIP context is zero-padded on the feature axis up to t5_dim."""
        B, L_clip, L_t5 = 1, 3, 4
        clip_dim, openclip_dim, t5_dim = 128, 128, 512
        clip_out = {
            "pooled": ops.zeros((B, clip_dim)),
            "last_hidden": ops.zeros((B, L_clip, clip_dim)),
            "penultimate": ops.ones((B, L_clip, clip_dim)),
        }
        openclip_out = {
            "pooled": ops.zeros((B, openclip_dim)),
            "last_hidden": ops.zeros((B, L_clip, openclip_dim)),
            "penultimate": ops.ones((B, L_clip, openclip_dim)),
        }
        t5_out = ops.zeros((B, L_t5, t5_dim))
        ehs, _ = assemble_prompt_features(clip_out, openclip_out, t5_out)
        arr = keras.ops.convert_to_numpy(ehs)
        # First L_clip rows: first 256 feats are ones (CLIP+OpenCLIP), rest pad 0.
        assert np.allclose(arr[:, :L_clip, : clip_dim + openclip_dim], 1.0)
        assert np.allclose(arr[:, :L_clip, clip_dim + openclip_dim:], 0.0)

    def test_overwide_clip_raises(self) -> None:
        """CLIP+OpenCLIP wider than T5 must fail loud."""
        B, L_clip, L_t5 = 1, 2, 2
        clip_out = {
            "pooled": ops.zeros((B, 400)),
            "last_hidden": ops.zeros((B, L_clip, 400)),
            "penultimate": ops.ones((B, L_clip, 400)),
        }
        openclip_out = {
            "pooled": ops.zeros((B, 400)),
            "last_hidden": ops.zeros((B, L_clip, 400)),
            "penultimate": ops.ones((B, L_clip, 400)),
        }
        t5_out = ops.ones((B, L_t5, 512))  # 400+400=800 > 512
        with pytest.raises(ValueError, match="must be <="):
            assemble_prompt_features(clip_out, openclip_out, t5_out)


# ---------------------------------------------------------------------
# (2) end-to-end generate
# ---------------------------------------------------------------------


@pytest.mark.integration
class TestGenerate:
    def test_generate_shape_and_no_nan(self, tiny_pipeline: SD3Pipeline) -> None:
        config, ae = get_sd3_config("tiny")
        B = 1
        L_clip, L_t5 = 8, 6

        clip_ids = _ids(B, L_clip, vocab=512)
        openclip_ids = _ids(B, L_clip, vocab=512)
        t5_ids = _ids(B, L_t5, vocab=512)

        image = tiny_pipeline.generate(
            clip_token_ids=clip_ids,
            openclip_token_ids=openclip_ids,
            t5_token_ids=t5_ids,
            num_inference_steps=3,
            seed=42,
        )

        # VAE upsample = 2 ** (len(ch_mult) - 1); tiny ch_mult=(1,2) -> 2.
        upsample = 2 ** (len(ae.ch_mult) - 1)
        expected_hw = config.sample_size * upsample  # 16 * 2 = 32
        assert tuple(image.shape) == (B, expected_hw, expected_hw, ae.out_ch)
        assert tuple(image.shape) == (1, 32, 32, 3)

        arr = keras.ops.convert_to_numpy(image)
        assert not np.any(np.isnan(arr)), "decoded image contains NaN"
        assert np.all(np.isfinite(arr)), "decoded image contains non-finite values"

    def test_generate_deterministic_with_seed(
        self, tiny_pipeline: SD3Pipeline
    ) -> None:
        """Same seed + same ids -> identical output (deterministic latent)."""
        B, L_clip, L_t5 = 1, 5, 4
        clip_ids = _ids(B, L_clip, vocab=512)
        openclip_ids = _ids(B, L_clip, vocab=512)
        t5_ids = _ids(B, L_t5, vocab=512)
        kw = dict(
            clip_token_ids=clip_ids,
            openclip_token_ids=openclip_ids,
            t5_token_ids=t5_ids,
            num_inference_steps=2,
            seed=7,
        )
        img_a = keras.ops.convert_to_numpy(tiny_pipeline.generate(**kw))
        img_b = keras.ops.convert_to_numpy(tiny_pipeline.generate(**kw))
        assert np.allclose(img_a, img_b, atol=1e-5)
