"""
Subjects for the R-088 / R-141 precision arm -- one builder per model package
============================================================================

This module is an *instrument*, not a test suite: like ``precision_arm_oracle``
beside it, it carries no ``test_`` prefix so pytest does not collect it. The
tests that consume it live in ``test_precision_arm_family.py``.

Why one registry instead of 51 files
------------------------------------
Rules R-088 ("a mixed-precision arm exists") and R-141 ("that arm has all four
parts") were charged against ~55 ``models/`` test directories. The *assertions*
are identical for every one of them and already live once, in
``precision_arm_oracle.assert_precision_arm``. What differs per package is only
two things: how to build a small instance, and what to feed it. That is what
this file holds -- a name to ``(build, make_inputs, kwargs)`` mapping -- so the
family is parameterized over a table rather than copied 51 times.

The completeness of the table is itself asserted (see
``test_precision_arm_family.py::test_every_charged_package_has_a_subject``)
against :data:`CHARGED_PACKAGES`, so a package cannot silently drop out of the
family by someone deleting a dict entry.

Sizing rule
-----------
Every subject is built at the smallest geometry that still exercises the
package's own code path. These are dtype guards, not capacity tests: a wider
model measures the same float16 behaviour and costs minutes.

Per-subject deviations
----------------------
``kwargs`` is forwarded verbatim to :func:`assert_precision_arm`. Any deviation
from the default four-part arm (``check_backward=False``,
``allowed_none_grads``, a relaxed ``expected_compute_dtype``) must carry a
comment at the entry stating what was MEASURED, never a guess.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np


__all__ = ["CHARGED_PACKAGES", "SUBJECTS", "subject_names",
           "XLA_OVERRIDES", "xla_subject",
           "FLOAT64_CHARGED", "FLOAT64_OVERRIDES", "float64_subject",
           "EXTRA_SUBJECTS", "ROUNDTRIP_NA", "roundtrip_charged",
           "roundtrip_subject", "NO_HEAD_SIBLINGS"]


#: Every package the audit charged with R-088 / R-141 and that step 18 left
#: open. ``coshnet``, ``darkir``, ``fftnet`` and ``pw_fnet`` are absent on
#: purpose: step 18 gave each its own ``test_precision_arm.py`` because each
#: was a REAL fp16 defect with a package-specific before/after to record.
CHARGED_PACKAGES: Tuple[str, ...] = (
    "SAM", "accunet", "beit", "bert", "bias_free_denoisers",
    "byte_latent_transformer", "capsnet", "cbam", "cliffordnet", "convnext",
    "convunext", "depth_anything", "fastvit", "fnet", "fractalnet", "gemma",
    "gpt2", "hierarchical_reasoning_model", "ideogram4", "lewm",
    "masked_autoencoder", "masked_language_model", "memory_bank",
    "mini_vec2vec", "mobilenet", "modern_bert", "mothnet", "nano_vlm", "ntm",
    "pft_sr", "power_mlp", "qwen", "relgt", "resnet", "scunet", "sd3_mmdit",
    "shgcn", "som", "squeezenet", "superpoint", "swin_transformer", "tabm",
    "time_series", "tiny_recursive_model", "vae", "video_jepa", "vit",
    "vit_hmlp", "vit_siglip", "vq_vae", "vq_vae_rotation", "wave_field",
    "yolo12",
)


def _f32(*shape: int, seed: int = 0) -> np.ndarray:
    """A deterministic float32 draw of the given shape."""
    return np.random.RandomState(seed).randn(*shape).astype("float32")


def _ids(vocab: int, *shape: int, seed: int = 0) -> np.ndarray:
    """A deterministic int32 token draw in ``[0, vocab)``."""
    return np.random.RandomState(seed).randint(0, vocab, shape).astype("int32")


SUBJECTS: Dict[str, Tuple[Callable[[], Any], Callable[[], Any], Dict[str, Any]]] = {}


def _sub(name: str, build, make_inputs, **kwargs: Any) -> None:
    SUBJECTS[name] = (build, make_inputs, kwargs)


def subject_names() -> Tuple[str, ...]:
    """Registered subject names, sorted -- the parameterization order."""
    return tuple(sorted(SUBJECTS))


# ---------------------------------------------------------------------------
# Vision backbones
# ---------------------------------------------------------------------------

def _b_convnext():
    # The CLASS, not ``create_convnext_v1``: the factory resolves ``depths`` /
    # ``dims`` from the variant table and then forwards ``**kwargs``, so an
    # override of either raises "got multiple values for keyword argument".
    from dl_techniques.models.convnext import ConvNeXtV1
    return ConvNeXtV1(num_classes=4, depths=[1, 1], dims=[8, 16])


_sub("convnext", _b_convnext, lambda: _f32(1, 32, 32, 3))


def _b_resnet():
    # The CLASS, for the same variant-table reason as ``convnext`` above.
    from dl_techniques.models.resnet import ResNet
    return ResNet(num_classes=4, blocks_per_stage=[1, 1],
                  filters_per_stage=[8, 16], block_type="basic")


_sub("resnet", _b_resnet, lambda: _f32(1, 32, 32, 3))


def _b_vit():
    from dl_techniques.models.vit import create_vit
    return create_vit(variant="vit_tiny", num_classes=4,
                      input_shape=(32, 32, 3), patch_size=16)


_sub("vit", _b_vit, lambda: _f32(1, 32, 32, 3))


def _b_vit_hmlp():
    from dl_techniques.models.vit_hmlp import create_vit_hmlp
    return create_vit_hmlp(input_shape=(32, 32, 3), num_classes=4,
                           scale="tiny", patch_size=16)


_sub("vit_hmlp", _b_vit_hmlp, lambda: _f32(1, 32, 32, 3))


def _b_vit_siglip():
    from dl_techniques.models.vit_siglip import create_siglip_vision_transformer
    return create_siglip_vision_transformer(
        input_shape=(32, 32, 3), num_classes=4, scale="tiny", patch_size=16)


_sub("vit_siglip", _b_vit_siglip, lambda: _f32(1, 32, 32, 3))


def _b_beit():
    from dl_techniques.models.beit import create_beit_classifier
    return create_beit_classifier("tiny", (32, 32, 3), 16, num_classes=4)


# ``allowed_none_grads=1``: MEASURED IDENTICAL under float32 -- 1 of 224 in
# BOTH arms (fp16 ``grad_norm_sum`` 7.594397e+00, float32 7.584365e+00).
_sub("beit", _b_beit, lambda: _f32(1, 32, 32, 3), allowed_none_grads=1)


def _b_fastvit():
    from dl_techniques.models.fastvit import create_fastvit_image_encoder
    return create_fastvit_image_encoder("mci0", input_shape=(64, 64, 3),
                                        projection_dim=32)


_sub("fastvit", _b_fastvit, lambda: _f32(1, 64, 64, 3))


def _b_mobilenet():
    from dl_techniques.models.mobilenet import create_mobilenetv2
    return create_mobilenetv2(variant="small", num_classes=4,
                              input_shape=(32, 32, 3))


_sub("mobilenet", _b_mobilenet, lambda: _f32(1, 32, 32, 3))


def _b_swin():
    from dl_techniques.models.swin_transformer import create_swin_transformer
    return create_swin_transformer("tiny", 4, input_shape=(32, 32, 3))


_sub("swin_transformer", _b_swin, lambda: _f32(1, 32, 32, 3))


def _b_squeezenet():
    from dl_techniques.models.squeezenet import create_squeezenet_v1
    return create_squeezenet_v1("1.0", num_classes=4, input_shape=(64, 64, 3))


_sub("squeezenet", _b_squeezenet, lambda: _f32(1, 64, 64, 3))


def _b_cbam():
    from dl_techniques.models.cbam import create_cbam_net
    return create_cbam_net("tiny", num_classes=4, input_shape=(32, 32, 3))


_sub("cbam", _b_cbam, lambda: _f32(1, 32, 32, 3))


def _b_fractalnet():
    from dl_techniques.models.fractalnet import create_fractal_net
    return create_fractal_net(variant="micro", num_classes=4,
                              input_shape=(32, 32, 3))


_sub("fractalnet", _b_fractalnet, lambda: _f32(1, 32, 32, 3))


def _b_capsnet():
    from dl_techniques.models.capsnet import create_capsnet
    return create_capsnet(num_classes=4, input_shape=(28, 28, 1))


_sub("capsnet", _b_capsnet, lambda: _f32(1, 28, 28, 1))


def _b_cliffordnet():
    from dl_techniques.models.cliffordnet import create_cliffordnet
    return create_cliffordnet(variant="nano", num_classes=4)


_sub("cliffordnet", _b_cliffordnet, lambda: _f32(1, 32, 32, 3))


def _b_superpoint():
    from dl_techniques.models.superpoint import create_superpoint
    return create_superpoint("tiny", input_shape=(64, 64, 1))


_sub("superpoint", _b_superpoint, lambda: _f32(1, 64, 64, 1))


def _b_yolo12():
    from dl_techniques.models.yolo12 import create_yolov12_multitask
    return create_yolov12_multitask(num_detection_classes=4, tasks=["detection"],
                                    input_shape=(64, 64, 3), scale="n")


# ``forward_training=True``: this package is the reason that parameter exists.
# Judged on the INFERENCE path at initialization, ``yolo12`` reports NaN in
# 5712 of 5712 outputs under ``mixed_float16`` -- entirely because its
# UNTRAINED BatchNorms leave the float32 activations at ``absmax``
# 2.997772e+08, three orders above float16's 65504 ceiling. In training mode
# BN uses the batch statistics and the same model measures ``absmax`` 4.703125
# fp16 / 4.644949 float32, both clean. See D-065.
_sub("yolo12", _b_yolo12, lambda: _f32(1, 64, 64, 3), forward_training=True)


# ---------------------------------------------------------------------------
# Dense prediction / restoration
# ---------------------------------------------------------------------------

def _b_bias_free_denoisers():
    from dl_techniques.models.bias_free_denoisers import create_convunext_variant
    return create_convunext_variant("tiny", (32, 32, 1),
                                    enable_deep_supervision=False)


_sub("bias_free_denoisers", _b_bias_free_denoisers, lambda: _f32(1, 32, 32, 1))


def _b_convunext():
    from dl_techniques.models.convunext import create_convunext_variant
    return create_convunext_variant("tiny", input_shape=(32, 32, 3))


_sub("convunext", _b_convunext, lambda: _f32(1, 32, 32, 3))


def _b_scunet():
    from dl_techniques.models.scunet import SCUNet
    return SCUNet(in_nc=3, config=[1, 1, 1, 1, 1, 1, 1], dim=8, head_dim=4,
                  window_size=4, input_resolution=64)


_sub("scunet", _b_scunet, lambda: _f32(1, 64, 64, 3))


def _b_accunet():
    from dl_techniques.models.accunet import create_acc_unet
    return create_acc_unet(input_channels=3, num_classes=1, base_filters=8,
                           input_shape=(32, 32))


_sub("accunet", _b_accunet, lambda: _f32(1, 32, 32, 3))


def _b_pft_sr():
    from dl_techniques.models.pft_sr import create_pft_sr
    return create_pft_sr(scale=2, variant="light")


_sub("pft_sr", _b_pft_sr, lambda: _f32(1, 32, 32, 3))


def _b_depth_anything():
    from dl_techniques.models.depth_anything import create_depth_anything
    return create_depth_anything(encoder_type="vit_s", image_shape=(64, 64, 3),
                                 encoder_kind="placeholder",
                                 decoder_dims=[16, 16, 16, 16])


_sub("depth_anything", _b_depth_anything, lambda: _f32(1, 64, 64, 3))


# ---------------------------------------------------------------------------
# Language / sequence
# ---------------------------------------------------------------------------

def _b_bert():
    from dl_techniques.models.bert import create_bert
    return create_bert("tiny", vocab_size=64, max_position_embeddings=32,
                       hidden_size=32, num_layers=1, num_heads=2,
                       intermediate_size=64)


_sub("bert", _b_bert, lambda: _ids(64, 2, 16))


def _b_fnet():
    from dl_techniques.models.fnet import FNet
    return FNet(vocab_size=64, hidden_size=32, num_layers=1,
                intermediate_size=64, max_position_embeddings=32)


_sub("fnet", _b_fnet, lambda: _ids(64, 2, 16))


def _b_modern_bert():
    from dl_techniques.models.modern_bert import ModernBERT
    return ModernBERT(vocab_size=64, hidden_size=32, num_layers=1, num_heads=2,
                      intermediate_size=64, max_position_embeddings=32,
                      local_attention_window_size=8)


_sub("modern_bert", _b_modern_bert, lambda: _ids(64, 2, 16))


def _b_gpt2():
    from dl_techniques.models.gpt2 import GPT2
    return GPT2(vocab_size=64, embed_dim=32, depth=1, num_heads=2,
                max_seq_len=32)


_sub("gpt2", _b_gpt2, lambda: _ids(64, 2, 16))


def _b_gemma():
    # ``create_gemma3`` returns a FUNCTIONAL wrapper that expects two input
    # tensors; the subject here is the ``Gemma3`` backbone itself.
    from dl_techniques.models.gemma import Gemma3
    return Gemma3(vocab_size=64, hidden_size=32, num_layers=1,
                  num_attention_heads=2, num_key_value_heads=1,
                  ffn_hidden_size=64, max_seq_len=32, sliding_window_size=8)


# ``_ids(64, ...)``, not ``_ids(256, ...)``: this subject's ``vocab_size`` is 64.
#
# DECISION plan-2026-08-19T163559-499b6f0e/D-076
# The original ``_ids(256, 2, 16)`` fed token ids up to 255 to a 64-row
# embedding table. Out-of-bounds gather is NOT an error on either backend, and
# eager-GPU and XLA resolve it DIFFERENTLY, so the R-143 arm read a spurious
# eager-vs-XLA divergence of 3.238814e-01 against an ``absmax`` of 3.200422e-01
# -- a "defect" larger than the signal. With in-range ids the same measurement
# is 3.565848e-04, a factor of ~900. The fp16 arm was gathering the same
# garbage. Keep every ``_ids`` bound equal to its subject's ``vocab_size``.
# See decisions.md D-076.
_sub("gemma", _b_gemma, lambda: _ids(64, 2, 16))


def _b_qwen():
    from dl_techniques.models.qwen import Qwen3Next
    return Qwen3Next(vocab_size=64, hidden_size=32, num_layers=1,
                     num_attention_heads=2, num_key_value_heads=1,
                     max_seq_len=32, num_experts=2, num_experts_per_tok=1,
                     moe_intermediate_size=32)


# DECISION plan-2026-08-19T163559-499b6f0e/D-076
# ``_ids(64, ...)``: same correction as ``gemma``. MEASURED eager-vs-XLA
# 4.184644e-01 with out-of-range ids against 4.322007e-04 with in-range ones.
# Every ``_ids`` bound must equal its subject's ``vocab_size``.
_sub("qwen", _b_qwen, lambda: _ids(64, 2, 16))


def _b_blt():
    from dl_techniques.models.byte_latent_transformer import create_blt_model
    return create_blt_model("micro", vocab_size=260, max_sequence_length=32)


# ``allowed_none_grads=54``: MEASURED IDENTICAL under float32 -- 54 of 254 in
# BOTH arms (fp16 ``grad_norm_sum`` 1.836834e+01, float32 1.848275e+01). BLT's
# entropy model and its patching branch are not reached by a plain forward.
_sub("byte_latent_transformer", _b_blt, lambda: _ids(256, 2, 32),
     allowed_none_grads=54)


def _b_wave_field():
    from dl_techniques.models.wave_field import create_wave_field_llm
    return create_wave_field_llm("small", vocab_size=64)


_sub("wave_field", _b_wave_field, lambda: _ids(64, 2, 16))


def _b_memory_bank():
    from dl_techniques.models.memory_bank import WaveFieldMemoryLLM
    return WaveFieldMemoryLLM(
        vocab_size=64, embed_dim=32, depth=4, num_heads=2, max_seq_len=16,
        d_k=8, d_v=16, s_lt=64, top_k=4, infonce_negatives=8,
        diversity_subsample=16,
    )


# ``allowed_none_grads=1``: MEASURED IDENTICAL under float32 -- 1 of 92 in
# BOTH arms (fp16 ``grad_norm_sum`` 5.537149e-01, float32 5.536436e-01).
_sub("memory_bank", _b_memory_bank, lambda: _ids(64, 2, 16),
     allowed_none_grads=1)


def _b_masked_language_model():
    from dl_techniques.models.bert import BERT
    from dl_techniques.models.masked_language_model import MaskedLanguageModel
    encoder = BERT(vocab_size=64, hidden_size=32, num_layers=1, num_heads=2,
                   intermediate_size=64, max_position_embeddings=32,
                   hidden_dropout_rate=0.0, attention_probs_dropout_rate=0.0)
    return MaskedLanguageModel(encoder=encoder, vocab_size=64, mask_token_id=3)


_sub("masked_language_model", _b_masked_language_model, lambda: _ids(64, 2, 16))


def _b_hrm():
    from dl_techniques.models.hierarchical_reasoning_model import (
        create_hierarchical_reasoning_model,
    )
    return create_hierarchical_reasoning_model(
        vocab_size=64, seq_len=16, variant="micro")


_sub("hierarchical_reasoning_model", _b_hrm,
     lambda: {"token_ids": _ids(64, 2, 16),
              "puzzle_ids": _ids(1000, 2, seed=1)})


def _b_trm():
    from dl_techniques.models.tiny_recursive_model import create_trm
    return create_trm(vocab_size=64, hidden_size=32, num_heads=2,
                      expansion=2.0, seq_len=16, puzzle_emb_len=4,
                      h_layers=1, l_layers=1, halt_max_steps=2)


def _trm_batch():
    return {"inputs": _ids(64, 2, 16),
            "puzzle_identifiers": _ids(1, 2, seed=1),
            "labels": _ids(64, 2, 16, seed=2)}


def _trm_call(model, inputs, training):
    """``TRM.call`` is ``(carry, batch, training)`` and returns ``(carry, outputs)``.

    Only ``outputs`` is judged: ``carry`` is ACT loop STATE, and its integer
    step counters and boolean halt flags are not model outputs at all.
    """
    carry = model.initial_carry(inputs)
    _new_carry, outputs = model(carry, inputs, training=training)
    return outputs


_sub("tiny_recursive_model", _b_trm, _trm_batch, call_fn=_trm_call)


# ---------------------------------------------------------------------------
# Generative
# ---------------------------------------------------------------------------

def _b_vae():
    from dl_techniques.models.vae import create_vae
    return create_vae(input_shape=(32, 32, 1), latent_dim=8, variant="small")


_sub("vae", _b_vae, lambda: _f32(1, 32, 32, 1))


def _tiny_conv_codec(latent: int = 8):
    """A 2-conv encoder / 2-deconv decoder pair, the smallest VQ-VAE harness."""
    import keras
    encoder = keras.Sequential([
        keras.layers.Input((32, 32, 1)),
        keras.layers.Conv2D(16, 3, strides=2, padding="same",
                            activation="relu", name="enc_conv_1"),
        keras.layers.Conv2D(latent, 3, strides=2, padding="same",
                            name="enc_conv_2"),
    ], name="tiny_encoder")
    decoder = keras.Sequential([
        keras.layers.Input((8, 8, latent)),
        keras.layers.Conv2DTranspose(16, 3, strides=2, padding="same",
                                     activation="relu", name="dec_deconv_1"),
        keras.layers.Conv2DTranspose(1, 3, strides=2, padding="same",
                                     name="dec_deconv_2"),
    ], name="tiny_decoder")
    return encoder, decoder


def _b_vq_vae():
    from dl_techniques.models.vq_vae import VQVAEModel
    encoder, decoder = _tiny_conv_codec()
    return VQVAEModel(encoder=encoder, decoder=decoder, num_embeddings=8,
                      embedding_dim=8)


# ``allowed_none_grads=1``: MEASURED IDENTICAL under float32 -- 1 of 9 in BOTH
# arms (fp16 ``grad_norm_sum`` 1.945405e-02, float32 1.936058e-02). The
# codebook is updated by the commitment path, not by this loss.
_sub("vq_vae", _b_vq_vae, lambda: _f32(1, 32, 32, 1), allowed_none_grads=1)


def _b_vq_vae_rotation():
    from dl_techniques.models.vq_vae_rotation import VQVAERotationTrick
    encoder, decoder = _tiny_conv_codec()
    return VQVAERotationTrick(num_embeddings=8, embedding_dim=8,
                              encoder=encoder, decoder=decoder)


# ``allowed_none_grads=1``: MEASURED IDENTICAL under float32 -- 1 of 9 in BOTH
# arms (fp16 ``grad_norm_sum`` 1.536274e-02, float32 1.534150e-02).
_sub("vq_vae_rotation", _b_vq_vae_rotation, lambda: _f32(1, 32, 32, 1),
     allowed_none_grads=1)


def _b_masked_autoencoder():
    import keras
    from dl_techniques.models.masked_autoencoder import create_mae_model
    # The encoder must downsample by exactly the decoder's upsample factor;
    # two stride-2 convs against two ``decoder_dims`` entries is the smallest
    # pair the factory's own scale check accepts.
    inp = keras.Input((32, 32, 3))
    x = keras.layers.Conv2D(16, 3, strides=2, padding="same",
                            name="tiny_enc_conv_1")(inp)
    x = keras.layers.Conv2D(16, 3, strides=2, padding="same",
                            name="tiny_enc_conv_2")(x)
    encoder = keras.Model(inp, x, name="tiny_encoder")
    return create_mae_model(encoder=encoder, patch_size=16,
                            input_shape=(32, 32, 3), decoder_dims=[16, 16])


# ``dtype_exempt_outputs=[1]``: output 1 is ``"mask"``, a BINARY patch
# indicator, not an activation. ``mae.py`` states the reason at its own line
# 61 -- the masking ops return float32 -- and its ``compute_loss`` casts
# target, reconstruction AND mask to float32 anyway, so a float16 mask would
# be cast straight back. MEASURED under ``mixed_float16``: reconstruction
# float16, mask float32, masked_input float16, encoded float16.
_sub("masked_autoencoder", _b_masked_autoencoder, lambda: _f32(1, 32, 32, 3),
     dtype_exempt_outputs=[1])


def _b_sd3_mmdit():
    from dl_techniques.models.sd3_mmdit import create_sd3_vae
    return create_sd3_vae("tiny")


_sub("sd3_mmdit", _b_sd3_mmdit, lambda: _f32(1, 32, 32, 3))


_IDEOGRAM4_BATCH, _IDEOGRAM4_SEQ, _IDEOGRAM4_TEXT = 2, 8, 4


def _b_ideogram4():
    from dl_techniques.models.ideogram4 import create_ideogram4_transformer
    return create_ideogram4_transformer("tiny")


def _ideogram4_inputs():
    """A packed text-then-image batch, the only shape ``call`` accepts."""
    from dl_techniques.models.ideogram4.config import get_ideogram4_config
    from dl_techniques.models.ideogram4.transformer import (
        LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR,
    )
    cfg, _ae = get_ideogram4_config("tiny")
    # Batch 2, not 1: ``ScalarSinusoidalEmbedding`` squeezes a length-1
    # leading axis and then fails its own ``min_ndim=2`` input spec.
    b, seq, text = _IDEOGRAM4_BATCH, _IDEOGRAM4_SEQ, _IDEOGRAM4_TEXT
    indicator = np.empty((b, seq), dtype="int32")
    indicator[:, :text] = LLM_TOKEN_INDICATOR
    indicator[:, text:] = OUTPUT_IMAGE_INDICATOR
    position_ids = np.zeros((b, seq, 3), dtype="int32")
    for l in range(seq):
        position_ids[:, l, 0] = l
        position_ids[:, l, 1] = l % 2
        position_ids[:, l, 2] = l % 3
    return {
        "llm_features": _f32(b, seq, cfg.llm_features_dim),
        "x": _f32(b, seq, cfg.in_channels, seed=1),
        "t": np.full((b,), 0.5, dtype="float32"),
        "position_ids": position_ids,
        "segment_ids": np.zeros((b, seq), dtype="int32"),
        "indicator": indicator,
    }


# ``expected_compute_dtype="float32"``: the velocity head casts to float32
# UNCONDITIONALLY and by design, mirroring the reference implementation's
# ``.float()`` return -- ``transformer.py`` line 86 and line 325 both say so,
# and the class docstring promises "always float32". This is a RULING, not an
# exemption: part 2 still runs, it just asserts the documented dtype, so a
# silent change in either direction fails here.
_sub("ideogram4", _b_ideogram4, _ideogram4_inputs,
     expected_compute_dtype="float32")


# ---------------------------------------------------------------------------
# Multimodal / world models
# ---------------------------------------------------------------------------

def _b_nano_vlm():
    from dl_techniques.models.nano_vlm import NanoVLM
    return NanoVLM(
        vision_config={"img_size": 32, "patch_size": 16, "embed_dim": 32,
                       "depth": 1, "num_heads": 2, "output_mode": "none"},
        text_config={"vocab_size": 64, "embed_dim": 32, "depth": 1,
                     "num_heads": 2, "max_seq_len": 16},
        # ``cross_attention``: the vision and text streams have DIFFERENT
        # sequence lengths (5 patches vs 16 tokens) and ``concatenation``
        # rejects that by design.
        fusion_config={"fusion_strategy": "cross_attention", "dim": 32,
                       "attention_config": {"num_heads": 2},
                       "num_fusion_layers": 1},
        vocab_size=64,
    )


_sub("nano_vlm", _b_nano_vlm,
     lambda: {"images": _f32(1, 32, 32, 3), "text_tokens": _ids(64, 1, 16)},
     # ``allowed_none_grads=1``: MEASURED IDENTICAL under float32 -- the same
     # single variable, ``shared_output_projection/kernel``, is ``None`` in
     # BOTH arms (fp16 normsum 2.867185e-01, float32 2.706649e-01).
     allowed_none_grads=1)


def _b_video_jepa():
    from dl_techniques.models.video_jepa import create_video_jepa
    return create_video_jepa(img_size=32, patch_size=16, num_frames=2,
                             embed_dim=32)


# ``allowed_none_grads=1``: MEASURED IDENTICAL under float32 -- 1 of 117 in
# BOTH arms (fp16 ``grad_norm_sum`` 2.909179e+00, float32 2.864443e+00).
_sub("video_jepa", _b_video_jepa, lambda: {"pixels": _f32(1, 2, 32, 32, 3)},
     allowed_none_grads=1)


def _b_lewm():
    from dl_techniques.models.lewm import create_lewm
    return create_lewm(img_size=56, patch_size=14, encoder_scale="tiny",
                       embed_dim=192, projector_hidden_dim=192,
                       history_size=2, num_preds=1, depth=1, heads=2,
                       dim_head=32, mlp_dim=128, dropout_rate=0.0, emb_dropout_rate=0.0,
                       action_dim=2, smoothed_dim=10, mlp_scale=2,
                       sigreg_knots=17, sigreg_num_proj=32)


_sub("lewm", _b_lewm, lambda: {"pixels": _f32(1, 3, 56, 56, 3),
                               "action": _f32(1, 2, 2, seed=1)})


def _b_sam():
    from .test_sam.test_correctness import build_reduced_sam
    return build_reduced_sam()


def _sam_inputs():
    from .test_sam.test_correctness import sam_inputs
    return sam_inputs()


# ``allowed_none_grads=12``: MEASURED IDENTICAL under float32 -- 12 of 201 in
# BOTH arms (fp16 ``grad_norm_sum`` 1.923222e+02, float32 1.922683e+02). These
# are SAM's unreached prompt-encoder embeddings on a point-only prompt, not an
# fp16 effect; the number is pinned so an fp16-SPECIFIC disconnection still
# fails this arm.
_sub("SAM", _b_sam, _sam_inputs, allowed_none_grads=12)


# ---------------------------------------------------------------------------
# Tabular / graph / classical
# ---------------------------------------------------------------------------

def _b_tabm():
    from dl_techniques.models.tabm import create_tabm_mini
    return create_tabm_mini(n_num_features=8, cat_cardinalities=[], n_classes=3,
                            k=4, hidden_dims=[16])


_sub("tabm", _b_tabm, lambda: _f32(4, 8))


def _b_power_mlp():
    from dl_techniques.models.power_mlp import create_power_mlp
    return create_power_mlp(hidden_units=[8, 8, 3])


_sub("power_mlp", _b_power_mlp, lambda: _f32(4, 6))


def _b_mothnet():
    from dl_techniques.models.mothnet import MothNet
    return MothNet(num_classes=4, al_units=16, mb_units=32)


_sub("mothnet", _b_mothnet, lambda: _f32(4, 16))


def _b_som():
    from dl_techniques.models.som import create_som
    return create_som(map_size=(4, 4), input_dim=8)


# ``check_backward=False``: MEASURED ``n_vars == 0``. A SOM has NO trainable
# variables at all -- Kohonen's competitive rule updates ``som_weights`` via
# ``assign_add`` inside ``call``, not via a gradient -- so part 4 of the arm
# has nothing to reach and the oracle rejects the model outright. The
# ``training=True`` FORWARD is still exercised by part 1, which is where this
# package's real fp16 defect lived (D-062).
_sub("som", _b_som, lambda: _f32(4, 8), check_backward=False)


def _b_mini_vec2vec():
    from dl_techniques.models.mini_vec2vec import create_mini_vec2vec_aligner
    return create_mini_vec2vec_aligner(embedding_dim=8)


_sub("mini_vec2vec", _b_mini_vec2vec, lambda: _f32(4, 8))


def _b_ntm():
    from dl_techniques.models.ntm import create_ntm_variant
    return create_ntm_variant(variant="tiny", input_shape=(10, 8), output_dim=4)


_sub("ntm", _b_ntm, lambda: _f32(2, 10, 8))


def _b_relgt():
    from dl_techniques.models.relgt import create_relgt_model
    return create_relgt_model(output_dim=2, model_size="small")


def _relgt_inputs():
    rng = np.random.default_rng(0)
    b, n, f = 2, 8, 16
    return {
        "node_features": rng.random((b, n, f)).astype("float32"),
        "node_types": rng.integers(0, 10, (b, n)).astype("int32"),
        "hop_distances": rng.integers(0, 3, (b, n)).astype("int32"),
        "relative_times": rng.random((b, n, 1)).astype("float32"),
        "subgraph_adjacency": rng.random((b, n, n)).astype("float32"),
    }


_sub("relgt", _b_relgt, _relgt_inputs)


def _b_shgcn():
    from dl_techniques.models.shgcn import SHGCNNodeClassifier
    return SHGCNNodeClassifier(num_classes=3, hidden_dims=[8], embedding_dim=8,
                               dropout_rate=0.0)


def _shgcn_inputs():
    """``[features, dense adjacency]`` -- a fixed 6-node ring, row-normalised."""
    import numpy as _np
    n = 6
    adj = _np.zeros((n, n), dtype="float32")
    for i in range(n):
        adj[i, (i + 1) % n] = 1.0
        adj[i, (i - 1) % n] = 1.0
        adj[i, i] = 1.0
    adj /= adj.sum(axis=1, keepdims=True)
    return [_f32(n, 8), adj]


_sub("shgcn", _b_shgcn, _shgcn_inputs)


def _b_time_series():
    from dl_techniques.models.time_series import create_nbeats_model
    return create_nbeats_model(backcast_length=16, forecast_length=4,
                               stack_types=["trend", "generic"],
                               nb_blocks_per_stack=1, hidden_layer_units=16)


_sub("time_series", _b_time_series, lambda: _f32(2, 16, 1))


# ===========================================================================
# R-143 -- the graph / XLA equivalence arm's per-subject deviations
# ===========================================================================
#
# The XLA arm reuses the table above: same builders, same inputs, same
# ``call_fn``, and ``forward_training`` carried over as ``training``. Only the
# entries below deviate, and each names the measurement that made it deviate.
# Everything not listed here runs the default arm at ``rtol=1e-2``, which the
# whole-registry sweep clears with roughly 2.7x margin (the largest relative
# eager-vs-XLA delta over the 51 compiling subjects is ``lewm``'s 3.6e-3).

def _b_sam_training_model():
    """``SAMTrainingModel``, not ``SAM`` -- the traceable half of the package.

    MEASURED: ``SAM.call`` raises under ``tf.function`` at ``jit_compile=False``
    AND ``jit_compile=True``, with the same ``TypeError: len is not well defined
    for a symbolic Tensor`` from ``model.py:634``, because ``postprocess_masks``
    hands a TENSOR ``original_size`` to ``ops.image.resize``. That is a
    graph-mode property, not an XLA one, and the package already documents it
    (``model.py:68-72``) along with the remedy: ``SAMTrainingModel`` drives the
    submodules directly and never reaches the postprocess. Pointing the XLA arm
    at ``SAM`` would test a documented eager-only contract and call it an XLA
    finding. MEASURED on the wrapper: ``jit_compile=True`` OK, max|delta|
    4.140615e-03 against ``absmax`` 1.475909e+00.
    """
    from .test_sam.test_correctness import build_reduced_sam
    from dl_techniques.models.SAM.SAM1.training_model import SAMTrainingModel
    return SAMTrainingModel(build_reduced_sam(), multimask_output=False)


def _sam_training_inputs():
    from .test_sam.test_training_model import _wrapper_inputs
    return _wrapper_inputs()


# DECISION plan-2026-08-19T163559-499b6f0e/D-075
# Every entry below is a DEVIATION and every one states the measurement that
# forced it. Do NOT add an entry that merely raises ``rtol`` to make a subject
# pass: a tolerance above the signal asserts nothing, and the two subjects with
# a genuinely large float32 delta (`yolo12`, `hierarchical_reasoning_model`)
# are judged at float64 -- where the SAME assertion still runs -- precisely so
# that no such entry is ever needed. See decisions.md D-075.
#: name -> ``{"build": ..., "make_inputs": ..., **assert_xla_equivalence kwargs}``
XLA_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "SAM": {
        "build": _b_sam_training_model,
        "make_inputs": _sam_training_inputs,
        "call_fn": None,
    },
    # ``expect="raises"``: superpoint is the ONE subject of 53 that genuinely
    # cannot be XLA-compiled, and the repo already ships the opt-out
    # (`src/train/superpoint/train_superpoint.py:873`, `jit_compile=False`,
    # anchored). MEASURED: it traces fine at ``jit_compile=False`` (max|delta|
    # 2.717853e-03 / 1.198910e-04) and raises `InvalidArgumentError` at
    # ``jit_compile=True`` -- "No registered 'ResizeBicubic' OpKernel for
    # XLA_GPU_JIT devices". Same missing kernel family as D-060's missing
    # float16 ResizeBicubic GRADIENT. Asserting the RAISE, keyed on the op
    # name, is what keeps the documented opt-out from becoming cargo: if TF
    # ever registers the kernel, this test fails and the `jit_compile=False`
    # becomes removable.
    "superpoint": {
        "expect": "raises",
        "expect_reason": "ResizeBicubic",
    },
    # ``scope="float64"``: yolo12 is the only compiling subject whose float32
    # eager-vs-XLA delta is large -- 1.188995e+00 against ``absmax``
    # 4.583885e+00, i.e. 2.594e-01 relative, with an eager-vs-eager spread of
    # exactly 0.0 so it is not stochasticity. It is CONDITIONING, and the
    # float64 control proves it: the SAME comparison on the SAME graph measures
    # 5.685979e-12 against ``absmax`` 3.787825e+00, a relative 1.501e-12 -- ten
    # orders of magnitude, from nothing but the mantissa. XLA computes the same
    # function; the model at initialization is ill-conditioned (the D-065
    # untrained-BatchNorm story again, which is also why ``training=True`` is
    # carried over from the precision arm). Judging at float64 keeps a REAL
    # assertion here instead of an rtol of 0.3, which would assert nothing.
    "yolo12": {
        "scope": "float64",
        "rtol": 1e-6,
    },
    # ``scope="float64"``: the second and last conditioning row, and the one
    # whose refutation needed its own probe. At float32, eager and XLA disagree
    # by 7.260822e+00 on an ``absmax`` of 5.541113e+00 -- a delta LARGER than
    # the signal -- with an eager-vs-eager spread of exactly 0.0 across three
    # independent trials, so it is neither noise nor stochasticity. It is also
    # not XLA: perturbing ONE element of
    # ``core/token_embedding/embeddings`` by ONE ULP (7.450581e-09) moves the
    # output by 9.618139e-02, an amplification of **1.3e7**, and restoring the
    # element returns the output to a delta of exactly 0.0. A recursive model
    # with that gain turns XLA's ULP-level reassociation into a whole-output
    # change. Corroborating: ``q_halt_logits`` and ``q_continue_logits``, which
    # read a shallower path, agree at EXACTLY 0.0 in both scopes and at both
    # precisions. At float64 the logits delta is 2.852738e-02 against
    # ``absmax`` 5.044708e+00, i.e. 5.655e-03 relative -- hence rtol 2e-2,
    # a 3.5x margin, chosen from the measurement rather than to make it pass.
    "hierarchical_reasoning_model": {
        "scope": "float64",
        "rtol": 2e-2,
    },
}


def xla_subject(name: str) -> Tuple[Callable[[], Any], Callable[[], Any], Dict[str, Any]]:
    """Resolve one subject for :func:`assert_xla_equivalence`.

    Interface contract (1 call site, ``test_xla_equivalence_family.py``):
    starts from :data:`SUBJECTS`, keeps ``call_fn``, renames the precision arm's
    ``forward_training`` to the XLA arm's ``training``, drops every other
    precision-only kwarg, then applies :data:`XLA_OVERRIDES` on top. Raises
    ``KeyError`` for an unregistered name.

    :param name: A key of :data:`SUBJECTS`.
    :return: ``(build, make_inputs, kwargs)`` ready to splat into the arm.
    """
    build, make_inputs, kwargs = SUBJECTS[name]
    resolved: Dict[str, Any] = {}
    if kwargs.get("call_fn") is not None:
        resolved["call_fn"] = kwargs["call_fn"]
    if kwargs.get("forward_training"):
        resolved["training"] = True
    override = dict(XLA_OVERRIDES.get(name, {}))
    build = override.pop("build", build)
    make_inputs = override.pop("make_inputs", make_inputs)
    if "call_fn" in override and override["call_fn"] is None:
        override.pop("call_fn")
        resolved.pop("call_fn", None)
    resolved.update(override)
    return build, make_inputs, resolved


# ===========================================================================
# R-142 -- the five directories charged with a float64 arm that never ran
# ===========================================================================
#
# Batch 1 charged exactly five: ``set_floatx`` count 0 across the directory
# while ``float64`` is NAMED in 14 test files for SAM and in 1-2 files for each
# of the others. The charge is "the arm is absent", not "float64 is broken" --
# so these five get a real one. ``SAM`` is again represented by
# ``SAMTrainingModel``: ``SAM.call``'s ``ops.image.resize`` is a graph-mode
# limitation that has nothing to do with precision, and the wrapper is the
# supervised path.

#: The five R-142 rows, as names into :data:`SUBJECTS`.
FLOAT64_CHARGED: Tuple[str, ...] = (
    "SAM", "bias_free_denoisers", "ideogram4", "sd3_mmdit", "time_series",
)

#: name -> ``{"build": ..., "make_inputs": ..., **assert_float64_arm kwargs}``
FLOAT64_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "SAM": {
        "build": _b_sam_training_model,
        "make_inputs": _sam_training_inputs,
    },
    # ``expected_output_dtype="float32"``: NOT an exemption -- the assertion
    # still runs, against the value the package DOCUMENTS. ``ideogram4``'s
    # velocity head casts to float32 unconditionally, stated three times in
    # `models/ideogram4/transformer.py` (module notes line 86, class docstring
    # line 160, call site line 325) as mirroring the reference implementation's
    # `.float()` return. D-058 ruled the same way for its mixed-precision arm.
    # MEASURED here: under a real float64 scope the model returns float32, so a
    # plain float64 assertion would convict a documented contract -- and
    # switching part 3 off would stop noticing if the cast ever disappeared.
    "ideogram4": {
        "expected_output_dtype": "float32",
    },
}


def float64_subject(name: str) -> Tuple[Callable[[], Any], Callable[[], Any], Dict[str, Any]]:
    """Resolve one subject for :func:`assert_float64_arm`.

    Interface contract (1 call site, ``test_float64_arm_family.py``): same shape
    as :func:`xla_subject` -- start from :data:`SUBJECTS`, keep ``call_fn``,
    carry ``forward_training`` across as ``training``, apply
    :data:`FLOAT64_OVERRIDES`.
    """
    build, make_inputs, kwargs = SUBJECTS[name]
    resolved: Dict[str, Any] = {}
    if kwargs.get("call_fn") is not None:
        resolved["call_fn"] = kwargs["call_fn"]
    if kwargs.get("forward_training"):
        resolved["training"] = True
    override = dict(FLOAT64_OVERRIDES.get(name, {}))
    build = override.pop("build", build)
    make_inputs = override.pop("make_inputs", make_inputs)
    resolved.update(override)
    return build, make_inputs, resolved


# ===========================================================================
# R-063 / R-072 / R-073 / R-135 -- the round-trip instrument family
# ===========================================================================
#
# The instrument family (``test_roundtrip_instrument_family.py``) is charged
# against ALL 73 ``models/`` packages, not just the 53 the precision arm
# covers. The 20 packages below are exactly the difference. They live here
# rather than in a second registry so that a package has ONE build/input pair
# in the tree: ``roundtrip_subject`` resolves ``SUBJECTS`` first and falls back
# to :data:`EXTRA_SUBJECTS`.
#
# Four of the twenty (``coshnet``, ``darkir``, ``fftnet``, ``pw_fnet``) are
# deliberately absent from :data:`CHARGED_PACKAGES` -- step 18 gave each its own
# ``test_precision_arm.py`` -- so their builders are transcribed from those
# files rather than duplicated logic.

EXTRA_SUBJECTS: Dict[str, Tuple[Callable[[], Any], Callable[[], Any], Dict[str, Any]]] = {}


def _extra(name: str, build, make_inputs, **kwargs: Any) -> None:
    EXTRA_SUBJECTS[name] = (build, make_inputs, kwargs)


def _b_coshnet():
    from dl_techniques.models.coshnet.model import create_coshnet
    return create_coshnet("tiny", num_classes=4, input_shape=(32, 32, 3))


_extra("coshnet", _b_coshnet, lambda: _f32(1, 32, 32, 3))


def _b_darkir():
    from dl_techniques.models.darkir.model import create_darkir_model
    return create_darkir_model(
        width=8, middle_blk_num_enc=1, middle_blk_num_dec=1,
        enc_blk_nums=[1, 1], dec_blk_nums=[1, 1], dilations=[1, 2],
    )


_extra("darkir", _b_darkir, lambda: _f32(1, 32, 32, 3))


def _b_fftnet():
    from dl_techniques.models.fftnet.model import create_fftnet
    return create_fftnet("tiny", image_size=32, patch_size=16)


_extra("fftnet", _b_fftnet, lambda: _f32(1, 32, 32, 3))


def _b_pw_fnet():
    from dl_techniques.models.pw_fnet.model import create_pw_fnet
    return create_pw_fnet(width=8, middle_blk_num=1,
                          enc_blk_nums=[1, 1], dec_blk_nums=[1, 1])


_extra("pw_fnet", _b_pw_fnet, lambda: _f32(1, 32, 32, 3))


def _b_clip():
    from dl_techniques.models.clip.model import CLIP
    return CLIP(image_size=32, patch_size=16, vision_layers=1, vision_width=32,
                vision_heads=2, vision_kv_heads=1, vocab_size=64,
                context_length=8, text_layers=1, text_width=32, text_heads=2,
                text_kv_heads=1, embed_dim=16)


def _clip_inputs():
    return {"image": _f32(1, 32, 32, 3), "text": _ids(64, 1, 8)}


_extra("clip", _b_clip, _clip_inputs)


def _b_detr():
    import keras
    from dl_techniques.models.detr.model import DETR, DetrTransformer
    backbone = keras.Sequential([
        keras.layers.Conv2D(32, 3, strides=2, padding="same",
                            activation="relu", name="stub_conv_1"),
        keras.layers.Conv2D(32, 3, strides=2, padding="same",
                            activation="relu", name="stub_conv_2"),
    ], name="stub_backbone")
    transformer = DetrTransformer(hidden_dim=32, num_heads=2,
                                  num_encoder_layers=1, num_decoder_layers=1,
                                  ffn_dim=32, dropout_rate=0.0,
                                  name="detr_transformer_subject")
    return DETR(num_classes=4, num_queries=5, backbone=backbone,
                transformer=transformer, hidden_dim=32, aux_loss=False)


def _detr_inputs():
    return [_f32(1, 32, 32, 3), np.zeros((1, 32, 32), dtype="bool")]


_extra("detr", _b_detr, _detr_inputs)


def _b_dino():
    from dl_techniques.models.dino.dino_v2 import create_dino_v2
    return create_dino_v2("tiny", image_size=28, patch_size=14, num_classes=4,
                          num_register_tokens=2)


# The backbone's input contract is a 2-element list ``[images, masks]`` -- its
# own module docstring states it -- with ``masks`` the (B, num_patches) iBOT
# mask. 28/14 gives 2x2 = 4 patches.
_extra("dino", _b_dino,
       lambda: [_f32(1, 28, 28, 3), np.zeros((1, 4), dtype="bool")])


def _b_distilbert():
    from dl_techniques.models.distilbert.model import DistilBERT
    return DistilBERT(vocab_size=64, hidden_size=32, num_layers=1, num_heads=2,
                      intermediate_size=64, max_position_embeddings=16)


_extra("distilbert", _b_distilbert, lambda: {"input_ids": _ids(64, 1, 8)})


def _b_energy_transformer():
    from dl_techniques.models.energy_transformer.model import (
        create_energy_transformer_classifier,
    )
    return create_energy_transformer_classifier(
        variant="tiny", input_shape=(32, 32, 3), patch_size=16, num_classes=4)


_extra("energy_transformer", _b_energy_transformer, lambda: _f32(1, 32, 32, 3))


def _b_fastvlm():
    from dl_techniques.models.fastvlm.model import FastVLM
    # Three stages: ``embed_dims must have 3 elements`` (measured, its own
    # constructor guard).
    return FastVLM(num_classes=4, embed_dims=[16, 32, 64], depths=[1, 1, 1],
                   num_heads=[1, 2, 4], mlp_ratio=2.0, dropout_rate=0.0,
                   drop_path_rate=0.0, include_top=True,
                   input_shape=(32, 32, 3))


_extra("fastvlm", _b_fastvlm, lambda: _f32(1, 32, 32, 3))


def _b_graph_energy_transformer():
    from dl_techniques.models.graph_energy_transformer.model import (
        create_graph_classifier,
    )
    return create_graph_classifier(node_feature_dim=8, num_classes=4,
                                   embed_dim=16, num_heads=2, head_dim=8,
                                   hopfield_dim=32, num_blocks=1, pe_dim=4)


def _get_inputs():
    n = 8
    return {
        "node_features": _f32(1, n, 8),
        "adjacency": np.ones((1, n, n), dtype="float32"),
        # The dict keys are ``pe`` / ``node_mask``, per the backbone's own
        # documented call signature -- not ``positional_encoding`` / ``mask``.
        "pe": _f32(1, n, 4, seed=1),
        "node_mask": np.ones((1, n), dtype="float32"),
    }


_extra("graph_energy_transformer", _b_graph_energy_transformer, _get_inputs)


def _b_kan():
    from dl_techniques.models.kan.model import KAN
    return KAN(layer_configs=[{"features": 8, "grid_size": 5,
                               "activation": "swish"},
                              {"features": 4, "grid_size": 4,
                               "activation": "linear"}],
               input_features=6)


_extra("kan", _b_kan, lambda: _f32(2, 6))


def _b_latent_gmm():
    from dl_techniques.models.latent_gmm_registration.model import (
        LatentGMMRegistration,
    )
    return LatentGMMRegistration(num_gaussians=4, k_neighbors=8)


_extra("latent_gmm_registration", _b_latent_gmm,
       lambda: (_f32(1, 16, 3), _f32(1, 16, 3, seed=1)))


def _b_mamba():
    from dl_techniques.models.mamba.mamba_v1 import Mamba
    return Mamba(vocab_size=64, d_model=32, num_layers=1, d_state=8, d_conv=4,
                 expand=2)


_extra("mamba", _b_mamba, lambda: _ids(64, 1, 8))


def _b_mobile_clip():
    from dl_techniques.models.mobile_clip.mobile_clip_v2 import (
        create_mobile_clip_v2,
    )
    return create_mobile_clip_v2("mobileclip2_s0")


def _mobile_clip_inputs():
    return {"image": _f32(1, 256, 256, 3), "text": _ids(100, 1, 77)}


_extra("mobile_clip", _b_mobile_clip, _mobile_clip_inputs)


def _b_nam():
    from dl_techniques.models.nam.model import NAM, NAMConfig
    return NAM(config=NAMConfig(hidden_size=32, num_heads=4,
                                num_tree_layers=1, intermediate_size=64,
                                memory_size=8, num_read_heads=2,
                                max_expression_len=16, halt_max_steps=2,
                                hidden_dropout_rate=0.0,
                                attention_dropout_rate=0.0))


def _nam_call(model, inputs, training):
    """``NAM.call(carry, batch, training)`` -- one ACT step, not ``model(x)``.

    Same three-positional contract as :func:`_trm_call` above. The carry is
    derived from the batch by the model's own ``initial_carry``, so the call
    stays deterministic for a given batch.
    """
    carry = model.initial_carry(inputs)
    _carry, outputs = model(carry, inputs, training=training)
    return outputs


_extra("nam", _b_nam, lambda: {"input_ids": _ids(16, 2, 16)},
       call_fn=_nam_call)


def _b_nano_vlm_world_model():
    from dl_techniques.models.nano_vlm_world_model.model import (
        create_score_based_nanovlm,
    )
    return create_score_based_nanovlm(variant="mini", vocab_size=256)


def _nvwm_inputs():
    return {"images": _f32(1, 224, 224, 3), "text": _ids(256, 1, 8)}


_extra("nano_vlm_world_model", _b_nano_vlm_world_model, _nvwm_inputs)


def _b_thera():
    from dl_techniques.models.thera.model import Thera
    from dl_techniques.models.thera.edsr_backbone import EDSRBackbone
    from dl_techniques.models.thera.tails import build_thera_tail
    backbone = EDSRBackbone(num_feats=16, num_blocks=1,
                            name="edsr_backbone_subject")
    feat_ch = backbone.compute_output_shape((None, None, None, 3))[-1]
    return Thera(hidden_dim=8, out_dim=3, backbone=backbone,
                 tail=build_thera_tail("air", in_channels=feat_ch))


def _thera_inputs():
    import keras
    from dl_techniques.layers.grid_sample import make_grid
    grid = np.asarray(keras.ops.convert_to_numpy(
        keras.ops.convert_to_tensor(make_grid((12, 12)))))
    coords = np.broadcast_to(grid[None, ...], (1, 12, 12, 2)).astype("float32")
    return (_f32(1, 8, 8, 3), coords, np.ones((1, 1), dtype="float32"))


_extra("thera", _b_thera, _thera_inputs)


def _b_tree_transformer():
    from dl_techniques.models.tree_transformer.model import TreeTransformer
    # ``max_len``, not ``max_position_embeddings``: the constructor rejects the
    # BERT spelling outright rather than ignoring it.
    return TreeTransformer(vocab_size=64, hidden_size=32, num_layers=1,
                           num_heads=2, intermediate_size=64, max_len=16)


_extra("tree_transformer", _b_tree_transformer,
       lambda: {"input_ids": _ids(64, 1, 8)})


#: The one ``models/`` package with NO round-trip subject, and why. Its own
#: ``__init__.py`` states it: "nothing subclasses ``keras.Model``, so there is
#: nothing for a model factory to build". ``PowerSampler`` is a sampling
#: procedure over a caller-supplied logits function; there is no archive, no
#: weight and therefore no round trip to instrument.
ROUNDTRIP_NA: Dict[str, str] = {
    "power_sampling": (
        "no keras.Model subclass in the package (stated in its own "
        "__init__.py); PowerSampler holds no weights, so R-063/R-072/R-073 "
        "have no subject here"
    ),
}


def roundtrip_charged() -> Tuple[str, ...]:
    """Every ``models/`` package that owns a round-trip subject, sorted.

    Interface contract (2 call sites, ``test_roundtrip_instrument_family.py``):
    returns the union of :data:`SUBJECTS` and :data:`EXTRA_SUBJECTS` keys. The
    completeness test asserts this union plus :data:`ROUNDTRIP_NA` is set-equal
    to the on-disk package list, so a package cannot leave the family silently.
    """
    return tuple(sorted(set(SUBJECTS) | set(EXTRA_SUBJECTS)))


def roundtrip_subject(name: str) -> Tuple[Callable[[], Any], Callable[[], Any], Dict[str, Any]]:
    """Resolve one subject for the round-trip instrument family.

    Interface contract (3 call sites, all in
    ``test_roundtrip_instrument_family.py``): returns
    ``(build, make_inputs, kwargs)`` where ``kwargs`` carries only the two keys
    the round-trip oracle understands -- ``call_fn`` and ``training`` -- so a
    precision-arm-only kwarg (``rtol_against_float32``, ``check_backward``)
    cannot leak into a round-trip assertion and silently do nothing.

    :raises KeyError: if ``name`` has no subject (including the
        :data:`ROUNDTRIP_NA` packages, which have none by ruling).
    """
    build, make_inputs, kwargs = (
        SUBJECTS[name] if name in SUBJECTS else EXTRA_SUBJECTS[name]
    )
    resolved: Dict[str, Any] = {}
    if kwargs.get("call_fn") is not None:
        resolved["call_fn"] = kwargs["call_fn"]
    if kwargs.get("forward_training"):
        resolved["training"] = True
    return build, make_inputs, resolved


# ---------------------------------------------------------------------------
# R-072(b) -- the no-sub-layer-config siblings
# ---------------------------------------------------------------------------
#
# Build parity (R-072(a)) is blind to OVER-building: it passes when both paths
# build everything. The sibling arm is the other half -- a component is turned
# OFF and its weights must be ABSENT.
#
# The subject set is not chosen by taste. It is every package whose model class
# accepts ``include_top``, derived mechanically by
# ``inspect.signature(type(model).__init__)`` -- 13 of the 72 -- and
# ``test_the_no_head_sibling_set_is_the_include_top_set`` re-derives it, so a
# package that GAINS the knob cannot stay out of the arm.

NO_HEAD_SIBLINGS: Dict[str, Callable[[], Any]] = {}


def _no_head(name: str):
    def register(fn):
        NO_HEAD_SIBLINGS[name] = fn
        return fn
    return register


@_no_head("cbam")
def _b_cbam_no_head():
    from dl_techniques.models.cbam import create_cbam_net
    return create_cbam_net("tiny", num_classes=4, input_shape=(32, 32, 3),
                           include_top=False)


@_no_head("convnext")
def _b_convnext_no_head():
    from dl_techniques.models.convnext import ConvNeXtV1
    return ConvNeXtV1(num_classes=4, depths=[1, 1], dims=[8, 16],
                      include_top=False)


@_no_head("coshnet")
def _b_coshnet_no_head():
    from dl_techniques.models.coshnet.model import create_coshnet
    return create_coshnet("tiny", num_classes=4, input_shape=(32, 32, 3),
                          include_top=False)


@_no_head("dino")
def _b_dino_no_head():
    from dl_techniques.models.dino.dino_v2 import create_dino_v2
    return create_dino_v2("tiny", image_size=28, patch_size=14, num_classes=4,
                          num_register_tokens=2, include_top=False)


@_no_head("fastvlm")
def _b_fastvlm_no_head():
    from dl_techniques.models.fastvlm.model import FastVLM
    return FastVLM(num_classes=4, embed_dims=[16, 32, 64], depths=[1, 1, 1],
                   num_heads=[1, 2, 4], mlp_ratio=2.0, dropout_rate=0.0,
                   drop_path_rate=0.0, include_top=False,
                   input_shape=(32, 32, 3))


@_no_head("fractalnet")
def _b_fractalnet_no_head():
    from dl_techniques.models.fractalnet import create_fractal_net
    return create_fractal_net(variant="micro", num_classes=4,
                              input_shape=(32, 32, 3), include_top=False)


@_no_head("mobilenet")
def _b_mobilenet_no_head():
    from dl_techniques.models.mobilenet import create_mobilenetv2
    return create_mobilenetv2(variant="small", num_classes=4,
                              input_shape=(32, 32, 3), include_top=False)


@_no_head("resnet")
def _b_resnet_no_head():
    from dl_techniques.models.resnet import ResNet
    return ResNet(num_classes=4, blocks_per_stage=[1, 1],
                  filters_per_stage=[8, 16], block_type="basic",
                  include_top=False)


@_no_head("squeezenet")
def _b_squeezenet_no_head():
    from dl_techniques.models.squeezenet import create_squeezenet_v1
    return create_squeezenet_v1("1.0", num_classes=4, input_shape=(64, 64, 3),
                                include_top=False)


@_no_head("swin_transformer")
def _b_swin_no_head():
    from dl_techniques.models.swin_transformer import create_swin_transformer
    return create_swin_transformer("tiny", 4, input_shape=(32, 32, 3),
                                   include_top=False)


@_no_head("vit")
def _b_vit_no_head():
    from dl_techniques.models.vit import create_vit
    return create_vit(variant="vit_tiny", num_classes=4,
                      input_shape=(32, 32, 3), patch_size=16,
                      include_top=False)


@_no_head("vit_hmlp")
def _b_vit_hmlp_no_head():
    from dl_techniques.models.vit_hmlp import create_vit_hmlp
    return create_vit_hmlp(input_shape=(32, 32, 3), num_classes=4,
                           scale="tiny", patch_size=16, include_top=False)


@_no_head("vit_siglip")
def _b_vit_siglip_no_head():
    from dl_techniques.models.vit_siglip import create_siglip_vision_transformer
    return create_siglip_vision_transformer(
        input_shape=(32, 32, 3), num_classes=4, scale="tiny", patch_size=16,
        include_top=False)
