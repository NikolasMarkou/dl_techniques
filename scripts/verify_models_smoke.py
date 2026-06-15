#!/usr/bin/env python
"""Data-driven smoke-test harness for every model package under
``src/dl_techniques/models/``.

For each registered package it builds the smallest variant and runs a single
forward pass, recording one of PASS / FAIL / XFAIL / XPASS / SKIP. The script
is a *report*, not a gate: it always exits 0 unless the harness scaffolding
itself throws.

Modes per registry entry:
  RUN   -> expected to build + forward + return finite output.
  XFAIL -> known-dead/known-broken; failure is expected (XFAIL),
           an unexpected success is reported as XPASS.
  SKIP  -> not exercised (no top-level model, or special orchestrator).

Each entry imports its model lazily inside the ``build`` callable so an
import error is caught per-entry and never aborts the whole matrix.

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src \
    .venv/bin/python scripts/verify_models_smoke.py [--only a,b] [--verbose]
"""
from __future__ import annotations

import os
import sys
import gc
import argparse
import traceback

# --- robust src/ on sys.path (works regardless of CWD) ---------------------
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
)

import numpy as np  # noqa: E402
import keras  # noqa: E402

# tensorflow imported lazily/guarded (only some recipes need it)
try:
    import tensorflow as tf  # noqa: E402
except Exception:  # pragma: no cover - tf should be present in this env
    tf = None


# ---------------------------------------------------------------------------
# Input factories
# ---------------------------------------------------------------------------
def _img(b=2, h=32, w=32, c=3):
    return np.random.rand(b, h, w, c).astype(np.float32)


def _tokens(b=2, s=16, vocab=256):
    return np.random.randint(0, vocab, (b, s)).astype(np.int32)


def _vec(b=2, d=16):
    return np.random.rand(b, d).astype(np.float32)


# ---------------------------------------------------------------------------
# Finite-check helper
# ---------------------------------------------------------------------------
def _all_finite(out):
    """Recursively verify every leaf tensor is finite.

    Returns (ok: bool, shapes: str).
    """
    shapes = []

    def _walk(o):
        if isinstance(o, dict):
            return all(_walk(v) for v in o.values())
        if isinstance(o, (list, tuple)):
            return all(_walk(v) for v in o)
        try:
            arr = keras.ops.convert_to_numpy(o)
        except Exception:
            arr = np.asarray(o)
        shapes.append(str(tuple(arr.shape)))
        return bool(np.all(np.isfinite(arr)))

    ok = _walk(out)
    return ok, ", ".join(shapes)


# ---------------------------------------------------------------------------
# REGISTRY — one entry per model package (70 packages)
# ---------------------------------------------------------------------------
# Each entry: {name, mode, build()->model, forward(model)->output, note}

REGISTRY = []


def _reg(name, mode, build, forward, note=""):
    REGISTRY.append(
        {"name": name, "mode": mode, "build": build, "forward": forward, "note": note}
    )


# --- accunet ---------------------------------------------------------------
def _b_accunet():
    from dl_techniques.models.accunet.model import create_acc_unet
    return create_acc_unet(input_channels=3, num_classes=1, input_shape=(64, 64))
_reg("accunet", "RUN", _b_accunet,
     lambda m: m(_img(h=64, w=64), training=False), "segmentation U-Net")


# --- bert ------------------------------------------------------------------
def _b_bert():
    from dl_techniques.models.bert.bert import BERT
    return BERT.from_variant("tiny")
def _f_bert(m):
    ids = _tokens(vocab=1000)
    mask = np.ones_like(ids)
    return m({"input_ids": ids, "attention_mask": mask}, training=False)
_reg("bert", "RUN", _b_bert, _f_bert, "tiny")


# --- bias_free_denoisers ---------------------------------------------------
def _b_bfcnn():
    from dl_techniques.models.bias_free_denoisers.bfcnn import create_bfcnn_variant
    return create_bfcnn_variant("small", input_shape=(32, 32, 3))
_reg("bias_free_denoisers", "RUN", _b_bfcnn,
     lambda m: m(_img(), training=False), "bfcnn small representative")


# --- byte_latent_transformer ----------------------------------------------
def _b_blt():
    from dl_techniques.models.byte_latent_transformer.model import ByteLatentTransformer
    return ByteLatentTransformer.from_variant("micro", vocab_size=260)
_reg("byte_latent_transformer", "RUN", _b_blt,
     lambda m: m(_tokens(vocab=260), training=False), "micro")


# --- capsnet ---------------------------------------------------------------
def _b_capsnet():
    from dl_techniques.models.capsnet.model import create_capsnet
    return create_capsnet(num_classes=10, input_shape=(28, 28, 1))
_reg("capsnet", "RUN", _b_capsnet,
     lambda m: m(_img(h=28, w=28, c=1), training=False), "")


# --- cbam ------------------------------------------------------------------
def _b_cbam():
    from dl_techniques.models.cbam.model import CBAMNet
    return CBAMNet.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))
_reg("cbam", "RUN", _b_cbam, lambda m: m(_img(), training=False), "tiny")


# --- ccnets ----------------------------------------------------------------
_reg("ccnets", "SKIP", None, None,
     "3-model orchestrator (Explainer/Reasoner/Producer); not model(x)")


# --- cliffordnet -----------------------------------------------------------
def _b_clifford():
    from dl_techniques.models.cliffordnet.model import CliffordNet
    return CliffordNet.from_variant("nano", num_classes=10)
_reg("cliffordnet", "RUN", _b_clifford,
     lambda m: m(_img(), training=False), "nano (input 32x32x3)")


# --- clip ------------------------------------------------------------------
def _b_clip():
    from dl_techniques.models.clip.model import CLIP
    return CLIP(image_size=32, patch_size=8, vision_layers=2, vision_width=64,
                vision_heads=4, vision_kv_heads=2, text_layers=2, text_width=64,
                text_heads=4, text_kv_heads=4, embed_dim=64, context_length=16,
                vocab_size=1000)
def _f_clip(m):
    inp = {"image": _img(), "text": _tokens(vocab=1000)}
    try:
        return m(inp, training=False)
    except Exception:
        inp2 = {"images": _img(h=32, w=32), "texts": _tokens(vocab=1000)}
        return m(inp2, training=False)
_reg("clip", "RUN", _b_clip, _f_clip, "small custom dims")


# --- convnext --------------------------------------------------------------
def _b_convnext():
    from dl_techniques.models.convnext.convnext_v1 import ConvNeXtV1
    return ConvNeXtV1.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))
_reg("convnext", "RUN", _b_convnext, lambda m: m(_img(), training=False), "v1 tiny")

def _b_convnext_v2():
    from dl_techniques.models.convnext.convnext_v2 import ConvNeXtV2
    return ConvNeXtV2.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))
_reg("convnext_v2", "RUN", _b_convnext_v2, lambda m: m(_img(), training=False), "v2 tiny")


# --- convnext_patch_vae ----------------------------------------------------
def _b_cpvae():
    from dl_techniques.models.convnext_patch_vae.model import ConvNeXtPatchVAE
    return ConvNeXtPatchVAE.from_variant("tiny", img_size=32, patch_size=8)
_reg("convnext_patch_vae", "RUN", _b_cpvae,
     lambda m: m(_img(h=32, w=32), training=False), "tiny, img32/patch8")


# --- convunext -------------------------------------------------------------
def _b_convunext():
    from dl_techniques.models.convunext.model import ConvUNextModel
    return ConvUNextModel.from_variant("tiny", output_channels=10, input_shape=(32, 32, 3))
_reg("convunext", "RUN", _b_convunext, lambda m: m(_img(), training=False), "tiny")


# --- coshnet ---------------------------------------------------------------
def _b_coshnet():
    from dl_techniques.models.coshnet.model import CoShNet
    return CoShNet.from_variant("tiny", num_classes=10, input_shape=(28, 28, 3))
_reg("coshnet", "RUN", _b_coshnet,
     lambda m: m(_img(h=28, w=28), training=False), "tiny")


# --- darkir ----------------------------------------------------------------
def _b_darkir():
    from dl_techniques.models.darkir.model import create_darkir_model
    return create_darkir_model(img_channels=3, width=16)
_reg("darkir", "RUN", _b_darkir,
     lambda m: m(_img(h=64, w=64), training=False), "low-light restoration")


# --- depth_anything --------------------------------------------------------
def _b_depth():
    from dl_techniques.models.depth_anything.model import create_depth_anything
    return create_depth_anything(image_shape=(224, 224, 3))
_reg("depth_anything", "RUN", _b_depth,
     lambda m: m(_img(h=224, w=224), training=False), "")


# --- detr ------------------------------------------------------------------
def _b_detr():
    from dl_techniques.models.detr.model import create_detr
    return create_detr(num_classes=80, num_queries=100)
def _f_detr(m):
    images = _img(h=128, w=128)              # (2,128,128,3)
    pad = np.zeros((2, 128, 128), dtype=bool)  # padding mask
    return m((images, pad), training=False)
_reg("detr", "RUN", _b_detr, _f_detr, "resnet50 backbone; (images, pad_mask)")


# --- dino (v1/v2/v3) -------------------------------------------------------
def _b_dino_v1():
    from dl_techniques.models.dino.dino_v1 import DINOv1
    return DINOv1.from_variant("tiny", image_size=32, patch_size=4)
_reg("dino_v1", "RUN", _b_dino_v1, lambda m: m(_img(), training=False), "tiny")

def _b_dino_v2():
    from dl_techniques.models.dino.dino_v2 import DINOv2
    return DINOv2.from_variant("small", num_classes=10)
def _f_dino_v2(m):
    imgs = _img(h=224, w=224)
    # masks: (B, N_patches) boolean -- infer patch count from model
    n = getattr(m, "num_patches", None)
    if n is None:
        n = (224 // 14) * (224 // 14)
    masks = np.zeros((2, int(n)), dtype=bool)
    return m([imgs, masks], training=False)
_reg("dino_v2", "RUN", _b_dino_v2, _f_dino_v2, "224x224, 2-input [images, masks]")

def _b_dino_v3():
    from dl_techniques.models.dino.dino_v3 import DINOv3
    return DINOv3.from_variant("tiny", num_classes=10)
_reg("dino_v3", "RUN", _b_dino_v3,
     lambda m: m(_img(h=224, w=224), training=False), "tiny @224")


# --- distilbert ------------------------------------------------------------
def _b_distilbert():
    from dl_techniques.models.distilbert.model import DistilBERT
    return DistilBERT.from_variant("base")
def _f_distilbert(m):
    ids = _tokens(vocab=1000)
    mask = np.ones_like(ids)
    return m({"input_ids": ids, "attention_mask": mask}, training=False)
_reg("distilbert", "RUN", _b_distilbert, _f_distilbert, "base (only variant)")


# --- fastvlm ---------------------------------------------------------------
def _b_fastvlm():
    from dl_techniques.models.fastvlm.model import FastVLM
    return FastVLM.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))
_reg("fastvlm", "RUN", _b_fastvlm, lambda m: m(_img(), training=False), "tiny")


# --- fftnet ----------------------------------------------------------------
def _b_fftnet():
    from dl_techniques.models.fftnet.model import FFTNet
    return FFTNet.from_variant("base")
_reg("fftnet", "XFAIL", _b_fftnet, lambda m: m(_img(), training=False),
     "FFTNet core should forward; XFAIL guard (SpectreHead is the dead part)")


# --- fnet ------------------------------------------------------------------
def _b_fnet():
    from dl_techniques.models.fnet.model import FNet
    return FNet.from_variant("base")
def _f_fnet(m):
    ids = _tokens(vocab=1000)
    mask = np.ones_like(ids)
    return m({"input_ids": ids, "attention_mask": mask}, training=False)
_reg("fnet", "RUN", _b_fnet, _f_fnet, "base (no tiny variant)")


# --- fractalnet ------------------------------------------------------------
def _b_fractal():
    from dl_techniques.models.fractalnet.model import FractalNet
    return FractalNet.from_variant("micro", num_classes=10, input_shape=(28, 28, 1))
_reg("fractalnet", "RUN", _b_fractal,
     lambda m: m(_img(h=28, w=28, c=1), training=False), "micro")


# --- gemma -----------------------------------------------------------------
def _b_gemma():
    from dl_techniques.models.gemma.gemma3 import Gemma3
    return Gemma3.from_variant("tiny")
_reg("gemma", "RUN", _b_gemma, lambda m: m(_tokens(vocab=256), training=False), "tiny")


# --- gpt2 ------------------------------------------------------------------
def _b_gpt2():
    from dl_techniques.models.gpt2.gpt2 import GPT2
    return GPT2.from_variant("tiny")
_reg("gpt2", "RUN", _b_gpt2, lambda m: m(_tokens(vocab=256), training=False), "tiny")


# --- hierarchical_reasoning_model (DEAD) -----------------------------------
def _b_hrm():
    from dl_techniques.models.hierarchical_reasoning_model.model import (
        HierarchicalReasoningModel,
    )
    return HierarchicalReasoningModel.from_variant("tiny")
_reg("hierarchical_reasoning_model", "XFAIL", _b_hrm,
     lambda m: m(_tokens(vocab=256), training=False),
     "DEAD: None->tensor bug in call() (SYSTEM.md)")


# --- ideogram4 -------------------------------------------------------------
def _b_ideo():
    from dl_techniques.models.ideogram4.transformer import create_ideogram4_transformer
    return create_ideogram4_transformer(variant="tiny")
def _f_ideo(m):
    # Packed-stream DiT: dict with llm_features, x, t, position_ids,
    # segment_ids, indicator. indicator: 3=LLM text token, 2=image token.
    from dl_techniques.models.ideogram4.constants import (
        LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR,
    )
    cfg = m.config
    in_ch = cfg.in_channels
    llm_dim = cfg.llm_features_dim
    B, L = 2, 8
    n_text = 2  # first 2 tokens are text, rest are image tokens
    llm = np.random.rand(B, L, llm_dim).astype(np.float32)
    x = np.random.rand(B, L, in_ch).astype(np.float32)
    t = np.random.rand(B).astype(np.float32)
    position_ids = np.zeros((B, L, 3), dtype=np.int32)
    position_ids[:, :, 1] = np.arange(L)  # arbitrary h coords
    segment_ids = np.zeros((B, L), dtype=np.int32)
    indicator = np.full((B, L), OUTPUT_IMAGE_INDICATOR, dtype=np.int32)
    indicator[:, :n_text] = LLM_TOKEN_INDICATOR
    return m({
        "llm_features": llm, "x": x, "t": t,
        "position_ids": position_ids, "segment_ids": segment_ids,
        "indicator": indicator,
    }, training=False)
_reg("ideogram4", "RUN", _b_ideo, _f_ideo, "tiny; packed-stream dict")


# --- jepa (SKIP) -----------------------------------------------------------
_reg("jepa", "SKIP", None, None,
     "no top-level keras.Model (JEPAEncoder/JEPAPredictor are layers)")


# --- kan -------------------------------------------------------------------
def _b_kan():
    from dl_techniques.models.kan.model import KAN
    return KAN.from_variant("small", input_features=784, output_features=10)
_reg("kan", "RUN", _b_kan, lambda m: m(_vec(d=784), training=False), "small")


# --- latent_gmm_registration ----------------------------------------------
def _b_latgmm():
    from dl_techniques.models.latent_gmm_registration.model import LatentGMMRegistration
    return LatentGMMRegistration(num_gaussians=4, k_neighbors=8)
def _f_latgmm(m):
    src = np.random.rand(2, 64, 3).astype(np.float32)
    tgt = np.random.rand(2, 64, 3).astype(np.float32)
    return m([src, tgt], training=False)
_reg("latent_gmm_registration", "RUN", _b_latgmm, _f_latgmm,
     "two point clouds [source, target]")


# --- lewm ------------------------------------------------------------------
def _b_lewm():
    from dl_techniques.models.lewm.model import LeWM
    return LeWM()
def _f_lewm(m):
    # dict {"pixels": (B,T,H,W,C), "action": (B,T-1,A)}; T = history+preds.
    cfg = m.config
    a = cfg.action_dim
    h = cfg.img_size            # square edge (224 default; internal ViT)
    c = cfg.img_channels
    t = cfg.history_size + cfg.num_preds
    pixels = np.random.rand(2, t, int(h), int(h), c).astype(np.float32)
    action = np.random.rand(2, t - 1, int(a)).astype(np.float32)
    return m({"pixels": pixels, "action": action}, training=False)
_reg("lewm", "RUN", _b_lewm, _f_lewm, "dict {pixels, action}; img_size from config")


# --- mamba (v1/v2) ---------------------------------------------------------
def _b_mamba():
    from dl_techniques.models.mamba.mamba_v1 import Mamba
    return Mamba.from_variant("base", vocab_size=1000)
def _f_mamba(m):
    ids = _tokens(vocab=1000)
    try:
        return m({"input_ids": ids}, training=False)
    except Exception:
        return m(ids, training=False)
_reg("mamba", "RUN", _b_mamba, _f_mamba, "v1 base")

def _b_mamba2():
    from dl_techniques.models.mamba.mamba_v2 import Mamba2
    return Mamba2.from_variant("base", vocab_size=1000)
_reg("mamba_v2", "RUN", _b_mamba2, _f_mamba, "v2 base")


# --- masked_autoencoder (needs encoder) ------------------------------------
def _b_mae():
    # Real class is MaskedAutoencoder (no create_* factory); encoder must
    # emit a 4D (B,H,W,C) feature map. patch_size must divide H/W (32).
    from dl_techniques.models.masked_autoencoder.mae import MaskedAutoencoder
    enc = keras.Sequential([
        keras.layers.Input((32, 32, 3)),
        keras.layers.Conv2D(32, 3, padding="same"),
        keras.layers.Conv2D(64, 3, padding="same"),
    ])
    return MaskedAutoencoder(encoder=enc, patch_size=16, input_shape=(32, 32, 3))
_reg("masked_autoencoder", "RUN", _b_mae,
     lambda m: m(_img(), training=False), "MaskedAutoencoder; inline conv encoder")


# --- masked_language_model (needs encoder) ---------------------------------
def _b_mlm():
    # Real class is MaskedLanguageModel (no create_* factory). MLM head
    # vocab_size must match the BERT encoder vocab so token ids stay valid.
    from dl_techniques.models.masked_language_model.mlm import MaskedLanguageModel
    from dl_techniques.models.bert.bert import BERT
    enc = BERT.from_variant("tiny", vocab_size=1000)
    return MaskedLanguageModel(encoder=enc, vocab_size=1000, mask_token_id=103)
def _f_mlm(m):
    ids = _tokens(vocab=1000)
    mask = np.ones_like(ids)
    return m({"input_ids": ids, "attention_mask": mask}, training=False)
_reg("masked_language_model", "RUN", _b_mlm, _f_mlm, "inline BERT-tiny encoder")


# --- memory_bank -----------------------------------------------------------
def _b_membank():
    from dl_techniques.models.memory_bank.wave_field_memory_llm import WaveFieldMemoryLLM
    return WaveFieldMemoryLLM.from_variant("tiny")
_reg("memory_bank", "RUN", _b_membank,
     lambda m: m(_tokens(vocab=256), training=False), "tiny")


# --- mini_vec2vec ----------------------------------------------------------
def _b_minivec():
    from dl_techniques.models.mini_vec2vec.model import create_mini_vec2vec_aligner
    return create_mini_vec2vec_aligner(embedding_dim=128)
_reg("mini_vec2vec", "RUN", _b_minivec,
     lambda m: m(_vec(d=128), training=False), "")


# --- mobile_clip -----------------------------------------------------------
def _b_mobileclip():
    from dl_techniques.models.mobile_clip.mobile_clip_v1 import MobileClipModel
    return MobileClipModel.from_variant("s0")
def _f_mobileclip(m):
    inp = {"images": _img(h=256, w=256), "texts": _tokens(vocab=1000, s=16)}
    try:
        return m(inp, training=False)
    except Exception:
        return m({"image": _img(h=256, w=256), "text": _tokens(vocab=1000, s=16)},
                 training=False)
_reg("mobile_clip", "RUN", _b_mobileclip, _f_mobileclip, "s0")


# --- mobilenet (v1-v4) -----------------------------------------------------
def _b_mnv1():
    from dl_techniques.models.mobilenet.mobilenet_v1 import MobileNetV1
    return MobileNetV1.from_variant("small", num_classes=10, input_shape=(32, 32, 3))
_reg("mobilenet_v1", "RUN", _b_mnv1, lambda m: m(_img(), training=False), "small")

def _b_mnv2():
    from dl_techniques.models.mobilenet.mobilenet_v2 import MobileNetV2
    return MobileNetV2.from_variant("small", num_classes=10, input_shape=(32, 32, 3))
_reg("mobilenet_v2", "RUN", _b_mnv2, lambda m: m(_img(), training=False), "small")

def _b_mnv3():
    from dl_techniques.models.mobilenet.mobilenet_v3 import MobileNetV3
    return MobileNetV3.from_variant("small", num_classes=10, input_shape=(32, 32, 3))
_reg("mobilenet_v3", "RUN", _b_mnv3, lambda m: m(_img(), training=False), "small")

def _b_mnv4():
    from dl_techniques.models.mobilenet.mobilenet_v4 import MobileNetV4
    return MobileNetV4.from_variant("small", num_classes=10, input_shape=(32, 32, 3))
_reg("mobilenet_v4", "RUN", _b_mnv4, lambda m: m(_img(), training=False), "small")


# --- modern_bert (+ blt + hrm) ---------------------------------------------
def _b_modernbert():
    from dl_techniques.models.modern_bert.modern_bert import ModernBERT
    return ModernBERT.from_variant("base")
def _f_modernbert(m):
    ids = _tokens(vocab=1000)
    try:
        return m({"input_ids": ids}, training=False)
    except Exception:
        return m(ids, training=False)
_reg("modern_bert", "RUN", _b_modernbert, _f_modernbert, "base")

def _b_modernbert_blt():
    from dl_techniques.models.modern_bert.modern_bert_blt import ModernBertBLT
    return ModernBertBLT.from_variant("base")
_reg("modern_bert_blt", "RUN", _b_modernbert_blt,
     lambda m: m(_tokens(vocab=256), training=False), "BLT base (byte input)")

def _b_modernbert_hrm():
    from dl_techniques.models.modern_bert.modern_bert_blt_hrm import (
        ReasoningByteBERT, create_reasoning_byte_bert_base,
    )
    return ReasoningByteBERT(config=create_reasoning_byte_bert_base())
_reg("modern_bert_blt_hrm", "RUN", _b_modernbert_hrm,
     lambda m: m(_tokens(vocab=256), training=False),
     "ReasoningByteBERT (now functional, commit 124d464b)")


# --- mothnet ---------------------------------------------------------------
def _b_mothnet():
    from dl_techniques.models.mothnet.model import MothNet
    return MothNet(num_classes=10)
_reg("mothnet", "RUN", _b_mothnet, lambda m: m(_vec(d=64), training=False),
     "flat feature vector")


# --- nam -------------------------------------------------------------------
def _b_nam():
    from dl_techniques.models.nam.model import NAM
    return NAM.from_variant("tiny")
def _f_nam(m):
    # ACT-step model: call(carry, batch). vocab_size=21 (tiny default).
    batch = {"input_ids": _tokens(vocab=21, s=16)}
    carry = m.initial_carry(batch)
    new_carry, outputs = m(carry, batch, training=False)
    return outputs
_reg("nam", "RUN", _b_nam, _f_nam, "tiny; ACT-step call(carry, batch)")


# --- nano_vlm --------------------------------------------------------------
def _b_nanovlm():
    from dl_techniques.models.nano_vlm.model import create_nanovlm
    return create_nanovlm(variant="mini", vocab_size=32000)
def _f_nanovlm(m):
    return m({"images": _img(h=224, w=224),
              "text_tokens": _tokens(vocab=32000, s=16)}, training=False)
_reg("nano_vlm", "RUN", _b_nanovlm, _f_nanovlm, "mini; {images, text_tokens}")


# --- nano_vlm_world_model (DEAD) -------------------------------------------
def _b_nanovlm_wm():
    from dl_techniques.models.nano_vlm_world_model.model import create_score_based_nanovlm
    return create_score_based_nanovlm(variant="mini")
def _f_nanovlm_wm(m):
    return m({"image": _img(h=224, w=224), "input_ids": _tokens(vocab=32000, s=16)},
             training=False)
_reg("nano_vlm_world_model", "XFAIL", _b_nanovlm_wm, _f_nanovlm_wm,
     "DEAD: keras.random.uniform int32 dtype bug (SYSTEM.md)")


# --- ntm -------------------------------------------------------------------
def _b_ntm():
    from dl_techniques.models.ntm.model import NTMModel
    return NTMModel.from_variant("tiny", input_shape=(8, 8), output_dim=10)
def _f_ntm(m):
    x = np.random.rand(2, 8, 8).astype(np.float32)  # (B, T, input_dim)
    return m(x, training=False)
_reg("ntm", "RUN", _b_ntm, _f_ntm, "small; (B,T,input_dim)")


# --- pft_sr ----------------------------------------------------------------
def _b_pftsr():
    from dl_techniques.models.pft_sr.model import create_pft_sr
    return create_pft_sr(scale=2, variant="light")
_reg("pft_sr", "RUN", _b_pftsr,
     lambda m: m(_img(h=32, w=32), training=False), "light, scale2")


# --- power_mlp -------------------------------------------------------------
def _b_powermlp():
    from dl_techniques.models.power_mlp.model import PowerMLP
    return PowerMLP.from_variant("small", num_classes=10, input_dim=784)
_reg("power_mlp", "RUN", _b_powermlp, lambda m: m(_vec(d=784), training=False), "small")


# --- pw_fnet ---------------------------------------------------------------
def _b_pwfnet():
    from dl_techniques.models.pw_fnet.model import PW_FNet
    return PW_FNet(img_channels=3, width=16)
_reg("pw_fnet", "RUN", _b_pwfnet,
     lambda m: m(_img(h=64, w=64), training=False), "denoiser/restoration")


# --- qwen ------------------------------------------------------------------
def _b_qwen():
    from dl_techniques.models.qwen.qwen3 import Qwen3
    return Qwen3.from_variant("tiny")
_reg("qwen", "RUN", _b_qwen, lambda m: m(_tokens(vocab=256), training=False), "qwen3 tiny")


# --- relgt -----------------------------------------------------------------
def _b_relgt():
    from dl_techniques.models.relgt.model import create_relgt_model
    return create_relgt_model(output_dim=10, model_size="small")
def _f_relgt(m):
    # RELGTTokenEncoder needs 5 keys: node_features, node_types,
    # hop_distances, relative_times, subgraph_adjacency.
    B, K, F = 2, 10, 32
    inp = {
        "node_features": np.random.rand(B, K, F).astype(np.float32),
        "node_types": np.random.randint(0, 10, (B, K)).astype(np.int32),
        "hop_distances": np.random.randint(0, 3, (B, K)).astype(np.int32),
        "relative_times": np.random.rand(B, K, 1).astype(np.float32),
        "subgraph_adjacency": np.random.rand(B, K, K).astype(np.float32),
    }
    return m(inp, training=False)
_reg("relgt", "RUN", _b_relgt, _f_relgt,
     "small; 5-key graph token dict")


# --- resnet ----------------------------------------------------------------
def _b_resnet():
    from dl_techniques.models.resnet.model import ResNet
    return ResNet.from_variant("resnet18", num_classes=10, input_shape=(32, 32, 3))
_reg("resnet", "RUN", _b_resnet, lambda m: m(_img(), training=False), "resnet18")


# --- sam -------------------------------------------------------------------
def _b_sam():
    from dl_techniques.models.sam.model import SAM
    return SAM.from_variant("vit_b")
def _f_sam(m):
    # dict {image, original_size}; original_size must be a TENSOR (a python
    # tuple trips Keras' positional-arg guard on the nested dict input).
    image = _img(h=1024, w=1024)
    osz = keras.ops.convert_to_tensor([1024, 1024])
    return m({"image": image, "original_size": osz}, training=False)
# Recipe is correct (tensor original_size clears the input guard); the residual
# FAIL is a GENUINE model bug: WindowedAttentionWithRelPos multiplies an int32
# tensor by a float (1.0) -> "Expected int32 ... got 1.0". Not a harness error.
_reg("sam", "RUN", _b_sam, _f_sam,
     "vit_b @1024; {image, original_size(tensor)} - GENUINE bug in encoder attn")


# --- scunet ----------------------------------------------------------------
def _b_scunet():
    from dl_techniques.models.scunet.model import SCUNet
    return SCUNet(in_nc=3, dim=64, input_resolution=64)
_reg("scunet", "RUN", _b_scunet,
     lambda m: m(_img(h=64, w=64), training=False), "input_resolution must match")


# --- sd3_mmdit -------------------------------------------------------------
def _b_sd3():
    from dl_techniques.models.sd3_mmdit.transformer import create_sd3_mmdit
    return create_sd3_mmdit(variant="tiny")
def _f_sd3(m):
    # MM-DiT dict: latent, encoder_hidden_states, pooled_projections, timestep.
    cfg = m.config
    s = cfg.sample_size            # latent grid edge (16 for tiny)
    in_ch = cfg.in_channels
    jdim = cfg.joint_attention_dim
    pdim = cfg.pooled_projection_dim
    B = 2
    return m({
        "latent": np.random.rand(B, s, s, in_ch).astype(np.float32),
        "encoder_hidden_states": np.random.rand(B, 16, jdim).astype(np.float32),
        "pooled_projections": np.random.rand(B, pdim).astype(np.float32),
        "timestep": np.array([1.0, 1.0], dtype=np.float32),
    }, training=False)
_reg("sd3_mmdit", "RUN", _b_sd3, _f_sd3,
     "tiny; {latent, encoder_hidden_states, pooled_projections, timestep}")


# --- shgcn (DEAD) ----------------------------------------------------------
def _b_shgcn():
    from dl_techniques.models.shgcn.model import SHGCNModel
    return SHGCNModel(hidden_dims=[64, 32], output_dim=10)
def _f_shgcn(m):
    feats = np.random.rand(2, 16, 8).astype(np.float32)
    adj = np.eye(16, dtype=np.float32)[None].repeat(2, 0)
    return m([feats, adj], training=False)
_reg("shgcn", "XFAIL", _b_shgcn, _f_shgcn,
     "DEAD: needs SparseTensor adjacency (SYSTEM.md)")


# --- som -------------------------------------------------------------------
def _b_som():
    from dl_techniques.models.som.model import SOMModel
    return SOMModel(map_size=(10, 10), input_dim=64)
_reg("som", "RUN", _b_som, lambda m: m(_vec(d=64), training=False), "SOM clustering")


# --- squeezenet (v1/v2) ----------------------------------------------------
def _b_sqv1():
    from dl_techniques.models.squeezenet.squeezenet_v1 import SqueezeNetV1
    return SqueezeNetV1.from_variant("1.0", num_classes=10, input_shape=(32, 32, 3))
_reg("squeezenet_v1", "RUN", _b_sqv1, lambda m: m(_img(), training=False), "1.0")

def _b_sqv2():
    from dl_techniques.models.squeezenet.squeezenet_v2 import SqueezeNoduleNetV2
    return SqueezeNoduleNetV2.from_variant("v2_3d", num_classes=2, input_shape=(32, 32, 32, 1))
def _f_sqv2(m):
    x = np.random.rand(2, 32, 32, 32, 1).astype(np.float32)  # (B,D,H,W,1) 3D volume
    return m(x, training=False)
# Recipe is correct (v2_3d + 4D volume builds & forwards). Residual FAIL is a
# GENUINE non-finite forward (NaN logits on untrained model, no NaN weights) --
# same family as squeezenet_v1 in the baseline. Not a harness error.
_reg("squeezenet_v2", "RUN", _b_sqv2, _f_sqv2,
     "v2_3d (B,D,H,W,1) - GENUINE non-finite forward bug")


# --- swin_transformer (DEAD) -----------------------------------------------
def _b_swin():
    from dl_techniques.models.swin_transformer.model import SwinTransformer
    return SwinTransformer.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))
_reg("swin_transformer", "XFAIL", _b_swin, lambda m: m(_img(), training=False),
     "DEAD: 4D/3D shape mismatch in block (SYSTEM.md)")


# --- tabm ------------------------------------------------------------------
def _b_tabm():
    from dl_techniques.models.tabm.model import create_tabm_mini
    return create_tabm_mini(n_num_features=16, cat_cardinalities=[], n_classes=2)
_reg("tabm", "RUN", _b_tabm, lambda m: m(_vec(d=16), training=False),
     "mini; 16 num features, no cat, 2 classes")


# --- thera -----------------------------------------------------------------
def _b_thera():
    from dl_techniques.models.thera.model import build_thera
    return build_thera(out_dim=3, backbone="edsr-baseline", size="air")
def _f_thera(m):
    src = _img(h=32, w=32)
    scale = np.array(2.0, dtype=np.float32)
    return m([src, scale], training=False)
_reg("thera", "RUN", _b_thera, _f_thera, "edsr-air; [source, scale]")


# --- time_series (7 subpackages) -------------------------------------------
def _b_xlstm():
    # xLSTMForecaster lives in forecaster.py (model.py has the LM xLSTM).
    from dl_techniques.models.time_series.xlstm.forecaster import xLSTMForecaster
    return xLSTMForecaster.from_variant("tiny", input_length=48,
                                        prediction_length=12, num_features=4)
def _f_ts3(m):
    x = np.random.rand(2, 48, 4).astype(np.float32)  # (B, T, features)
    return m(x, training=False)
_reg("time_series_xlstm", "RUN", _b_xlstm, _f_ts3,
     "tiny forecaster; (B,48,4)")

def _b_nbeats():
    # Factory lives in nbeats.py (no nbeats/model.py); args are
    # backcast_length / forecast_length.
    from dl_techniques.models.time_series.nbeats.nbeats import create_nbeats_model
    return create_nbeats_model(backcast_length=96, forecast_length=24)
def _f_ts_uni(m):
    x = np.random.rand(2, 96).astype(np.float32)  # (B, T) univariate
    return m(x, training=False)
_reg("time_series_nbeats", "RUN", _b_nbeats, _f_ts_uni,
     "backcast96/forecast24; (B,96)")

def _b_prism():
    from dl_techniques.models.time_series.prism.model import PRISMModel
    return PRISMModel.from_variant("tiny", context_len=48, forecast_len=12,
                                   num_features=4)
_reg("time_series_prism", "RUN", _b_prism, _f_ts3,
     "tiny; (B,48,4) context window")

def _b_tirex():
    from dl_techniques.models.time_series.tirex.model import TiRexCore
    return TiRexCore.from_variant("tiny", prediction_length=24)
_reg("time_series_tirex", "RUN", _b_tirex, _f_ts_uni,
     "tiny (UNSURE: input rank)")

def _b_deepar():
    from dl_techniques.models.time_series.deepar.model import create_deepar
    return create_deepar(hidden_dim=16, num_layers=1, target_dim=1, covariate_dim=4)
def _f_deepar(m):
    # training-mode dict: target (B,T,target_dim), covariates (B,T,cov_dim).
    return m({
        "target": np.random.rand(2, 48, 1).astype(np.float32),
        "covariates": np.random.rand(2, 48, 4).astype(np.float32),
    }, training=False)
_reg("time_series_deepar", "RUN", _b_deepar, _f_deepar,
     "{target, covariates}")

def _f_ts_feat(m):
    x = np.random.rand(2, 48, 10).astype(np.float32)
    return m(x, training=False)

def _b_mdn():
    from dl_techniques.models.time_series.mdn.model import create_mdn_model
    return create_mdn_model(hidden_layers=[32, 32], output_dimension=1,
                            num_mixtures=5, input_dimension=10)
def _f_mdn(m):
    x = np.random.rand(2, 10).astype(np.float32)  # (B, input_dimension)
    return m(x, training=False)
_reg("time_series_mdn", "RUN", _b_mdn, _f_mdn, "flat (B,10) features")

def _b_adaptive_ema():
    from dl_techniques.models.time_series.adaptive_ema.model import (
        create_adaptive_ema_slope_filter,
    )
    return create_adaptive_ema_slope_filter()
_reg("time_series_adaptive_ema", "RUN", _b_adaptive_ema, _f_ts_uni,
     "(UNSURE: constructor args + input rank)")


# --- tiny_recursive_model --------------------------------------------------
def _b_trm():
    from dl_techniques.models.tiny_recursive_model.model import create_trm
    return create_trm(vocab_size=256, hidden_size=64, num_heads=4,
                      expansion=2.0, seq_len=16)
def _f_trm(m):
    # ACT-step model: call(carry, batch); batch dict key "inputs" (B, seq_len).
    batch = {"inputs": _tokens(vocab=256, s=16)}
    carry = m.initial_carry(batch)
    new_carry, outputs = m(carry, batch, training=False)
    return outputs
_reg("tiny_recursive_model", "RUN", _b_trm, _f_trm,
     "ACT-step call(carry, batch)")


# --- tree_transformer ------------------------------------------------------
def _b_treetf():
    from dl_techniques.models.tree_transformer.model import create_tree_transformer
    return create_tree_transformer("tiny")
def _f_treetf(m):
    ids = _tokens(vocab=1000)
    mask = np.ones_like(ids)
    return m({"input_ids": ids, "attention_mask": mask}, training=False)
_reg("tree_transformer", "RUN", _b_treetf, _f_treetf, "tiny")


# --- vae -------------------------------------------------------------------
def _b_vae():
    from dl_techniques.models.vae.model import VAE
    return VAE.from_variant("small", input_shape=(28, 28, 1), latent_dim=64)
_reg("vae", "RUN", _b_vae,
     lambda m: m(_img(h=28, w=28, c=1), training=False), "small")


# --- video_jepa ------------------------------------------------------------
def _b_videojepa():
    from dl_techniques.models.video_jepa.model import VideoJEPA
    from dl_techniques.models.video_jepa.config import VideoJEPAConfig
    return VideoJEPA(config=VideoJEPAConfig())
def _f_videojepa(m):
    x = np.random.rand(2, 4, 32, 32, 3).astype(np.float32)  # (B,T,H,W,C)
    return m(x, training=False)
_reg("video_jepa", "RUN", _b_videojepa, _f_videojepa,
     "default config (UNSURE: T/H/W from config)")


# --- vit -------------------------------------------------------------------
def _b_vit():
    from dl_techniques.models.vit.model import ViT
    return ViT.from_variant("vit_tiny", num_classes=10, input_shape=(32, 32, 3))
_reg("vit", "RUN", _b_vit, lambda m: m(_img(), training=False), "vit_tiny")


# --- vit_hmlp --------------------------------------------------------------
def _b_vithmlp():
    from dl_techniques.models.vit_hmlp.model import create_vit_hmlp
    return create_vit_hmlp(input_shape=(32, 32, 3), num_classes=10, scale="tiny")
_reg("vit_hmlp", "RUN", _b_vithmlp, lambda m: m(_img(), training=False), "tiny")


# --- vit_siglip ------------------------------------------------------------
def _b_vitsiglip():
    from dl_techniques.models.vit_siglip.model import create_siglip_vision_transformer
    return create_siglip_vision_transformer(input_shape=(32, 32, 3), num_classes=10, scale="tiny")
_reg("vit_siglip", "RUN", _b_vitsiglip, lambda m: m(_img(), training=False), "tiny")


# --- vq_vae (needs encoder+decoder) ----------------------------------------
def _b_vqvae():
    from dl_techniques.models.vq_vae.model import VQVAEModel
    enc = keras.Sequential([
        keras.layers.Input((32, 32, 3)),
        keras.layers.Conv2D(32, 3, strides=2, padding="same"),
        keras.layers.Conv2D(64, 3, strides=2, padding="same"),
    ])
    dec = keras.Sequential([
        keras.layers.Input((8, 8, 64)),
        keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same"),
        keras.layers.Conv2DTranspose(3, 3, strides=2, padding="same"),
    ])
    return VQVAEModel(encoder=enc, decoder=dec, num_embeddings=512, embedding_dim=64)
_reg("vq_vae", "RUN", _b_vqvae, lambda m: m(_img(), training=False),
     "inline tiny encoder/decoder")


# --- vq_vae_rotation -------------------------------------------------------
def _b_vqvae_rot():
    from dl_techniques.models.vq_vae_rotation.model import VQVAERotationTrick
    return VQVAERotationTrick(num_embeddings=512, embedding_dim=64, input_shape=(32, 32, 3))
_reg("vq_vae_rotation", "RUN", _b_vqvae_rot, lambda m: m(_img(), training=False),
     "auto-build path via input_shape")


# --- wave_field_llm --------------------------------------------------------
def _b_wavellm():
    from dl_techniques.models.wave_field_llm.wave_field_llm import WaveFieldLLM
    return WaveFieldLLM.from_variant("tiny")
_reg("wave_field_llm", "RUN", _b_wavellm,
     lambda m: m(_tokens(vocab=256), training=False), "tiny")


# --- yolo12 ----------------------------------------------------------------
def _b_yolo12():
    from dl_techniques.models.yolo12.feature_extractor import (
        create_yolov12_feature_extractor,
    )
    return create_yolov12_feature_extractor(input_shape=(256, 256, 3), scale="n")
_reg("yolo12", "RUN", _b_yolo12,
     lambda m: m(_img(h=256, w=256), training=False),
     "feature extractor scale n; multi-scale pyramid output")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run(only=None, verbose=False):
    results = []  # (name, status, info)
    fail_tracebacks = {}

    entries = REGISTRY
    if only:
        wanted = {n.strip() for n in only.split(",") if n.strip()}
        entries = [e for e in REGISTRY if e["name"] in wanted]
        missing = wanted - {e["name"] for e in entries}
        for m in sorted(missing):
            results.append((m, "SKIP", "not in registry (--only filter)"))

    for e in entries:
        name, mode = e["name"], e["mode"]
        if mode == "SKIP":
            results.append((name, "SKIP", e["note"]))
            continue

        model = None
        try:
            model = e["build"]()
            out = e["forward"](model)
            ok, shapes = _all_finite(out)
            if not ok:
                raise ValueError(f"non-finite output (shapes: {shapes})")
            if mode == "XFAIL":
                results.append((name, "XPASS", f"unexpectedly passed; {shapes}"))
            else:
                results.append((name, "PASS", shapes))
        except Exception:
            tb = traceback.format_exc()
            first = ""
            for line in reversed(tb.strip().splitlines()):
                if line.strip():
                    first = line.strip()
                    break
            if mode == "XFAIL":
                results.append((name, "XFAIL", f"{e['note']} | {first}"))
            else:
                results.append((name, "FAIL", first))
                fail_tracebacks[name] = tb[-1500:]
        finally:
            del model
            gc.collect()
            try:
                keras.backend.clear_session()
            except Exception:
                pass

    # --- print matrix ------------------------------------------------------
    print("=" * 78)
    print("MODEL SMOKE MATRIX")
    print("=" * 78)
    for name, status, info in results:
        print(f"{status:6s}  {name:30s}  {info}")

    # --- summary -----------------------------------------------------------
    counts = {k: 0 for k in ("PASS", "FAIL", "XFAIL", "XPASS", "SKIP")}
    for _, status, _ in results:
        counts[status] = counts.get(status, 0) + 1

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  total entries : {len(results)}")
    for k in ("PASS", "FAIL", "XFAIL", "XPASS", "SKIP"):
        print(f"  {k:6s}: {counts[k]}")

    fails = [n for n, s, _ in results if s == "FAIL"]
    xpass = [n for n, s, _ in results if s == "XPASS"]
    if fails:
        print(f"\n  FAIL (need attention): {', '.join(fails)}")
    if xpass:
        print(f"  XPASS (now works, retire XFAIL?): {', '.join(xpass)}")
    if not fails and not xpass:
        print("\n  no FAIL / XPASS — matrix clean")

    # --- verbose tracebacks ------------------------------------------------
    if verbose and fail_tracebacks:
        print("\n" + "=" * 78)
        print("FAIL TRACEBACKS")
        print("=" * 78)
        for name, tb in fail_tracebacks.items():
            print(f"\n----- {name} -----")
            print(tb)

    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None,
                    help="comma-separated entry names to run (default: all)")
    ap.add_argument("--verbose", action="store_true",
                    help="print full stored tracebacks for FAIL entries")
    args = ap.parse_args()
    run(only=args.only, verbose=args.verbose)
    # Always exit 0: this is a report, not a gate.
    sys.exit(0)


if __name__ == "__main__":
    main()
