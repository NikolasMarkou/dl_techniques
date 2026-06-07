"""End-to-end VAE-GAN for convnext_patch_vae vMF — generation fix trained jointly.

Motivation (epistemic analysis analysis_2026-06-06_0c7feade):
  - H7 (eval bug): prior samples were drawn off-sphere -> baked in correctly here
    (Uniform(S^{d-1}) per patch).
  - H10 (joint aggregate-posterior mismatch): a factorized prior samples OFF the
    joint manifold -> coherent-but-unstructured generation. The two-stage diffusion
    prior fixed this *post-hoc* on a FROZEN decoder (latent-space objective only),
    which is why samples stayed blurry.

This trains the fix END-TO-END: encoder + decoder + discriminator jointly. The
adversarial loss is applied to BOTH reconstructions AND decoded *prior samples*, so
the decoder is explicitly optimized (in image space) to map the vMF uniform-sphere
prior to realistic images. L1 recon + small vMF-KL anchor the autoencoder (VAE-GAN
is far more stable than a pure GAN); R1 gradient penalty stabilizes D.

  G loss = L1(x, x_rec) + beta*KL(kappa) + adv_w*[adv(x_rec)+adv(x_prior)] + fm_w*FM(x_rec)
  D loss = nonsat( D(real)=1, D(x_rec)=0, D(x_prior)=0 ) + r1*R1(real)

Run:
  CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m experiments.convnext_patchvae_vaegan
"""
import os, glob
import numpy as np
import tensorflow as tf
import keras
from keras import layers, ops
import matplotlib.pyplot as plt

from dl_techniques.models.convnext_patch_vae.config import ConvNeXtPatchVAEConfig
from dl_techniques.models.convnext_patch_vae.model import ConvNeXtPatchVAE
from dl_techniques.layers.sampling import vmf_kl_divergence
from dl_techniques.utils.logger import logger

IMG_DIR_COCO = "/media/arxwn/data0_4tb/datasets/coco_2017/train2017"
IMG_DIR_ADE = "/media/arxwn/data0_4tb/datasets/ade20k/images/ADE/training"
OUT_DIR = "/media/arxwn/data_fast/repositories/dl_techniques/results/vaegan_e2e"
os.makedirs(OUT_DIR, exist_ok=True)

IMG = 256
PATCH = int(os.environ.get("PATCH", "16"))
LATENT = 32
EMBED = int(os.environ.get("EMBED", "192"))
DEPTH = int(os.environ.get("DEPTH", "6"))
BATCH = int(os.environ.get("BATCH", "16"))
STEPS = int(os.environ.get("STEPS", "24000"))
N_CACHE = int(os.environ.get("N_CACHE", "12000"))
ADV_W = float(os.environ.get("ADV_W", "0.5"))
FM_W = float(os.environ.get("FM_W", "5.0"))
BETA = float(os.environ.get("BETA", "1e-4"))
R1 = float(os.environ.get("R1", "5.0"))


def load_images(n, size=IMG, seed=7):
    import cv2
    paths = (sorted(glob.glob(os.path.join(IMG_DIR_COCO, "*.jpg")))
             + sorted(glob.glob(os.path.join(IMG_DIR_ADE, "**", "*.jpg"), recursive=True)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    out = []
    for i in idx:
        img = cv2.imread(paths[i])
        if img is None:
            continue
        out.append(cv2.cvtColor(cv2.resize(img, (size, size)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    return np.array(out)


def build_discriminator(size=IMG):
    x = keras.Input((size, size, 3))
    h = x
    feats = []
    for c, s in [(64, 2), (128, 2), (256, 2), (256, 2), (512, 2)]:
        h = layers.Conv2D(c, 4, strides=s, padding="same")(h)
        h = layers.GroupNormalization(groups=min(8, c))(h)
        h = layers.LeakyReLU(0.2)(h)
        feats.append(h)
    h = layers.Conv2D(1, 4, padding="same")(h)            # PatchGAN logits (8x8)
    return keras.Model(x, [h] + feats, name="discriminator")


class VAEGAN(keras.Model):
    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = ConvNeXtPatchVAEConfig(img_size=IMG, img_channels=3, patch_size=PATCH,
                                     embed_dim=EMBED, encoder_depth=DEPTH, decoder_depth=DEPTH,
                                     latent_dim=LATENT, sampling_type="vmf", lambda_sigreg=0.0,
                                     recon_loss_type="bce")
        self.vae = ConvNeXtPatchVAE(cfg)
        self.D = build_discriminator()
        self.hp = IMG // PATCH
        self.g_loss = keras.metrics.Mean(name="g")
        self.d_loss = keras.metrics.Mean(name="d")
        self.r_loss = keras.metrics.Mean(name="recon")
        self.a_loss = keras.metrics.Mean(name="adv")

    def compile(self, g_opt, d_opt):
        # vMF sampler uses keras.random.beta -> no XLA-GPU kernel in TF 2.18.
        super().compile(optimizer=g_opt, jit_compile=False)
        self.d_opt = d_opt

    def _encode_decode(self, x, training):
        mu, kappa = self.vae.encoder(x, training=training)
        B = ops.shape(mu)[0]
        mu_f = ops.reshape(mu, (-1, LATENT)); k_f = ops.reshape(kappa, (-1, 1))
        z = ops.reshape(self.vae.sampling([mu_f, k_f], training=training), (B, self.hp, self.hp, LATENT))
        x_rec = ops.sigmoid(self.vae.decoder(z, training=training))
        return x_rec, kappa

    def _prior_decode(self, n, training):
        eps = keras.random.normal((n, self.hp, self.hp, LATENT))
        z = eps / ops.maximum(ops.sqrt(ops.sum(ops.square(eps), -1, keepdims=True)), 1e-8)  # F0: on-sphere
        return ops.sigmoid(self.vae.decoder(z, training=training))

    def train_step(self, data):
        x = data[0] if isinstance(data, (tuple, list)) else data
        n = ops.shape(x)[0]

        # ---- D step ----
        x_rec, _ = self._encode_decode(x, training=True)
        x_pri = self._prior_decode(n, training=True)
        with tf.GradientTape() as dt:
            d_real = self.D(x, training=True)[0]
            d_rec = self.D(ops.stop_gradient(x_rec), training=True)[0]
            d_pri = self.D(ops.stop_gradient(x_pri), training=True)[0]
            d_loss = (ops.mean(keras.ops.softplus(-d_real))
                      + 0.5 * ops.mean(keras.ops.softplus(d_rec))
                      + 0.5 * ops.mean(keras.ops.softplus(d_pri)))
        dv = self.D.trainable_variables
        dg = dt.gradient(d_loss, dv)
        self.d_opt.apply_gradients((g, v) for g, v in zip(dg, dv) if g is not None)

        # ---- G step ----
        with tf.GradientTape() as gt:
            x_rec, kappa = self._encode_decode(x, training=True)
            x_pri = self._prior_decode(n, training=True)
            recon = ops.mean(ops.abs(x - x_rec))
            kl = ops.mean(vmf_kl_divergence(ops.reshape(kappa, (-1, 1)), dim=LATENT))
            out_rec = self.D(x_rec, training=True); out_pri = self.D(x_pri, training=True)
            out_real = self.D(x, training=True)
            adv = ops.mean(keras.ops.softplus(-out_rec[0])) + ops.mean(keras.ops.softplus(-out_pri[0]))
            fm = 0.0
            for fr, fk in zip(out_real[1:], out_rec[1:]):
                fm = fm + ops.mean(ops.abs(ops.stop_gradient(fr) - fk))
            g_loss = recon + BETA * kl + ADV_W * adv + FM_W * fm
        gv = self.vae.trainable_variables
        gg = gt.gradient(g_loss, gv)
        self.optimizer.apply_gradients((g, v) for g, v in zip(gg, gv) if g is not None)

        self.g_loss.update_state(g_loss); self.d_loss.update_state(d_loss)
        self.r_loss.update_state(recon); self.a_loss.update_state(adv)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.g_loss, self.d_loss, self.r_loss, self.a_loss]


def main():
    logger.info(f"loading images (cache {N_CACHE})")
    imgs = load_images(N_CACHE)
    logger.info(f"images {imgs.shape}; patch={PATCH} grid={IMG//PATCH} batch={BATCH} steps={STEPS}")
    ds = tf.data.Dataset.from_tensor_slices(imgs).shuffle(8192).repeat().batch(BATCH, drop_remainder=True)

    model = VAEGAN()
    model.compile(keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9),
                  keras.optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.9))
    _ = model._encode_decode(tf.constant(imgs[:2]), training=False)  # build submodules
    vae_p = sum(int(np.prod(w.shape)) for w in
                (model.vae.encoder.weights + model.vae.decoder.weights + model.vae.sampling.weights))
    logger.info(f"VAE params={vae_p:,}  D params={model.D.count_params():,}")

    epochs = max(1, STEPS // 1500)
    eval_imgs = imgs[:8]
    for ep in range(epochs):
        model.fit(ds, steps_per_epoch=1500, epochs=1, verbose=2)
        # snapshot grid: real | recon | prior-sample
        rec, _ = model._encode_decode(tf.constant(eval_imgs), training=False)
        pri = np.array(model._prior_decode(8, training=False))
        rec = np.clip(np.array(rec), 0, 1)
        fig, ax = plt.subplots(3, 8, figsize=(16, 6))
        for c in range(8):
            ax[0, c].imshow(eval_imgs[c]); ax[1, c].imshow(rec[c]); ax[2, c].imshow(pri[c])
            for r in range(3):
                ax[r, c].axis("off")
        for r, lab in enumerate(["real", "recon", "prior sample"]):
            ax[r, 0].axis("on"); ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
            ax[r, 0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=9)
        plt.suptitle(f"VAE-GAN end-to-end (vMF) — epoch {ep+1}/{epochs}")
        plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f"vaegan_epoch_{ep+1:03d}.png"), dpi=110); plt.close()
        logger.info(f"epoch {ep+1}/{epochs} grid saved")
    model.vae.save(os.path.join(OUT_DIR, "vaegan_vae.keras"))
    logger.info(f"saved -> {OUT_DIR}/vaegan_vae.keras")


if __name__ == "__main__":
    main()
