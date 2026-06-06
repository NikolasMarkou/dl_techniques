"""F2/F3 fix for the convnext_patch_vae vMF generative prior.

Diagnosis (epistemic-deconstructor analysis_2026-06-06_0c7feade):
  - H7  the viz callback decoded off-sphere N(0,I) -> noise (fixed separately, F0).
  - H10 the residual: the factorized Uniform(S^31) prior samples OFF the thin joint
        manifold where the encoder put real images -> coherent-but-unstructured samples.
        Phase-3 dose-response: the needed joint context is LOCAL (~3-4 patches).

This script learns a JOINT prior over the frozen encoder's latent grid (16x16x32):
  a small *spatial* conv two-stage VAE (decoder frozen; Dai & Wipf 2019 two-stage VAE).
Sampling the learned prior -> stage-1 decode should land ON the manifold.

It also evaluates a zero-training STRUCTURED-NOISE baseline (blur N(0,I) across the
grid, then L2-normalize) which injects local correlation for free.

Scored with the Phase-2 metric: re-encoded adjacent-vs-distant patch cosine gap
(real images ~0.10; factorized uniform ~0.03). Higher = more realistic structure.

Run:
  CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m experiments.convnext_patchvae_latent_prior
"""
import os, glob
import numpy as np
import tensorflow as tf
import keras
from keras import layers, ops
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from dl_techniques.utils.logger import logger

CKPT = ("/media/arxwn/data_fast/repositories/dl_techniques/results/"
        "convnext_patch_vae_ade20k+coco_large_20260606_094857/best_model.keras")
IMG_DIR = "/media/arxwn/data0_4tb/datasets/coco_2017/val2017"
OUT_DIR = "/media/arxwn/data_fast/repositories/dl_techniques/results/latent_prior_fix"
os.makedirs(OUT_DIR, exist_ok=True)

N_TRAIN = int(os.environ.get("N_TRAIN", "3000"))
N_EVAL = int(os.environ.get("N_EVAL", "64"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))
BETA = float(os.environ.get("BETA", "0.005"))


# --------------------------------------------------------------------------- #
# helpers (metric + io reused from the analysis probe)
# --------------------------------------------------------------------------- #
def load_images(paths, n, size=256, seed=42):
    import cv2
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=min(n * 2, len(paths)), replace=False)
    imgs = []
    for i in idx:
        img = cv2.imread(paths[i])
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size))
        imgs.append(img.astype(np.float32) / 255.0)
        if len(imgs) == n:
            break
    return np.array(imgs)


def unit(a, axis=-1):
    return a / np.maximum(np.linalg.norm(a, axis=axis, keepdims=True), 1e-8)


def structure_gap(z_grid):
    """adjacent-patch cosine minus distant-patch cosine on a (B,Hp,Wp,D) grid."""
    zn = unit(z_grid, axis=-1)
    h = np.sum(zn[:, :, :-1, :] * zn[:, :, 1:, :], axis=-1).mean()
    v = np.sum(zn[:, :-1, :, :] * zn[:, 1:, :, :], axis=-1).mean()
    adj = float((h + v) / 2)
    B, Hp, Wp, D = zn.shape
    rng = np.random.default_rng(0)
    sims = []
    for _ in range(2000):
        b = rng.integers(B)
        i1, j1, i2, j2 = rng.integers(Hp), rng.integers(Wp), rng.integers(Hp), rng.integers(Wp)
        if abs(i1 - i2) + abs(j1 - j2) < 3:
            continue
        sims.append(float(np.sum(zn[b, i1, j1] * zn[b, i2, j2])))
    dist = float(np.mean(sims))
    return adj, dist, adj - dist


def total_variation(imgs):
    return float(np.mean(np.abs(np.diff(imgs, axis=1))) + np.mean(np.abs(np.diff(imgs, axis=2))))


# --------------------------------------------------------------------------- #
# stage-2 learned prior: small spatial conv VAE over the (16,16,32) latent grid
# --------------------------------------------------------------------------- #
def build_latent_prior(Hp, Wp, D, w_ch=16):
    """Spatial conv two-stage VAE. Latent w is spatial (Hp/4, Wp/4, w_ch) to
    preserve the LOCAL structure the dose-response said matters (~4 patches)."""
    def conv(x, c, s=1):
        x = layers.Conv2D(c, 3, strides=s, padding="same")(x)
        x = layers.GroupNormalization(groups=min(8, c))(x)
        return layers.Activation("gelu")(x)

    # encoder z(16,16,D) -> w(4,4,w_ch)
    zin = keras.Input((Hp, Wp, D))
    h = conv(zin, 64)
    h = conv(h, 96, s=2)          # 8x8
    h = conv(h, 128, s=2)         # 4x4
    mu = layers.Conv2D(w_ch, 1)(h)
    logvar = layers.Conv2D(w_ch, 1)(h)
    enc = keras.Model(zin, [mu, logvar], name="prior_enc")

    # decoder w(4,4,w_ch) -> zhat(16,16,D)
    win = keras.Input((Hp // 4, Wp // 4, w_ch))
    g = conv(win, 128)
    g = layers.UpSampling2D()(g)   # 8x8
    g = conv(g, 96)
    g = layers.UpSampling2D()(g)   # 16x16
    g = conv(g, 64)
    zout = layers.Conv2D(D, 1)(g)
    dec = keras.Model(win, zout, name="prior_dec")
    return enc, dec


class LatentPrior(keras.Model):
    def __init__(self, Hp, Wp, D, beta=0.005, **kw):
        super().__init__(**kw)
        self.enc, self.dec = build_latent_prior(Hp, Wp, D)
        self.beta = beta
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.cos_tracker = keras.metrics.Mean(name="cos")
        self.kl_tracker = keras.metrics.Mean(name="kl")

    def _normalize(self, zraw):
        n = ops.sqrt(ops.sum(ops.square(zraw), axis=-1, keepdims=True))
        return zraw / ops.maximum(n, 1e-8)

    def call(self, z, training=False):
        mu, logvar = self.enc(z, training=training)
        if training:
            eps = keras.random.normal(ops.shape(mu))
            w = mu + ops.exp(0.5 * ops.clip(logvar, -10, 10)) * eps
        else:
            w = mu
        return self._normalize(self.dec(w, training=training)), mu, logvar

    def train_step(self, data):
        z = data[0] if isinstance(data, (tuple, list)) else data
        with tf.GradientTape() as tape:
            zhat, mu, logvar = self(z, training=True)
            cos = ops.mean(1.0 - ops.sum(z * zhat, axis=-1))          # per-patch cosine
            kl = ops.mean(0.5 * ops.sum(
                ops.exp(ops.clip(logvar, -10, 10)) + ops.square(mu) - 1.0 - logvar,
                axis=[1, 2, 3]))
            loss = cos + self.beta * kl
        self.optimizer.apply_gradients(zip(
            tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.cos_tracker.update_state(cos)
        self.kl_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.loss_tracker, self.cos_tracker, self.kl_tracker]

    def sample(self, n, seed=0):
        w = keras.random.normal((n,) + self.dec.input_shape[1:], seed=seed)
        return self._normalize(self.dec(w, training=False))


# --------------------------------------------------------------------------- #
def main():
    logger.info(f"Loading frozen VAE: {CKPT}")
    model = keras.models.load_model(CKPT, compile=False)
    model.trainable = False
    paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))

    # ---- cache latents from real images (frozen encoder) ----
    logger.info(f"Encoding {N_TRAIN} train + {N_EVAL} eval images -> latent grids")
    imgs = load_images(paths, n=N_TRAIN + N_EVAL, seed=7)
    zs = []
    for k in range(0, len(imgs), 64):
        zs.append(model(tf.constant(imgs[k:k + 64]), training=False)["z"].numpy())
    z_all = np.concatenate(zs).astype(np.float32)
    z_eval, z_train = z_all[:N_EVAL], z_all[N_EVAL:]
    eval_imgs = imgs[:N_EVAL]
    B, Hp, Wp, D = z_train.shape
    logger.info(f"latent grids: train={z_train.shape} eval={z_eval.shape} "
                f"norm_mean={np.linalg.norm(z_train, axis=-1).mean():.4f}")

    def decode_reencode(z_grid):
        dec = np.clip(np.array(model.decode(tf.constant(z_grid))), 0, 1)
        re = model(tf.constant(dec.astype(np.float32)), training=False)["z"].numpy()
        return dec, structure_gap(re)

    results = {}
    # ---- baselines ----
    _, real_gap = decode_reencode(z_eval)
    results["A_real"] = real_gap
    z_unif = unit(np.random.default_rng(1).standard_normal((N_EVAL, Hp, Wp, D)).astype(np.float32))
    dec_unif, gap_unif = decode_reencode(z_unif)
    results["B_uniform_F0"] = gap_unif
    # structured noise: blur N(0,I) across the grid then normalize (free local correlation)
    raw = np.random.default_rng(2).standard_normal((N_EVAL, Hp, Wp, D)).astype(np.float32)
    blurred = gaussian_filter(raw, sigma=(0, 1.5, 1.5, 0))
    z_struct = unit(blurred)
    dec_struct, gap_struct = decode_reencode(z_struct)
    results["C_structured_noise"] = gap_struct

    # ---- train learned prior ----
    logger.info(f"Training latent prior ({EPOCHS} epochs, beta={BETA}) on cached latents")
    prior = LatentPrior(Hp, Wp, D, beta=BETA)
    prior.compile(optimizer=keras.optimizers.Adam(1e-3))
    ds = tf.data.Dataset.from_tensor_slices(z_train).shuffle(4096).batch(64)
    hist = prior.fit(ds, epochs=EPOCHS, verbose=2)
    z_learned = np.array(prior.sample(N_EVAL, seed=3))
    logger.info(f"learned-prior sample norm_mean={np.linalg.norm(z_learned, axis=-1).mean():.4f} "
                f"latent-space gap={structure_gap(z_learned)[2]:.4f} (real latents ~{structure_gap(z_train)[2]:.4f})")
    dec_learned, gap_learned = decode_reencode(z_learned)
    results["D_learned_prior"] = gap_learned

    # ---- report ----
    logger.info("=" * 64)
    logger.info("RE-ENCODED STRUCTURE GAP (real ~0.10; factorized-uniform ~0.03; higher=better)")
    for name, (adj, dist, gap) in results.items():
        logger.info(f"  {name:20s} gap={gap:.4f}  (adj={adj:.4f} dist={dist:.4f})")
    recov = (results["D_learned_prior"][2] - results["B_uniform_F0"][2]) / \
            max(results["A_real"][2] - results["B_uniform_F0"][2], 1e-6)
    logger.info(f"  learned-prior recovery toward real: {100 * recov:.1f}%")

    # ---- visual grids ----
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    rows = [("real recon", np.clip(np.array(model.decode(tf.constant(z_eval))), 0, 1)),
            ("uniform (F0)", dec_unif),
            ("structured noise", dec_struct),
            ("learned prior (F3)", dec_learned)]
    for r, (label, grid) in enumerate(rows):
        for c in range(8):
            axes[r, c].imshow(grid[c]); axes[r, c].axis("off")
        axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
    plt.suptitle("convnext_patch_vae vMF generation: F0 vs structured-noise vs learned prior")
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "generation_comparison.png")
    plt.savefig(out_png, dpi=110); plt.close()
    logger.info(f"Saved comparison grid -> {out_png}")
    prior.save(os.path.join(OUT_DIR, "latent_prior.keras"))
    logger.info(f"Saved learned prior -> {OUT_DIR}/latent_prior.keras")


if __name__ == "__main__":
    main()
