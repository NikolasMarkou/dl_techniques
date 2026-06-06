"""F3 (stronger): a latent DIFFUSION prior over the convnext_patch_vae vMF latent grid.

Why diffusion, not a 2-stage VAE: the frozen encoder's latents are WEAKLY correlated
but HIGH-entropy (adjacent-patch cosine ~0.12, per-patch ~uniform R_bar~0.13). A
compressive VAE bottleneck collapses that detail to smooth color blobs (verified:
convnext_patchvae_latent_prior.py -> mode collapse, cos-recon 0.11). A diffusion model
is non-compressive (same-resolution denoiser) and models subtle high-dim correlations
without posterior collapse.

Pipeline: freeze VAE -> cache latent grids z (16,16,32, unit-sphere per patch) ->
train a small conv UNet DDPM in the ambient R^32 grid space (predict-noise, cosine
schedule) -> DDIM-sample -> L2-normalize per patch (project to sphere) -> frozen decode.

Scored with the Phase-2 re-encoded structure-gap (real ~0.11; factorized-uniform ~0.03;
NOTE: gap is necessary-not-sufficient — too-high gap = over-smoothed; judge with the
grid PNG too).

Run:
  CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m experiments.convnext_patchvae_latent_diffusion
"""
import os, glob
import numpy as np
import tensorflow as tf
import keras
from keras import layers, ops
import matplotlib.pyplot as plt

from dl_techniques.utils.logger import logger

CKPT = ("/media/arxwn/data_fast/repositories/dl_techniques/results/"
        "convnext_patch_vae_ade20k+coco_large_20260606_094857/best_model.keras")
IMG_DIR = "/media/arxwn/data0_4tb/datasets/coco_2017/val2017"
OUT_DIR = "/media/arxwn/data_fast/repositories/dl_techniques/results/latent_prior_fix"
os.makedirs(OUT_DIR, exist_ok=True)

N_TRAIN = int(os.environ.get("N_TRAIN", "5000"))
N_EVAL = int(os.environ.get("N_EVAL", "64"))
EPOCHS = int(os.environ.get("EPOCHS", "120"))
T = 1000
DDIM_STEPS = 50


# ----------------------------- io + metric -------------------------------- #
def load_images(paths, n, size=256, seed=42):
    import cv2
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=min(n * 2, len(paths)), replace=False)
    imgs = []
    for i in idx:
        img = cv2.imread(paths[i])
        if img is None:
            continue
        img = cv2.cvtColor(cv2.resize(img, (size, size)), cv2.COLOR_BGR2RGB)
        imgs.append(img.astype(np.float32) / 255.0)
        if len(imgs) == n:
            break
    return np.array(imgs)


def unit(a, axis=-1):
    return a / np.maximum(np.linalg.norm(a, axis=axis, keepdims=True), 1e-8)


def structure_gap(z):
    zn = unit(z, -1)
    h = np.sum(zn[:, :, :-1, :] * zn[:, :, 1:, :], -1).mean()
    v = np.sum(zn[:, :-1, :, :] * zn[:, 1:, :, :], -1).mean()
    adj = float((h + v) / 2)
    B, Hp, Wp, D = zn.shape
    rng = np.random.default_rng(0); s = []
    for _ in range(2000):
        b = rng.integers(B); i1, j1, i2, j2 = rng.integers(Hp), rng.integers(Wp), rng.integers(Hp), rng.integers(Wp)
        if abs(i1 - i2) + abs(j1 - j2) < 3:
            continue
        s.append(float(np.sum(zn[b, i1, j1] * zn[b, i2, j2])))
    dist = float(np.mean(s))
    return adj, dist, adj - dist


# --------------------------- diffusion model ------------------------------ #
def sinusoidal_embed(t, dim=128):
    half = dim // 2
    freqs = ops.exp(-np.log(10000.0) * ops.arange(0, half, dtype="float32") / half)
    args = ops.cast(t, "float32")[:, None] * freqs[None, :]
    return ops.concatenate([ops.sin(args), ops.cos(args)], axis=-1)


def build_unet(Hp, Wp, D, base=96, tdim=128):
    zin = keras.Input((Hp, Wp, D)); tin = keras.Input((), dtype="int32")
    temb = layers.Dense(base * 2, activation="gelu")(sinusoidal_embed(tin, tdim))
    temb = layers.Dense(base * 2)(temb)

    def res(x, c):
        h = layers.GroupNormalization(groups=min(8, c))(x); h = layers.Activation("gelu")(h)
        h = layers.Conv2D(c, 3, padding="same")(h)
        h = h + layers.Dense(c)(temb)[:, None, None, :]
        h = layers.GroupNormalization(groups=min(8, c))(h); h = layers.Activation("gelu")(h)
        h = layers.Conv2D(c, 3, padding="same")(h)
        s = layers.Conv2D(c, 1)(x) if x.shape[-1] != c else x
        return h + s

    x0 = layers.Conv2D(base, 3, padding="same")(zin)        # 16
    x1 = res(x0, base)
    d1 = layers.Conv2D(base * 2, 3, strides=2, padding="same")(x1)   # 8
    x2 = res(d1, base * 2)
    d2 = layers.Conv2D(base * 4, 3, strides=2, padding="same")(x2)   # 4
    m = res(res(d2, base * 4), base * 4)
    u2 = layers.UpSampling2D()(m)                                    # 8
    u2 = res(layers.Concatenate()([u2, x2]), base * 2)
    u1 = layers.UpSampling2D()(u2)                                   # 16
    u1 = res(layers.Concatenate()([u1, x1]), base)
    out = layers.Conv2D(D, 3, padding="same")(
        layers.Activation("gelu")(layers.GroupNormalization(groups=8)(u1)))
    return keras.Model([zin, tin], out, name="latent_unet")


class LatentDDPM(keras.Model):
    def __init__(self, Hp, Wp, D, **kw):
        super().__init__(**kw)
        self.net = build_unet(Hp, Wp, D)
        s = 0.008
        t = np.linspace(0, T, T + 1) / T
        a = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        abar = (a / a[0]).astype("float32")
        self.abar = ops.convert_to_tensor(np.clip(abar, 1e-5, 0.9999))
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def train_step(self, data):
        z = data[0] if isinstance(data, (tuple, list)) else data
        b = ops.shape(z)[0]
        ti = keras.random.randint((b,), 1, T + 1)
        ab = ops.take(self.abar, ti)[:, None, None, None]
        eps = keras.random.normal(ops.shape(z))
        zt = ops.sqrt(ab) * z + ops.sqrt(1 - ab) * eps
        with tf.GradientTape() as tape:
            pred = self.net([zt, ti], training=True)
            loss = ops.mean(ops.square(eps - pred))
        self.optimizer.apply_gradients(zip(
            tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]

    def generate(self, n, Hp, Wp, D, seed=0):
        abar = np.array(self.abar)
        steps = np.linspace(T, 1, DDIM_STEPS).astype(int)
        z = np.array(keras.random.normal((n, Hp, Wp, D), seed=seed))
        for k, ti in enumerate(steps):
            ab = abar[ti]
            pred = np.array(self.net([tf.constant(z), tf.constant(np.full(n, ti, np.int32))], training=False))
            z0 = (z - np.sqrt(1 - ab) * pred) / np.sqrt(ab)
            z0 = unit(z0, -1)  # project to sphere each step (data lives on the sphere)
            if k < len(steps) - 1:
                ab_next = abar[steps[k + 1]]
                z = np.sqrt(ab_next) * z0 + np.sqrt(1 - ab_next) * pred
            else:
                z = z0
        return unit(z, -1)


# --------------------------------- main ----------------------------------- #
def main():
    logger.info(f"Loading frozen VAE: {CKPT}")
    model = keras.models.load_model(CKPT, compile=False); model.trainable = False
    paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    imgs = load_images(paths, n=N_TRAIN + N_EVAL, seed=7)
    zs = [model(tf.constant(imgs[k:k + 64]), training=False)["z"].numpy() for k in range(0, len(imgs), 64)]
    z_all = np.concatenate(zs).astype(np.float32)
    z_eval, z_train = z_all[:N_EVAL], z_all[N_EVAL:]
    B, Hp, Wp, D = z_train.shape
    logger.info(f"cached latents train={z_train.shape} norm={np.linalg.norm(z_train,axis=-1).mean():.3f} "
                f"real latent-gap={structure_gap(z_train)[2]:.4f}")

    def dec_reenc(z):
        dec = np.clip(np.array(model.decode(tf.constant(z.astype(np.float32)))), 0, 1)
        re = model(tf.constant(dec.astype(np.float32)), training=False)["z"].numpy()
        return dec, structure_gap(re)

    res = {}
    _, res["A_real"] = dec_reenc(z_eval)
    z_unif = unit(np.random.default_rng(1).standard_normal((N_EVAL, Hp, Wp, D)).astype(np.float32))
    dec_unif, res["B_uniform_F0"] = dec_reenc(z_unif)

    logger.info(f"Training latent DDPM ({EPOCHS} epochs)")
    ddpm = LatentDDPM(Hp, Wp, D)
    ddpm.compile(optimizer=keras.optimizers.Adam(2e-4))
    ds = tf.data.Dataset.from_tensor_slices(z_train).shuffle(8192).batch(128)
    ddpm.fit(ds, epochs=EPOCHS, verbose=2)
    z_diff = ddpm.generate(N_EVAL, Hp, Wp, D, seed=3)
    logger.info(f"DDPM sample latent-gap={structure_gap(z_diff)[2]:.4f} norm={np.linalg.norm(z_diff,axis=-1).mean():.3f}")
    dec_diff, res["D_diffusion_prior"] = dec_reenc(z_diff)

    logger.info("=" * 60)
    logger.info("RE-ENCODED STRUCTURE GAP (real~0.11; uniform~0.03; target=match real, not exceed)")
    for k, (a, d, g) in res.items():
        logger.info(f"  {k:20s} gap={g:.4f} (adj={a:.4f} dist={d:.4f})")

    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    rows = [("real recon", np.clip(np.array(model.decode(tf.constant(z_eval))), 0, 1)),
            ("uniform (F0)", dec_unif), ("diffusion prior (F3)", dec_diff)]
    for r, (lab, g) in enumerate(rows):
        for c in range(8):
            axes[r, c].imshow(g[c]); axes[r, c].axis("off")
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=9)
    plt.suptitle("convnext_patch_vae vMF: F0 (uniform) vs learned diffusion prior")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "diffusion_comparison.png")
    plt.savefig(p, dpi=110); plt.close()
    logger.info(f"Saved -> {p}")
    ddpm.net.save(os.path.join(OUT_DIR, "latent_ddpm.keras"))


if __name__ == "__main__":
    main()
