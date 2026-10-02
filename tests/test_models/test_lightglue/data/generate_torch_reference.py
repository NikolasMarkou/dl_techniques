"""Generator of ``torch_reference.npz``, the frozen independent LightGlue reference.

NOT a test and NOT collected by pytest (no ``test_`` prefix); it needs ``torch``, which the
test environment does not, so ``torch`` is imported inside ``main`` only. The committed
``torch_reference.npz`` is what ``test_torch_reference.py`` consumes without torch.

Provenance
----------
* Oracle: the official ``cvg/LightGlue`` ``lightglue/lightglue.py`` at commit
  ``eb42fee2d71449efb0aa5c10549752b5d75384d8`` (committed 2026-02-18, fetched 2026-10-02,
  sha256 ``dcb75b9cad1985c5e7a537c512d4318d47fbcc99df75b39a513de60e027c039f``). It is not
  vendored; pass its path.
* Run on CPU, torch ``2.14.1+cpu``, ``flash=False``. The static run is float64 end to end;
  the adaptive runs are float32 (the official pruned path hard-codes float32 buffers).
* Weights are random (normal, std 0.3 for matrices and 0.2 for vectors; the positional
  encoding keeps the reference init), ROUNDED TO FLOAT32 and then used in float64 by the
  official model, so the stored float32 arrays are exactly the values the oracle ran on.

Usage (from the repo root)::

    CUDA_VISIBLE_DEVICES= <python-with-torch> \
        tests/test_models/test_lightglue/data/generate_torch_reference.py \
        <path/to/official/lightglue.py> tests/test_models/test_lightglue/data/torch_reference.npz

Content of the npz
------------------
``w__<torch name>`` float32 state dict; ``k0 k1 d0 d1 s0 s1`` inputs (keypoints xy pixels
float32, descriptors, image sizes (w, h)); ``config`` ``[num_layers, descriptor_dim,
num_heads, input_dim]``; ``la`` ``(L, M+1, N+1)`` and ``c0 c1`` ``(L-1, M)`` / ``(L-1, N)``
float64 static log assignments and token confidences; ``m0 m1 ms0`` static matches and
scores; for each adaptive setting ``<tag>`` the keys ``<tag>__depth``, ``__width``,
``__bias`` (added to every token-confidence ``token.0`` bias), ``__mbias`` (added to the
layer-0 matchability bias), ``__m0 __m1 __s0 __stop __p0 __p1`` (matches, scores, stop
layer, prune counters).
"""

import importlib.util
import sys

import numpy as np

NUM_LAYERS, DIM, HEADS, IN_DIM = 3, 32, 4, 16
M, N = 24, 20
CORRESPOND = 16
SIZE0, SIZE1 = (160.0, 120.0), (150.0, 130.0)
# tag, depth_confidence, width_confidence, token-confidence bias add, layer-0 matchability bias add
SETTINGS = [
    ("adapt_stop", 0.5, -1, 3.0, 0.0),
    ("adapt_prune", -1, 0.9, 0.0, -2.0),
    ("adapt_both", 0.95, 0.6, 1.0, -2.0),
]


def _load_official(path):
    spec = importlib.util.spec_from_file_location("lightglue_official", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(official_path, out_path):
    import torch

    lg = _load_official(official_path)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    def make(depth, width):
        return lg.LightGlue(
            features=None, input_dim=IN_DIM, descriptor_dim=DIM, n_layers=NUM_LAYERS,
            num_heads=HEADS, flash=False, depth_confidence=depth, width_confidence=width,
        ).eval()

    base = make(-1, -1)
    with torch.no_grad():
        for name, p in base.named_parameters():
            if "posenc" in name:
                continue
            p.copy_(torch.randn_like(p) * (0.3 if p.dim() > 1 else 0.2))
    weights = {k: v.detach().numpy().astype(np.float32) for k, v in base.state_dict().items()}
    exact = {k: torch.from_numpy(v.astype(np.float64)) for k, v in weights.items()}

    def pts(n, size):
        return np.stack([rng.uniform(0, size[0], n), rng.uniform(0, size[1], n)], -1).astype(np.float32)[None]

    k0, k1 = pts(M, SIZE0), pts(N, SIZE1)
    d0 = rng.normal(size=(1, M, IN_DIM)).astype(np.float32)
    d1 = rng.normal(size=(1, N, IN_DIM)).astype(np.float32)
    # The first CORRESPOND points of image 1 are noisy copies of image-0 points, so the
    # random-weight model produces a non-trivial set of mutual matches to compare.
    k1[:, :CORRESPOND] = np.clip(
        k0[:, :CORRESPOND] * np.array([SIZE1]) / np.array([SIZE0])
        + rng.normal(scale=1.0, size=(1, CORRESPOND, 2)), 0.0, np.array([SIZE1]) - 1.0)
    d1[:, :CORRESPOND] = d0[:, :CORRESPOND] + 0.05 * rng.normal(size=(1, CORRESPOND, IN_DIM))
    d1 = d1.astype(np.float32)
    k1 = k1.astype(np.float32)
    s0 = np.array([SIZE0], np.float32)
    s1 = np.array([SIZE1], np.float32)
    t = lambda a: torch.from_numpy(a).double()
    data = {
        "image0": {"keypoints": t(k0), "descriptors": t(d0), "image_size": t(s0)},
        "image1": {"keypoints": t(k1), "descriptors": t(d1), "image_size": t(s1)},
    }

    static = make(-1, -1).double()
    static.load_state_dict(exact)
    with torch.no_grad():
        kp0 = lg.normalize_keypoints(t(k0), t(s0))
        kp1 = lg.normalize_keypoints(t(k1), t(s1))
        e0, e1 = static.posenc(kp0), static.posenc(kp1)
        x0, x1 = static.input_proj(t(d0)), static.input_proj(t(d1))
        las, c0s, c1s = [], [], []
        for i in range(NUM_LAYERS):
            x0, x1 = static.transformers[i](x0, x1, e0, e1)
            la, _ = static.log_assignment[i](x0, x1)
            las.append(la.numpy()[0])
            if i < NUM_LAYERS - 1:
                c0, c1 = static.token_confidence[i](x0, x1)
                c0s.append(c0.numpy()[0])
                c1s.append(c1.numpy()[0])
        out = static(data)

    arrays = {"w__" + k: v for k, v in weights.items()}
    arrays.update(
        k0=k0, k1=k1, d0=d0, d1=d1, s0=s0, s1=s1,
        config=np.array([NUM_LAYERS, DIM, HEADS, IN_DIM]),
        la=np.stack(las), c0=np.stack(c0s), c1=np.stack(c1s),
        m0=out["matches0"].numpy()[0], m1=out["matches1"].numpy()[0],
        ms0=out["matching_scores0"].numpy()[0],
    )
    for tag, depth, width, bias, mbias in SETTINGS:
        # The official pruned path hard-codes float32 buffers (``mscores0_ = zeros(float32)``),
        # so the adaptive runs are float32 on the float32 weights (exact, no rounding).
        model = make(depth, width)
        model.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
        data32 = {k: {kk: vv.float() for kk, vv in v.items()} for k, v in data.items()}
        with torch.no_grad():
            for i in range(NUM_LAYERS - 1):
                model.token_confidence[i].token[0].bias += bias
            model.log_assignment[0].matchability.bias += mbias
            o = model(data32)
        arrays.update({
            f"{tag}__depth": np.array(depth), f"{tag}__width": np.array(width),
            f"{tag}__bias": np.array(bias), f"{tag}__mbias": np.array(mbias),
            f"{tag}__m0": o["matches0"].numpy()[0], f"{tag}__m1": o["matches1"].numpy()[0],
            f"{tag}__s0": o["matching_scores0"].numpy()[0], f"{tag}__stop": np.array(int(o["stop"])),
            f"{tag}__p0": o["prune0"].numpy()[0], f"{tag}__p1": o["prune1"].numpy()[0],
        })
        sys.stdout.write(
            f"{tag}: stop {int(o['stop'])} matches {int((o['matches0'] > -1).sum())} "
            f"prune0 max {int(o['prune0'].max())}\n")
    np.savez_compressed(out_path, **arrays)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
