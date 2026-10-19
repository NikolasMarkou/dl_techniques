"""Generator of ``torch_reference.npz``, the frozen independent Timestral reference.

NOT a test and NOT collected by pytest (no ``test_`` prefix); it needs ``torch``, which the
test environment does not have, so ``torch`` is imported inside ``main`` only. The committed
``torch_reference.npz`` is what ``test_torch_reference.py`` consumes without torch.

Provenance
----------
* Oracle: the official ``mistralai/timestral`` package (``src/timestral/``) at commit
  ``a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`` (fetched 2026-10-19). It is not vendored;
  pass the directory that holds its ``timestral/`` package. sha256 of the files run:

  - ``model.py`` ``988121c6a089e843c4941732cbc5c64b10cb0eccd9310e228abe38ecf2dfb668``
  - ``layers.py`` ``480ba93e7e3681dea185e5fa08a96dcce327cf938931c715ae54e6b02934527d``
  - ``forecaster.py`` ``12a712f9be777d1e160513b7f8c81d545b991fcfe6cd4a46004a03533b7cb8fd``
  - ``scaler.py`` ``838da836f70b1640bcaed7701a70a1a358e30442693e63522a93f878e157a8ca``
  - ``masks.py`` ``1792898d809b99034bb70e2cc1ab74f897ce8478886f83140e705fc67e6762b3``
  - ``config.py`` ``d46e37020875c536e7a75e1b6ace3f67419ba2f2c9d9e8465f4b07f837e92ed6``
* Run on CPU, torch ``2.14.1+cpu``, float32 (the dtype the reference validated).
* Weights are random normal (std 0.1, seed 0) in the reference's own parameter layout.
  The config enables the attention bias so the fused-QKV bias layout is exercised.

Usage (from the repo root)::

    CUDA_VISIBLE_DEVICES= <python-with-torch> \\
        tests/test_models/test_timestral/data/generate_torch_reference.py \\
        <dir holding the official timestral/ package> \\
        tests/test_models/test_timestral/data/torch_reference.npz

Content of the npz
------------------
``config_json`` (the TimestralConfig as JSON, a 0-d unicode array); ``w__<torch name>``
float32 state dict; ``ctx`` ``ctx_obs`` ``groups`` (3 packed rows, two series, a missing
stretch) and ``block_h1`` ``block_h3`` (``predict_block``, context only); ``roles``
``known`` ``known_mask`` and ``block_cov_h2`` (``predict_block`` with a past-only and a
known-future covariate row); ``series`` ``(2, 2, 100)`` with NaNs, ``forecast_raw``
(``forecast``, prediction_length 70, three decode blocks, no test-time augmentation) and
``forecast_clampsym`` (the same with symmetric averaging and non-negative clipping).
"""

import json
import sys

import numpy as np

CONFIG = dict(
    patch_size=8,
    d_model=32,
    num_heads=4,
    mlp_hidden_dim=64,
    block_schedule=("time", "time_variate", "time", "time_variate"),
    quantile_knots=(0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95),
    block_size=4,
    context_length=256,
    attn_bias=True,
    quantile_width_factor=3.0,
)


def main(package_dir: str, out: str) -> None:
    sys.path.insert(0, package_dir)
    import torch
    from timestral.config import TimestralConfig
    from timestral.forecaster import TimestralForecaster
    from timestral.model import TimestralModel

    torch.manual_seed(0)
    config = TimestralConfig(**CONFIG)
    model = TimestralModel(config)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.1)
    model.eval()
    forecaster = TimestralForecaster(model)

    rng = np.random.RandomState(0)
    res = {"config_json": np.array(json.dumps(config.to_dict()))}
    for name, value in model.state_dict().items():
        res[f"w__{name}"] = value.numpy().astype(np.float32)

    ctx = np.cumsum(rng.randn(3, 64), axis=-1).astype(np.float32)
    ctx_obs = np.ones((3, 64), dtype=bool)
    ctx_obs[1, 3:11] = False
    ctx[1, 3:11] = 0.0
    groups = np.array([0, 0, 1], dtype=np.int64)
    roles = np.array([0, 1, 2, 0], dtype=np.int64)
    known = np.zeros((4, 16), dtype=np.float32)
    known[2] = rng.randn(16)
    known_mask = np.zeros((4, 16), dtype=bool)
    known_mask[2] = True
    cov_ctx = np.cumsum(rng.randn(4, 64), axis=-1).astype(np.float32)
    cov_groups = np.array([0, 0, 0, 1], dtype=np.int64)
    t = np.arange(100, dtype=np.float32)
    series = (5 * np.sin(2 * np.pi * t / 12) + np.cumsum(rng.randn(2, 2, 100), axis=-1)).astype(np.float32)
    series[0, 1, 20:27] = np.nan

    with torch.no_grad():
        T = torch.from_numpy
        for h in (1, 3):
            res[f"block_h{h}"] = model.predict_block(T(ctx), T(ctx_obs), T(groups), h).numpy()
        res["block_cov_h2"] = model.predict_block(
            T(cov_ctx), T(np.ones_like(cov_ctx, dtype=bool)), T(cov_groups), 2,
            T(roles), T(known), T(known_mask)).numpy()
        res["forecast_raw"] = forecaster.forecast(
            T(series), prediction_length=70,
            symmetric_averaging=False, clip_nonnegative=False).quantiles.numpy()
        res["forecast_clampsym"] = forecaster.forecast(T(series), prediction_length=70).quantiles.numpy()

    res.update(ctx=ctx, ctx_obs=ctx_obs, groups=groups.astype(np.int32), cov_ctx=cov_ctx,
               cov_groups=cov_groups.astype(np.int32), roles=roles.astype(np.int32),
               known=known, known_mask=known_mask, series=series)
    np.savez_compressed(out, **res)
    print({k: v.shape for k, v in res.items() if not k.startswith("w__")})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
