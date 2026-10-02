"""Test-only torch -> Keras weight mapping for the LightGlue model.

Not collected (no ``test_`` prefix). The mapping is a TRANSPOSE of every ``Linear`` weight
plus the norm affine copy; sublayer names mirror the torch state dict, so no permutation is
needed. It is the converter rule the model docstring promises, expressed as code that a
parity test exercises.
"""

import numpy as np

from tests import lightglue_reference_numpy as ref


def _assign(variable, value):
    variable.assign(np.asarray(value, dtype=variable.dtype))


def _load_dense(layer, weights, prefix):
    _assign(layer.kernel, weights[prefix + ".weight"].T)
    _assign(layer.bias, weights[prefix + ".bias"])


def _load_ffn(block, w, prefix):
    _load_dense(block.ffn_0, w, prefix + ".ffn.0")
    _assign(block.ffn_1.gamma, w[prefix + ".ffn.1.weight"])
    _assign(block.ffn_1.beta, w[prefix + ".ffn.1.bias"])
    _load_dense(block.ffn_3, w, prefix + ".ffn.3")


def load_torch_weights(model, w):
    """Load a torch-named, torch-layout weight dict (the oracle's) into ``model``."""
    if model.input_proj is not None:
        _load_dense(model.input_proj, w, "input_proj")
    _assign(model.posenc.kernel, w["posenc.Wr.weight"].T)
    for i in range(model.num_layers):
        p = f"transformers.{i}"
        s, c = model.self_blocks[i], model.cross_blocks[i]
        _load_dense(s.Wqkv, w, p + ".self_attn.Wqkv")
        _load_dense(s.out_proj, w, p + ".self_attn.out_proj")
        _load_ffn(s, w, p + ".self_attn")
        for name in ("to_qk", "to_v", "to_out"):
            _load_dense(getattr(c, name), w, f"{p}.cross_attn.{name}")
        _load_ffn(c, w, p + ".cross_attn")
        _load_dense(model.assignments[i].matchability, w, f"log_assignment.{i}.matchability")
        _load_dense(model.assignments[i].final_proj, w, f"log_assignment.{i}.final_proj")
        if i < model.num_layers - 1:
            _load_dense(model.confidences[i].token_0, w, f"token_confidence.{i}.token.0")


def random_inputs(rng, batch, m_pts, n_pts, input_dim, size=(96.0, 64.0), extra=False):
    """Random pixel keypoints and descriptors for a pair, float32 numpy dict."""
    def pts(n):
        return np.stack([rng.uniform(0, size[0], (batch, n)), rng.uniform(0, size[1], (batch, n))], -1)
    data = {
        "keypoints0": pts(m_pts).astype("float32"),
        "keypoints1": pts(n_pts).astype("float32"),
        "descriptors0": rng.normal(size=(batch, m_pts, input_dim)).astype("float32"),
        "descriptors1": rng.normal(size=(batch, n_pts, input_dim)).astype("float32"),
        "image_size0": np.tile(np.array(size, "float32"), (batch, 1)),
        "image_size1": np.tile(np.array([size[0], size[1] + 8.0], "float32"), (batch, 1)),
    }
    if extra:
        for side, n in (("0", m_pts), ("1", n_pts)):
            data["scales" + side] = rng.uniform(1, 4, (batch, n)).astype("float32")
            data["oris" + side] = rng.uniform(-3, 3, (batch, n)).astype("float32")
    return data


def oracle_forward(w, data, num_layers, num_heads, real0=None, real1=None, extra=False):
    """Oracle log assignments / confidences for one sample on the REAL keypoints only."""
    outs = []
    for b in range(data["keypoints0"].shape[0]):
        r0 = slice(None) if real0 is None else real0[b]
        r1 = slice(None) if real1 is None else real1[b]
        e0 = e1 = None
        if extra:
            e0 = np.stack([data["scales0"][b][r0], data["oris0"][b][r0]], -1).astype("float64")
            e1 = np.stack([data["scales1"][b][r1], data["oris1"][b][r1]], -1).astype("float64")
        outs.append(ref.forward_single(
            w, data["keypoints0"][b][r0].astype("float64"), data["descriptors0"][b][r0],
            data["image_size0"][b].astype("float64"),
            data["keypoints1"][b][r1].astype("float64"), data["descriptors1"][b][r1],
            data["image_size1"][b].astype("float64"), num_layers, num_heads, e0, e1))
    return outs
