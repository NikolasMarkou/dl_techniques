"""Generator of ``beit_relpos_reference.npz``, the frozen unilm reference of the BEiT relative position bias resize.

NOT a test and NOT collected by pytest (no ``test_`` prefix). It needs ``torch`` and a scipy below 1.14 (which still
has ``scipy.interpolate.interp2d``), which the test environment does not have, so numpy, scipy and torch are imported
inside the functions only. The committed ``beit_relpos_reference.npz`` is what ``test_init_from_engine.py`` reads
without them, to hold ``train.api.init_from.beit_relpos_cubic`` to unilm's own code at 1e-6
(plan-2026-11-19T081214-5c6cc105, D-004).

Provenance
----------
* Oracle: microsoft/unilm ``beit/run_class_finetuning.py`` at commit ``5255d52de86dad642810f5849dd357769346c1d7``,
  https://github.com/microsoft/unilm/blob/5255d52de86dad642810f5849dd357769346c1d7/beit/run_class_finetuning.py
  (fetched 2026-11-23, 593 lines, sha256 ``193c43aa800b088cdfff1a05678365f41a841b1ad42e996252b7c60e654607f6``).
  Its lines 351-411, the resize of one ``relative_position_bias_table`` (the 3 extra rows kept, a cubic ``interp2d``
  per head through the source displacements set on a geometric grid), are copied VERBATIM between the two marker
  comments of :func:`unilm_resize`, indented 4 spaces less than upstream; nothing else of the file runs. ``main``
  refuses to run unless that block, dedented (``textwrap.dedent``, UTF-8), hashes to :data:`UNILM_BLOCK_SHA256`, the
  sha256 of upstream lines 351-411 dedented the same way::

      sed -n 351,411p run_class_finetuning.py | python3 -c \\
          "import sys, textwrap, hashlib; print(hashlib.sha256(textwrap.dedent(sys.stdin.read()).encode()).hexdigest())"

* The copied lines are MIT-licensed (notice below, the ``LICENSE`` of microsoft/unilm at that commit; the file's own
  header reads "Copyright (c) 2021 Microsoft, Licensed under The MIT License").
* Run on the CPU by the system ``python3`` (3.10.12, scipy 1.13.1, torch 2.7.0+cu126, numpy 1.26.4); the npz records
  the versions of the run that wrote it.
* Inputs: per case, a seeded random normal float32 table ``((2 S - 1)^2 + 3, heads)`` (``numpy.random.default_rng``
  seeded with the case's index in :data:`CASES`), resized from a grid of ``S x S`` patches to ``D x D``. The cases:
  8 to 14, 8 to 16, 4 to 8 (unilm's search for the ratio ``q`` saturates at its upper bound 1.5, so the outer targets
  lie beyond the last source position), 14 to 28, each with 3 heads (BEiT tiny), and 8 to 16 with 12 heads (base).

Usage (from the repo root)::

    CUDA_VISIBLE_DEVICES= <python3 with torch and scipy < 1.14> \\
        tests/test_train/test_api/data/generate_beit_relpos_reference.py \\
        tests/test_train/test_api/data/beit_relpos_reference.npz

Content of the npz
------------------
``versions`` (0-d text: the interpreter, scipy, torch and numpy versions of the run); per case ``<S>_<D>_<heads>``:
``<case>__table`` the source table (float32), ``<case>__resized`` unilm's output ``((2 D - 1)^2 + 3, heads)``
(float32), ``<case>__q`` the ratio unilm's search found (float64, 0-d).

The MIT License of the copied lines
-----------------------------------
The MIT License (MIT)

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import hashlib
import sys
import textwrap
import types
from pathlib import Path

#: ``(S, D, heads)``: a table of a grid of ``S x S`` patches resized to ``D x D``; its index seeds its table.
CASES = ((8, 14, 3), (8, 16, 3), (4, 8, 3), (14, 28, 3), (8, 16, 12))

#: sha256 of upstream lines 351-411, dedented; the block between the markers must hash to it.
UNILM_BLOCK_SHA256 = "f245772ddadc2016e17eb2b39e9909b0c6f865c817acb39ca9c454708c1e59b9"
BEGIN, END = "# ---- unilm beit/run_class_finetuning.py L351-411 begin", "# ---- unilm L351-411 end"


class Target:
    """What lines 351-411 read of unilm's model: the target table's size and the grid of patches."""

    def __init__(self, key: str, window: int, heads: int):
        import torch

        self.patch_embed = types.SimpleNamespace(patch_shape=(window, window))
        self._state = {key: torch.zeros((2 * window - 1) ** 2 + 3, heads)}

    def state_dict(self):
        return self._state


def unilm_resize(table, window: int):
    """Run lines 351-411 on a checkpoint holding ``table`` for a model of ``window x window`` patches.

    :return: ``(resized table as float32 numpy, q)``.
    """
    import numpy as np
    import torch
    from scipy import interpolate

    checkpoint_model = {"blocks.0.attn.relative_position_bias_table": torch.from_numpy(table)}
    model = Target("blocks.0.attn.relative_position_bias_table", window, table.shape[1])
    q = None
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        # ---- unilm beit/run_class_finetuning.py L351-411 begin
        if "relative_position_bias_table" in key:
            rel_pos_bias = checkpoint_model[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = model.state_dict()[key].size()
            dst_patch_shape = model.patch_embed.patch_shape
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                print("Position interpolate for %s from %dx%d to %dx%d" % (
                    key, src_size, src_size, dst_size, dst_size))
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # if q > 1.090307:
                #     q = 1.090307

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                print("Original positions = %s" % str(x))
                print("Target positions = %s" % str(dx))

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                checkpoint_model[key] = new_rel_pos_bias
        # ---- unilm L351-411 end
    return checkpoint_model[all_keys[0]].numpy(), q


def embedded_block_sha256() -> str:
    """The sha256 of the lines between the two markers of :func:`unilm_resize`, dedented."""
    lines = Path(__file__).read_text(encoding="utf-8").splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.strip() == BEGIN) + 1
    stop = next(i for i, line in enumerate(lines) if line.strip() == END)
    return hashlib.sha256(textwrap.dedent("".join(lines[start:stop])).encode("utf-8")).hexdigest()


def main(out: str) -> None:
    import platform

    import numpy as np
    import scipy
    import torch

    found = embedded_block_sha256()
    if found != UNILM_BLOCK_SHA256:
        raise SystemExit(f"the unilm block hashes to {found}, not {UNILM_BLOCK_SHA256}: it is no longer verbatim")
    arrays = {"versions": np.array(f"python {platform.python_version()}; scipy {scipy.__version__}; "
                                   f"torch {torch.__version__}; numpy {np.__version__}")}
    for seed, (src, dst, heads) in enumerate(CASES):
        table = np.random.default_rng(seed).standard_normal(((2 * src - 1) ** 2 + 3, heads)).astype(np.float32)
        resized, q = unilm_resize(table, dst)
        name = f"{src}_{dst}_{heads}"
        arrays.update({f"{name}__table": table, f"{name}__resized": resized, f"{name}__q": np.array(q)})
        sys.stdout.write(f"{name}: table {table.shape} -> {resized.shape} {resized.dtype}, q = {q!r}\n")
    np.savez_compressed(out, **arrays)
    sys.stdout.write(f"{arrays['versions']}; block sha256 {found}; wrote {out}\n")


if __name__ == "__main__":
    main(sys.argv[1])
