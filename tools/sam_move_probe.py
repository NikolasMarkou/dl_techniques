#!/usr/bin/env python
"""SAM-family checkpoint equivalence probe (THROWAWAY -- deleted at plan CLOSE).

Plan: plans/plan-2026-08-08T014604-1d2b7cb4 (step 2 captures BEFORE, step 6
captures AFTER, step 12 deletes this file).

Usage -- identical before and after the `models/{sam,sam2,sam3}` ->
`models/SAM/{SAM1,SAM2,SAM3}` move::

    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python tools/sam_move_probe.py \
        --registrar dl_techniques.models.sam3 \
        --checkpoint results/sam3_tiny_20260808_040020/final_model.keras \
        --out plans/<plan>/checkpoints/sam3_before.json [--perturb-ulp]

    # after the move:  --registrar dl_techniques.models.SAM.SAM3  --out ..._after.json
    # registrar-less control:  --registrar dl_techniques
    # comparison (no TF/keras import at all):
    #     tools/sam_move_probe.py --compare before.json after.json

Exit status
-----------
0   record written, or `--compare` matched
1   `--compare` mismatched (the mismatching field is PRINTED), or a bad argument
2   the checkpoint FAILED TO LOAD -- the verbatim traceback is printed to stderr
    and the marker line ``LOAD_FAILED: <ExcType>: <msg>`` to stdout. This is the
    expected shape of the post-move registrar-less control (plan F-08).

Design notes (why it is shaped like this)
-----------------------------------------
* `n_weights` / `n_params` are sampled AT LOAD TIME, BEFORE any forward pass.
  A build-only load path once materialised 138 of 202 weights and read 202
  after a forward pass, the 64-weight gap having been silently filled with
  fresh random values. Both samples are recorded (`n_weights`,
  `n_weights_post_forward`) so that gap is visible instead of averaged away.
* The comparator asserts the record COUNTS are EQUAL **and NON-ZERO**, so a
  zero-record comparison cannot exit 0-shaped having compared nothing.
* `--perturb-ulp` nudges one float weight by one ULP so the comparator can be
  proven RED before any PASS is trusted.
* The config digest strips `"module"` VALUES recursively (the whole point of
  the refactor is to change them) and records the stripped module strings
  separately under `config_modules`, so the change is REPORTED rather than
  hidden. Renaming the key alone -- leaving the value in the dump -- would
  make the digest mismatch for exactly the reason it is meant to tolerate.
* Inputs are drawn from a fixed seed AND from geometry read off the LOADED
  model (not hardcoded), so the same command works either side of the move.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

BATCH = 2
FORWARD_SEED = 0


# ---------------------------------------------------------------------
# digests
# ---------------------------------------------------------------------

def digest(chunks: List[bytes]) -> str:
    """sha256 over an ordered sequence of byte chunks."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def strip_modules(obj: Any, found: List[str]) -> Any:
    """Recursively drop every ``"module"`` entry, collecting its values.

    :param obj: Any JSON-shaped structure.
    :param found: Accumulator the removed module strings are appended to.
    :return: The same structure with every ``"module"`` key removed.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "module":
                found.append(str(v))
                continue
            out[k] = strip_modules(v, found)
        return out
    if isinstance(obj, list):
        return [strip_modules(v, found) for v in obj]
    return obj


# ---------------------------------------------------------------------
# family detection + inputs
# ---------------------------------------------------------------------

_FAMILY_BY_CLASS = {
    "SAMTrainingModel": "sam",
    "SAM": "sam",
    "SAM2TrainingModel": "sam2",
    "SAM2": "sam2",
    "Sam3TrainingModel": "sam3",
    "Sam3Image": "sam3",
}


def detect_family(model: Any) -> str:
    """Name the family from the loaded model's class, not from its path."""
    name = type(model).__name__
    if name not in _FAMILY_BY_CLASS:
        raise ValueError(f"unknown SAM family for loaded class {name!r}")
    return _FAMILY_BY_CLASS[name]


def build_inputs(model: Any, family: str, rng: np.random.Generator
                 ) -> Dict[str, np.ndarray]:
    """Build one deterministic input dict, geometry read off ``model``.

    Shapes come from the LOADED object's own attributes so the probe does not
    carry a second, independently-rotting copy of each checkpoint's geometry.
    """
    if family == "sam3":
        img = int(model.sam3.backbone.img_size)
        vocab = int(model.sam3.text_encoder.vocab_size)
        ctx = int(model.sam3.text_encoder.context_length)
        return {
            "image": rng.standard_normal((BATCH, img, img, 3)).astype("float32"),
            "token_ids": rng.integers(0, vocab, (BATCH, ctx)).astype("int32"),
        }
    if family == "sam":
        height, width = tuple(model.sam.prompt_encoder.input_image_size)
        return {
            "image": rng.uniform(
                0.0, 255.0, (BATCH, int(height), int(width), 3)).astype("float32"),
            "point_coords": rng.uniform(
                0.0, float(min(height, width)), (BATCH, 2, 2)).astype("float32"),
            "point_labels": np.ones((BATCH, 2), dtype="int32"),
        }
    if family == "sam2":
        size = int(model.sam2.image_size)
        frames = int(model.num_frames)
        grid = int(model.sam2.feature_grid) * 4
        masks = (rng.random((BATCH, frames, grid, grid)) > 0.7).astype("float32")
        # Every frame non-empty on purpose: an accidentally empty frame is
        # indistinguishable from a deliberately occluded one.
        masks[:, :, 0, 0] = 1.0
        return {
            "image": rng.uniform(
                0.0, 255.0, (BATCH, frames, size, size, 3)).astype("float32"),
            "point_coords": rng.uniform(
                4.0, float(size) - 4.0, (BATCH, 1, 2)).astype("float32"),
            "point_labels": np.ones((BATCH, 1), dtype="int32"),
            "gt_masks": masks,
        }
    raise ValueError(f"no input builder for family {family!r}")


# ---------------------------------------------------------------------
# comparator (imports NOTHING heavy)
# ---------------------------------------------------------------------

_COUNT_FIELDS = ("n_weights", "n_params", "n_outputs", "n_weights_post_forward")
_DIGEST_FIELDS = ("weight_sha256", "output_sha256", "config_sha256_module_stripped")


def compare(a_path: str, b_path: str) -> int:
    """Compare two records. Exit 1 naming the field that differs."""
    with open(a_path) as fh:
        a = json.load(fh)
    with open(b_path) as fh:
        b = json.load(fh)

    mismatched: List[str] = []

    # Vacuity gate FIRST: a record that measured nothing cannot license a PASS.
    if not (isinstance(a.get("n_weights"), int) and a["n_weights"] > 0):
        print(f"VACUOUS: A ({a_path}) has n_weights={a.get('n_weights')!r}; "
              f"a zero-weight record cannot pass")
        return 1
    if not (isinstance(a.get("n_params"), int) and a["n_params"] > 0):
        print(f"VACUOUS: A ({a_path}) has n_params={a.get('n_params')!r}")
        return 1

    for field in _COUNT_FIELDS:
        if a.get(field) != b.get(field):
            mismatched.append(f"{field} ({a.get(field)!r} != {b.get(field)!r})")

    for field in _DIGEST_FIELDS:
        if a.get(field) != b.get(field):
            mismatched.append(f"{field} ({str(a.get(field))[:16]}... != "
                              f"{str(b.get(field))[:16]}...)")

    n_out = a.get("n_outputs")
    if n_out == 0 and b.get("n_outputs") == 0:
        print("NOTE: n_outputs == 0 on BOTH sides -- output equivalence is "
              f"UNMEASURED for this family (A reason: {a.get('forward_error')}; "
              f"B reason: {b.get('forward_error')})")

    print(f"compared {a['n_weights']} weights / {a['n_params']} params / "
          f"{n_out} outputs; A.registrar={a.get('registrar')!r} "
          f"B.registrar={b.get('registrar')!r}; "
          f"MISMATCH={mismatched or 'none'}")
    if a.get("config_modules") != b.get("config_modules"):
        print("INFO (not a mismatch, this is what the move changes): "
              f"config_modules A={a.get('config_modules')} "
              f"B={b.get('config_modules')}")
    return 1 if mismatched else 0


# ---------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------

def capture(args: argparse.Namespace) -> int:
    """Import the registrar, load the checkpoint, write one record."""
    importlib.import_module(args.registrar)  # registrar-first (INV-2 idiom)

    import keras
    from keras.src.saving import object_registration as objreg

    try:
        model = keras.models.load_model(args.checkpoint, compile=False)
    except Exception as exc:  # noqa: BLE001 -- the traceback IS the finding
        traceback.print_exc()
        print(f"LOAD_FAILED: {type(exc).__name__}: {exc}")
        return 2

    # ---- sampled AT LOAD TIME, before ANY forward pass -------------------
    weights = sorted(model.weights, key=lambda w: w.path)
    n_weights_at_load = len(weights)
    n_params_at_load = int(model.count_params())

    if args.perturb_ulp:
        target = next((w for w in weights
                       if "float" in str(np.asarray(w).dtype)), None)
        if target is None:
            print("LOAD_FAILED: no float weight to perturb")
            return 2
        value = np.array(np.asarray(target))
        original = value.flat[0]
        value.flat[0] = np.nextafter(original, np.asarray(np.inf, value.dtype))
        target.assign(value)
        print(f"PERTURBED 1 ULP: {target.path} [0] {original!r} -> "
              f"{value.flat[0]!r}")

    weight_chunks: List[bytes] = []
    for w in weights:
        v = np.asarray(w)
        weight_chunks += [w.path.encode(), str(v.shape).encode(),
                          str(v.dtype).encode(), np.ascontiguousarray(v).tobytes()]

    family = args.family or detect_family(model)

    # ---- deterministic forward pass -------------------------------------
    output_chunks: List[bytes] = []
    n_outputs = 0
    forward_error: Optional[str] = None
    keras.utils.set_random_seed(FORWARD_SEED)
    try:
        inputs = build_inputs(model, family, np.random.default_rng(FORWARD_SEED))
        # training=False EXPLICIT: at a non-zero drop_path_rate the default
        # None drops paths and is NOT inference.
        out = model(inputs, training=False)
        if isinstance(out, dict):
            flat = [out[k] for k in sorted(out)]
        elif isinstance(out, (list, tuple)):
            flat = list(out)
        else:
            flat = [out]
        for t in flat:
            v = np.asarray(keras.ops.convert_to_numpy(t)).astype("float32")
            output_chunks += [str(v.shape).encode(),
                              np.ascontiguousarray(v).tobytes()]
        n_outputs = len(flat)
    except Exception as exc:  # noqa: BLE001 -- degrade LOUDLY, never fake it
        traceback.print_exc()
        forward_error = f"{type(exc).__name__}: {exc}"
        n_outputs = 0
        output_chunks = []
        print(f"FORWARD_DEGRADED: n_outputs=0 -- {forward_error}")

    # ---- post-forward re-sample: catches a silently-filled weight gap ----
    n_weights_post_forward = len(model.weights)

    # ---- config, with module VALUES stripped and reported ----------------
    config_modules: List[str] = []
    try:
        raw = json.loads(json.dumps(model.get_config(), default=str))
        stripped = json.dumps(strip_modules(raw, config_modules), sort_keys=True)
        config_sha = hashlib.sha256(stripped.encode()).hexdigest()
    except Exception as exc:  # noqa: BLE001
        config_sha = f"UNAVAILABLE: {type(exc).__name__}"

    record = {
        "checkpoint": args.checkpoint,
        "family": family,
        "registrar": args.registrar,
        "model_class": type(model).__name__,
        "keras": keras.__version__,
        "perturbed": bool(args.perturb_ulp),
        "n_weights": n_weights_at_load,
        "n_params": n_params_at_load,
        "n_weights_post_forward": n_weights_post_forward,
        "n_outputs": n_outputs,
        "forward_error": forward_error,
        "n_params_from_weights": int(sum(np.asarray(w).size for w in weights)),
        "registered_keys": sorted(
            k for k in objreg.GLOBAL_CUSTOM_OBJECTS if "am" in k or "AM" in k),
        "config_modules": sorted(set(config_modules)),
        "weight_sha256": digest(weight_chunks),
        "output_sha256": digest(output_chunks),
        "config_sha256_module_stripped": config_sha,
    }
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(record, fh, indent=1)
    print(json.dumps(record, indent=1))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--registrar",
                   help="module imported BEFORE load_model (registrar-first)")
    p.add_argument("--checkpoint", help="path to the .keras file")
    p.add_argument("--out", help="where to write the JSON record")
    p.add_argument("--family", choices=("sam", "sam2", "sam3"),
                   help="override; normally detected from the loaded class")
    p.add_argument("--perturb-ulp", action="store_true",
                   help="nudge one float weight by 1 ULP (RED proof)")
    p.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = p.parse_args()

    if args.compare:
        return compare(*args.compare)
    if not args.registrar or not args.checkpoint:
        p.error("--registrar and --checkpoint are required unless --compare")
    return capture(args)


if __name__ == "__main__":
    sys.exit(main())
