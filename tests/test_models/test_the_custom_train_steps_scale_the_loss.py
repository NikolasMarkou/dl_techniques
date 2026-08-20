"""Every custom `train_step` under `models/` must call `optimizer.scale_loss`.

Rationale
---------
Under `mixed_float16` Keras wraps the optimizer in a `LossScaleOptimizer` whose
`apply()` DIVIDES every gradient by `dynamic_scale` (2**15 initially),
UNCONDITIONALLY. Keras' own default TF `train_step` pairs that division with an
`optimizer.scale_loss(loss)` multiplication inside the tape; a class that
OVERRIDES `train_step` silently opts out of the multiplication and keeps the
division. The result is not a precision loss -- it is the entire weight update
divided by 32768, with no warning.

MEASURED, total |dW| over 5 SGD steps (Adam is deliberately NOT used: its
per-parameter normalisation HIDES a uniform gradient rescale):

    site   float32      mixed_float16   ratio BEFORE   ratio AFTER
    clm    1.859214e+01 6.757726e-04    2.7513e+04     1.0000016
    blt    2.156337e+03 1.294069e-01    1.6664e+04     1.0354
    mlm    3.561846e+01 1.617076e-03    2.2027e+04     1.0617
    mae    2.506961e+02 2.850456e-02    8.7950e+03     1.00015

with `depth_anything` -- the one site that ALREADY called `scale_loss` -- as the
real-site GREEN calibration arm at 1.094 on GPU and 0.996 on CPU, and a CPU
control for every row.

This guard is STATIC on purpose: the runtime A/B costs minutes per site and
needs a GPU, but the omission is a one-line textual fact about a `train_step`
body, and it is the ADDITION of a new unscaled `train_step` that this file
exists to stop.

See decisions.md D-036 (plan-2026-08-19T163559-499b6f0e).
"""

import ast
import pathlib

import pytest

# ---------------------------------------------------------------------

MODELS_ROOT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src" / "dl_techniques" / "models"
)

# Sites measured LIVE but NOT repaired in this pass, each with its measured
# ratio. Every entry is a real defect; the waiver records that it is KNOWN and
# owed, not that it is acceptable. Removing an entry must be accompanied by the
# `scale_loss` call, and `test_every_waiver_is_still_a_real_site` fails the
# moment an entry stops naming a live unscaled `train_step`.
KNOWN_UNSCALED = {
    "vae/model.py": "3.6692e+04 (step 5.8 / D-011); routed to step 18",
    "vq_vae/model.py": "2.8040e+04 (step 5.8 / D-011); routed to step 18",
    "vq_vae_rotation/model.py": "1.3553e+04 (step 5.8 / D-011); routed to step 18",
    "video_jepa/model.py": "7.5446e+03 (step 5.8 / D-011); routed to step 18",
    "capsnet/model.py": "2.0166e+03 (step 5.8 / D-011) -- an order of magnitude "
                        "short of the 2**15 signature, cause NOT established",
    "latent_gmm_registration/model.py": "fp16-unreachable by design (D-004 arm b)",
    "memory_bank/wave_field_memory_llm.py": "scale_loss is an unconditional "
                                            "no-op here; see that file's own anchor",
    "nano_vlm_world_model/train.py": "never probed; not a `keras.Model.train_step` "
                                     "on the shipped forward path",
}


def _train_step_bodies():
    """Yield (relative path, FunctionDef, {name: FunctionDef}) per `train_step`.

    The third element is every `def` in the same module, which the predicate
    needs in order to follow `train_step` into a helper -- see
    `_calls_scale_loss`.
    """
    for path in sorted(MODELS_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - a parse failure is its own bug
            continue
        module_defs = {
            n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "train_step":
                yield str(path.relative_to(MODELS_ROOT)), node, module_defs


def _calls_scale_loss(node: ast.FunctionDef, module_defs=None) -> bool:
    """True if `node` -- or a same-module helper it calls -- uses `scale_loss`.

    The one-level callee closure is NOT optional. A purely lexical predicate
    reports `depth_anything/model.py:975` as an offender, and it is the one
    COMPLIANT site in the tree: its `train_step` dispatches to
    `_train_step_supervised` / `_train_step_semi_supervised`, and those hold the
    `scale_loss` call. A guard that cannot see one level of indirection would
    have demanded a "fix" to the only correct site in `models/`.
    """
    module_defs = module_defs or {}
    bodies = [node]
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            callee = module_defs.get(sub.func.attr)
            if callee is not None and callee is not node:
                bodies.append(callee)
    return any(
        isinstance(sub, ast.Attribute) and sub.attr == "scale_loss"
        for body in bodies for sub in ast.walk(body)
    )


ALL_SITES = list(_train_step_bodies())


def test_the_predicate_follows_one_level_of_indirection():
    """The property that makes `depth_anything` come out COMPLIANT.

    Pinned separately from the sweep so that narrowing the predicate back to a
    lexical scan fails HERE, with this name, rather than silently turning the
    tree's only correct site into an offender.
    """
    module = ast.parse(
        "def train_step(self, data):\n"
        "    return self._real_step(data)\n"
        "def _real_step(self, data):\n"
        "    with tf.GradientTape() as tape:\n"
        "        loss = self.compute_loss(y_pred=self(data))\n"
        "        scaled = self.optimizer.scale_loss(loss)\n"
        "    return tape.gradient(scaled, self.trainable_variables)\n"
    )
    defs = {n.name: n for n in module.body if isinstance(n, ast.FunctionDef)}
    assert _calls_scale_loss(defs["train_step"], defs)
    assert not _calls_scale_loss(defs["train_step"], {}), (
        "the lexical reading must NOT see the call -- otherwise this test is "
        "not testing the closure"
    )


def test_the_subject_set_is_not_empty():
    """Anti-vacuity floor, derived from the population measured when this
    guard landed (12 `train_step` definitions under `models/`)."""
    floor = int(0.8 * 12)
    assert len(ALL_SITES) >= floor, (
        f"only {len(ALL_SITES)} `train_step` definitions found under "
        f"{MODELS_ROOT}; the sweep has stopped seeing the tree"
    )


@pytest.mark.parametrize(
    "relative_path,node,module_defs",
    [(p, n, d) for p, n, d in ALL_SITES if p not in KNOWN_UNSCALED],
    ids=[p for p, _, _ in ALL_SITES if p not in KNOWN_UNSCALED],
)
def test_the_train_step_scales_the_loss(relative_path, node, module_defs):
    assert _calls_scale_loss(node, module_defs), (
        f"{relative_path}:{node.lineno} overrides `train_step` without calling "
        f"`self.optimizer.scale_loss(loss)` inside the tape. Under "
        f"`mixed_float16` the LossScaleOptimizer divides every gradient by "
        f"2**15 regardless, so the whole update is silently scaled down by "
        f"~32768x. Add the call, or add this path to KNOWN_UNSCALED with its "
        f"MEASURED ratio."
    )


def test_every_waiver_is_still_a_real_site():
    """Liveness: a waiver that no longer names a live unscaled site is stale."""
    present = {p: (n, d) for p, n, d in ALL_SITES}
    stale = []
    for waived in KNOWN_UNSCALED:
        if waived not in present:
            stale.append(f"{waived} (no `train_step` there any more)")
        elif _calls_scale_loss(*present[waived]):
            stale.append(f"{waived} (it now DOES call scale_loss -- drop the waiver)")
    assert not stale, "stale waivers: " + "; ".join(stale)


def test_the_predicate_fires_on_an_injected_defect():
    """Injected twin: a `train_step` with no `scale_loss` must be REJECTED."""
    injected = ast.parse(
        "def train_step(self, data):\n"
        "    with tf.GradientTape() as tape:\n"
        "        loss = self.compute_loss(y_pred=self(data))\n"
        "    g = tape.gradient(loss, self.trainable_variables)\n"
    ).body[0]
    assert not _calls_scale_loss(injected, {})
    with pytest.raises(AssertionError, match="without calling"):
        test_the_train_step_scales_the_loss("injected.py", injected, {})


def test_the_predicate_is_silent_on_the_fixed_twin():
    """Fixed twin: the same body WITH the call must be accepted."""
    fixed = ast.parse(
        "def train_step(self, data):\n"
        "    with tf.GradientTape() as tape:\n"
        "        loss = self.compute_loss(y_pred=self(data))\n"
        "        scaled = self.optimizer.scale_loss(loss)\n"
        "    g = tape.gradient(scaled, self.trainable_variables)\n"
    ).body[0]
    assert _calls_scale_loss(fixed, {})
    test_the_train_step_scales_the_loss("fixed.py", fixed, {})
