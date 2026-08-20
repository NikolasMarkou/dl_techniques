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

Step 19.2 closed the five sites step 18 owed. Under the DEFAULT dynamic
`LossScaleOptimizer` the ratio BEFORE, GPU 1, was:

    vae             9.680747e+01 / 1.549095e-03 = 6.249e+04
    vq_vae          1.380558e+00 / 4.532856e-05 = 3.046e+04
    vq_vae_rotation 9.799837e-01 / 2.713639e-05 = 3.611e+04
    video_jepa      1.433110e+04 / 1.204127e-02 = 1.190e+06
    capsnet         1.281255e+02 / 3.832428e-02 = 3.343e+03

The AFTER arm needed the instrument corrected TWICE, and both corrections are
part of the evidence rather than footnotes to it:

1. `model.get_weights()` includes NON-trainable state. BatchNorm moving
   averages move identically in both arms and MASKED the whole defect --
   `vae` read a ratio of 5.7 that way against 6.2e+04 over
   `model.trainable_weights`. The |dW| statistic must be over trainable
   weights only.
2. The DEFAULT `initial_scale=32768` overflows the fp16 backward of `vae` and
   `video_jepa`, so the dynamic scaler SKIPS steps inside a 6-step window --
   `dynamic_scale` fell 32768 -> 2048 for `vae`, i.e. 4 of 6 steps never
   applied, and the ratio read 4.3 for a correct fix. The AFTER arm therefore
   pins a STATIC scale (`LossScaleOptimizer(SGD(0.1), initial_scale=8,
   dynamic_growth_steps=10**9)`) and asserts `dynamic_scale` is unchanged
   across the measured window, so every step applies.

At that no-skip static scale, with `scale_loss` monkeypatched to the identity
to reproduce the pre-fix source exactly (BEFORE) -- CPU:

    site            float32      fp16 BEFORE  BEFORE  fp16 AFTER   AFTER
    vae             9.165381e+01 5.285929e+00 17.34   9.147439e+01 1.0020
    vq_vae          1.380598e+00 2.007510e-01  6.877  1.383238e+00 0.99809
    vq_vae_rotation 9.800340e-01 1.384036e-01  7.081  9.874229e-01 0.99252
    video_jepa      3.669095e+02 1.952298e+01 18.79   3.664513e+02 1.00125
    capsnet         1.281065e+02 2.242368e+02  0.5713 1.278453e+02 1.00204

GPU 1 AFTER, same protocol: vae 1.0881, vq_vae 0.99817, vq_vae_rotation
0.99252, video_jepa 1.0622, capsnet 1.00566 -- all inside the +-9.4% band the
`depth_anything` calibration arm establishes for this instrument.

`capsnet`'s BEFORE ratio of 0.5713 is BELOW 1: unscaled, its fp16 update was
LARGER than its float32 one. That is the same anomaly D-036 refused to smooth
over and it is still unexplained; what IS established is that the AFTER ratio
is 1.002 on CPU and 1.006 on GPU.

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

# Sites measured LIVE but NOT repaired, each with its measured ratio. Every
# entry is a real defect; the waiver records that it is KNOWN and owed, not
# that it is acceptable. Removing an entry must be accompanied by the
# `scale_loss` call, and `test_every_waiver_is_still_a_real_site` fails the
# moment an entry stops naming a live unscaled `train_step`.
#
# The five sites D-036 routed to step 18 -- `vae`, `vq_vae`,
# `vq_vae_rotation`, `video_jepa`, `capsnet` -- are GONE from this set as of
# step 19.2, repaired with the before/after table in the module docstring.
# Nothing here is "routed to" a step any more; the three that remain are
# ruled, not owed.
KNOWN_UNSCALED = {
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


# ---------------------------------------------------------------------
# The second half of the rule: what you do with the SCALED gradient
# ---------------------------------------------------------------------

_CLIP_CALLS = {"clip", "clip_by_value", "clip_by_norm", "clip_by_global_norm"}


def _literal_clip_of_scaled_gradients(node: ast.FunctionDef) -> list:
    """Clip calls with a CONSTANT bound inside a `train_step` that scales.

    Adding `scale_loss` is only half the repair. `tape.gradient(scaled_loss)`
    returns gradients in the SCALED domain, and anything that compares them
    against a fixed threshold before `apply()` unscales them is now comparing
    against a threshold 2**15 times too small. MEASURED in `models/vae`: with
    the `scale_loss` call correctly in place, a surviving
    `ops.clip(grad, -1.0, 1.0)` saturated every component and the per-element
    |dW| came out at exactly 3.051758e-06 == 0.1 * 2**-15, for an fp16/float32
    ratio of 64.8 instead of 1.0. The repair is to express the bound in the
    scaled domain -- `self.optimizer.scale_loss(1.0)` is the current loss
    scale, and is exactly 1.0 for a plain optimizer.

    A constant bound is the detectable form; a bound derived from
    `scale_loss` is not a Constant, so it passes.
    """
    hits = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        name = sub.func.attr if isinstance(sub.func, ast.Attribute) else (
            sub.func.id if isinstance(sub.func, ast.Name) else None)
        if name not in _CLIP_CALLS:
            continue
        for arg in sub.args[1:]:
            inner = arg.operand if isinstance(arg, ast.UnaryOp) else arg
            if isinstance(inner, ast.Constant):
                hits.append((name, sub.lineno))
                break
    return hits


@pytest.mark.parametrize(
    "relative_path,node,module_defs",
    [(p, n, d) for p, n, d in ALL_SITES if p not in KNOWN_UNSCALED],
    ids=[p for p, _, _ in ALL_SITES if p not in KNOWN_UNSCALED],
)
def test_a_scaling_train_step_does_not_clip_against_a_constant(
        relative_path, node, module_defs):
    if not _calls_scale_loss(node, module_defs):
        pytest.skip("not a scaling train_step")
    hits = _literal_clip_of_scaled_gradients(node)
    assert not hits, (
        f"{relative_path}:{node.lineno} calls `scale_loss` and then clips "
        f"against a CONSTANT bound at {hits}. Those gradients are in the "
        f"scaled domain; a literal bound saturates them and the subsequent "
        f"unscale divides the whole update by the loss scale. Express the "
        f"bound as `self.optimizer.scale_loss(<bound>)`."
    )


def test_the_clip_predicate_fires_on_the_pre_fix_vae_body():
    """RED proof, taken from the real pre-fix `models/vae` source."""
    pre_fix = ast.parse(
        "def train_step(self, data):\n"
        "    with tf.GradientTape() as tape:\n"
        "        total_loss = self.compute_loss(y_pred=self(data))\n"
        "        scaled_loss = self.optimizer.scale_loss(total_loss)\n"
        "    gradients = tape.gradient(scaled_loss, self.trainable_weights)\n"
        "    gradients = [ops.clip(g, -1.0, 1.0) for g in gradients]\n"
    ).body[0]
    assert _literal_clip_of_scaled_gradients(pre_fix) == [("clip", 6)]
    with pytest.raises(AssertionError, match="CONSTANT bound"):
        test_a_scaling_train_step_does_not_clip_against_a_constant(
            "pre_fix_vae.py", pre_fix, {})


def test_the_clip_predicate_is_silent_on_the_fixed_vae_body():
    """Fixed twin: the shipped form, whose bound is not a Constant."""
    fixed = ast.parse(
        "def train_step(self, data):\n"
        "    with tf.GradientTape() as tape:\n"
        "        total_loss = self.compute_loss(y_pred=self(data))\n"
        "        scaled_loss = self.optimizer.scale_loss(total_loss)\n"
        "    gradients = tape.gradient(scaled_loss, self.trainable_weights)\n"
        "    lim = self.optimizer.scale_loss(1.0)\n"
        "    gradients = [ops.clip(g, -lim, lim) for g in gradients]\n"
    ).body[0]
    assert _literal_clip_of_scaled_gradients(fixed) == []
    test_a_scaling_train_step_does_not_clip_against_a_constant(
        "fixed_vae.py", fixed, {})
