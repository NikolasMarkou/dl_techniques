# Migrations

Checkpoint-affecting changes to this repository, newest first.

A change belongs here when a `.keras` archive written *before* it may not load *after* it
— a registered key that moves, a module that moves, a config field that is renamed or
dropped. `plans/` is gitignored, so a note there does not reach the next reader; this file
is tracked and does.

---

## 2026-08-29 — every registered class moved to a package-qualified key

### What changed

Every `@keras.saving.register_keras_serializable()` call site under `src/` was replaced
with `@register_dl_technique("<package>")`, from
`src/dl_techniques/utils/keras_registration.py`.

**744 sites across 493 files** (494 files changed counting the helper itself), in
`src/dl_techniques/{layers,models,losses,metrics,initializers,regularizers,optimization,
callbacks,constraints}` and `src/train/`. `src/applications/` holds zero registration
sites.

The bare decorator mints the key `Custom>{ClassName}`. That key is **module-independent**:
two classes with the same bare name, anywhere in the tree, claim the identical registry
slot and whichever module is imported LAST silently wins every deserialization of both.
The v2 guide's §2.2 mandates the explicit `package=` that closes it. The new key is
`{package}>{ClassName}`.

**The rule for `package`** is the defining module's dotted path, with two things stripped
from `dl_techniques.models`:

- its **12 family directories** — `common`, `embeddings_experimental`, `general_purpose`,
  `graph`, `language`, `memory`, `neural_computer`, `point_cloud`, `tabular`,
  `time_series`, `vision`, `vision_language`;
- its **4 subfamily containers** — `image_restoration`, `keypoints`, `super_resolution`,
  `sam`.

Both are a filing decision rather than a namespace (`src/dl_techniques/models/README.md`),
and they have already been reshuffled once: the 2026-08-24 family-nesting reorg. A key
derived from them would have broken every archive at that moment. So
`src/dl_techniques/models/vision_language/sam/sam2/hiera.py` registers under
`dl_techniques.models.sam2.hiera`, not under its full import path.

The 38 sites that already carried an explicit `package=` kept their string unchanged, with
one exception (`yolov12_losses`, below).

### Why it is safe

`register_dl_technique` binds **two** keys per object: the package-qualified one, which is
what a NEW save writes, and the pre-migration `Custom>{ClassName}` as an alias to the same
object, which is what an OLD archive reads.

The alias is load-bearing, not decorative. Measured with a control on Keras 3.8.0: an
archive written under `Custom>X` and reloaded after X was re-keyed **with the alias
suppressed** is REFUSED with `TypeError`. Keras does **not** silently fall back to the
module path recorded alongside `registered_name`.

The measured proof that the migration is safe on this repository's actual data:
**all 14 custom class names present in the repo's `.keras` archives resolve under BOTH
their new key and `Custom>X`, to the IDENTICAL OBJECT** (`a is b`, not merely both
non-`None`) — `AsciiBert`, `AsciiCliffordBert`, `AsciiConvNextBert`, `AsciiConvNextV2Bert`,
`ResNet`, `Sam3DetectionLoss`, `Sam3DotProductScoring`, `Sam3DualViTDetNeck`, `Sam3Image`,
`Sam3SegmentationHead`, `Sam3TextEncoder`, `Sam3TrainingModel`, `Sam3TransformerDecoder`,
`Sam3ViTDetBackbone`. Across the whole rewrite the archive-load guard never lost a single
passing archive.

The registry census after the migration: **1452 keys, 0 duplicates** (up from 728, because
each aliased object now holds two keys).

### The four exceptions — no legacy alias at all

`ConvBlock`, `Downsample`, `MLPBlock` and `Upsample` each name **two** distinct registered
classes:

| bare name | the two qualified keys |
|---|---|
| `ConvBlock` | `dl_techniques.layers.yolo12_blocks>ConvBlock`, `dl_techniques.standard_blocks>ConvBlock` |
| `Downsample` | `dl_techniques.ideogram4>Downsample`, `dl_techniques.pw_fnet>Downsample` |
| `MLPBlock` | `dl_techniques.layers.ffn.mlp>MLPBlock`, `dl_techniques.tabm>MLPBlock` |
| `Upsample` | `dl_techniques.ideogram4>Upsample`, `dl_techniques.pw_fnet>Upsample` |

Aliasing either side would put **both** of them back on one `Custom>X` key and recreate,
in the legacy namespace, exactly the import-order collision this migration removes. So all
eight carry `legacy_alias=False` and `Custom>ConvBlock`, `Custom>Downsample`,
`Custom>MLPBlock` and `Custom>Upsample` resolve to **nothing**. `register_dl_technique`
raises `AliasCollisionError` at import time rather than letting one side quietly take the
key — Keras itself accepts the second write and makes the winner import-order-dependent.

**None of these eight appears in any archive in this repository.** Nothing was orphaned.

### `yolov12_losses` — the one re-keyed non-bare package

The four losses in `src/dl_techniques/losses/yolo12_multitask_loss.py`
(`YOLOv12ObjectDetectionLoss`, `DiceFocalSegmentationLoss`, `ClassificationFocalLoss`,
`YOLOv12MultiTaskLoss`) were the tree's only registrations under a package string with no
owning namespace: `"yolov12_losses"`. An un-namespaced key is claimable by any other
library that picks the same word — the same §2.2 hazard, one step removed — so they moved
to `dl_techniques.losses.yolo12_multitask_loss`.

They needed a mechanism the default alias does not provide. Their sites were **already
explicit**, so an archive holding one of them stores `yolov12_losses>X` and **never**
`Custom>X`. The default `Custom>` alias therefore covers nothing for them and the re-key
would have been a silent break. They pass
`legacy_packages=("yolov12_losses",)`, which binds that old key too. All four now resolve
under three keys to the same object.

`dl_techniques.train.ideogram4` (`src/train/ideogram4/train_ideogram4.py`) was
deliberately **left alone** by the same reasoning read the other way: it is already
namespaced and already unique, so re-keying it would be a checkpoint-affecting change
bought for nothing.

### What a caller must do

**For loading: nothing.** Every archive that loaded before this change loads after it. The
one standing requirement is unchanged and predates this migration: the module defining a
custom class must be imported before `keras.models.load_model` is called on an archive
containing it (`src/applications/bias_free_denoiser/denoiser_prior.py` documents this as
"registrar-first import must precede `load_model`").

**For writing a NEW registered class:** use the helper, not the stock decorator.

```python
from dl_techniques.utils.keras_registration import register_dl_technique

@register_dl_technique("dl_techniques.layers.attention.multi_head")
class MultiHeadAttention(keras.layers.Layer):
    ...
```

Pass `legacy_alias=False` if the bare class name is already registered by another class —
`AliasCollisionError` will tell you at import time if it is. A brand-new class that has
never been saved does not strictly need the alias, but leaving the default on costs
nothing and keeps the tree uniform.

**Do not add a bare `package=` to the stock decorator** as a repair for a collision. That
moves the key without minting an alias, and every archive storing the old
`registered_name` stops resolving.

### The 9 archives that do not load — and did not before this change either

These fail at HEAD, failed identically at the base commit, and are **not** caused by this
migration:

- `best_model.keras` (repo root)
- `results/sam3_tiny_20260813_143539/final_model.keras`
- `results/sam3_tiny_20260813_153331/final_model.keras`
- `results/sam3_tiny_20260813_163550/final_model.keras`
- `results/sam3_tiny_20260813_173620/final_model.keras`
- `results/sam3_tiny_20260813_183807/final_model.keras`
- `results/sam3_tiny_20260813_200815/final_model.keras`
- `results/sam3_tiny_20260814_091812/final_model.keras`
- `results/sam3_tiny_20260814_091832/final_model.keras`

The cause is the **2026-08-24 `models/` family-nesting reorg**, which moved modules without
a compatibility path. Each archive stores a `module` field naming a path that no longer
exists: `best_model.keras` names `dl_techniques.models.resnet.model` (now
`dl_techniques.models.vision.resnet.model`), and the eight `sam3_tiny_*` archives name
`dl_techniques.models.SAM.SAM3.training_model` (now
`dl_techniques.models.vision_language.sam.sam3.training_model`). All nine raise
`TypeError: Could not deserialize class '<X>' because its parent module <old.path> cannot
be imported`, which is a **module-path** failure, not a registered-key one — importing
those module paths directly fails the same way with no Keras involved.

Repairing them is a separate job: it needs either a module-path alias in `sys.modules` or a
rewrite of the stored `config.json`. This migration deliberately did not attempt it, and
the archive guard was graded against the measured base of "these nine fail" throughout, by
**set** rather than by count.

### Not fixed here, because there was nothing to fix

An "8 duplicate class names" concern was carried into this work. It was re-derived by AST
walk over `src/` and **no defect existed**. The real count is **15** duplicated class
names, in three groups, none of which contains a registry-key collision:

- **4 names with both sides registered** — `ConvBlock`, `Downsample`, `MLPBlock`,
  `Upsample` — already resolving to 8 distinct keys *before* this change;
- **2 names with only one side registered** — `BitErrorRate`, `SequenceAccuracy` (the
  other side is an undecorated class in `src/train/ntm/metrics.py`) — one claimant each;
- **9 names with neither side registered** — plain dataclasses and per-trainer configs,
  headed by `TrainingConfig` with **23** definitions and zero registry keys.

A duplicated class *name* and a duplicated registry *key* are unrelated properties. The
registry census reports 0 duplicate keys at HEAD, as it did before. The migration did not
repair any of these; the four-exception table above is the only duplicate-name handling
this change actually introduced, and it exists to avoid *creating* a collision in the
legacy namespace, not to fix one.
