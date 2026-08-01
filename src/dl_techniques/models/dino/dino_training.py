"""
DINO self-distillation training model — student + EMA teacher under stock `fit()`.

This module holds `DINOTrainingModel`, the object that turns the DINO backbones in
this package into something `model.compile(loss=DINOLoss(...))` + `model.fit()` can
actually train, with **no custom `train_step` and no bespoke training loop**.

Multi-crop input contract (the whole contract, stated once)
-----------------------------------------------------------
`call()` takes ONE fixed-shape tensor::

    (batch, n_views, height, width, channels)

* `n_views == n_global_views + n_local_views`, and `n_global_views` is 2 (DINO).
* **Views `0` and `1` are the GLOBAL crops**; views `2 ...` are the local crops.
  The teacher sees only views 0 and 1; the student sees all of them.
* Every view is at the SAME pixel resolution `(height, width)`. This is plan
  decision D-002: local crops crop a smaller AREA of the source image and are
  resized up to the global size, so one backbone, one positional-embedding table
  and `tf.data`'s fixed-shape batching serve every view with no
  positional-embedding interpolation. The cost is that local views are as
  expensive as global ones — the paper's compute saving on locals is given up.
* `src/dl_techniques/datasets/vision/multi_crop.py` produces exactly this element
  shape (built in the next step of the same plan).

Output contract
---------------
`call()` returns a SINGLE tensor of shape ``(batch * n_pairs, 2 * out_dim)``,
where ``n_pairs == n_global_views * n_views - n_global_views`` — every
(teacher global view, student view) pair with the same-view pair removed. Each
row is ``concatenate([student_logits, teacher_logits], axis=-1)``, built by
`dl_techniques.losses.dino_loss.pack_student_teacher`, and split again inside
`DINOLoss`.

Why a single tensor rather than the obvious `{"student_logits": ...,
"teacher_logits": ...}` dict: see the D-009 anchor on `DINOTrainingModel.call`.

Teacher updates
---------------
The teacher is NOT trained by backpropagation. It is an EMA of the student,
advanced once per training batch by
`dl_techniques.models.depth_anything.teacher_ema.TeacherEMACallback`, which calls
`update_teacher_ema(decay=...)` on this model. That module's
`cosine_ema_schedule` / `linear_ema_schedule` / `TeacherEMACallback` are reused
UNCHANGED — do not copy them here.

Example
-------
```python
from dl_techniques.models.dino import create_dino_training_model
from dl_techniques.losses.dino_loss import DINOLoss
from dl_techniques.models.depth_anything.teacher_ema import (
    TeacherEMACallback, cosine_ema_schedule,
)

model = create_dino_training_model(
    "tiny", image_size=96, patch_size=16, n_local_views=4, dino_out_dim=4096,
)
model.compile(optimizer="adamw", loss=DINOLoss(out_dim=4096))
model.fit(                          # NOTE: never pass validation_data -- the
    train_ds,                       # centering EMA fires on validation batches
    epochs=100,                     # too (see DINOLoss's class docstring).
    callbacks=[TeacherEMACallback(cosine_ema_schedule(0.996, 0.9999, 10000))],
)
```
"""

import keras
from keras import ops
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.losses.dino_loss import pack_student_teacher
from dl_techniques.models.dino.common import reject_input_shape
from dl_techniques.models.dino.dino_v1 import (
    ModelVariant,
    create_dino_teacher_student_pair,
)
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

# DINO always uses exactly two global crops (Caron et al. 2021 §3, and every
# follow-up). It is a named constant rather than a constructor argument because
# the loss's pairing, the "views 0 and 1 are global" input contract and the
# multi-crop dataset map all assume it; making it configurable would create
# three places that must agree and no caller that needs them to differ.
N_GLOBAL_VIEWS = 2


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DINOTrainingModel(keras.Model):
    """
    Student + frozen EMA teacher over a multi-crop batch, trainable by `fit()`.

    The model owns two structurally identical backbones-with-projection-heads.
    The student sees every view and is trained by backpropagation; the teacher
    sees only the global views, runs under `keras.ops.stop_gradient`, is
    `trainable=False`, and is advanced by `update_teacher_ema()`.

    Args:
        student: A built `keras.Model` mapping `(batch, H, W, C)` ->
            `(batch, out_dim)`. In practice a `DINOv1` / `DINOv2` / `DINOv3`
            with `include_projection_head=True`.
        teacher: A structurally identical `keras.Model`. Must be a DISTINCT
            object with DISTINCT weight variables — a shared pair would make
            the EMA a silent no-op, which is checked at construction.
        n_local_views: Number of local crops per sample, `>= 0`. Total views
            per sample is `N_GLOBAL_VIEWS + n_local_views`.
        **kwargs: Forwarded to `keras.Model`.

    Input shape:
        `(batch, N_GLOBAL_VIEWS + n_local_views, height, width, channels)`;
        views `0` and `1` are the global crops. See the module docstring for
        the full contract.

    Output shape:
        `(batch * n_pairs, 2 * out_dim)` — see the module docstring.

    Attributes:
        out_dim: Width of ONE network's logits, derived from `student`.
        n_views: `N_GLOBAL_VIEWS + n_local_views`.
        n_pairs: `N_GLOBAL_VIEWS * n_views - N_GLOBAL_VIEWS`.

    Raises:
        ValueError: If `student` and `teacher` are the same object, share
            weight variables, or disagree on weight count / shapes; if either
            is unbuilt; if their output is not rank-2; or if `n_local_views`
            is negative.

    Example:
        ```python
        from dl_techniques.models.dino import (
            create_dino_teacher_student_pair, DINOTrainingModel,
        )

        teacher, student = create_dino_teacher_student_pair(
            "tiny", image_size=32, patch_size=16, dino_out_dim=128,
        )
        model = DINOTrainingModel(
            student=student, teacher=teacher, n_local_views=2,
        )
        ```
    """

    def __init__(
            self,
            student: keras.Model,
            teacher: keras.Model,
            n_local_views: int = 0,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if n_local_views < 0:
            raise ValueError(
                f"n_local_views must be >= 0, got {n_local_views}")

        _validate_student_teacher_pair(student, teacher)

        self.student = student
        self.teacher = teacher

        # The teacher is never trained by backpropagation. This is re-applied
        # on EVERY construction, INCLUDING `from_config`, on purpose: a
        # `keras.Model`'s `trainable` flag is not part of the config a
        # sub-model round-trips through, so a teacher that silently reloaded
        # trainable would produce bit-identical outputs and a quietly wrong
        # training run (a recorded repo gotcha).
        self.teacher.trainable = False

        self.n_local_views = int(n_local_views)
        self.n_views = N_GLOBAL_VIEWS + self.n_local_views
        self.out_dim = int(student.output_shape[-1])
        self._image_shape: Tuple[int, ...] = tuple(student.input_shape[1:])

        # Every (teacher global view, student view) pair except the same-view
        # one, as two static index lists. Python-level loops over STATIC
        # integers -- nothing here reads a tensor value.
        teacher_index: List[int] = []
        student_index: List[int] = []
        for t_view in range(N_GLOBAL_VIEWS):
            for s_view in range(self.n_views):
                if s_view == t_view:
                    continue
                teacher_index.append(t_view)
                student_index.append(s_view)
        self._teacher_pair_index = teacher_index
        self._student_pair_index = student_index
        self.n_pairs = len(teacher_index)

        if self.n_pairs == 0:
            raise ValueError(
                f"n_views={self.n_views} leaves no cross-view pair to train "
                f"on (every pair would be a view against itself). Increase "
                f"n_local_views."
            )

        logger.info(
            f"Created DINOTrainingModel: {self.n_views} views "
            f"({N_GLOBAL_VIEWS} global + {self.n_local_views} local), "
            f"{self.n_pairs} cross-view pairs, out_dim={self.out_dim}, "
            f"image shape {self._image_shape}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Validate the multi-crop input contract; the sub-models are built."""
        if len(input_shape) != 5:
            raise ValueError(
                f"DINOTrainingModel expects a rank-5 multi-crop input "
                f"(batch, n_views, height, width, channels), got shape "
                f"{tuple(input_shape)}."
            )
        if input_shape[1] != self.n_views:
            raise ValueError(
                f"DINOTrainingModel was configured for n_views="
                f"{self.n_views} ({N_GLOBAL_VIEWS} global + "
                f"{self.n_local_views} local) but received an input with "
                f"{input_shape[1]} views (shape {tuple(input_shape)}). Views "
                f"0 and 1 must be the global crops."
            )
        if tuple(input_shape[2:]) != self._image_shape:
            raise ValueError(
                f"DINOTrainingModel's student takes images of shape "
                f"{self._image_shape} but the input's per-view shape is "
                f"{tuple(input_shape[2:])}. Per D-002 every view -- global "
                f"and local alike -- must be rendered at the SAME pixel "
                f"resolution."
            )
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Run the student on all views and the teacher on the global views.

        Args:
            inputs: `(batch, n_views, height, width, channels)`; views 0 and 1
                are the global crops.
            training: Standard Keras training flag, forwarded to the STUDENT
                only. The teacher always runs with `training=False` — it is a
                target-producing network, so its dropout / stochastic depth
                must be off regardless of the caller's flag.

        Returns:
            `(batch * n_pairs, 2 * out_dim)` — see the module docstring.
        """
        # DECISION plan-2026-08-01T105809-dc0c402e/D-009
        # This returns a SINGLE, RANK-2 tensor. Do NOT "improve" it into the
        # structured dict `{"student_logits": ..., "teacher_logits": ...}`
        # that plan decision D-005 originally specified, and do NOT return the
        # un-flattened `(batch, n_pairs, 2 * out_dim)` form.
        #   * The DICT is refused by Keras 3.8.0: `CompileLoss.build`
        #     broadcasts one `Loss` object across every leaf of a nested
        #     `y_pred` (`tree.map_structure(lambda x: loss, y_pred)`,
        #     compile_utils.py:653) and raises `KeyError: "The path:
        #     ('student_logits',) in the 'loss' argument, can't be found in
        #     either the model's output ('y_pred') or in the labels
        #     ('y_true')."` The `CLIPContrastiveLoss` precedent cited for the
        #     dict has only ever run under a hand-rolled loop, never stock
        #     `fit()`, so it was never evidence.
        #   * The RANK-3 form was MEASURED to fail inside the loss: `DINOLoss`
        #     reduces its centering statistic over `axis=0` only, so a
        #     `(batch, n_pairs, ...)` y_pred makes the batch centre
        #     `(1, n_pairs, out_dim)` and `center.assign()` dies with
        #     `NotImplementedError: numpy() is only available when eager
        #     execution is enabled` -- a shape error disguised as a backend
        #     error, mid-`fit()`.
        #   * `sample_weight` (the other candidate) cannot carry the teacher
        #     at all: MEASURED, `fit()` sources `sample_weight` from the
        #     DATASET tuple, and the teacher's logits are produced by this
        #     model from the same batch. `iBOTPatchLoss` additionally REFUSES
        #     a non-None `sample_weight` (D-008).
        # The flatten below is therefore load-bearing, not cosmetic.
        image_shape = self._image_shape

        # Student: every view. Fold views into the batch axis so one forward
        # pass covers them all.
        student_views = ops.reshape(inputs, (-1, *image_shape))
        student_logits = self.student(student_views, training=training)
        student_logits = ops.reshape(
            student_logits, (-1, self.n_views, self.out_dim))

        # Teacher: global views only, no gradient.
        global_views = inputs[:, :N_GLOBAL_VIEWS]
        teacher_views = ops.reshape(global_views, (-1, *image_shape))
        teacher_logits = self.teacher(teacher_views, training=False)
        teacher_logits = ops.stop_gradient(teacher_logits)
        teacher_logits = ops.reshape(
            teacher_logits, (-1, N_GLOBAL_VIEWS, self.out_dim))

        # Expand into cross-view pairs, then pack student+teacher on the last
        # axis and flatten the pair axis into the batch axis.
        student_pairs = ops.take(
            student_logits, self._student_pair_index, axis=1)
        teacher_pairs = ops.take(
            teacher_logits, self._teacher_pair_index, axis=1)
        packed = pack_student_teacher(student_pairs, teacher_pairs)
        return ops.reshape(packed, (-1, 2 * self.out_dim))

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """`(batch, n_views, H, W, C)` -> `(batch * n_pairs, 2 * out_dim)`."""
        batch = input_shape[0]
        rows = None if batch is None else batch * self.n_pairs
        return (rows, 2 * self.out_dim)

    def update_teacher_ema(self, decay: float) -> None:
        """
        Move every teacher weight toward its student counterpart.

        Interface contract:
            Parameters:
                decay: EMA coefficient in `[0, 1]`. `teacher <- decay *
                    teacher + (1 - decay) * student`, so `decay=1.0` freezes
                    the teacher and `decay=0.0` copies the student outright.
            Returns:
                ``None``. Every teacher variable is assigned in place.
            Failure mode:
                ``ValueError`` when ``decay`` is outside `[0, 1]`.

        **This method name is a hard contract with
        `dl_techniques.models.depth_anything.teacher_ema.TeacherEMACallback`,
        which calls `self.model.update_teacher_ema(decay=...)` by name.** If
        it is renamed or removed, that callback logs ONE warning and
        SELF-DISABLES: the run completes, the loss curve looks fine, and the
        teacher is never updated for the whole of pretraining. Keep the name
        and the `decay` KEYWORD spelling.

        All weights are updated, non-trainable ones included (normalization
        moving statistics are part of what the teacher must track).
        """
        decay = float(decay)
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}")

        for teacher_weight, student_weight in zip(
                self.teacher.weights, self.student.weights):
            teacher_weight.assign(
                decay * teacher_weight
                + (1.0 - decay) * ops.cast(student_weight, teacher_weight.dtype)
            )

    def get_config(self) -> Dict[str, Any]:
        """Serialize both sub-models plus the view layout."""
        config = super().get_config()
        config.update({
            "student": keras.saving.serialize_keras_object(self.student),
            "teacher": keras.saving.serialize_keras_object(self.teacher),
            "n_local_views": self.n_local_views,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DINOTrainingModel":
        """Rebuild both sub-models, then re-freeze the teacher (see `__init__`)."""
        config = dict(config)
        config["student"] = keras.saving.deserialize_keras_object(
            config["student"])
        config["teacher"] = keras.saving.deserialize_keras_object(
            config["teacher"])
        return cls(**config)


# ---------------------------------------------------------------------


def _validate_student_teacher_pair(
        student: keras.Model,
        teacher: keras.Model,
) -> None:
    """
    Refuse a student/teacher pair that would make the EMA meaningless.

    Interface contract:
        Parameters:
            student: The trainable network.
            teacher: The EMA target network.
        Returns:
            ``None`` when the pair is usable.
        Failure mode:
            ``ValueError``, naming which property failed.

    The checks exist because each failure is SILENT rather than loud: a shared
    object (or shared variables) makes `update_teacher_ema` a no-op that still
    moves "both" networks; a weight-list mismatch makes the zip() silently
    update a prefix; a non-rank-2 output makes the packed layout wrong in a way
    the loss cannot see.
    """
    if student is teacher:
        raise ValueError(
            "student and teacher must be two DISTINCT models; the same object "
            "was passed twice, which would make update_teacher_ema a no-op "
            "(the teacher would already equal the student)."
        )
    if not student.built or not teacher.built:
        raise ValueError(
            "student and teacher must both be built before constructing "
            "DINOTrainingModel (their weights are zipped positionally)."
        )
    if len(student.weights) != len(teacher.weights):
        raise ValueError(
            f"student and teacher must be structurally identical, but the "
            f"student has {len(student.weights)} weights and the teacher has "
            f"{len(teacher.weights)}."
        )

    student_ids = {id(w) for w in student.weights}
    shared = [w.path for w in teacher.weights if id(w) in student_ids]
    if shared:
        raise ValueError(
            f"student and teacher share {len(shared)} weight variable(s) "
            f"(e.g. {shared[0]}). The EMA would then be a no-op and the "
            f"teacher would receive the student's gradients."
        )

    for index, (student_weight, teacher_weight) in enumerate(
            zip(student.weights, teacher.weights)):
        if student_weight.shape != teacher_weight.shape:
            raise ValueError(
                f"student and teacher weight {index} disagree on shape: "
                f"{student_weight.shape} vs {teacher_weight.shape} "
                f"({student_weight.path} vs {teacher_weight.path})."
            )

    if len(student.output_shape) != 2:
        raise ValueError(
            f"student must produce rank-2 logits (batch, out_dim), got "
            f"{student.output_shape}. Build it with "
            f"include_projection_head=True."
        )
    if student.output_shape[-1] != teacher.output_shape[-1]:
        raise ValueError(
            f"student and teacher must produce the same out_dim, got "
            f"{student.output_shape[-1]} and {teacher.output_shape[-1]}."
        )
    if tuple(student.input_shape[1:]) != tuple(teacher.input_shape[1:]):
        raise ValueError(
            f"student and teacher must take the same input shape, got "
            f"{student.input_shape} and {teacher.input_shape}."
        )


# ---------------------------------------------------------------------


def create_dino_training_model(
        variant: ModelVariant = "small",
        *,
        image_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        n_local_views: int = 4,
        dino_out_dim: int = 65536,
        name: Optional[str] = None,
        **kwargs: Any
) -> DINOTrainingModel:
    """
    Build a `DINOTrainingModel` from a DINOv1 variant name.

    Follows the converged DINO factory scheme: `image_size` accepts an int or a
    `(height, width)` tuple, `patch_size=None` defers to the variant, and
    `input_shape` is refused (use `image_size`).

    Args:
        variant: One of `"tiny"`, `"small"`, `"base"`, `"large"`, `"giant"`.
        image_size: The GLOBAL crop size. Every view — local crops included —
            is rendered at this resolution (D-002).
        patch_size: Patch size, or `None` to defer to the variant's own.
        n_local_views: Number of local crops per sample.
        dino_out_dim: Projection-head output width. The paper uses 65536; the
            smoke scale of this repo's trainer uses 4096-8192.
        name: Optional model name.
        **kwargs: Forwarded to `create_dino_teacher_student_pair`, hence to
            both backbone constructors.

    Returns:
        An unbuilt-but-ready `DINOTrainingModel` (its two sub-models are built).

    Raises:
        TypeError: If `input_shape` is passed — use `image_size`.

    Example:
        ```python
        model = create_dino_training_model(
            "tiny", image_size=96, patch_size=16,
            n_local_views=4, dino_out_dim=4096,
        )
        ```
    """
    reject_input_shape(kwargs, "create_dino_training_model")

    teacher, student = create_dino_teacher_student_pair(
        variant=variant,
        image_size=image_size,
        patch_size=patch_size,
        dino_out_dim=dino_out_dim,
        **kwargs
    )
    return DINOTrainingModel(
        student=student,
        teacher=teacher,
        n_local_views=n_local_views,
        name=name,
    )

# ---------------------------------------------------------------------
