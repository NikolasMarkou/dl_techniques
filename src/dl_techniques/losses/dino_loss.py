"""
A self-supervised loss from the DINO framework.

This loss function is the core objective of the DINO (self-DIstillation
with NO labels) framework. It enables a model to learn rich visual
representations from images alone, without any human-provided labels.
The method is based on a student-teacher knowledge distillation paradigm,
where the student network is trained to match the output of a momentum
teacher network when shown different augmented views of the same image.

Conceptual Overview:
    The fundamental idea is to enforce consistency in the representations of
    different distorted versions ("views") of an image. The student network
    processes a set of views (including small local crops and large global
    crops), while the teacher network only processes the global crops. The
    training objective is to make the student's output distribution for any
    view match the teacher's output distribution for the global views. This
    forces the student to learn features that are invariant to these
    augmentations, capturing high-level semantic content.

Architectural Design & Collapse Prevention:
    A naive implementation would be prone to "collapse," where both networks
    learn a trivial solution, such as outputting a constant vector for all
    inputs. DINO employs two key mechanisms to prevent this:
    1.  Momentum Teacher: The teacher's weights are not updated by back-
        propagation. Instead, they are an exponential moving average (EMA) of
        the student's weights. This provides more stable and slowly evolving
        targets for the student to learn from.
    2.  Centering: The teacher's outputs are centered by subtracting a
        running average of all batch outputs. This normalization prevents any
        single dimension from dominating the output and encourages the model
        to produce features that are uniformly distributed, effectively
        avoiding collapse.

Mathematical Formulation:
    The loss is a cross-entropy calculated between the probability
    distributions produced by the student and teacher networks. Let `z_s` and
    `z_t` be the output logits from the student and teacher, respectively.

    First, the logits are converted to probabilities using a softmax function
    with different temperature parameters (`τ_s` for student, `τ_t` for
    teacher). The teacher's output is also centered using a momentum-updated
    center vector `C`.

        p_s = softmax(z_s / τ_s)
        p_t = softmax((z_t - C) / τ_t)

    A low teacher temperature `τ_t` sharpens its output distribution, creating
    confident targets for the student to match. The loss is then the
    cross-entropy between these two distributions:

        Loss = - Σ p_t * log(p_s)

    The center `C` is updated via an EMA of the teacher's outputs over many
    batches.

References:
    -   Caron, M., et al. (2021). "Emerging Properties in Self-Supervised
        Vision Transformers." https://arxiv.org/abs/2104.14294
"""

import keras
import numpy as np
from keras import ops
from typing import Optional, Any, Dict, Tuple, Union

# ---------------------------------------------------------------------


def pack_student_teacher(
        student_logits: keras.KerasTensor,
        teacher_logits: keras.KerasTensor,
) -> keras.KerasTensor:
    """
    Pack aligned student and teacher logits into ONE tensor for stock `fit()`.

    Interface contract:
        Parameters:
            student_logits: `(..., out_dim)`.
            teacher_logits: `(..., out_dim)`, the SAME shape as
                `student_logits` -- already broadcast into row-for-row
                correspondence by the caller. Row `i` of the result is the
                (student view, teacher view) pair the loss will score.
        Returns:
            `(..., 2 * out_dim)` -- student first, teacher second. This layout
            is the single source of truth for the packed convention; the
            matching unpack lives in `_resolve_student_teacher` in this module,
            and NOWHERE else.
        Failure mode:
            No validation is performed here (it must stay trace-safe). A width
            mismatch surfaces in the loss: `_resolve_student_teacher` checks the
            packed last dimension against `2 * out_dim` and, when it does not
            match, raises a `ValueError` naming the observed width, `out_dim`
            and `2 * out_dim`. **The one width it cannot see is a tensor that
            is `2 * out_dim` wide by accident** (e.g. a student-only tensor fed
            to a loss whose `out_dim` is half the real one) -- that is read as
            packed and scored, because the conventions are told apart by width
            alone. See `_resolve_student_teacher`'s "Residual ambiguity".

    Why this exists (D-009, MEASURED on keras 3.8.0): a model whose `call()`
    returns a dict/tuple cannot be scored by ONE `compile(loss=...)` object --
    `CompileLoss.build` broadcasts the loss across every leaf of a nested
    `y_pred` and then raises `KeyError: The path: ('student_logits',) ...`. A
    single tensor is the only shape that works, so the two networks' outputs
    are concatenated on the last axis and split again inside the loss.

    Example:
        ```python
        packed = pack_student_teacher(student_logits, teacher_logits)
        # -> feed as the model's single output; DINOLoss(out_dim=D) unpacks it.
        ```
    """
    return ops.concatenate([student_logits, teacher_logits], axis=-1)

# ---------------------------------------------------------------------


def _resolve_student_teacher(
        loss_name: str,
        y_true: Optional[keras.KerasTensor],
        y_pred: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        out_dim: int,
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """
    Resolve `(teacher_logits, student_logits)` from any calling convention.

    Shared by `DINOLoss` and `iBOTPatchLoss`, which accept the same three
    conventions:

    1. **Packed single tensor** -- `y_pred` has a last dimension of exactly
       `2 * out_dim`: student first, teacher second (see
       `pack_student_teacher`). `y_true` is IGNORED. **This is the only
       convention usable under stock `compile(loss=...)` / `fit()`** when the
       teacher's logits are produced by the model rather than by the dataset.
    2. **Structured dict `y_pred`** -- carries `"student_logits"` and
       `"teacher_logits"`; `y_true` is IGNORED. Direct invocation only -- a
       model returning this dict is refused by Keras (D-009; see `DINOLoss`'s
       "Input shapes").
    3. **Direct two-tensor call** -- `y_true` is the teacher's logits and
       `y_pred` the student's, each of last dimension `out_dim`.

    Conventions 1 and 3 are told apart by the last dimension alone, which is
    unambiguous because `2 * out_dim != out_dim` for every legal `out_dim`
    (`out_dim > 0` is enforced by both constructors). A statically-unknown last
    dimension falls through to convention 3.

    Width validation (the contract `pack_student_teacher` advertises): a
    statically-known last dimension that is neither `out_dim` nor `2 * out_dim`
    raises a `ValueError` naming the observed width, `out_dim` and
    `2 * out_dim`. Under convention 3 the SAME check is applied to `y_true`,
    because under `fit()` `y_true` is the passed-through label, never the
    teacher's logits -- so a packed `y_pred` whose width misses `2 * out_dim`
    would otherwise fall through to convention 3 and be scored against the
    label. **MEASURED before this check existed** (`DINOLoss` on a correctly
    packed `(6, 16)` tensor plus a `(6, 1)` label): at `out_dim=9` the fall-
    through produced `InvalidArgumentError: required broadcastable shapes
    [Op:Mul]`, naming neither width; at `out_dim=16` it produced **no error at
    all**, returning `16.5552` computed from the LABEL as the teacher's logits.
    The silent case is the one this check exists for.

    Residual ambiguity (known, and NOT closable by a width check): a tensor
    that is `2 * out_dim` wide for the wrong reason is read as packed. MEASURED:
    a student-only `(6, 8)` tensor fed to `DINOLoss(out_dim=4)` is split into
    two halves of one network's logits and scored, returning `7.3828`. Width is
    the only signal available at trace time, so this is documented rather than
    detected; it is pinned by a characterization test in
    `tests/test_losses/test_dino_loss.py`.

    Args:
        loss_name: Name of the calling loss, used in error messages only.
        y_true: Teacher logits, or ignored under conventions 1 and 2.
        y_pred: Packed logits, student logits, or a dict carrying
            `"student_logits"` and `"teacher_logits"` (plus, for
            `iBOTPatchLoss`, an optional `"mask"`, which this helper does not
            read).
        out_dim: The calling loss's `out_dim`, i.e. the width of ONE network's
            logits. Used to split the packed form and to validate widths.

    Returns:
        Tuple of `(teacher_logits, student_logits)`.

    Raises:
        KeyError: If `y_pred` is a dict missing `"student_logits"` or
            `"teacher_logits"`.
        ValueError: If `y_pred` is a plain tensor and `y_true` is None, so no
            teacher logits are available from either argument; or if a
            statically-known last dimension of `y_pred` or of `y_true` is
            neither `out_dim` nor `2 * out_dim`.
    """
    if isinstance(y_pred, dict):
        missing = {'student_logits', 'teacher_logits'} - set(y_pred)
        if missing:
            raise KeyError(
                f"{loss_name} received a dict y_pred missing required "
                f"key(s) {sorted(missing)}; got keys {sorted(y_pred)}"
            )
        return y_pred['teacher_logits'], y_pred['student_logits']

    def _static_last_dim(t: Any) -> Optional[int]:
        """Last dimension of `t` if it is statically known, else None."""
        shape = getattr(t, 'shape', None)
        if shape is None or len(shape) == 0:
            return None
        value = shape[-1]
        return value if isinstance(value, int) else None

    last_dim = _static_last_dim(y_pred)

    if last_dim == 2 * out_dim:
        # Packed convention -- student first, teacher second.
        return y_pred[..., out_dim:], y_pred[..., :out_dim]

    if y_true is None:
        raise ValueError(
            f"{loss_name} called with a plain-tensor y_pred requires y_true to "
            f"be the teacher's logits, but y_true is None. Either pass the "
            f"teacher logits as y_true, pass a PACKED y_pred whose last "
            f"dimension is 2 * out_dim = {2 * out_dim} (see "
            f"pack_student_teacher), or pass a dict y_pred with keys "
            f"'student_logits' and 'teacher_logits'. Got a y_pred with last "
            f"dimension {last_dim}."
        )

    # DECISION plan-2026-08-01T105809-dc0c402e/D-038
    # The width check `pack_student_teacher`'s contract promises. Do NOT delete
    # this and go back to falling through to convention 3: under `fit()`
    # `y_true` is the passed-through LABEL, so a packed tensor that misses
    # `2 * out_dim` is scored against the label instead of against the teacher.
    # MEASURED before the check existed, on a correct `(6, 16)` packed tensor
    # plus a `(6, 1)` label: `out_dim=9` -> `InvalidArgumentError: required
    # broadcastable shapes [Op:Mul]` naming neither width; `out_dim=16` -> NO
    # ERROR AT ALL, value 16.5552 computed from the label. Do NOT weaken this
    # to a `y_pred`-only check either -- `out_dim=16` passes that one.
    # See decisions.md D-038.
    true_last_dim = _static_last_dim(y_true)
    for name, observed in (('y_pred', last_dim), ('y_true', true_last_dim)):
        if observed is None or observed == out_dim:
            continue
        raise ValueError(
            f"{loss_name} received a {name} whose last dimension is "
            f"{observed}, which is neither out_dim = {out_dim} (the two-tensor "
            f"convention, y_true = teacher logits and y_pred = student logits) "
            f"nor 2 * out_dim = {2 * out_dim} (the PACKED convention, see "
            f"pack_student_teacher). Under fit() y_true is the label, so a "
            f"packed y_pred that misses 2 * out_dim = {2 * out_dim} would "
            f"otherwise be scored against the label. Check that this loss's "
            f"out_dim = {out_dim} matches the model head's width."
        )

    return y_true, y_pred

# ---------------------------------------------------------------------
# Schedulable teacher temperature.
#
# DECISION plan-2026-08-01T105809-dc0c402e/D-022
# `teacher_temp` is a non-trainable `keras.Variable`, NOT a Python float, and
# the public attribute is a read-only property. Do NOT "simplify" it back to
# `self.teacher_temp = teacher_temp`: a Python float is CONSTANT-FOLDED into
# the traced training graph, so DINO's teacher-temperature warmup -- which the
# reference implementation ramps during training -- becomes a SILENT no-op.
# MEASURED under a real 2-epoch `fit()` with the centering EMA frozen
# (`center_momentum=0.999999`) so the temperature is the only thing that could
# move the loss: setting `loss.teacher_temp = 4.0` (a 100x change) between
# epochs moved the reported loss 9.953619 -> 9.953612, i.e. 7e-7 relative --
# nothing. With this Variable, `set_teacher_temp(4.0)` moved it
# 16.215630 -> 12.618570. The setter raises instead of assigning, so the
# silent path is now a loud one.
#
# These three module-level objects are installed on BOTH `DINOLoss` and
# `iBOTPatchLoss` (one definition, two bindings) rather than copy-pasted, and
# deliberately not a mixin -- a `keras.losses.Loss` subclass with an extra base
# class complicates `from_config` for no gain here.
# ---------------------------------------------------------------------


def _create_teacher_temp_variable(value: float) -> keras.Variable:
    """
    Create the non-trainable scalar `keras.Variable` backing `teacher_temp`.

    It is **float64**, not float32, and deliberately so: the value round-trips
    through `get_config()` as a Python float, and a float32 store silently
    changes it (MEASURED: `teacher_temp=0.05` came back as
    `0.05000000074505806`, failing an exact-equality config round-trip
    assertion that had been green since step 2). It is cast to the logits'
    dtype at the single point of use, so the storage dtype costs nothing on
    the forward path.

    The initializer is a raw `np.float64` scalar and NOT
    `keras.initializers.Constant(value)`: MEASURED, `Constant` materializes at
    float32 first and then upcasts, so it reproduces the very rounding this
    float64 store exists to avoid (`Constant(0.05)` in a float64 Variable
    reads back as 0.05000000074505806).
    """
    return keras.Variable(
        initializer=np.float64(value),
        dtype='float64',
        trainable=False,
        name='teacher_temp',
    )


def _get_teacher_temp(self) -> float:
    """Current teacher temperature as a plain Python float."""
    return float(ops.convert_to_numpy(self._teacher_temp))


def _refuse_teacher_temp_assignment(self, value: Any) -> None:
    raise AttributeError(
        f"{type(self).__name__}.teacher_temp is read-only; use "
        f"set_teacher_temp({value!r}) instead. A plain attribute assignment "
        f"would be silently ignored inside a compiled training step, because "
        f"the temperature is read from a keras.Variable that the traced graph "
        f"already holds (MEASURED: a 100x plain-attribute change moved the "
        f"training loss by 7e-7)."
    )


_TEACHER_TEMP_PROPERTY = property(
    _get_teacher_temp,
    _refuse_teacher_temp_assignment,
    doc="Teacher sharpening temperature (read-only; see set_teacher_temp).",
)


def _set_teacher_temp(self, value: float) -> None:
    """
    Set the teacher temperature so a running training step sees the new value.

    Interface contract:
        Parameters:
            value: The new temperature. Must be > 0.
        Returns:
            ``None``. The change takes effect on the next batch, inside an
            already-traced `fit()` graph.
        Failure mode:
            ``ValueError`` when ``value <= 0`` -- the same bound the
            constructor enforces.

    This is the seam a teacher-temperature warmup schedule drives. Reuse
    `dl_techniques.models.depth_anything.teacher_ema.linear_ema_schedule` (or
    `cosine_ema_schedule`) with a `keras.callbacks.LambdaCallback` rather than
    adding another schedule-callback class -- see
    `src/dl_techniques/models/dino/README.md` § "Training".
    """
    if value <= 0:
        raise ValueError(f"teacher_temp must be positive, got {value}")
    self._teacher_temp.assign(float(value))

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DINOLoss(keras.losses.Loss):
    """
    DINO consistency loss for self-supervised learning with momentum-based center.

    This loss enforces consistency between student and teacher networks' CLS token
    outputs by matching probability distributions. It uses a momentum-updated center
    vector to prevent feature collapse and applies different temperature scaling
    to student and teacher outputs for effective knowledge distillation.

    **Intent**: Implement the core DINO loss mechanism that enables effective
    self-supervised learning by enforcing cross-view consistency while preventing
    trivial solutions through statistical centering.

    **Mathematical Operations**:
    1. **Teacher Processing**: logits_t = teacher_logits - center
    2. **Temperature Scaling**: p_t = softmax(logits_t / τ_teacher)
    3. **Student Processing**: p_s = log_softmax(student_logits / τ_student)
    4. **Cross-Entropy**: L = -Σ p_t * p_s
    5. **Center Update**: center ← α * center + (1-α) * mean(teacher_logits)

    **Architecture**:
    ```
    Teacher Logits → Center → Temp Scale → Softmax → Target Dist
                                                          ↓
    Student Logits → Temp Scale → LogSoftmax → CrossEntropy Loss
    ```

    Args:
        out_dim: Dimensionality of the model's output embeddings/logits.
        student_temp: Temperature for sharpening student's output distribution.
            Lower values create sharper distributions. Defaults to 0.1. This
            one is a plain float -- DINO does not schedule it.
        teacher_temp: INITIAL temperature for sharpening teacher's output
            distribution. Should be lower than student_temp. Defaults to 0.04.
            Stored as a `keras.Variable` so a warmup schedule can move it;
            read it back via the `teacher_temp` property and change it via
            `set_teacher_temp()`.
        center_momentum: Momentum coefficient for EMA center updates.
            Higher values create more stable centers. Defaults to 0.9.

    Input shapes:
        Three calling conventions are supported.

        0. **Packed single tensor -- the ONE that works under stock `fit()`
           when the teacher's logits come from the model.** `y_pred` has last
           dimension `2 * out_dim`: `[student_logits, teacher_logits]`
           concatenated on the last axis (build it with
           `pack_student_teacher`), and `y_true` is IGNORED. Every leading
           dimension is treated as a batch dimension, so the model must have
           already flattened its (view, pair) structure into rows -- see
           `DINOTrainingModel` in
           `src/dl_techniques/models/dino/dino_training.py`, which emits
           `(batch * n_pairs, 2 * out_dim)`. **A rank > 2 packed `y_pred` is
           NOT supported**: the centering EMA reduces only `axis=0`, so a
           `(batch, n_pairs, 2 * out_dim)` input makes the batch centre
           `(1, n_pairs, out_dim)` and the `center.assign()` fails.
        1. **Direct two-tensor call** -- `y_true` is the teacher's logits
           `(batch_size, out_dim)` and `y_pred` the student's
           `(batch_size, out_dim)`. Usable under `compile(loss=...)` only when
           the DATASET supplies the teacher's logits as the label, which DINO
           cannot do (the teacher is part of the model).
        2. **Structured `y_pred`** -- `y_pred` is a dict carrying both
           networks' outputs and `y_true` is IGNORED (as in
           `CLIPContrastiveLoss` and `KoLeoLoss`)::

               y_pred = {
                   "student_logits": (batch_size, out_dim),
                   "teacher_logits": (batch_size, out_dim),
               }

           **This convention is for direct invocation, NOT for a model whose
           `call()` returns that dict.** MEASURED on keras 3.8.0: when the
           model's output is nested, `CompileLoss.build` broadcasts a single
           `Loss` object across every leaf
           (`loss = tree.map_structure(lambda x: loss, y_pred)`), so each copy
           receives one leaf instead of the whole structure and the build then
           fails with `KeyError: The path: ('student_logits',) in the 'loss'
           argument, can't be found in either the model's output (y_pred) or in
           the labels (y_true)`. A model that wants one loss to see several
           tensors must return them as a SINGLE tensor.

        Note that `y_true` may not be `None` when going through `__call__`:
        Keras converts it to a tensor before dispatching to `call()`. Pass a
        dummy tensor (the repo's `get_dummy_labels` shape) when it is ignored.

    Output shape:
        Scalar loss tensor.

    Attributes:
        center: Non-trainable momentum-updated `keras.Variable` of shape
            `(1, out_dim)`, dtype float32, created eagerly in `__init__`.
        teacher_temp: **Read-only property** returning the current teacher
            temperature as a Python float. It is backed by a non-trainable
            scalar `keras.Variable`, so a warmup schedule can move it during
            training; assign through `set_teacher_temp(value)`. A plain
            `loss.teacher_temp = x` raises `AttributeError` on purpose --
            MEASURED, a Python-float temperature is constant-folded into the
            traced training step, so such an assignment is a SILENT no-op
            (a 100x change moved the training loss by 7e-7). See D-022.

    Example:
        ```python
        # Initialize for vision_heads transformer with 65k dimensional output
        dino_loss = DINOLoss(out_dim=65536, student_temp=0.1, teacher_temp=0.04)

        # Under stock fit(), via a model whose call() returns the PACKED
        # single tensor (a dict y_pred does NOT work here -- see "Input
        # shapes" convention 0 and D-009):
        model.compile(optimizer=..., loss=dino_loss)
        model.fit(train_ds, epochs=100)          # NOTE: no validation_data

        # Or called directly (teacher first, student second):
        loss = dino_loss(teacher_cls, student_cls)
        ```

    Note:
        **The centering EMA is applied inside `call()`** -- there is no public
        `update_center()` method and none is needed. The repo forbids a custom
        Keras `train_step`, so the center is maintained the same way a
        BatchNormalization moving average is: a non-trainable variable
        `.assign()`-ed from inside the forward path. MEASURED under keras 3.8.0
        / tensorflow 2.18.0: the center reaches the hand-computed EMA value to
        1.6e-08, bit-identically across `jit_compile` auto/False/True.

    Note:
        **SSL pretraining with this loss MUST run without `validation_data`.**
        This is a hard requirement, not hygiene. The centering EMA fires on
        EVERY invocation of `call()`, and Keras runs the compiled loss on
        validation batches too. MEASURED: `validation_batch_size` defaults to
        `batch_size`, so a validation set covering the same number of samples
        as one training epoch DOUBLES the number of centering updates per
        epoch. In the step-1 probe this pushed the center 81% past its correct
        training-only value -- silently, with a finite loss and a clean exit.
        The corruption scales with the validation batch COUNT. Use a separate
        evaluation callback (e.g. k-NN on frozen features) instead.

    Note:
        **The center's value does NOT survive a `.keras` model checkpoint.**
        Keras does not checkpoint loss-owned variables. It is therefore
        serialized explicitly through `get_config()` / `from_config()` (see
        those methods). The cost is a config blob proportional to `out_dim`
        (MEASURED as JSON: ~83 KB at `out_dim=4096`, ~1.3 MiB at 65536).

    Note:
        **Single-device only.** No cross-replica reduction of the batch centre
        is performed. keras 3.8.0's `Distribution` API exposes neither
        `num_replicas_in_sync` nor `reduce` (MEASURED), so a data-parallel run
        would maintain a per-replica center. Untested under any distribution
        strategy.
    """

    # Installed from the module-level definitions above (D-022): a read-only
    # property over a keras.Variable, plus the setter that actually works
    # inside a compiled step.
    teacher_temp = _TEACHER_TEMP_PROPERTY
    set_teacher_temp = _set_teacher_temp

    def __init__(
            self,
            out_dim: int,
            student_temp: float = 0.1,
            teacher_temp: float = 0.04,
            center_momentum: float = 0.9,
            name: str = 'dino_loss',
            **kwargs: Any
    ) -> None:
        """
        Initialize DINO loss with specified parameters.

        Args:
            out_dim: Output dimensionality for center vector initialization.
            student_temp: Temperature for student distribution sharpening.
            teacher_temp: Temperature for teacher distribution sharpening.
            center_momentum: EMA momentum for center updates.
            name: Name for this loss instance.
            **kwargs: Additional arguments for Loss base class.

        Raises:
            ValueError: If out_dim <= 0, temperatures <= 0, or center_momentum
                       not in [0, 1).
        """
        super().__init__(name=name, **kwargs)

        # Validate input parameters
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        if student_temp <= 0:
            raise ValueError(f"student_temp must be positive, got {student_temp}")
        if teacher_temp <= 0:
            raise ValueError(f"teacher_temp must be positive, got {teacher_temp}")
        if not (0 <= center_momentum < 1):
            raise ValueError(f"center_momentum must be in [0, 1), got {center_momentum}")

        # Store configuration
        self.out_dim = out_dim
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        # Schedulable teacher temperature -- see D-022 above the property.
        self._teacher_temp = _create_teacher_temp_variable(teacher_temp)

        # Create momentum-updated center as non-trainable weight
        self.center = keras.Variable(
            initializer=keras.initializers.Zeros(),
            shape=(1, out_dim),
            dtype='float32',
            trainable=False,
            name='center',
        )

    def call(
            self,
            y_true: Optional[keras.KerasTensor],
            y_pred: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
    ) -> keras.KerasTensor:
        """
        Compute the DINO loss and advance the centering EMA.

        Args:
            y_true: Teacher's output logits with shape (batch_size, out_dim)
                when `y_pred` is a plain student-width tensor; IGNORED when
                `y_pred` is PACKED or a structured dict (see the class
                docstring's "Input shapes").
            y_pred: The PACKED tensor `(batch_size, 2 * out_dim)`, or the
                student's logits `(batch_size, out_dim)`, or a dict with keys
                `"student_logits"` and `"teacher_logits"`.

        Returns:
            Scalar tensor representing the computed DINO loss.

        Note:
            This method ALSO advances the centering EMA -- see the class
            docstring. The returned loss uses the center as it stood BEFORE
            this call's update, matching the reference DINO ordering.
        """
        teacher_logits, student_logits = _resolve_student_teacher(
            'DINOLoss', y_true, y_pred, self.out_dim)

        # DECISION plan-2026-08-01T105809-dc0c402e/D-006
        # The centering EMA lives HERE, inside call(), maintained by .assign()
        # on a non-trainable keras.Variable -- the BatchNormalization
        # moving-average pattern. Do NOT move it back out into a public
        # update_center() called from a train_step(): this repo forbids a
        # custom Keras train_step, so such a method has no caller and the
        # center would silently stay at zero for an entire pretraining run
        # (a finite, decreasing loss and a collapsed representation). The
        # branch was chosen on a measurement, not a prediction -- see D-006.
        #
        # Read the center BEFORE assigning so the loss provably uses the
        # pre-update value regardless of how the graph orders the stateful op.
        center = ops.convert_to_tensor(self.center)

        # Process teacher output: center and sharpen.
        # Read the temperature from the VARIABLE (`self._teacher_temp`), never
        # through the `self.teacher_temp` property -- the property returns a
        # Python float, which a traced step constant-folds (D-022).
        teacher_probs = ops.softmax(
            (teacher_logits - ops.cast(center, teacher_logits.dtype))
            / ops.cast(self._teacher_temp, teacher_logits.dtype),
            axis=-1,
        )

        # Process student output: sharpen to log probabilities
        student_log_probs = ops.log_softmax(
            student_logits / self.student_temp, axis=-1)

        # Compute cross-entropy loss: -sum(p_teacher * log_p_student)
        loss = -ops.sum(teacher_probs * student_log_probs, axis=-1)

        # EMA update: center <- a * center + (1-a) * mean(teacher_logits)
        batch_center = ops.mean(teacher_logits, axis=0, keepdims=True)
        self.center.assign(
            center * self.center_momentum
            + ops.cast(batch_center, center.dtype) * (1.0 - self.center_momentum)
        )

        return ops.mean(loss)

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization, INCLUDING the center's value.

        Returns:
            Config dict carrying every constructor argument plus a `center`
            entry holding the centering vector as a nested list.
        """
        config = super().get_config()
        config.update({
            'out_dim': self.out_dim,
            'student_temp': self.student_temp,
            'teacher_temp': self.teacher_temp,
            'center_momentum': self.center_momentum,
            # DECISION plan-2026-08-01T105809-dc0c402e/D-007
            # The center's VALUE is carried in the config on purpose. Keras
            # does NOT checkpoint loss-owned variables (MEASURED: a .keras
            # round-trip returns the right Loss subclass with a live .center
            # that reads back as zeros), so without this the centering
            # statistic silently resets on every resume. Do NOT "slim down"
            # get_config() by dropping this key -- that reintroduces a silent
            # state loss no test that checks only hyperparameters can see.
            'center': ops.convert_to_numpy(self.center).tolist(),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'DINOLoss':
        """
        Reconstruct the loss, restoring the center's value when present.

        Args:
            config: A config dict as produced by `get_config()`. A missing
                `center` key is tolerated (the center starts at zeros).

        Returns:
            The reconstructed `DINOLoss`.
        """
        config = dict(config)
        center = config.pop('center', None)
        instance = cls(**config)
        if center is not None:
            instance.center.assign(
                ops.reshape(
                    ops.convert_to_tensor(center, dtype='float32'),
                    instance.center.shape,
                )
            )
        return instance

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class iBOTPatchLoss(keras.losses.Loss):
    """
    iBOT masked patch prediction loss for self-supervised learning.

    This loss extends DINO's consistency objective to patch-level predictions
    using masked image modeling. It matches student predictions for masked
    patches against teacher outputs for the same patches (unmasked), enabling
    learning of local image features through reconstruction tasks.

    **Intent**: Enable patch-level self-supervised learning by predicting
    masked patch representations, combining global consistency (DINO) with
    local reconstruction objectives for comprehensive visual understanding.

    **Mathematical Operations**:
    1. **Mask Selection**: Select only logits for masked patches
    2. **Teacher Processing**: p_t = softmax((logits_t - center) / τ_teacher)
    3. **Student Processing**: p_s = log_softmax(logits_s / τ_student)
    4. **Cross-Entropy**: L = -Σ p_t * p_s (only for masked patches)
    5. **Center Update**: center ← α * center + (1-α) * mean(all_teacher_patches)

    **Architecture**:
    ```
    Teacher Patches → Center → Temp Scale → Softmax → Target Dist
                                                          ↓
    Student Patches → Mask Select → Temp Scale → LogSoftmax → Loss
           ↑
    Boolean Mask (True for masked patches)
    ```

    Args:
        out_dim: Dimensionality of patch token embeddings/logits.
        student_temp: Temperature for student distribution sharpening.
        teacher_temp: INITIAL temperature for teacher distribution sharpening.
            Backed by a `keras.Variable`; see `DINOLoss` and
            `set_teacher_temp()`.
        center_momentum: EMA momentum for center updates.

    Input shapes:
        The same three calling conventions as `DINOLoss`, and the same
        MEASURED constraint: the structured form is for direct invocation, not
        for a model whose `call()` returns that dict (see `DINOLoss`'s "Input
        shapes" for the exact Keras behaviour and error).

        0. **Packed single tensor** -- last dimension `2 * out_dim`,
           `[student, teacher]`, `y_true` IGNORED. This is the stock-`fit()`
           convention, but note it carries NO mask, so every patch
           participates; a masked iBOT objective under stock `fit()` needs the
           mask packed into the same tensor by a future caller, which is not
           built here (no consumer yet -- `DINOTrainingModel` is DINO-only).
        1. **Structured `y_pred` (required to use a mask at all)** -- `y_true`
           is IGNORED and `y_pred` is a dict::

               y_pred = {
                   "student_logits": (batch_size, num_patches, out_dim),
                   "teacher_logits": (batch_size, num_patches, out_dim),
                   "mask":           (batch_size, num_patches),  # optional, bool
               }

           An omitted `"mask"` means every patch participates.
        2. **Direct two-tensor call** -- `y_true` is the teacher's patch logits
           and `y_pred` the student's; every patch participates.

        The mask CANNOT be passed as a third positional argument. Keras 3's
        `Loss.__call__` signature is `(y_true, y_pred, sample_weight=None)` and
        it dispatches to a TWO-argument `self.call`, so a third positional
        argument is silently swallowed as `sample_weight` and never reaches
        the masking logic. `__call__` is overridden here to raise on a
        non-None `sample_weight` rather than let that happen quietly.

    Output shape:
        Scalar loss tensor.

    Attributes:
        center: Non-trainable `keras.Variable` of shape `(1, 1, out_dim)`,
            dtype float32, created eagerly in `__init__`.

    Example:
        ```python
        # Initialize for vision_heads transformer patches
        ibot_loss = iBOTPatchLoss(out_dim=65536, student_temp=0.1)

        loss = ibot_loss(
            None,
            {
                "teacher_logits": teacher_patches,  # (B, 196, 65536)
                "student_logits": student_patches,  # (B, 196, 65536)
                "mask": mask,                       # (B, 196), bool
            },
        )
        ```

    Note:
        Every note on `DINOLoss` applies verbatim to this class as well: the
        centering EMA is applied inside `call()` with no public
        `update_center()`; **pretraining must run without `validation_data`**;
        the center's value is carried through `get_config()` because Keras does
        not checkpoint loss-owned variables; and no cross-replica reduction is
        performed (single-device only, untested under any distribution
        strategy).
    """

    # Installed from the module-level definitions above (D-022) -- identical
    # contract to DINOLoss's.
    teacher_temp = _TEACHER_TEMP_PROPERTY
    set_teacher_temp = _set_teacher_temp

    def __init__(
            self,
            out_dim: int,
            student_temp: float = 0.1,
            teacher_temp: float = 0.04,
            center_momentum: float = 0.9,
            name: str = 'ibot_loss',
            **kwargs: Any
    ) -> None:
        """
        Initialize iBOT patch loss with specified parameters.

        Args:
            out_dim: Patch token embedding dimensionality.
            student_temp: Temperature for student sharpening.
            teacher_temp: Temperature for teacher sharpening.
            center_momentum: EMA momentum for center updates.
            name: Name for this loss instance.
            **kwargs: Additional arguments for Loss base class.

        Raises:
            ValueError: If parameters are outside valid ranges.
        """
        super().__init__(name=name, **kwargs)

        # Validate parameters
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        if student_temp <= 0:
            raise ValueError(f"student_temp must be positive, got {student_temp}")
        if teacher_temp <= 0:
            raise ValueError(f"teacher_temp must be positive, got {teacher_temp}")
        if not (0 <= center_momentum < 1):
            raise ValueError(f"center_momentum must be in [0, 1), got {center_momentum}")

        # Store configuration
        self.out_dim = out_dim
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        # Schedulable teacher temperature -- see D-022 above the property.
        self._teacher_temp = _create_teacher_temp_variable(teacher_temp)

        # Create center for patch tokens (shape for broadcasting over patches)
        self.center = keras.Variable(
            initializer=keras.initializers.Zeros(),
            shape=(1, 1, out_dim),
            dtype='float32',
            trainable=False,
            name='center',
        )

    def __call__(
            self,
            y_true: Any,
            y_pred: Any,
            sample_weight: Any = None
    ) -> keras.KerasTensor:
        """
        Invoke the loss, refusing a `sample_weight` outright.

        The pre-fix API asked callers to write `loss(teacher, student, mask)`.
        Under Keras 3 that third argument lands in `sample_weight` and is
        applied as a weighting of the already-reduced scalar loss -- the mask
        never reaches the masking logic, and nothing complains. Refusing
        `sample_weight` turns that silent-wrong path into a loud one.

        Args:
            y_true: Teacher patch logits, or ignored when `y_pred` is packed
                or a dict.
            y_pred: The PACKED tensor (last dim `2 * out_dim`), the student's
                patch logits, or the structured dict.
            sample_weight: Must be None.

        Returns:
            The scalar loss.

        Raises:
            TypeError: If `sample_weight` is not None.
        """
        if sample_weight is not None:
            raise TypeError(
                "iBOTPatchLoss does not accept a sample_weight. If you meant "
                "to supply the patch mask, pass it inside the structured "
                "y_pred dict as y_pred['mask'] -- a third positional argument "
                "is swallowed by Keras as sample_weight and never reaches the "
                "masking logic."
            )
        return super().__call__(y_true, y_pred)

    def call(
            self,
            y_true: Optional[keras.KerasTensor],
            y_pred: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
    ) -> keras.KerasTensor:
        """
        Compute the iBOT loss over the masked patches and advance the EMA.

        Args:
            y_true: Teacher patch logits `(B, num_patches, out_dim)` when
                `y_pred` is a plain tensor; IGNORED when `y_pred` is a dict.
            y_pred: Student patch logits, or a dict with `"student_logits"`,
                `"teacher_logits"` and an optional boolean `"mask"` of shape
                `(B, num_patches)` (True = the patch participates).

        Returns:
            Scalar loss tensor, the mask-weighted mean over participating
            patches. Exactly `0.0` when no patch participates.
        """
        teacher_logits, student_logits = _resolve_student_teacher(
            'iBOTPatchLoss', y_true, y_pred, self.out_dim)
        mask = y_pred.get('mask') if isinstance(y_pred, dict) else None

        center = ops.convert_to_tensor(self.center)

        # Process teacher output: center and sharpen.
        # Read the temperature from the VARIABLE (`self._teacher_temp`), never
        # through the `self.teacher_temp` property -- the property returns a
        # Python float, which a traced step constant-folds (D-022).
        teacher_probs = ops.softmax(
            (teacher_logits - ops.cast(center, teacher_logits.dtype))
            / ops.cast(self._teacher_temp, teacher_logits.dtype),
            axis=-1,
        )

        # Process student output: sharpen to log probabilities
        student_log_probs = ops.log_softmax(
            student_logits / self.student_temp, axis=-1)

        # Per-patch cross-entropy, shape (B, num_patches)
        per_patch_loss = -ops.sum(teacher_probs * student_log_probs, axis=-1)

        # DECISION plan-2026-08-01T105809-dc0c402e/D-008
        # Mask-weighted mean, with NO branch. Do NOT restore the previous
        # `ops.boolean_mask(...)` + `if ops.shape(...)[0] == 0:` form: MEASURED
        # on keras 3.8.0, `keras.ops.boolean_mask` DOES NOT EXIST
        # (AttributeError in eager AND under tf.function), so the whole body
        # was unreachable and the zero-masked branch behind it was never even
        # evaluated. A gather-based rewrite would reintroduce the original
        # defect: a data-dependent shape that a Python `if` cannot read at
        # trace time. This form is correct at zero participating patches by
        # construction -- 0 / eps == 0.0 -- with no dynamic shape anywhere.
        # (`mask is None` is a Python branch on a STATIC value -- the presence
        # of a dict key -- not on a tensor, so it is trace-time safe.)
        if mask is None:
            loss = ops.mean(per_patch_loss)
        else:
            mask_weights = ops.cast(mask, per_patch_loss.dtype)
            num_participating = ops.sum(mask_weights)
            loss = ops.sum(per_patch_loss * mask_weights) / ops.maximum(
                num_participating, ops.cast(1e-8, per_patch_loss.dtype))

        # EMA update over ALL teacher patches (masked and unmasked alike),
        # matching the reference iBOT centering statistic.
        batch_center = ops.mean(teacher_logits, axis=[0, 1], keepdims=True)
        self.center.assign(
            center * self.center_momentum
            + ops.cast(batch_center, center.dtype) * (1.0 - self.center_momentum)
        )

        return loss

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization, INCLUDING the center's value.

        Returns:
            Config dict carrying every constructor argument plus a `center`
            entry holding the centering vector as a nested list.
        """
        config = super().get_config()
        config.update({
            'out_dim': self.out_dim,
            'student_temp': self.student_temp,
            'teacher_temp': self.teacher_temp,
            'center_momentum': self.center_momentum,
            # DECISION plan-2026-08-01T105809-dc0c402e/D-007
            # See DINOLoss.get_config -- the center's value is carried here
            # because Keras does not checkpoint loss-owned variables.
            'center': ops.convert_to_numpy(self.center).tolist(),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'iBOTPatchLoss':
        """
        Reconstruct the loss, restoring the center's value when present.

        Args:
            config: A config dict as produced by `get_config()`. A missing
                `center` key is tolerated (the center starts at zeros).

        Returns:
            The reconstructed `iBOTPatchLoss`.
        """
        config = dict(config)
        center = config.pop('center', None)
        instance = cls(**config)
        if center is not None:
            instance.center.assign(
                ops.reshape(
                    ops.convert_to_tensor(center, dtype='float32'),
                    instance.center.shape,
                )
            )
        return instance

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class KoLeoLoss(keras.losses.Loss):
    """
    Kozachenko-Leonenko entropic regularizer for uniform distribution on unit sphere.

    This regularization loss prevents feature collapse by encouraging embeddings
    to be uniformly distributed on the unit hypersphere. It maximizes the distance
    to nearest neighbors, promoting diverse feature representations and preventing
    the network from converging to trivial solutions.

    **Intent**: Provide unsupervised regularization that maintains embedding
    diversity without requiring labels or additional supervision, essential
    for self-supervised learning to avoid representational collapse.

    **Mathematical Operations**:
    1. **Normalize**: x̂ = x / ||x||₂ (project to unit sphere)
    2. **Similarity**: S = x̂ᵀx̂ (cosine similarity matrix)
    3. **Nearest Neighbor**: s_nn = max(S - I) (largest off-diagonal)
    4. **Distance**: d = √(2 - 2s_nn) (L2 distance from similarity)
    5. **Loss**: L = -log(d + ε) (maximize log distance)

    **Architecture**:
    ```
    Input Embeddings → L2 Normalize → Unit Sphere
                                         ↓
    Similarity Matrix ← Cosine Similarity Computation
                                         ↓
    Nearest Neighbors ← Max Off-diagonal Selection
                                         ↓
    L2 Distances ← Distance Conversion
                                         ↓
    -Log(Distance) ← Final Loss Computation
    ```

    Args:
        epsilon: Small value for numerical stability in log computation.
                Higher values provide more stability but may affect gradients.

    Input shapes:
        y_true: Ignored (unsupervised loss).
        y_pred: Embeddings with shape `(batch_size, embedding_dim)`.

    Output shape:
        Scalar loss tensor.

    Example:
        ```python
        # Initialize with default stability
        koleo_loss = KoLeoLoss(epsilon=1e-8)

        # Apply to CLS token embeddings
        student_cls = student_model.cls_token(inputs)  # Shape: (batch, 768)

        # y_true is ignored for this unsupervised loss
        reg_loss = koleo_loss(None, student_cls)

        # Combine with main loss
        total_loss = main_loss + 0.1 * reg_loss
        ```

    Note:
        This is an unsupervised regularizer that ignores y_true. It can be
        applied to any embedding layer to encourage diversity. The loss
        magnitude depends on embedding dimensionality and batch size.
    """

    def __init__(
            self,
            epsilon: float = 1e-8,
            name: str = 'koleo_loss',
            **kwargs: Any
    ) -> None:
        """
        Initialize KoLeo loss with numerical stability parameter.

        Args:
            epsilon: Small value added to distances before log for stability.
            name: Name for this loss instance.
            **kwargs: Additional arguments for Loss base class.

        Raises:
            ValueError: If epsilon <= 0.
        """
        super().__init__(name=name, **kwargs)

        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.epsilon = epsilon

    def call(
            self,
            y_true: Optional[keras.KerasTensor],
            y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute KoLeo regularization loss.

        Args:
            y_true: Ignored (this is an unsupervised loss).
            y_pred: Embeddings to regularize with shape (batch_size, dim).

        Returns:
            Scalar regularization loss encouraging uniform distribution.
        """
        # L2 normalize embeddings to unit sphere.
        #
        # DECISION plan-2026-08-01T105809-dc0c402e/D-023
        # The normalization runs at float32 whenever the input is a 16-bit
        # float. Do NOT "simplify" this back to a bare
        # `ops.normalize(y_pred, axis=-1)`: that reduces `sum(x**2)` over the
        # embedding width in the INPUT dtype, and at fp16 the sum overflows
        # 65504 long before any individual activation does. Overflow gives
        # `x / inf == 0`, so every embedding becomes the zero vector, every
        # pairwise cosine similarity becomes 0, and the loss collapses to the
        # CONSTANT `-log(sqrt(2) + eps) = -0.34657` with an EXACTLY ZERO
        # gradient -- a silently dead anti-collapse regularizer, precisely
        # when collapse is what it exists to detect.
        # MEASURED at width 256, activation scale 300 (`sum(x**2)` = 2.25e7),
        # `KoLeoLoss(dtype='float16')`: a SPREAD batch and a fully COLLAPSED
        # batch both return -0.346436, where float32 returns -0.279918 and
        # +15.257816 respectively; fp16 gradient absmax 0.0 vs float32
        # 3.42e-06.
        # Reachability, also MEASURED: this does NOT fire under a global
        # `mixed_float16` policy through `Loss.__call__`, because
        # `keras.losses.Loss` casts `y_pred` to `self.dtype`, which stays
        # float32 (`floatx()`) under that policy. It DOES fire via
        # `KoLeoLoss(dtype='float16')`, `KoLeoLoss(dtype='mixed_float16')`,
        # and any direct `loss.call(...)` on fp16 activations -- e.g. from
        # inside a model's own `call()`. float64 is left alone (never
        # downcast); this is the same defect class as D-020 in `DINOHead`.
        features = ops.convert_to_tensor(y_pred)
        input_dtype = keras.backend.standardize_dtype(features.dtype)
        safe_dtype = (
            'float32' if input_dtype in ('float16', 'bfloat16') else input_dtype
        )
        features = ops.normalize(ops.cast(features, safe_dtype), axis=-1)
        batch_size = ops.shape(features)[0]

        # Compute pairwise cosine similarity matrix
        similarity_matrix = ops.matmul(features, ops.transpose(features))

        # Mask diagonal to exclude self-similarity
        # Set diagonal to large negative value to ignore in max reduction
        eye = ops.eye(batch_size, dtype=similarity_matrix.dtype)
        masked_similarity = similarity_matrix - 2.0 * eye

        # Find nearest neighbor similarity for each embedding
        nearest_neighbor_sim = ops.max(masked_similarity, axis=1)

        # Convert cosine similarity to L2 distance on unit sphere
        # For unit vectors: ||a - b||² = 2 - 2(a·b)
        # Clamp similarity to valid range to avoid numerical issues
        clamped_sim = ops.clip(nearest_neighbor_sim, -1.0, 1.0)
        distances_squared = 2.0 - 2.0 * clamped_sim
        distances = ops.sqrt(ops.maximum(distances_squared, 0.0))

        # Compute loss: maximize log distance = minimize -log(distance)
        loss = -ops.log(distances + self.epsilon)

        # Return in the caller's dtype: only the INTERMEDIATE normalization was
        # promoted (D-023 above), so the method's dtype contract is unchanged.
        return ops.cast(ops.mean(loss), input_dtype)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
        })
        return config

# ---------------------------------------------------------------------

