"""Guards for `src/train/sam2/data.py` -- the synthetic moving-instance video.

Every guard here is a DATA guard: no `SAM2` is constructed and no forward pass
runs, so nothing in this module can be made vacuously green by the
carried-forward fact that at random init every object score is negative and the
whole mask output collapses to the constant `-1024`. The one place the model is
touched is by importing its exported output-key constants, which is the point
of `test_the_target_keys_are_exactly_the_wrappers_output_keys`.

RED-proof ledger -- EXECUTED, on CPU (this file constructs no model, so the
gate device is irrelevant to every number in it)
--------------------------------------------------------------------------
| # | mutation | what ACTUALLY fired |
|---|---|---|
| G2.1 | freeze the velocity to `(0, 0)` | `test_every_visible_transition_moves_the_mask_centroid` AND `test_the_pixels_move_with_the_mask` -- the latter measured the two frames BIT-IDENTICAL (`abs(image[t] - image[t+1]).max() == 0.0`), because at the shipped sweep speed the bar has already left the canvas |
| G2.2 | `absent[start:start + occlusion_frames + 1]` | `test_the_window_flag_count_equals_the_requested_length[1]` and `[2]`, plus 4 downstream guards. NOTE the measured surprise: `test_every_emitted_clip_has_exactly_that_many_empty_frames` stayed GREEN, because the per-clip retry budget re-rolls the drawn window start until one happens to clip to the right length. The direct assertion on `_occlusion_window` is what makes this guard discriminating; the emitted-count assertion alone would have absorbed the off-by-one |
| G2.3 | draw the presence target from `tf.random.uniform > 0.5` | `test_the_presence_target_equals_mask_nonempty_at_every_frame` and `test_a_never_occluded_clip_is_present_at_every_frame` |
| G2.4 | `frame_zero = mask[-1]` | `test_the_prompt_lands_inside_frame_zeros_mask_and_outside_the_last_frames` and `test_background_points_are_also_drawn_from_frame_zero` |
| G2.5 | delete the `absent[0]` check | `test_an_occlusion_window_reaching_frame_zero_is_refused`, and NOTHING else -- exactly one assertion, which is the point |
| G2.6 | the renderer returns an all-zero canvas | 24 of 35 RED; see `TestDeadComponentProbe` for the measured partition |
"""

import os
from typing import List, Optional

import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.SAM.SAM2.training_model import (
    INPUT_BOXES,
    INPUT_GT_MASKS,
    INPUT_IMAGE,
    INPUT_POINT_COORDS,
    INPUT_POINT_LABELS,
    OUTPUT_KEYS,
    SAM2_LOW_RES_LOGITS,
    SAM2_OBJECT_SCORE_LOGITS,
)

from train.sam.data import RECORD_BOX, RECORD_IMAGE, RECORD_MASK
from train.sam2 import data as sam2_data
from train.sam2.data import (
    MAX_CLIP_ATTEMPTS,
    _bar_geometry,
    _occlusion_window,
    build_sam2_video_dataset,
    synthetic_moving_instance_video,
    to_video_training_record,
)

#: The `tiny` geometry the trainer runs at: `Hiera.MODEL_VARIANTS['tiny']`'s
#: `image_size` 64, and SAM 2's `feature_grid * 4` = 64 / 16 * 4 = 16.
IMAGE_SIZE = 64
MASK_SIZE = 16
NUM_FRAMES = 4
STRIDE = IMAGE_SIZE // MASK_SIZE


def _clips(
        num_clips: int = 12,
        occlusion_frames: int = 1,
        occlusion_start: Optional[int] = None,
        seed: int = 0,
        num_frames: int = NUM_FRAMES,
) -> List[dict]:
    """Render a list of clips at the trainer's geometry."""
    return list(
        synthetic_moving_instance_video(
            num_clips=num_clips,
            num_frames=num_frames,
            image_size=IMAGE_SIZE,
            mask_size=MASK_SIZE,
            occlusion_frames=occlusion_frames,
            occlusion_start=occlusion_start,
            seed=seed,
        )
    )


def _centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    """Mask centroid in mask cells, or ``None`` when the mask is empty."""
    indices = np.argwhere(mask > 0)
    return None if indices.size == 0 else indices.mean(axis=0)


def _cell_of_prompt(coords: np.ndarray) -> tuple:
    """Invert `_sample_from_indicator`'s cell-centre mapping.

    It emits ``(col + 0.5) * scale - 0.5``; `PromptEncoder._embed_points` adds
    its own ``+0.5``. The inverse is therefore exact, not a rounding.
    """
    scale = IMAGE_SIZE / MASK_SIZE
    col = int(round((float(coords[0]) + 0.5) / scale - 0.5))
    row = int(round((float(coords[1]) + 0.5) / scale - 0.5))
    return row, col


# ---------------------------------------------------------------------------
# record shape and domain
# ---------------------------------------------------------------------------
class TestClipRecordShape:
    """The record contract the `tf.data` signature is built from."""

    def test_a_clip_carries_the_advertised_shapes_and_dtypes(self) -> None:
        clip = _clips(num_clips=1)[0]
        assert clip[RECORD_IMAGE].shape == (NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3)
        assert clip[RECORD_MASK].shape == (NUM_FRAMES, MASK_SIZE, MASK_SIZE)
        assert clip[RECORD_BOX].shape == (4,)
        assert clip[RECORD_IMAGE].dtype == np.float32
        assert clip[RECORD_MASK].dtype == np.float32

    def test_the_image_lives_in_the_zero_to_255_domain(self) -> None:
        # `SAM2.preprocess` normalizes in the 0-255 domain; a [0, 1] image
        # would be 255x under-exposed and every shape assertion would accept it.
        images = np.stack([clip[RECORD_IMAGE] for clip in _clips(num_clips=6)])
        assert images.min() >= 0.0
        assert images.max() <= 255.0
        assert images.max() > 128.0

    def test_the_mask_is_binary(self) -> None:
        masks = np.stack([clip[RECORD_MASK] for clip in _clips(num_clips=6)])
        assert set(np.unique(masks).tolist()) <= {0.0, 1.0}

    def test_the_box_is_frame_zeros_tight_box(self) -> None:
        # The box is the FRAME-0 box: it prompts the conditioning frame. A box
        # taken from any other frame would point the prompt encoder at where
        # the object is not.
        for clip in _clips(num_clips=6):
            mask_cells = np.argwhere(clip[RECORD_MASK][0] > 0)
            x1, y1, x2, y2 = clip[RECORD_BOX]
            # The image-pixel box must contain every frame-0 mask cell.
            assert x1 <= mask_cells[:, 1].min() * STRIDE + STRIDE
            assert y1 <= mask_cells[:, 0].min() * STRIDE + STRIDE
            assert x2 >= mask_cells[:, 1].max() * STRIDE
            assert y2 >= mask_cells[:, 0].max() * STRIDE


# ---------------------------------------------------------------------------
# G2.1 -- motion
# ---------------------------------------------------------------------------
class TestMotion:
    """The object really moves, measured on the MASK, not on the pixels.

    A "consecutive frames differ" assertion on the image would pass on a frozen
    object as soon as the distractor bar moves. The centroid of the ground-truth
    mask is the quantity that a frozen instance cannot fake.
    """

    def test_every_visible_transition_moves_the_mask_centroid(self) -> None:
        # The window is pinned to the LAST frame so every clip contributes
        # exactly `T - 1 - occlusion_frames` visible transitions; with a drawn
        # window the count depends on where it landed, and a weakened source
        # could hide behind a smaller sample.
        deltas: List[float] = []
        num_clips = 16
        for clip in _clips(
                num_clips=num_clips, occlusion_frames=1,
                occlusion_start=NUM_FRAMES - 1, seed=5,
        ):
            mask = clip[RECORD_MASK]
            centroids = [_centroid(mask[t]) for t in range(NUM_FRAMES)]
            for t in range(NUM_FRAMES - 1):
                if centroids[t] is None or centroids[t + 1] is None:
                    continue
                deltas.append(
                    float(np.abs(centroids[t] - centroids[t + 1]).sum())
                )
        assert len(deltas) == num_clips * (NUM_FRAMES - 1 - 1)
        # MEASURED floor over seeds 0/5/11/77 at this geometry: 1.0 mask cell,
        # which is the construction guarantee (the velocity is forced to at
        # least one ground-truth cell per frame). A frozen object gives 0.0.
        assert min(deltas) >= 1.0

    def test_the_pixels_move_with_the_mask(self) -> None:
        clip = _clips(num_clips=1, seed=5)[0]
        image = clip[RECORD_IMAGE]
        for t in range(NUM_FRAMES - 1):
            assert float(np.abs(image[t] - image[t + 1]).max()) > 0.0


# ---------------------------------------------------------------------------
# G2.2 -- the occlusion count is EXACT
# ---------------------------------------------------------------------------
class TestOcclusionCountIsExact:
    """The all-zero-frame count is the quantity the loss gating depends on.

    A range would not do: `SAM2GatedMaskLoss` (step 3) zeroes the mask/dice
    terms on exactly the absent rows, and the object-score BCE's positive/
    negative balance IS this count. An off-by-one here is a silent change of
    what the trainer measures.
    """

    @pytest.mark.parametrize("occlusion_frames", [0, 1, 2, 3])
    def test_the_window_flag_count_equals_the_requested_length(
            self, occlusion_frames: int
    ) -> None:
        absent = _occlusion_window(NUM_FRAMES, occlusion_frames, start=1)
        assert absent.shape == (NUM_FRAMES,)
        assert int(absent.sum()) == occlusion_frames

    @pytest.mark.parametrize("occlusion_frames", [0, 1, 2])
    def test_every_emitted_clip_has_exactly_that_many_empty_frames(
            self, occlusion_frames: int
    ) -> None:
        counts = []
        for clip in _clips(num_clips=12, occlusion_frames=occlusion_frames):
            mask = clip[RECORD_MASK]
            counts.append(
                int((mask.reshape(NUM_FRAMES, -1).sum(axis=1) == 0.0).sum())
            )
        assert counts == [occlusion_frames] * 12

    def test_the_empty_frames_are_exactly_the_requested_window(self) -> None:
        for clip in _clips(num_clips=8, occlusion_frames=2, occlusion_start=1):
            empty = clip[RECORD_MASK].reshape(NUM_FRAMES, -1).sum(axis=1) == 0.0
            assert empty.tolist() == [False, True, True, False]

    def test_an_empty_frame_is_genuinely_all_zero(self) -> None:
        # Not "small": exactly zero. A transparent distractor -- one that paints
        # pixels but leaves the ground truth alone -- would leave a nonzero mask
        # here while the image looked occluded.
        for clip in _clips(num_clips=6, occlusion_frames=1, occlusion_start=2):
            assert float(clip[RECORD_MASK][2].sum()) == 0.0
            assert float(clip[RECORD_MASK][0].sum()) > 0.0

    def test_the_bar_geometry_covers_the_window_and_nothing_else(self) -> None:
        # The exactness above rests on this arithmetic, so it is asserted
        # directly rather than only through its rendered consequence.
        for occlusion_frames in (1, 2, 3):
            for patch_width in (4, 11, 20):
                margin, speed = _bar_geometry(occlusion_frames, patch_width)
                start = 1
                for t in range(8):
                    offset = speed * (
                        2 * t - 2 * start - occlusion_frames + 1
                    )
                    inside = start <= t < start + occlusion_frames
                    covers = abs(offset) <= margin
                    disjoint = abs(offset) >= patch_width + margin
                    assert covers is inside
                    # No partial erosion anywhere: a frame is either fully
                    # covered or fully untouched.
                    assert covers or disjoint


# ---------------------------------------------------------------------------
# G2.3 -- presence is derived from the mask
# ---------------------------------------------------------------------------
class TestPresenceIsDerived:
    """`target_obj` is `any(mask > 0)`, computed at the point of use.

    There is no presence field in the record, so there is nothing for a
    disagreeing flag to be written into. This asserts the derivation itself,
    at a clip where the object is genuinely absent -- the only place where a
    flag and a mask CAN disagree.
    """

    def test_the_presence_target_equals_mask_nonempty_at_every_frame(
            self,
    ) -> None:
        for clip in _clips(num_clips=8, occlusion_frames=1, occlusion_start=2):
            record = {k: tf.convert_to_tensor(v) for k, v in clip.items()}
            _, targets = to_video_training_record(record, IMAGE_SIZE)
            presence = np.asarray(targets[SAM2_OBJECT_SCORE_LOGITS])
            derived = (
                clip[RECORD_MASK].reshape(NUM_FRAMES, -1).max(axis=1) > 0.0
            ).astype("float32")[:, None]
            assert presence.shape == (NUM_FRAMES, 1)
            np.testing.assert_array_equal(presence, derived)
            # The absent frame is exactly 0.0, and it is the ONLY one.
            assert presence[2, 0] == 0.0
            assert presence.sum() == float(NUM_FRAMES - 1)

    def test_a_never_occluded_clip_is_present_at_every_frame(self) -> None:
        clip = _clips(num_clips=1, occlusion_frames=0, seed=4)[0]
        record = {k: tf.convert_to_tensor(v) for k, v in clip.items()}
        _, targets = to_video_training_record(record, IMAGE_SIZE)
        presence = np.asarray(targets[SAM2_OBJECT_SCORE_LOGITS])
        np.testing.assert_array_equal(presence, np.ones((NUM_FRAMES, 1), "float32"))

    def test_the_record_carries_no_presence_field(self) -> None:
        clip = _clips(num_clips=1)[0]
        assert set(clip) == {RECORD_IMAGE, RECORD_MASK, RECORD_BOX}


# ---------------------------------------------------------------------------
# G2.4 -- the prompt is frame-0's
# ---------------------------------------------------------------------------
class TestFrameZeroPrompt:
    """The prompt is sampled from FRAME 0's mask, and from no other frame."""

    def test_the_prompt_lands_inside_frame_zeros_mask_and_outside_the_last_frames(
            self,
    ) -> None:
        # The fixture is chosen so the two answers DIFFER: only clips whose
        # frame-0 and frame-(T-1) masks are disjoint are used, so "sampled from
        # frame T-1" is detectable. Asserting on an overlapping clip would be a
        # coincidence point where the correct and the mutated source agree.
        checked = 0
        for clip in _clips(num_clips=24, seed=9):
            mask = clip[RECORD_MASK]
            if float((mask[0] * mask[NUM_FRAMES - 1]).sum()) > 0.0:
                continue
            if float(mask[NUM_FRAMES - 1].sum()) == 0.0:
                continue
            record = {k: tf.convert_to_tensor(v) for k, v in clip.items()}
            inputs, _ = to_video_training_record(record, IMAGE_SIZE)
            coords = np.asarray(inputs[INPUT_POINT_COORDS])
            labels = np.asarray(inputs[INPUT_POINT_LABELS])
            assert coords.shape == (1, 2)
            assert labels.tolist() == [1]
            row, col = _cell_of_prompt(coords[0])
            assert mask[0][row, col] > 0.0
            assert mask[NUM_FRAMES - 1][row, col] == 0.0
            checked += 1
        assert checked >= 4, (
            f"only {checked} disjoint frame-0/frame-{NUM_FRAMES - 1} clips were "
            f"found; the guard needs a fixture where the two frames' masks "
            f"differ or it proves nothing"
        )

    def test_background_points_are_also_drawn_from_frame_zero(self) -> None:
        clip = _clips(num_clips=1, seed=9)[0]
        record = {k: tf.convert_to_tensor(v) for k, v in clip.items()}
        inputs, _ = to_video_training_record(
            record, IMAGE_SIZE, num_background_points=2
        )
        coords = np.asarray(inputs[INPUT_POINT_COORDS])
        labels = np.asarray(inputs[INPUT_POINT_LABELS])
        assert coords.shape == (3, 2)
        assert labels.tolist() == [1, 0, 0]
        mask = clip[RECORD_MASK]
        row, col = _cell_of_prompt(coords[0])
        assert mask[0][row, col] > 0.0
        for index in (1, 2):
            row, col = _cell_of_prompt(coords[index])
            assert mask[0][row, col] == 0.0

    def test_the_box_prompt_is_optional_and_frame_zero_shaped(self) -> None:
        clip = _clips(num_clips=1, seed=9)[0]
        record = {k: tf.convert_to_tensor(v) for k, v in clip.items()}
        without, _ = to_video_training_record(record, IMAGE_SIZE)
        assert INPUT_BOXES not in without
        with_box, _ = to_video_training_record(
            record, IMAGE_SIZE, include_box=True
        )
        assert np.asarray(with_box[INPUT_BOXES]).shape == (1, 4)


# ---------------------------------------------------------------------------
# G2.5 -- a frame-0 occlusion is refused
# ---------------------------------------------------------------------------
class TestFrameZeroOcclusionRefusal:
    """An occluded frame 0 has no prompt region, and is refused, not emitted.

    `sample_point_in_mask` answers an empty mask with a PADDING label rather
    than an exception, so a frame-0-occluded clip would train the model on a
    clip with no prompt at all and nothing downstream would report it.
    """

    def test_an_occlusion_window_reaching_frame_zero_is_refused(self) -> None:
        with pytest.raises(ValueError, match="reaches FRAME 0"):
            _occlusion_window(NUM_FRAMES, 2, start=0)

    def test_a_whole_clip_occlusion_is_not_expressible(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, num_frames - 1\]"):
            _occlusion_window(NUM_FRAMES, NUM_FRAMES, start=0)
        with pytest.raises(ValueError, match=r"\[0, num_frames - 1\]"):
            list(_clips(num_clips=1, occlusion_frames=NUM_FRAMES))

    def test_a_window_past_the_end_is_refused(self) -> None:
        with pytest.raises(ValueError, match="does not fit"):
            _occlusion_window(NUM_FRAMES, 2, start=3)

    def test_the_control_a_late_window_still_yields(self) -> None:
        # The paired positive arm. A `raises`-only guard is the shape that went
        # permanently green on GPU in iteration 1.
        clips = _clips(num_clips=4, occlusion_frames=2, occlusion_start=2)
        assert len(clips) == 4
        for clip in clips:
            empty = clip[RECORD_MASK].reshape(NUM_FRAMES, -1).sum(axis=1) == 0.0
            assert empty.tolist() == [False, False, True, True]

    def test_every_emitted_clip_has_a_nonempty_frame_zero(self) -> None:
        for clip in _clips(num_clips=16, occlusion_frames=1):
            assert float(clip[RECORD_MASK][0].sum()) > 0.0


# ---------------------------------------------------------------------------
# determinism, as a PARTITION over seeds
# ---------------------------------------------------------------------------
class TestSeedPartition:
    """Reproducible per seed, and not collapsed to one clip across seeds.

    A single same-seed pair is a coin flip when the sampled space is small; the
    partition over N seeds is the claim worth making. Determinism is asserted
    on the GENERATOR's output only -- the prompt sampler is `tf.random`, which
    this pipeline does not seed.
    """

    def test_a_seed_reproduces_its_clips_and_seeds_partition(self) -> None:
        digests = []
        for seed in range(8):
            first = _clips(num_clips=2, seed=seed)
            second = _clips(num_clips=2, seed=seed)
            for left, right in zip(first, second):
                np.testing.assert_array_equal(
                    left[RECORD_IMAGE], right[RECORD_IMAGE]
                )
                np.testing.assert_array_equal(
                    left[RECORD_MASK], right[RECORD_MASK]
                )
                np.testing.assert_array_equal(left[RECORD_BOX], right[RECORD_BOX])
            digests.append(first[0][RECORD_MASK].tobytes())
        assert len(set(digests)) >= 2, (
            "8 seeds produced one single clip; the source is seed-insensitive"
        )


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------
class TestArgumentValidation:
    """Refusals that keep a wrong geometry from becoming a silent result."""

    def test_a_mask_size_that_does_not_divide_the_image_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive divisor"):
            list(
                synthetic_moving_instance_video(
                    num_clips=1,
                    num_frames=NUM_FRAMES,
                    image_size=IMAGE_SIZE,
                    mask_size=15,
                )
            )

    def test_a_zero_length_clip_is_refused(self) -> None:
        with pytest.raises(ValueError, match="num_frames must be"):
            list(
                synthetic_moving_instance_video(
                    num_clips=1,
                    num_frames=0,
                    image_size=IMAGE_SIZE,
                    mask_size=MASK_SIZE,
                    occlusion_frames=0,
                )
            )


# ---------------------------------------------------------------------------
# G2.6 -- dead-component probe
# ---------------------------------------------------------------------------
class TestDeadComponentProbe:
    """A dead renderer is reported, not absorbed into a shorter epoch.

    MEASURED partition under `stencil -> all-zero canvas`, the whole file:
    **24 of 35 RED, 11 GREEN**. Every guard that renders a clip goes RED,
    because the generator raises `RuntimeError` after `MAX_CLIP_ATTEMPTS`
    rather than emitting a short or approximate epoch. The 11 survivors are
    exactly the guards that never render anything:

    * `test_the_window_flag_count_equals_the_requested_length[0..3]` (4) and
      `test_the_bar_geometry_covers_the_window_and_nothing_else` (1) --
      pure integer arithmetic;
    * the three `TestFrameZeroOcclusionRefusal` window-validation raises and
      the two `TestArgumentValidation` raises (5) -- argument checks that
      fire before a pixel is drawn;
    * this test itself (1).

    That is the honest shape of a dead-component probe on a data module: the
    partition is between "renders" and "does arithmetic", not between "good"
    and "bad" tests. It is recorded because iteration 1 falsified the
    hypothesis "all guards go RED under a dead component" at five consecutive
    steps, and step 1 of this iteration left 26 of 30 GREEN.
    """

    def test_a_dead_renderer_is_reported_not_absorbed(
            self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _dead(size: int, rng: np.random.Generator) -> np.ndarray:
            return np.zeros((size, size), dtype=np.uint8)

        monkeypatch.setattr(sam2_data, "INSTANCE_RENDERERS", (_dead,))
        with pytest.raises(RuntimeError, match=str(MAX_CLIP_ATTEMPTS)):
            _clips(num_clips=1)


# ---------------------------------------------------------------------------
# the `tf.data` assembly
# ---------------------------------------------------------------------------
class TestDatasetAssembly:
    """`build_sam2_video_dataset` produces what `SAM2TrainingModel` consumes."""

    def test_the_batched_shapes_match_the_wrappers_input_contract(self) -> None:
        dataset = build_sam2_video_dataset(
            num_clips=4,
            num_frames=NUM_FRAMES,
            image_size=IMAGE_SIZE,
            batch_size=2,
            mask_size=MASK_SIZE,
            occlusion_frames=1,
            include_box=True,
            seed=1,
        )
        inputs, targets = next(iter(dataset))
        assert tuple(inputs[INPUT_IMAGE].shape) == (
            2, NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3
        )
        assert tuple(inputs[INPUT_POINT_COORDS].shape) == (2, 1, 2)
        assert tuple(inputs[INPUT_POINT_LABELS].shape) == (2, 1)
        assert tuple(inputs[INPUT_BOXES].shape) == (2, 1, 4)
        assert tuple(inputs[INPUT_GT_MASKS].shape) == (
            2, NUM_FRAMES, MASK_SIZE, MASK_SIZE
        )
        assert tuple(targets[SAM2_LOW_RES_LOGITS].shape) == (
            2, NUM_FRAMES, MASK_SIZE, MASK_SIZE
        )
        assert tuple(targets[SAM2_OBJECT_SCORE_LOGITS].shape) == (
            2, NUM_FRAMES, 1
        )

    def test_the_target_keys_are_exactly_the_wrappers_output_keys(self) -> None:
        # A CANARY, deliberately coupled: when the wrapper grows the
        # `iou_supervision` output at step 4, this assertion goes RED here and
        # the pipeline is forced to grow the matching target. A pipeline target
        # the model does not emit, or a model output nothing supervises, both
        # fail `compile(loss={...})` in ways that name a key, not a cause.
        clip = _clips(num_clips=1)[0]
        record = {k: tf.convert_to_tensor(v) for k, v in clip.items()}
        _, targets = to_video_training_record(record, IMAGE_SIZE)
        assert set(targets) == set(OUTPUT_KEYS)

    def test_the_dataset_epoch_is_the_requested_clip_count(self) -> None:
        dataset = build_sam2_video_dataset(
            num_clips=6,
            num_frames=NUM_FRAMES,
            image_size=IMAGE_SIZE,
            batch_size=2,
            mask_size=MASK_SIZE,
            seed=2,
        )
        assert sum(int(batch[0][INPUT_IMAGE].shape[0]) for batch in dataset) == 6

    def test_the_source_writes_nothing_to_disk(self, tmp_path) -> None:
        # Datasets never go on the repo SSD; this one is generated in memory and
        # touches no path at all. Asserted, not assumed.
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            dataset = build_sam2_video_dataset(
                num_clips=4,
                num_frames=NUM_FRAMES,
                image_size=IMAGE_SIZE,
                batch_size=2,
                mask_size=MASK_SIZE,
                seed=3,
            )
            for _ in dataset:
                pass
        finally:
            os.chdir(cwd)
        assert list(tmp_path.iterdir()) == []
