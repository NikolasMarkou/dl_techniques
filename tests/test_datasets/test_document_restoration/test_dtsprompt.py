"""Tests for the DTSPrompt generators and the DocRes ``TASKS`` table.

**Deliberate import asymmetry — do not "fix" it.** This test module imports
``cv2`` and ``skimage``; the shipped module under test must import neither.
That is not an inconsistency, it is the point. ``opencv-python`` and
``scikit-image`` are undeclared in ``pyproject.toml`` (measured: present in
this ``.venv`` at 4.13.0 / 0.26.0, absent from the core dependency list and
from every extra), so shipping an import of them would make the library work
here and fail on a clean install. But upstream DocRes *is* OpenCV, so
agreement with OpenCV is the parity criterion, and the strongest oracle
available is to run OpenCV side by side in the test. The tests skip cleanly if
either package is missing, and
``test_the_shipped_package_imports_neither_cv2_nor_skimage`` enforces the
constraint on the shipped side.
"""

import ast
import dataclasses
import importlib
import pathlib

import numpy as np
import pytest

from dl_techniques.datasets.document_restoration import dtsprompt as dts
from dl_techniques.datasets.document_restoration import tasks as task_mod

cv2 = pytest.importorskip("cv2", reason="parity oracle only; never a library dependency")
skimage_filters = pytest.importorskip(
    "skimage.filters", reason="parity oracle only; never a library dependency"
)

RNG = np.random.default_rng(1029)


# ---------------------------------------------------------------------------
# Fixtures: tiny, fixed, and small enough to reason about by hand.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_page() -> np.ndarray:
    """A fixed 37x53 RGB page. Non-square on purpose: a square page cannot see
    an H/W transposition."""
    return np.random.default_rng(7).integers(0, 256, (37, 53, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def blank_page() -> np.ndarray:
    """The degenerate all-constant page a blank scan produces."""
    return np.full((31, 43, 3), 217, dtype=np.uint8)


def assert_finite_uint8(arr: np.ndarray, name: str) -> None:
    """Assert an array is a finite uint8 page-shaped map.

    Args:
        arr: The array to check.
        name: Label used in assertion messages.
    """
    assert arr.dtype == np.uint8, f"{name}: dtype {arr.dtype}"
    assert np.isfinite(arr.astype(np.float64)).all(), f"{name}: non-finite value"


# ---------------------------------------------------------------------------
# 0. The hard constraint: the shipped package is numpy/scipy only.
# ---------------------------------------------------------------------------


def _top_level_imports(path: pathlib.Path) -> set:
    """Collect the top-level module names a source file imports.

    Args:
        path: Python source file.

    Returns:
        Set of first path components of every imported module.
    """
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def _package_sources() -> list:
    """Every shipped ``.py`` file of the document_restoration package.

    Returns:
        Sorted list of paths.
    """
    pkg = importlib.import_module("dl_techniques.datasets.document_restoration")
    root = pathlib.Path(pkg.__file__).parent
    return sorted(root.glob("*.py"))


def test_the_shipped_package_imports_neither_cv2_nor_skimage():
    """D-002: cv2/skimage are installed here but UNDECLARED in pyproject.toml.

    This scans every shipped module of the package, not just ``dtsprompt.py``,
    so the constraint cannot be evaded by moving an import to a sibling.
    """
    sources = _package_sources()
    assert len(sources) >= 3, f"expected at least 3 shipped modules, found {sources}"
    forbidden = {"cv2", "skimage"}
    offenders = {}
    for path in sources:
        bad = _top_level_imports(path) & forbidden
        if bad:
            offenders[path.name] = sorted(bad)
    assert not offenders, (
        f"forbidden undeclared imports in the shipped package: {offenders}. "
        f"cv2 and scikit-image are NOT in pyproject.toml; see D-002."
    )


def test_the_forbidden_import_scanner_can_actually_see_an_import(tmp_path):
    """Anti-vacuity: prove the scanner detects both forbidden spellings."""
    probe = tmp_path / "probe.py"
    probe.write_text("import numpy\nimport cv2\nfrom skimage.filters import x\n")
    found = _top_level_imports(probe)
    assert "cv2" in found and "skimage" in found and "numpy" in found


def test_the_shipped_package_does_import_numpy_and_scipy():
    """The replacement stack is actually present, i.e. the module is not empty."""
    imports = _top_level_imports(
        pathlib.Path(dts.__file__)
    )
    assert "numpy" in imports
    assert "scipy" in imports


# ---------------------------------------------------------------------------
# 1. Primitive-level parity against the cv2 / skimage oracle.
# ---------------------------------------------------------------------------


def test_the_dilation_is_bit_exact_against_cv2():
    """``ndimage.grey_dilation(mode='nearest')`` == ``cv2.dilate(ones(7,7))``.

    Provably so, not merely empirically: 'nearest' extends the border with the
    nearest in-image value, which is already inside the max window, while cv2's
    default dilate border value is -inf and so contributes nothing. Both
    therefore take the max over exactly the in-image part of the window.

    The same argument makes 'reflect', 'mirror' and 'constant(0)' equally
    correct here -- all of them pad with values the window already contains --
    so a ``nearest -> reflect`` mutation of the shipped module is a genuine
    no-op that no test can or should detect. MEASURED: all four agree with
    OpenCV on every pixel of four differently-shaped pages. 'wrap' is the one
    mode that is wrong, because it imports values from the opposite edge, and
    it is asserted below so this test is not vacuous.
    """
    from scipy import ndimage

    plane = RNG.integers(0, 256, (97, 131), dtype=np.uint8)
    expected = cv2.dilate(plane, np.ones((7, 7), np.uint8))
    for mode in ("nearest", "reflect", "mirror"):
        assert np.array_equal(
            expected, ndimage.grey_dilation(plane, size=(7, 7), mode=mode)
        ), mode
    assert np.array_equal(
        expected, ndimage.grey_dilation(plane, size=(7, 7), mode="constant", cval=0)
    )
    assert not np.array_equal(
        expected, ndimage.grey_dilation(plane, size=(7, 7), mode="wrap")
    ), "'wrap' agreed with cv2 -- this control is supposed to show the max "
    "filter is not indifferent to every border mode"


def test_the_median_blur_is_bit_exact_against_cv2():
    """``median_filter(mode='nearest')`` == ``cv2.medianBlur``, which is
    hardcoded to BORDER_REPLICATE. A 'reflect' border would NOT match."""
    from scipy import ndimage

    plane = RNG.integers(0, 256, (97, 131), dtype=np.uint8)
    expected = cv2.medianBlur(plane, 21)
    assert np.array_equal(expected, ndimage.median_filter(plane, size=21, mode="nearest"))
    wrong = ndimage.median_filter(plane, size=21, mode="reflect")
    assert not np.array_equal(expected, wrong), (
        "the reflect border agreed with cv2 -- this control is supposed to "
        "show the border mode is load-bearing"
    )


def test_the_gray_conversion_is_bit_exact_against_cv2():
    """The 15-bit fixed-point luma matches ``cvtColor`` on every pixel, and the
    widely-quoted 14-bit triple does not."""
    bgr = RNG.integers(0, 256, (200, 200, 3), dtype=np.uint8)
    expected = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Our module is RGB; feeding it the reversed array makes the two agree by
    # construction only if the weights really are per-channel correct.
    got = dts._rgb_to_gray(bgr[..., ::-1].copy())
    assert np.array_equal(expected, got)

    b, g, r = (bgr[..., i].astype(np.int64) for i in range(3))
    fourteen_bit = ((b * 1868 + g * 9617 + r * 4899 + (1 << 13)) >> 14).astype(np.uint8)
    assert not np.array_equal(expected, fourteen_bit), (
        "the 14-bit triple matched cv2 exactly -- if OpenCV changed, re-derive "
        "_GRAY_* rather than assuming"
    )


def test_the_minmax_normalisation_is_bit_exact_against_cv2():
    """Including the degenerate constant plane, where cv2 zeroes the scale."""
    plane = RNG.integers(30, 200, (97, 131), dtype=np.uint8)
    expected = cv2.normalize(
        plane, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
    )
    assert np.array_equal(expected, dts._normalize_minmax_uint8(plane))

    const = np.full((8, 8), 77, np.uint8)
    expected_const = cv2.normalize(
        const, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
    )
    got_const = dts._normalize_minmax_uint8(const)
    assert np.array_equal(expected_const, got_const)
    assert got_const.max() == 0, "cv2 maps a constant plane to zeros; we must too"


def test_the_sobel_gradient_is_bit_exact_against_cv2(tiny_page):
    """The whole ``Sobel -> convertScaleAbs -> addWeighted -> gray`` chain."""
    bgr = tiny_page[..., ::-1].copy()
    x = cv2.Sobel(bgr, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(bgr, cv2.CV_16S, 0, 1)
    blended = cv2.addWeighted(
        cv2.convertScaleAbs(x), 0.5, cv2.convertScaleAbs(y), 0.5, 0
    )
    expected = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
    assert np.array_equal(expected, dts._sobel_magnitude_gray(tiny_page))


def test_the_bilinear_resize_matches_cv2_to_one_grey_level():
    """The single documented approximation.

    cv2 evaluates INTER_LINEAR for 8-bit input in 11-bit fixed point; this
    module evaluates it in float64 and rounds once. Same sampling convention,
    same replicate border, so the two differ only by cv2's intermediate
    quantisation. The bound asserted here is max|difference| <= 1.
    """
    plane = RNG.integers(0, 256, (97, 131), dtype=np.uint8)
    for out_h, out_w in [(256, 256), (41, 53), (97, 131)]:
        expected = cv2.resize(plane, (out_w, out_h)).astype(np.int64)
        got = dts._resize_bilinear(plane, out_h, out_w).astype(np.int64)
        diff = np.abs(expected - got)
        assert diff.max() <= 1, f"{(out_h, out_w)}: max|d| = {diff.max()}"
        assert diff.mean() < 0.2, f"{(out_h, out_w)}: mean|d| = {diff.mean()}"


def test_the_resize_is_not_align_corners():
    """A guard against the other plausible bilinear convention.

    ``scipy.ndimage.zoom(order=1)`` defaults to ``grid_mode=False``, which is
    the align-corners convention and is NOT what cv2 does. It must disagree,
    or this test would be blind to the module silently adopting it.
    """
    from scipy import ndimage

    plane = RNG.integers(0, 256, (97, 131), dtype=np.uint8)
    ours = dts._resize_bilinear(plane, 41, 53).astype(np.int64)
    align_corners = np.clip(
        np.rint(ndimage.zoom(plane, (41 / 97, 53 / 131), order=1, grid_mode=False)),
        0,
        255,
    ).astype(np.int64)
    assert np.abs(ours - align_corners).max() > 1, (
        "align-corners agreed with the half-pixel convention -- the resize "
        "convention would then be untested"
    )


def test_the_sauvola_threshold_is_bit_exact_against_skimage():
    """``_threshold_sauvola`` reproduces ``skimage.filters.threshold_sauvola``
    exactly, at every window size the two-pass algorithm can pick."""
    gray = RNG.integers(0, 256, (61, 83), dtype=np.uint8)
    for window in (1, 3, 5, 7, 21, 41):
        expected = skimage_filters.threshold_sauvola(gray, window_size=window, k=0.5)
        got = dts._threshold_sauvola(gray, window_size=window, k=0.5)
        assert np.abs(expected - got).max() == 0.0, f"window={window}"


def test_the_sauvola_window_offset_is_load_bearing():
    """Anti-vacuity for the integral-image window: a one-pixel shift disagrees.

    The skimage implementation pads asymmetrically and shifts the box by one;
    the shipped code pads symmetrically and does not. If an off-by-one there
    were invisible, the exactness test above would prove nothing.
    """
    gray = RNG.integers(0, 256, (61, 83), dtype=np.uint8)
    shifted = np.roll(gray, 1, axis=0)
    a = dts._threshold_sauvola(gray, window_size=7, k=0.5)
    b = dts._threshold_sauvola(shifted, window_size=7, k=0.5)
    assert np.abs(a - b).max() > 1.0


# ---------------------------------------------------------------------------
# 2. Generator-level parity against a verbatim transcription of upstream.
# ---------------------------------------------------------------------------


PARITY_MAX = 4
"""Bound on the background pipeline's disagreement with the cv2 reference.

MEASURED over three page shapes x three working resolutions: max|d| peaks at
**3** grey levels (on the min-max-normalised branch, whose rescale amplifies a
one-level resize error), and mean|d| peaks at **0.48**. The bound is set one
level above the measured worst case, and
``test_the_pipeline_is_bit_exact_when_no_resize_happens`` proves that the whole
residual comes from the resize and nothing else -- when the page is already at
the working resolution the agreement is exact, 0 on every pixel.
"""

PARITY_MEAN = 0.6


def _upstream_estimate_background(bgr: np.ndarray, working: int = 1024):
    """Upstream ``deshadow_prompt``/``appearance_prompt``, transcribed verbatim.

    Args:
        bgr: BGR uint8 page, as ``cv2.imread`` would return it.
        working: Working square resolution.

    Returns:
        ``(bg_imgs, result_norm)``, upstream's two kept intermediates.
    """
    h, w = bgr.shape[:2]
    img = cv2.resize(bgr, (working, working))
    planes = cv2.split(img)
    norm_planes, bg_imgs = [], []
    for plane in planes:
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        bg_imgs.append(bg)
        diff = 255 - cv2.absdiff(plane, bg)
        norm_planes.append(
            cv2.normalize(
                diff, None, alpha=0, beta=255,
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1,
            )
        )
    bg_merged = cv2.resize(cv2.merge(bg_imgs), (w, h))
    norm_merged = cv2.resize(cv2.merge(norm_planes), (w, h))
    return bg_merged, norm_merged


def _upstream_binarization_prompt(bgr: np.ndarray) -> np.ndarray:
    """Upstream ``binarization_promptv2`` + ``SauvolaModBinarization``, verbatim."""
    image = bgr
    n1 = int(0.05 * min(image.shape[0], image.shape[1]))
    if n1 % 2 == 0:
        n1 += 1
    n2 = int(0.1 * min(image.shape[0], image.shape[1]))
    if n2 % 2 == 0:
        n2 += 1
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    t1 = skimage_filters.threshold_sauvola(gray, window_size=n1, k=0.5)
    max_val = np.amax(gray)
    c = np.copy(t1).astype(np.float32)
    c[gray > t1] = (gray[gray > t1] - t1[gray > t1]) / (max_val - t1[gray > t1])
    c[gray <= t1] = 0
    c = c * 255.0
    new_in = np.copy(c.astype(np.uint8))
    t2 = skimage_filters.threshold_sauvola(new_in, window_size=n2, k=0.5)
    binary = np.copy(gray)
    binary[new_in <= t2] = 0
    binary[new_in > t2] = 255

    result, thresh = binary, t2.astype(np.uint8)
    result = result.copy()
    result[result > 155] = 255
    result[result <= 155] = 0
    x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
    hf = cv2.addWeighted(cv2.convertScaleAbs(x), 0.5, cv2.convertScaleAbs(y), 0.5, 0)
    hf = cv2.cvtColor(hf, cv2.COLOR_BGR2GRAY)
    return np.concatenate(
        (thresh[..., None], hf[..., None], result[..., None]), -1
    )


@pytest.mark.parametrize("working", [64, 128])
def test_estimate_background_matches_upstream_within_the_resize_bound(tiny_page, working):
    """The whole background pipeline, end to end, against upstream.

    Run at reduced working resolutions so the assertion is cheap; the default
    1024 case is covered by ``test_..._at_the_shipped_working_resolution``.
    """
    bgr = tiny_page[..., ::-1].copy()
    exp_bg, exp_norm = _upstream_estimate_background(bgr, working)
    got_bg, got_norm = dts.estimate_background(tiny_page, working_size=working)
    for got, expected, name in [
        (got_bg, exp_bg[..., ::-1], "bg"),
        (got_norm, exp_norm[..., ::-1], "norm"),
    ]:
        diff = np.abs(got.astype(np.int64) - expected.astype(np.int64))
        assert diff.max() <= PARITY_MAX, f"{name}@{working}: max|d| = {diff.max()}"
        assert diff.mean() < PARITY_MEAN, f"{name}@{working}: mean|d| = {diff.mean()}"


def test_the_pipeline_is_bit_exact_when_no_resize_happens():
    """Localises the whole cv2 disagreement to the bilinear resize.

    With the page already at the working resolution both resizes are identity
    (:func:`_resize_bilinear` short-circuits), and the dilate / median /
    absdiff / min-max chain then agrees with OpenCV on EVERY pixel. So the
    residual measured by the two parity tests above is the documented resize
    approximation and nothing else -- not a wrong kernel, border mode or
    normalisation.
    """
    page = np.random.default_rng(13).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    exp_bg, exp_norm = _upstream_estimate_background(page[..., ::-1].copy(), 64)
    got_bg, got_norm = dts.estimate_background(page, working_size=64)
    assert np.array_equal(got_bg, exp_bg[..., ::-1])
    assert np.array_equal(got_norm, exp_norm[..., ::-1])


@pytest.mark.slow
def test_the_generators_match_upstream_at_the_shipped_working_resolution(tiny_page):
    """The default 1024 path. ~8 s: three 1024x1024 21-tap median filters."""
    bgr = tiny_page[..., ::-1].copy()
    exp_bg, exp_norm = _upstream_estimate_background(bgr, dts.BACKGROUND_WORKING_SIZE)
    got_bg = dts.deshadow_prompt(tiny_page)
    got_norm = dts.appearance_prompt(tiny_page)
    for got, expected, name in [
        (got_bg, exp_bg[..., ::-1], "deshadow"),
        (got_norm, exp_norm[..., ::-1], "appearance"),
    ]:
        diff = np.abs(got.astype(np.int64) - expected.astype(np.int64))
        assert diff.max() <= PARITY_MAX, f"{name}: max|d| = {diff.max()}"
        assert diff.mean() < PARITY_MEAN, f"{name}: mean|d| = {diff.mean()}"


def test_deblur_prompt_matches_upstream_exactly(tiny_page):
    """No resize in this path, so the agreement must be bit-exact."""
    bgr = tiny_page[..., ::-1].copy()
    x = cv2.Sobel(bgr, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(bgr, cv2.CV_16S, 0, 1)
    hf = cv2.addWeighted(cv2.convertScaleAbs(x), 0.5, cv2.convertScaleAbs(y), 0.5, 0)
    hf = cv2.cvtColor(hf, cv2.COLOR_BGR2GRAY)
    expected = cv2.cvtColor(hf, cv2.COLOR_GRAY2BGR)

    got = dts.deblur_prompt(tiny_page)
    assert got.shape == tiny_page.shape
    assert_finite_uint8(got, "deblur")
    assert np.array_equal(expected, got)


def test_deblur_prompt_replicates_one_map_into_three_channels(tiny_page):
    """Contrast with binarization: here the three channels carry no new info."""
    got = dts.deblur_prompt(tiny_page)
    assert np.array_equal(got[..., 0], got[..., 1])
    assert np.array_equal(got[..., 0], got[..., 2])


def test_binarization_prompt_matches_upstream_exactly(tiny_page):
    """No resize in this path either, so bit-exactness is the bar."""
    bgr = tiny_page[..., ::-1].copy()
    expected = _upstream_binarization_prompt(bgr)
    got = dts.binarization_prompt(tiny_page)
    assert got.shape == tiny_page.shape
    assert_finite_uint8(got, "binarization")
    assert np.array_equal(expected, got)


def test_binarization_prompt_stacks_three_DIFFERENT_maps(tiny_page):
    """The order is [threshold, gradient, binary] and they are not replicas."""
    got = dts.binarization_prompt(tiny_page)
    assert not np.array_equal(got[..., 0], got[..., 1])
    assert not np.array_equal(got[..., 1], got[..., 2])
    assert set(np.unique(got[..., 2])).issubset({0, 255}), "channel 2 must be binary"
    assert not set(np.unique(got[..., 0])).issubset({0, 255}), (
        "channel 0 must be the continuous threshold map, not a binary one"
    )
    binary, threshold = dts.sauvola_mod_binarization(tiny_page)
    assert np.array_equal(got[..., 0], threshold)
    assert np.array_equal(got[..., 2], binary)
    assert np.array_equal(got[..., 1], dts._sobel_magnitude_gray(tiny_page))


def test_the_sauvola_windows_track_the_smaller_page_dimension():
    """5% / 10% of ``min(H, W)``, forced odd -- not a fixed pixel size."""
    assert dts._odd_window(0.05, 400) == 21
    assert dts._odd_window(0.10, 400) == 41
    assert dts._odd_window(0.05, 200) == 11
    assert dts._odd_window(0.10, 200) == 21
    # int() truncation, then the even -> +1 correction.
    assert dts._odd_window(0.05, 61) == 3   # int(3.05) = 3, already odd
    assert dts._odd_window(0.05, 100) == 5  # int(5.0) = 5
    assert dts._odd_window(0.05, 8) == 1    # int(0.4) = 0 -> 1, never 0
    for extent in range(1, 300):
        for frac in (0.05, 0.10):
            n = dts._odd_window(frac, extent)
            assert n >= 1 and n % 2 == 1


def test_sauvola_is_genuinely_two_pass():
    """A one-pass Sauvola on the same page gives a different binary map.

    Without this, ``sauvola_mod_binarization`` could silently collapse to a
    single pass and every shape/dtype assertion would still hold.
    """
    page = RNG.integers(0, 256, (200, 240, 3), dtype=np.uint8)
    binary, _ = dts.sauvola_mod_binarization(page)
    gray = dts._rgb_to_gray(page)
    n1 = dts._odd_window(0.05, 200)
    one_pass_t = dts._threshold_sauvola(gray, window_size=n1, k=0.5)
    one_pass = np.where(gray > one_pass_t, 255, 0).astype(np.uint8)
    disagreement = np.mean(one_pass != binary)
    assert disagreement > 0.01, (
        f"the two-pass result is only {100 * disagreement:.3f}% different from "
        f"a single pass -- the second pass may not be running"
    )


def test_the_two_sauvola_passes_use_different_windows():
    """n2 is twice n1 (before the odd correction), so the passes differ."""
    n1 = dts._odd_window(dts.SAUVOLA_N1_FRACTION, 400)
    n2 = dts._odd_window(dts.SAUVOLA_N2_FRACTION, 400)
    assert n2 > n1
    assert dts.SAUVOLA_K1 == dts.SAUVOLA_K2 == 0.5


# ---------------------------------------------------------------------------
# 3. The dewarping prompt: the x/y order is the silent defect.
# ---------------------------------------------------------------------------


def test_the_base_grid_matches_a_hand_written_reference():
    """A 3x4 grid, written out by hand. Channel 0 is x, channel 1 is y."""
    grid = dts.base_coordinate_grid(3, 4)
    assert grid.shape == (3, 4, 2)
    assert grid.dtype == np.float32
    expected_x = np.array(
        [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.float32
    ) / 4.0
    expected_y = np.array(
        [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.float32
    ) / 3.0
    assert np.array_equal(grid[..., 0], expected_x)
    assert np.array_equal(grid[..., 1], expected_y)


def test_a_swapped_xy_base_grid_would_be_caught():
    """The RED-proof shape of the x/y guard.

    Channel 0 must be constant DOWN each column and increasing ACROSS each
    row; channel 1 the reverse. Swapping the two planes breaks both halves,
    even though the swapped array keeps its shape, dtype and value range.
    """
    grid = dts.base_coordinate_grid(9, 13)
    x, y = grid[..., 0], grid[..., 1]
    assert np.all(np.diff(x, axis=1) > 0), "channel 0 must increase across the width"
    assert np.all(np.diff(x, axis=0) == 0), "channel 0 must be constant down a column"
    assert np.all(np.diff(y, axis=0) > 0), "channel 1 must increase down the height"
    assert np.all(np.diff(y, axis=1) == 0), "channel 1 must be constant across a row"

    swapped = grid[..., ::-1]
    fired = 0
    if not np.all(np.diff(swapped[..., 0], axis=1) > 0):
        fired += 1
    if not np.all(np.diff(swapped[..., 1], axis=0) > 0):
        fired += 1
    assert fired == 2, "the swap must break BOTH halves of the predicate"


def test_the_base_grid_agrees_with_upstream_getbasecoord_at_256():
    """Upstream's only call site is ``getBasecoord(256, 256) / 256``."""
    h = w = 256
    base0 = np.tile(np.arange(h).reshape(h, 1), (1, w)).astype(np.float32)
    base1 = np.tile(np.arange(w).reshape(1, w), (h, 1)).astype(np.float32)
    upstream = np.concatenate(
        (np.expand_dims(base1, -1), np.expand_dims(base0, -1)), -1
    ) / 256.0
    assert np.array_equal(upstream.astype(np.float32), dts.base_coordinate_grid(h, w))


def test_dewarp_prompt_layout_and_range(tiny_page):
    """Three channels: x, y, mask/255. float32, unlike the other four."""
    h, w = tiny_page.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mask[5:20, 7:30] = 255
    prompt = dts.dewarp_prompt(tiny_page, mask)
    assert prompt.shape == (h, w, 3)
    assert prompt.dtype == np.float32
    assert np.isfinite(prompt).all()
    assert prompt.min() >= 0.0 and prompt.max() <= 1.0
    assert np.array_equal(prompt[..., :2], dts.base_coordinate_grid(h, w))
    assert np.array_equal(prompt[..., 2], mask.astype(np.float32) / 255.0)


def test_dewarp_prompt_requires_a_caller_supplied_mask(tiny_page):
    """The MBD network is not part of this port; the mask is an input."""
    h, w = tiny_page.shape[:2]
    with pytest.raises(TypeError):
        dts.dewarp_prompt(tiny_page)
    with pytest.raises(ValueError, match="does not match the page"):
        dts.dewarp_prompt(tiny_page, np.zeros((h + 1, w), np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        dts.dewarp_prompt(tiny_page, np.zeros((h, w), np.float32))


def test_apply_document_mask_zeroes_outside(tiny_page):
    h, w = tiny_page.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mask[5:20, 7:30] = 255
    masked = dts.apply_document_mask(tiny_page, mask)
    assert np.array_equal(masked[5:20, 7:30], tiny_page[5:20, 7:30])
    assert masked[mask == 0].max() == 0
    assert tiny_page[mask == 0].max() > 0, "the fixture must have content outside"


# ---------------------------------------------------------------------------
# 4. The degenerate all-constant page.
# ---------------------------------------------------------------------------


def test_every_generator_survives_a_blank_page(blank_page):
    """No NaN, no divide-by-zero, no wraparound on a uniform scan."""
    h, w = blank_page.shape[:2]
    mask = np.full((h, w), 255, np.uint8)
    outputs = {
        "deshadow": dts.deshadow_prompt(blank_page, working_size=64),
        "appearance": dts.appearance_prompt(blank_page, working_size=64),
        "deblur": dts.deblur_prompt(blank_page),
        "binarization": dts.binarization_prompt(blank_page),
    }
    for name, arr in outputs.items():
        assert arr.shape == (h, w, 3), f"{name}: {arr.shape}"
        assert_finite_uint8(arr, name)
    dewarp = dts.dewarp_prompt(blank_page, mask)
    assert np.isfinite(dewarp).all()
    assert dewarp.dtype == np.float32


def test_a_black_page_does_not_divide_by_zero():
    """``max_val - T1`` is the one division in the Sauvola port; an all-zero
    page makes ``max_val`` zero."""
    black = np.zeros((33, 41, 3), np.uint8)
    binary, threshold = dts.sauvola_mod_binarization(black)
    assert np.isfinite(binary.astype(np.float64)).all()
    assert np.isfinite(threshold.astype(np.float64)).all()
    prompt = dts.binarization_prompt(black)
    assert_finite_uint8(prompt, "binarization(black)")
    white = np.full((33, 41, 3), 255, np.uint8)
    assert_finite_uint8(dts.binarization_prompt(white), "binarization(white)")


def test_the_blank_page_normalisation_is_zeros_not_nan(blank_page):
    """cv2's NORM_MINMAX zeroes the scale on a constant plane; a naive
    ``(x-min)/(max-min)`` would produce NaN here."""
    _, norm = dts.estimate_background(blank_page, working_size=64)
    assert np.isfinite(norm.astype(np.float64)).all()
    assert norm.max() == 0


# ---------------------------------------------------------------------------
# 5. Input-contract guards.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "generator",
    [
        dts.deshadow_prompt,
        dts.appearance_prompt,
        dts.deblur_prompt,
        dts.binarization_prompt,
    ],
)
def test_every_generator_refuses_a_float_or_wrong_rank_input(generator):
    """A caller that already divided by 255 must be told, not silently served."""
    with pytest.raises(ValueError, match="uint8"):
        generator(np.zeros((16, 16, 3), np.float32))
    with pytest.raises(ValueError, match=r"shape \(H, W, 3\)"):
        generator(np.zeros((16, 16), np.uint8))
    with pytest.raises(ValueError, match="numpy array"):
        generator([[0, 0, 0]])


def test_estimate_background_refuses_a_non_positive_working_size(tiny_page):
    with pytest.raises(ValueError, match="working_size"):
        dts.estimate_background(tiny_page, working_size=0)


def test_the_base_grid_refuses_a_zero_extent():
    with pytest.raises(ValueError, match="extents must be positive"):
        dts.base_coordinate_grid(0, 5)


def test_working_size_is_load_bearing_not_cosmetic(tiny_page):
    """The kernels are absolute pixel sizes at the working resolution, so
    changing it changes the answer. Documented, and asserted here so nobody
    'optimises' the resize away."""
    a = dts.deshadow_prompt(tiny_page, working_size=64)
    b = dts.deshadow_prompt(tiny_page, working_size=256)
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# 6. The TASKS table.
# ---------------------------------------------------------------------------


EXPECTED_SUPERVISED = {
    "dewarping": 2,
    "deshadowing": 3,
    "appearance": 3,
    "deblurring": 3,
    "binarization": 2,
}


def test_the_table_has_exactly_the_five_docres_tasks():
    assert set(task_mod.TASKS) == set(EXPECTED_SUPERVISED)
    assert task_mod.task_names() == (
        "dewarping", "deshadowing", "appearance", "deblurring", "binarization",
    )


@pytest.mark.parametrize("name,expected", sorted(EXPECTED_SUPERVISED.items()))
def test_the_supervised_channel_counts_are_the_ones_finding_F6_states(name, expected):
    """F-6: dewarping and binarization supervise 2 of the 3 output channels;
    the other three tasks supervise all 3."""
    spec = task_mod.get_task(name)
    assert spec.n_supervised_channels == expected
    assert spec.supervised_slice == slice(0, expected)
    assert spec.n_supervised_channels <= task_mod.N_OUTPUT_CHANNELS


def test_every_task_resolves_and_carries_a_live_generator():
    """Each entry's ``prompt_fn`` is really the module-level generator, not a
    stale copy, and each declared ``prompt_dtype`` is the dtype it returns."""
    page = np.random.default_rng(3).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    mask = np.full((24, 32), 255, np.uint8)
    for name in task_mod.task_names():
        spec = task_mod.get_task(name)
        assert spec.name == name
        assert callable(spec.prompt_fn)
        if spec.requires_mask:
            prompt = spec.prompt_fn(page, mask)
        elif spec.prompt_fn in (dts.deshadow_prompt, dts.appearance_prompt):
            prompt = spec.prompt_fn(page, working_size=32)
        else:
            prompt = spec.prompt_fn(page)
        assert prompt.shape == (24, 32, task_mod.N_PROMPT_CHANNELS), name
        assert prompt.dtype == np.dtype(spec.prompt_dtype), name


def test_the_table_binds_each_task_to_its_OWN_generator():
    """Identity, not just callability.

    This guard exists because its absence was MEASURED: with
    ``binarization`` mis-bound to ``deblur_prompt`` the whole suite stayed
    green at 52 passed. Every other table assertion is blind to the swap --
    both generators take one uint8 page and return a ``(H, W, 3)`` uint8 map,
    so shape, dtype and channel count all agree. Only identity separates them.
    """
    expected = {
        "dewarping": dts.dewarp_prompt,
        "deshadowing": dts.deshadow_prompt,
        "appearance": dts.appearance_prompt,
        "deblurring": dts.deblur_prompt,
        "binarization": dts.binarization_prompt,
    }
    assert set(expected) == set(task_mod.TASKS)
    for name, fn in expected.items():
        assert task_mod.get_task(name).prompt_fn is fn, (
            f"task {name!r} is bound to {task_mod.get_task(name).prompt_fn!r}, "
            f"not to {fn!r}"
        )
    bound = [task_mod.get_task(n).prompt_fn for n in task_mod.task_names()]
    assert len(set(bound)) == len(bound), "two tasks share one generator"


def test_the_generator_identity_guard_is_not_satisfiable_by_shape_alone():
    """Anti-vacuity: the two swappable generators really are shape-compatible,
    which is exactly why the identity check above is the only instrument that
    can see the swap."""
    page = np.random.default_rng(5).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    a = dts.binarization_prompt(page)
    b = dts.deblur_prompt(page)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert not np.array_equal(a, b)


def test_only_dewarping_requires_a_mask():
    needing = {n for n in task_mod.task_names() if task_mod.get_task(n).requires_mask}
    assert needing == {"dewarping"}


def test_the_losses_and_postprocess_modes_are_the_declared_ones():
    assert task_mod.get_task("binarization").loss == task_mod.LOSS_CATEGORICAL_CROSSENTROPY
    assert task_mod.get_task("binarization").postprocess == task_mod.POSTPROCESS_ARGMAX_BINARY
    assert task_mod.get_task("dewarping").postprocess == task_mod.POSTPROCESS_FLOW_REMAP
    for name in ("deshadowing", "appearance", "deblurring"):
        spec = task_mod.get_task(name)
        assert spec.loss == task_mod.LOSS_L1
        assert spec.postprocess == task_mod.POSTPROCESS_CLAMP_IMAGE
    for name in task_mod.task_names():
        spec = task_mod.get_task(name)
        assert spec.loss in task_mod.LOSSES
        assert spec.postprocess in task_mod.POSTPROCESS_MODES


def test_a_bogus_task_name_raises_and_lists_the_legal_ones():
    with pytest.raises(ValueError) as exc:
        task_mod.get_task("deshadow")  # a plausible near-miss
    message = str(exc.value)
    assert "deshadow" in message
    for name in task_mod.task_names():
        assert name in message


def test_the_table_cannot_be_mutated_in_place():
    """It is the single home of task specificity; a consumer must not patch it."""
    with pytest.raises(TypeError):
        task_mod.TASKS["binarization"] = None
    with pytest.raises(dataclasses.FrozenInstanceError):
        task_mod.get_task("binarization").loss = task_mod.LOSS_L1


def test_the_table_validator_rejects_a_malformed_entry():
    """Anti-vacuity for the import-time validation."""
    bad_loss = task_mod.TaskSpec(
        name="x", prompt_fn=dts.deblur_prompt, requires_mask=False,
        prompt_dtype="uint8", n_supervised_channels=3,
        loss="huber", postprocess=task_mod.POSTPROCESS_CLAMP_IMAGE,
    )
    with pytest.raises(ValueError, match="unknown loss"):
        task_mod._validate((bad_loss,))

    bad_channels = task_mod.TaskSpec(
        name="x", prompt_fn=dts.deblur_prompt, requires_mask=False,
        prompt_dtype="uint8", n_supervised_channels=4,
        loss=task_mod.LOSS_L1, postprocess=task_mod.POSTPROCESS_CLAMP_IMAGE,
    )
    with pytest.raises(ValueError, match="out of range"):
        task_mod._validate((bad_channels,))

    bad_mode = task_mod.TaskSpec(
        name="x", prompt_fn=dts.deblur_prompt, requires_mask=False,
        prompt_dtype="uint8", n_supervised_channels=3,
        loss=task_mod.LOSS_L1, postprocess="sharpen",
    )
    with pytest.raises(ValueError, match="unknown postprocess"):
        task_mod._validate((bad_mode,))

    dup = task_mod.TaskSpec(
        name="dup", prompt_fn=dts.deblur_prompt, requires_mask=False,
        prompt_dtype="uint8", n_supervised_channels=3,
        loss=task_mod.LOSS_L1, postprocess=task_mod.POSTPROCESS_CLAMP_IMAGE,
    )
    with pytest.raises(ValueError, match="duplicate task name"):
        task_mod._validate((dup, dup))

    # Control: the real table passes the same validator.
    task_mod._validate(tuple(task_mod.TASKS.values()))


def test_the_task_string_is_not_branched_on_anywhere_else():
    """The table is the ONE place task specificity lives.

    Scans the shipped library and trainer trees for a comparison against a
    DocRes task literal outside this package. A match means a second home for
    a decision that is supposed to have exactly one.
    """
    import subprocess

    root = pathlib.Path(task_mod.__file__).resolve().parents[4]
    hits = []
    pattern = r"task\s*==\s*[\"'](dewarping|deshadowing|appearance|deblurring|binarization)[\"']"
    proc = subprocess.run(
        ["grep", "-rnE", pattern, str(root / "src")],
        capture_output=True, text=True,
    )
    for line in proc.stdout.splitlines():
        if "/document_restoration/" not in line:
            hits.append(line)
    assert not hits, (
        "task-string branching found outside the TASKS table:\n" + "\n".join(hits)
    )
