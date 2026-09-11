r"""Guards for the UVDoc sidecar writer and the pipeline read path.

The defect class this exists to catch
-------------------------------------
A precomputed corpus is a SECOND producer of the rectifier's supervision, and
every way of getting it wrong is silent:

1. **The writer and the reader drift.** One stores ``f_gt`` where the other
   reads ``g``; both are ``(H, W, 2)`` float32 absolute-pixel maps, both are
   finite, both are in range, and the loss consumes either without complaint.
   The model then trains to predict the FORWARD map while every shape, dtype
   and range check stays green. So the staged halves are asserted with
   ``tests/test_datasets/test_document_rectification/backward_map_convention.py``
   -- the shared instrument that IS the contract, already applied to both
   dataset modules -- and not against a comment restating it here.

2. **The geometry cache silently stops working.** Keying the sidecars by RENDER
   instead of by GEOMETRY costs nothing observable: the staged corpus is
   identical, the training curve is identical, and the only symptom is that a
   full staging run takes ~6 h instead of ~1.2 h. Nothing about the OUTPUT can
   see it, so :class:`TestTheGeometryCacheIsUsed` counts ``load_geometry``
   calls directly and asserts M, not N.

3. **The cache is staged and then not used.** A reader that falls back to the
   archive whenever anything is slightly off looks perfect in every test that
   has the archive. :class:`TestTheStagedCorpusIsReadWithoutTheArchive` points
   ``uvdoc_root`` at a path that DOES NOT EXIST, so a fallback is an exception
   rather than a slow success.

Every always-run arm builds a stand-in UVDoc tree on disk. The arms that need
the real 27.5 GB ``UVDoc_final.zip`` are ``skipif``-gated and are the only ones
that touch it -- read-only, two renders, into ``tmp_path``.

Nothing here writes into repo-root ``results/`` or into the real staging root:
every sidecar goes to ``tmp_path``.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dl_techniques.datasets.document_rectification import (
    UVDocSource,
    load_geometry,
    load_image,
)
from train.doc_scanner import stage_uvdoc_samples as writer
from train.doc_scanner.common import (
    DEFAULT_UVDOC_ROOT,
    SOURCE_UVDOC,
    STAGE_RECTIFIER,
    STAGE_SEGMENTER,
    UVDOC_CACHE_GEOMETRY_DIR,
    UVDOC_CACHE_MANIFEST,
    UVDOC_CACHE_RENDER_DIR,
    UVDOC_CACHE_SCHEMA,
    DocScannerDataError,
    DocScannerTrainingConfig,
    collect_uvdoc_sample_ids,
    read_uvdoc_sidecar,
    require_training_data,
    uvdoc_cache_dir,
    uvdoc_sample,
)

from ...test_datasets.test_document_rectification.backward_map_convention import (  # noqa: TID252
    assert_is_backward_map,
    assert_is_forward_map,
)

# Imported rather than re-written: `_write_standin_corpus` and `_control_lattice`
# encode the one genuinely load-bearing fixture fact -- that a MATLAB v7.3 `.mat`
# is HDF5 with every axis REVERSED, so a stand-in must write `array.T`. A second
# copy of that here is a second place for the transpose to be right or wrong, and
# `uvdoc.py`'s whole orientation guard rests on it. This module adds only what
# that helper does not write: several geometries, several renders per geometry,
# and the `img/<id>.png` files (the geometry suite never loads an image).
from ...test_datasets.test_document_rectification.test_uvdoc import (  # noqa: TID252
    STANDIN_HEIGHT,
    STANDIN_WIDTH,
    _control_lattice,
    _write_standin_corpus,
)
from dl_techniques.datasets.document_rectification import synthetic_warp as sw
from dl_techniques.datasets.document_rectification import uvdoc as uvdoc_module

Image = pytest.importorskip(
    "PIL.Image",
    reason="UVDoc renders are PNGs; Pillow is in the 'data' extra",
)

#: Three geometries, seven renders. The whole point of the layout is that these
#: two numbers are DIFFERENT, so a per-render writer and a per-geometry writer
#: produce measurably different work for identical output.
STANDIN_GEOMETRIES = 3
STANDIN_RENDERS = 7

#: Target grid for the arms that go through `DocScannerTrainingConfig`: square,
#: a multiple of 32 (the config's own constraint), and small so three
#: densifications plus an inversion each cost well under a second.
STANDIN_SIZE = 32

#: Target grid for the arms that assert the CONVENTION. Deliberately NON-SQUARE
#: and deliberately not the same numbers as the square grid.
#:
#: MEASURED, and the reason this constant exists: at a square 32x32 the shared
#: `backward_map_convention` instrument accepts `g` in `f_gt`'s place on ALL
#: THREE stand-in geometries, because its swap check is a per-channel RANGE
#: check against the frame extent and at H == W the two extents are the same
#: number. A square convention arm here would therefore have passed a writer
#: that stored the two halves the wrong way round. See D-051.
CONVENTION_SIZE = (49, 83)

#: An 8-bit render agrees with `load_image`'s float32 to within half a grey
#: level. Asserted from BOTH sides -- see `TestTheSidecarRoundTrips`.
QUANTISATION_TOLERANCE = 1.0 / 255.0

UVDOC_ARCHIVE = DEFAULT_UVDOC_ROOT
requires_archive = pytest.mark.skipif(
    not os.path.exists(UVDOC_ARCHIVE),
    reason=f"needs the staged 27.5 GB UVDoc archive at {UVDOC_ARCHIVE}",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_render_png(tree, sample_id, rng):
    """Write one ``img/<sample_id>.png`` of the stand-in frame size."""
    folder = os.path.join(tree, uvdoc_module.IMAGE_DIR)
    os.makedirs(folder, exist_ok=True)
    pixels = rng.integers(
        0, 256, size=(STANDIN_HEIGHT, STANDIN_WIDTH, 3), dtype=np.uint8
    )
    Image.fromarray(pixels).save(os.path.join(folder, f"{sample_id}.png"))


@pytest.fixture
def standin_fanout(tmp_path):
    """A stand-in UVDoc tree with ``STANDIN_RENDERS`` renders over
    ``STANDIN_GEOMETRIES`` geometries, plus the id mapping it encodes.

    Returns ``(root, {sample_id: geom_name})``.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    rng = np.random.default_rng(20260910)

    geometries = []
    for index in range(STANDIN_GEOMETRIES):
        warp = sw.sample_warp(
            np.random.default_rng(100 + index),
            shear_stages=2,
            shear_amplitude=0.02,
            perspective_amplitude=0.02,
        )
        name = f"geo_{index}"
        lattice = _control_lattice(warp, STANDIN_HEIGHT, STANDIN_WIDTH)
        tree = _write_standin_corpus(
            str(root),
            lattice,
            STANDIN_HEIGHT,
            STANDIN_WIDTH,
            geometry=name,
            sample=f"{index:05d}",
        )
        geometries.append(name)

    mapping = {}
    for render in range(STANDIN_RENDERS):
        sample_id = f"{render:05d}"
        geometry = geometries[render % STANDIN_GEOMETRIES]
        mapping[sample_id] = geometry
        with open(
            os.path.join(
                tree, uvdoc_module.SAMPLE_METADATA_DIR, f"{sample_id}.json"
            ),
            "w",
        ) as handle:
            json.dump({"geom_name": geometry, "sample_id": sample_id}, handle)
        _write_render_png(tree, sample_id, rng)

    return str(root), mapping


@pytest.fixture
def staged(tmp_path, standin_fanout):
    """Stage the whole stand-in fan-out SQUARE and return
    ``(root, mapping, cache_root, report)``."""
    root, mapping = standin_fanout
    cache_root = tmp_path / "cache"
    report = writer.stage_samples(
        root, str(cache_root), size=(STANDIN_SIZE, STANDIN_SIZE), seed=7
    )
    return root, mapping, str(cache_root), report


@pytest.fixture
def staged_nonsquare(tmp_path, standin_fanout):
    """The same fan-out staged on a NON-SQUARE grid, plus its directory."""
    root, mapping = standin_fanout
    cache_root = tmp_path / "cache-nonsquare"
    writer.stage_samples(root, str(cache_root), size=CONVENTION_SIZE, seed=7)
    directory = cache_root / f"{CONVENTION_SIZE[0]}x{CONVENTION_SIZE[1]}"
    return root, mapping, directory


def _config(cache_root, **overrides):
    """A rectifier config pointed at a staged sidecar corpus."""
    base = dict(
        stage=STAGE_RECTIFIER,
        data_source=SOURCE_UVDOC,
        uvdoc_root=DEFAULT_UVDOC_ROOT,
        uvdoc_cache_root=str(cache_root),
        image_size=STANDIN_SIZE,
    )
    base.update(overrides)
    return DocScannerTrainingConfig(**base)


# ---------------------------------------------------------------------------
# 1. THE POINT OF THE STEP: what the writer wrote is what the reader reads.
# ---------------------------------------------------------------------------


class TestTheSidecarRoundTrips:
    """Values, not shapes. Asserted against the in-process producers."""

    def test_the_maps_are_bit_identical_to_load_geometry(self, staged):
        root, mapping, cache_root, _ = staged
        directory = Path(cache_root) / f"{STANDIN_SIZE}x{STANDIN_SIZE}"
        with UVDocSource(root) as source:
            for sample_id, geometry_name in mapping.items():
                reference = load_geometry(
                    source,
                    geometry_name,
                    np.random.default_rng(0),
                    size=(STANDIN_SIZE, STANDIN_SIZE),
                    strict=False,
                )
                _, f_gt, g, mask = read_uvdoc_sidecar(directory, sample_id)
                np.testing.assert_array_equal(
                    f_gt, reference.f_gt,
                    err_msg=f"staged f_gt for {sample_id} is not load_geometry's",
                )
                np.testing.assert_array_equal(g, reference.g)
                np.testing.assert_array_equal(mask, reference.mask)
                assert f_gt.dtype == np.float32 and g.dtype == np.float32
                assert mask.dtype == np.float32

    def test_the_render_is_eight_bit_and_says_so_from_both_sides(self, staged):
        root, mapping, cache_root, _ = staged
        directory = Path(cache_root) / f"{STANDIN_SIZE}x{STANDIN_SIZE}"
        sample_id = sorted(mapping)[0]
        with UVDocSource(root) as source:
            reference = load_image(
                source, sample_id, size=(STANDIN_SIZE, STANDIN_SIZE)
            )
        image, _, _, _ = read_uvdoc_sidecar(directory, sample_id)

        assert image.dtype == np.float32
        assert image.shape == (STANDIN_SIZE, STANDIN_SIZE, 3)
        deviation = float(np.abs(image - reference).max())
        assert deviation <= QUANTISATION_TOLERANCE, (
            "the staged render must agree with load_image to within one grey "
            f"level; worst element differs by {deviation:.6f}"
        )
        # The other half of D-050, so nobody later "tightens" this to exact and
        # quietly quadruples the corpus to make it pass.
        assert deviation > 0.0, (
            "the staged render is EXACTLY load_image's output, so the 8-bit "
            "storage this bound exists to describe is not happening -- either "
            "the writer changed to float32 or the fixture is degenerate"
        )

    def test_reading_the_stored_bytes_back_is_exact(self, staged):
        """The lossy step is at STAGING; the artefact itself round-trips."""
        root, mapping, cache_root, _ = staged
        directory = Path(cache_root) / f"{STANDIN_SIZE}x{STANDIN_SIZE}"
        sample_id = sorted(mapping)[0]
        path = directory / UVDOC_CACHE_RENDER_DIR / f"{sample_id}.npz"
        with np.load(path) as payload:
            stored = np.asarray(payload["image"])
        image, _, _, _ = read_uvdoc_sidecar(directory, sample_id)
        assert stored.dtype == np.uint8
        np.testing.assert_array_equal(
            (image * 255.0).round().astype(np.uint8), stored
        )


class TestTheStagedMapsObeyTheSharedConvention:
    """Asserted with the shared instrument, on a NON-SQUARE grid.

    Non-square is not decoration: the anti-vacuity arm below MEASURES that the
    same instrument is blind to an ``f_gt`` / ``g`` swap at 32x32 and sees it
    at 49x83, which is the whole reason `stage_samples` takes a size pair.
    """

    def test_f_gt_is_a_backward_map_and_g_is_a_forward_map(
            self, staged_nonsquare
    ):
        _, mapping, directory = staged_nonsquare
        height, width = CONVENTION_SIZE
        for sample_id in sorted(mapping):
            _, f_gt, g, _ = read_uvdoc_sidecar(directory, sample_id)
            assert_is_backward_map(
                f_gt, height, width, f"staged UVDoc sidecar {sample_id}"
            )
            assert_is_forward_map(
                g, height, width, f"staged UVDoc sidecar {sample_id}"
            )

    def test_the_mask_is_binary(self, staged):
        _, mapping, cache_root, _ = staged
        directory = Path(cache_root) / f"{STANDIN_SIZE}x{STANDIN_SIZE}"
        _, _, _, mask = read_uvdoc_sidecar(directory, sorted(mapping)[0])
        assert mask.shape == (STANDIN_SIZE, STANDIN_SIZE, 1)
        assert set(np.unique(mask)).issubset({0.0, 1.0})
        assert 0.0 < float(mask.mean()) < 1.0, (
            "a mask that is all page or all background pins nothing"
        )

    def test_a_channel_swap_is_visible_here(self, staged_nonsquare):
        """Anti-vacuity: the instrument must REJECT an ``(x, y)`` swap.

        Without this the convention arm above could be passing on shape, dtype
        and finiteness alone.
        """
        _, mapping, directory = staged_nonsquare
        height, width = CONVENTION_SIZE
        caught = 0
        for sample_id in sorted(mapping):
            _, f_gt, _, _ = read_uvdoc_sidecar(directory, sample_id)
            try:
                assert_is_backward_map(
                    f_gt[..., ::-1], height, width, "channel-swapped f_gt"
                )
            except AssertionError:
                caught += 1
        assert caught == len(mapping), (
            f"a swapped (x, y) f_gt was accepted on {len(mapping) - caught} of "
            f"{len(mapping)} staged renders at {height}x{width}"
        )

    def test_the_same_swap_is_MEASURED_invisible_at_a_square_size(self, staged):
        """The pinned negative result that forces the non-square fixture.

        This is not a defect in the instrument: it documents its own swap check
        as a per-channel RANGE check and says outright that at ``H == W``
        nothing can catch a swap. This arm pins that it is still true for THIS
        producer, so nobody simplifies ``stage_samples`` back to one square
        ``image_size`` int and quietly removes the only power the arm above
        has. See D-051.
        """
        _, mapping, cache_root, _ = staged
        directory = Path(cache_root) / f"{STANDIN_SIZE}x{STANDIN_SIZE}"
        blind = 0
        for sample_id in sorted(mapping):
            _, f_gt, _, _ = read_uvdoc_sidecar(directory, sample_id)
            try:
                assert_is_backward_map(
                    f_gt[..., ::-1], STANDIN_SIZE, STANDIN_SIZE,
                    "channel-swapped f_gt",
                )
                blind += 1
            except AssertionError:
                pass
        assert blind == len(mapping), (
            "the square-size blindness this fixture works around no longer "
            f"holds ({blind} of {len(mapping)} accepted) -- re-measure D-051 "
            "before relying on either arm"
        )

    def test_the_instrument_is_MEASURED_blind_to_an_f_gt_g_direction_swap(
            self, staged_nonsquare
    ):
        """The OTHER pinned negative, and the more dangerous one.

        ``g`` is ``(H, W, 2)`` float32 with channel 0 an x in the width's range
        and channel 1 a y in the height's -- exactly like ``f_gt``. The shared
        instrument therefore ACCEPTS ``g`` in ``f_gt``'s place, at every size,
        square or not: it pins units and channel order, never direction. So a
        writer that stored the two halves the wrong way round would satisfy
        every convention arm in this class.

        What actually pins direction is
        ``TestTheSidecarRoundTrips::test_the_maps_are_bit_identical_to_load_geometry``,
        which compares element for element against ``load_geometry``'s own
        ``f_gt`` and ``g``. Deleting that arm removes the ONLY guard here that
        can see a direction swap; this test exists so that fact is written down
        where someone deleting it will read it.
        """
        _, mapping, directory = staged_nonsquare
        height, width = CONVENTION_SIZE
        accepted = 0
        for sample_id in sorted(mapping):
            _, _, g, _ = read_uvdoc_sidecar(directory, sample_id)
            try:
                assert_is_backward_map(
                    g, height, width, "g mislabelled as f_gt"
                )
                accepted += 1
            except AssertionError:
                pass
        assert accepted == len(mapping), (
            "the convention instrument now rejects g in f_gt's place on "
            f"{len(mapping) - accepted} of {len(mapping)} renders. That is an "
            "IMPROVEMENT, not a failure -- re-measure and rewrite this pin "
            "rather than weakening anything."
        )


# ---------------------------------------------------------------------------
# 2. THE OTHER POINT: the per-geometry cache is real, and it is measured.
# ---------------------------------------------------------------------------


class TestTheGeometryCacheIsUsed:
    """M densifications for N renders, counted -- not inferred from output."""

    def test_staging_n_renders_performs_m_densifications(
            self, tmp_path, standin_fanout, monkeypatch
    ):
        root, mapping = standin_fanout
        calls = []
        real = writer.load_geometry

        def counting(source, name, rng, **kwargs):
            calls.append(name)
            return real(source, name, rng, **kwargs)

        monkeypatch.setattr(writer, "load_geometry", counting)
        report = writer.stage_samples(
            root, str(tmp_path / "cache"),
            size=(STANDIN_SIZE, STANDIN_SIZE), seed=7,
        )

        assert report.requested == STANDIN_RENDERS
        assert report.distinct_geometries == STANDIN_GEOMETRIES
        assert len(calls) == STANDIN_GEOMETRIES, (
            f"{STANDIN_RENDERS} renders over {STANDIN_GEOMETRIES} geometries "
            f"cost {len(calls)} densifications: {calls}. A per-render layout "
            "would cost 7 here and 20,000 on the real corpus (~6 h instead of "
            "~1.2 h) with byte-identical output, so only this count can see it."
        )
        assert sorted(set(calls)) == sorted(set(mapping.values()))
        assert report.densifications == STANDIN_GEOMETRIES
        assert report.densifications < report.requested

    def test_limit_by_geometry_stages_whole_geometries(
            self, tmp_path, standin_fanout, monkeypatch
    ):
        """The knob that makes a SUBSET exercise the cache (D-052).

        ``--limit`` takes a prefix of render ids, and on the real corpus that
        prefix shares no geometries at all -- measured: the first 4,032 ids
        span 4,032 distinct geometries, one apiece. ``--limit-geometries``
        takes whole geometries instead, so a subset carries the corpus's real
        fan-out and pays one densification per geometry.
        """
        root, mapping = standin_fanout
        calls = []
        real = writer.load_geometry
        monkeypatch.setattr(
            writer, "load_geometry",
            lambda source, name, rng, **kw: (
                calls.append(name) or real(source, name, rng, **kw)
            ),
        )
        report = writer.stage_samples(
            root, str(tmp_path / "cache"),
            size=(STANDIN_SIZE, STANDIN_SIZE), seed=7, limit_geometries=1,
        )
        expected = [i for i, g in mapping.items() if g == "geo_0"]
        assert len(expected) > 1, "the fixture must fan geo_0 out"
        assert report.requested == len(expected)
        assert report.distinct_geometries == 1
        assert len(calls) == 1
        assert report.densifications < report.requested, (
            "a --limit-geometries subset must cost FEWER densifications than "
            "renders; that is the whole point of the flag"
        )

    def test_every_render_still_gets_its_own_sidecar(self, staged):
        _, mapping, cache_root, report = staged
        directory = Path(cache_root) / f"{STANDIN_SIZE}x{STANDIN_SIZE}"
        renders = sorted(
            p.stem for p in (directory / UVDOC_CACHE_RENDER_DIR).glob("*.npz")
        )
        geometries = sorted(
            p.stem for p in (directory / UVDOC_CACHE_GEOMETRY_DIR).glob("*.npz")
        )
        assert renders == sorted(mapping)
        assert geometries == sorted(set(mapping.values()))
        assert report.renders_written == STANDIN_RENDERS
        assert report.geometries_written == STANDIN_GEOMETRIES

    def test_two_renders_of_one_geometry_share_their_maps(self, staged):
        _, mapping, cache_root, _ = staged
        directory = Path(cache_root) / f"{STANDIN_SIZE}x{STANDIN_SIZE}"
        by_geometry = {}
        for sample_id, geometry in mapping.items():
            by_geometry.setdefault(geometry, []).append(sample_id)
        shared = [ids for ids in by_geometry.values() if len(ids) > 1]
        assert shared, "the fixture must fan at least one geometry out"
        for ids in shared:
            first = read_uvdoc_sidecar(directory, ids[0])
            for other in ids[1:]:
                rest = read_uvdoc_sidecar(directory, other)
                np.testing.assert_array_equal(first[1], rest[1])
                np.testing.assert_array_equal(first[2], rest[2])
                assert not np.array_equal(first[0], rest[0]), (
                    "two DIFFERENT renders of one geometry must still carry "
                    "different images"
                )


# ---------------------------------------------------------------------------
# 3. Additive, idempotent, resumable.
# ---------------------------------------------------------------------------


class TestStagingIsIdempotent:
    def test_a_second_run_writes_nothing(self, staged, monkeypatch):
        root, _, cache_root, first = staged
        directory = Path(cache_root) / f"{STANDIN_SIZE}x{STANDIN_SIZE}"
        before = {
            path: (path.stat().st_size, path.stat().st_mtime_ns)
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        }

        calls = []
        monkeypatch.setattr(
            writer, "load_geometry",
            lambda *args, **kwargs: calls.append(args[1]),
        )
        second = writer.stage_samples(
            root, cache_root, size=(STANDIN_SIZE, STANDIN_SIZE), seed=7
        )

        assert calls == [], "a no-op re-run must not densify anything"
        assert second.renders_written == 0
        assert second.geometries_written == 0
        assert second.bytes_written == 0
        assert second.renders_skipped == first.renders_written
        assert second.geometries_skipped == first.geometries_written

        after = {
            path: (path.stat().st_size, path.stat().st_mtime_ns)
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        }
        assert after == before, "a no-op re-run rewrote or added a file"

    def test_a_limited_run_extends_rather_than_replaces(
            self, tmp_path, standin_fanout
    ):
        root, mapping = standin_fanout
        cache_root = tmp_path / "cache"
        partial = writer.stage_samples(
            root, str(cache_root), size=(STANDIN_SIZE, STANDIN_SIZE), seed=7,
            limit=2,
        )
        assert partial.renders_written == 2

        full = writer.stage_samples(
            root, str(cache_root), size=(STANDIN_SIZE, STANDIN_SIZE), seed=7
        )
        assert full.renders_skipped == 2
        assert full.renders_written == STANDIN_RENDERS - 2
        directory = Path(cache_root) / f"{STANDIN_SIZE}x{STANDIN_SIZE}"
        assert len(
            list((directory / UVDOC_CACHE_RENDER_DIR).glob("*.npz"))
        ) == STANDIN_RENDERS

    def test_a_stale_temporary_file_is_not_read_as_a_sidecar(self, staged):
        _, mapping, cache_root, _ = staged
        directory = Path(cache_root) / f"{STANDIN_SIZE}x{STANDIN_SIZE}"
        (directory / UVDOC_CACHE_RENDER_DIR / "99999.npz.tmp").write_bytes(
            b"not an npz"
        )
        config = _config(cache_root)
        assert "99999" not in collect_uvdoc_sample_ids(config), (
            "a `.npz.tmp` left by a killed run must not enter the worklist"
        )


class TestDryRunAndDiskBudget:
    def test_dry_run_writes_nothing_at_all(self, tmp_path, standin_fanout):
        root, _ = standin_fanout
        cache_root = tmp_path / "cache"
        report = writer.stage_samples(
            root, str(cache_root), size=(STANDIN_SIZE, STANDIN_SIZE),
            dry_run=True,
        )
        assert report.dry_run is True
        assert report.requested == STANDIN_RENDERS
        assert report.distinct_geometries == STANDIN_GEOMETRIES
        assert report.renders_written == 0
        assert report.densifications == 0
        assert not cache_root.exists(), (
            f"--dry-run created {cache_root}; it must allocate nothing"
        )

    def test_an_impossible_floor_refuses_before_writing(
            self, tmp_path, standin_fanout
    ):
        root, _ = standin_fanout
        cache_root = tmp_path / "cache"
        with pytest.raises(writer.DiskBudgetError) as info:
            writer.stage_samples(
                root, str(cache_root), size=(STANDIN_SIZE, STANDIN_SIZE),
                min_free_bytes=1 << 60,
            )
        assert "refusing to stage" in str(info.value)
        assert not cache_root.exists()


class TestTheManifestIsAContract:
    def test_a_missing_manifest_is_not_a_cache(self, tmp_path):
        config = _config(tmp_path / "nothing")
        assert uvdoc_cache_dir(config) is None

    def test_an_empty_cache_root_forces_the_archive_path(self, staged):
        _, _, cache_root, _ = staged
        assert uvdoc_cache_dir(_config(cache_root)) is not None
        assert uvdoc_cache_dir(_config("")) is None

    def test_a_foreign_schema_is_refused_not_ignored(self, staged):
        _, _, cache_root, _ = staged
        directory = Path(cache_root) / f"{STANDIN_SIZE}x{STANDIN_SIZE}"
        path = directory / UVDOC_CACHE_MANIFEST
        manifest = json.loads(path.read_text())
        manifest["schema"] = UVDOC_CACHE_SCHEMA + 1
        path.write_text(json.dumps(manifest))
        with pytest.raises(DocScannerDataError) as info:
            uvdoc_cache_dir(_config(cache_root))
        assert "schema" in str(info.value)

    def test_a_manifest_that_disagrees_with_its_own_directory_is_refused(
            self, staged
    ):
        _, _, cache_root, _ = staged
        directory = Path(cache_root) / f"{STANDIN_SIZE}x{STANDIN_SIZE}"
        path = directory / UVDOC_CACHE_MANIFEST
        manifest = json.loads(path.read_text())
        manifest["height"] = STANDIN_SIZE * 2
        path.write_text(json.dumps(manifest))
        with pytest.raises(DocScannerDataError):
            uvdoc_cache_dir(_config(cache_root))

    def test_the_writer_refuses_a_foreign_schema_too(self, staged):
        root, _, cache_root, _ = staged
        directory = Path(cache_root) / f"{STANDIN_SIZE}x{STANDIN_SIZE}"
        path = directory / UVDOC_CACHE_MANIFEST
        manifest = json.loads(path.read_text())
        manifest["schema"] = UVDOC_CACHE_SCHEMA + 1
        path.write_text(json.dumps(manifest))
        with pytest.raises(DocScannerDataError):
            writer.stage_samples(
                root, cache_root, size=(STANDIN_SIZE, STANDIN_SIZE), limit=1
            )


# ---------------------------------------------------------------------------
# 4. The pipeline actually reads it -- proven by removing the archive.
# ---------------------------------------------------------------------------


class TestTheStagedCorpusIsReadWithoutTheArchive:
    """``uvdoc_root`` points at a path that does not exist, on purpose."""

    @pytest.fixture
    def config(self, staged):
        _, _, cache_root, _ = staged
        return _config(
            cache_root, uvdoc_root="/nonexistent/UVDoc_final.zip"
        )

    def test_the_worklist_comes_from_the_sidecars(self, config, staged):
        _, mapping, _, _ = staged
        assert collect_uvdoc_sample_ids(config) == sorted(mapping)

    def test_the_gate_passes_with_no_archive_on_disk(self, config):
        require_training_data(config)

    def test_the_rectifier_target_is_the_staged_f_gt_and_g(
            self, config, staged
    ):
        _, mapping, cache_root, _ = staged
        directory = Path(cache_root) / f"{STANDIN_SIZE}x{STANDIN_SIZE}"
        ids = collect_uvdoc_sample_ids(config)
        image, target = uvdoc_sample(config, ids, 0)
        expected_image, f_gt, g, _ = read_uvdoc_sidecar(directory, ids[0])

        assert target.shape == (STANDIN_SIZE, STANDIN_SIZE, 4)
        np.testing.assert_array_equal(image, expected_image)
        np.testing.assert_array_equal(target[..., :2], f_gt)
        np.testing.assert_array_equal(target[..., 2:], g)

    def test_the_segmenter_target_is_the_staged_mask(self, config, staged):
        _, _, cache_root, _ = staged
        directory = Path(cache_root) / f"{STANDIN_SIZE}x{STANDIN_SIZE}"
        segmenter = replace(config, stage=STAGE_SEGMENTER)
        ids = collect_uvdoc_sample_ids(segmenter)
        _, target = uvdoc_sample(segmenter, ids, 0)
        _, _, _, mask = read_uvdoc_sidecar(directory, ids[0])
        np.testing.assert_array_equal(target, mask)

    def test_a_size_with_no_sidecars_does_not_silently_use_another_size(
            self, staged
    ):
        _, _, cache_root, _ = staged
        other = _config(
            cache_root,
            uvdoc_root="/nonexistent/UVDoc_final.zip",
            image_size=STANDIN_SIZE * 2,
        )
        assert uvdoc_cache_dir(other) is None
        with pytest.raises(Exception):
            require_training_data(other)


# ---------------------------------------------------------------------------
# 5. The CLI.
# ---------------------------------------------------------------------------


class TestTheCli:
    def test_help_prints_usage_and_allocates_nothing(self, capsys):
        with pytest.raises(SystemExit) as info:
            writer.main(["--help"])
        assert info.value.code == 0
        assert "usage:" in capsys.readouterr().out

    def test_main_stages_and_returns_zero(self, tmp_path, standin_fanout):
        root, mapping = standin_fanout
        cache_root = tmp_path / "cache"
        code = writer.main([
            "--uvdoc-root", root,
            "--cache-root", str(cache_root),
            "--image-size", str(STANDIN_SIZE),
            "--limit", "4",
        ])
        assert code == 0
        directory = Path(cache_root) / f"{STANDIN_SIZE}x{STANDIN_SIZE}"
        assert len(
            list((directory / UVDOC_CACHE_RENDER_DIR).glob("*.npz"))
        ) == 4

    def test_an_unreadable_corpus_returns_one_rather_than_raising(
            self, tmp_path
    ):
        assert writer.main([
            "--uvdoc-root", str(tmp_path / "nope.zip"),
            "--cache-root", str(tmp_path / "cache"),
            "--dry-run",
        ]) == 1


# ---------------------------------------------------------------------------
# 6. The real archive. Read-only, two renders, into tmp_path.
# ---------------------------------------------------------------------------


@requires_archive
class TestTheRealArchiveStagesAndReadsBack:
    def test_two_real_renders_stage_and_round_trip(self, tmp_path):
        cache_root = tmp_path / "cache"
        report = writer.stage_samples(
            UVDOC_ARCHIVE, str(cache_root), size=(288, 288), seed=42, limit=2
        )
        assert report.requested == 2
        assert report.renders_written == 2
        assert report.densifications == report.geometries_written
        assert report.densifications <= 2

        directory = Path(cache_root) / "288x288"
        ids = sorted(
            p.stem for p in (directory / UVDOC_CACHE_RENDER_DIR).glob("*.npz")
        )
        for sample_id in ids:
            image, f_gt, g, mask = read_uvdoc_sidecar(directory, sample_id)
            assert image.shape == (288, 288, 3)
            assert_is_backward_map(
                f_gt, 288, 288, f"real UVDoc sidecar {sample_id}"
            )
            assert_is_forward_map(
                g, 288, 288, f"real UVDoc sidecar {sample_id}"
            )
            assert mask.shape == (288, 288, 1)
            assert 0.0 < float(mask.mean()) < 1.0

    def test_a_limit_prefix_really_does_span_one_geometry_per_render(self):
        """The measurement behind D-052, re-derived rather than quoted.

        If UVDoc were ever re-published with ids that group by geometry, this
        goes RED and the ``--limit`` warning in ``--help`` becomes wrong.
        """
        with UVDocSource(UVDOC_ARCHIVE) as source:
            ids = sorted(source.sample_ids())[:64]
            geometries = {source.geometry_for_sample(i) for i in ids}
        assert len(geometries) == len(ids), (
            f"the first {len(ids)} render ids now span {len(geometries)} "
            "geometries, not one apiece -- re-measure D-052"
        )

    def test_the_archive_is_never_modified(self, tmp_path):
        before = os.stat(UVDOC_ARCHIVE)
        writer.stage_samples(
            UVDOC_ARCHIVE, str(tmp_path / "cache"), size=(288, 288), limit=1
        )
        after = os.stat(UVDOC_ARCHIVE)
        assert (before.st_size, before.st_mtime_ns) == (
            after.st_size, after.st_mtime_ns
        ), "staging touched the 27.5 GB archive; it is read-only input"
