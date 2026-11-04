"""ADE20K, the full 2021_17_01 release: class ids from ``_seg.png`` and a reduced class set.

The release on disk is NOT the SceneParse150 benchmark: ``index_ade20k.pkl`` lists 3,688 object
classes and every ``<stem>_seg.png`` is an RGB picture whose red and green channels code the class of
a pixel as ``(R // 10) * 256 + G`` (0 is unlabelled) and whose blue channel codes the instance.
This module reads that coding and turns it into a mask of ``k`` classes, with ``255`` for everything
else. It needs numpy and PIL only; it never imports TensorFlow and never unpickles anything.

Interface contract
------------------
``decode_class_ids(seg)``
    :param seg: ``uint8`` ``(H, W, 3)`` array of a ``_seg.png``.
    :return: ``int32`` ``(H, W)`` ids in ``0 .. MAX_CLASS_ID`` (0 is unlabelled).
    :raises AdeError: for an array that is not ``(H, W, 3)`` or more channels.

``list_release(root)``
    :param root: the release directory (``images/ADE/training`` and ``images/ADE/validation`` under it).
    :return: ``{"train": [AdePair, ...], "val": [AdePair, ...]}``, each sorted by stem; a pair is a
        ``<stem>.jpg`` and its ``<stem>_seg.png``. Per-image instance directories are not entered.
    :raises AdeError: for a root with no ``images/ADE/<split>`` directory or a split with no pair.

``read_object_names(objects_txt)``
    :return: ``{class id: name}`` from the release's ``objects.txt`` (tab separated, latin-1; the
        column "Name index" is the id in a ``_seg.png``). The file lists the classes the release counts
        objects for (1,599 of the 3,688), so a class absent from it has no name here.

``class_pixel_counts(seg_paths, workers=1)``
    :return: ``int64`` ``(MAX_CLASS_ID + 1,)``, the pixels of every class over the files. ``workers``
        above 1 reads the files in that many processes.

``official_mapping(names)`` / ``frequency_mapping(counts, names, k)``
    :return: a :class:`ClassMapping`. ``official`` is the MIT SceneParse150 set (``SCENE_PARSE_150``);
        it is refused, naming the class, if the release's name for an id shares no synonym with the
        benchmark's name for the class. ``frequency`` is the ``k`` classes with the most pixels
        (ties: the smaller id), most frequent first. :raises AdeError: ``k`` above the number of
        classes with a pixel, or a name check that fails.

``ClassMapping.lut()``
    :return: ``uint8`` ``(MAX_CLASS_ID + 1,)`` that maps a release id to its class index, ``255`` for
        every id the mapping does not name and for 0.

``convert_seg(seg, lut)``
    :return: ``(mask, pixels)``: the ``uint8`` ``(H, W)`` mask and the ``int64`` ``(256,)`` count of each value.
"""

import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image

#: The red channel is ``10 * (id // 256) + (0 to 9)`` and 255 is its top, so ids stop at ``25 * 256 + 255``.
MAX_CLASS_ID = 25 * 256 + 255

#: The mask value of a pixel that belongs to none of the classes (unlabelled, or a class left out).
VOID_INDEX = 255

SEG_SUFFIX = "_seg.png"
#: A pair's stem: ``ADE_train_00000001``, ``ADE_frame_00000316`` (video frames inside ``training``) or ``ADE_val_00000001``.
_STEM = re.compile(r"^(ADE_(?:train|frame|val)_\d+)$")
_SPLIT_DIRS = {"train": "training", "val": "validation"}


class AdeError(ValueError):
    """A release that cannot be read as ADE20K, or a mapping that cannot be built; the message says why."""


@dataclass(frozen=True)
class AdePair:
    """One annotated image of the release: its ``stem`` and the two absolute paths."""

    stem: str
    image: str
    seg: str


@dataclass(frozen=True)
class ClassMapping:
    """A reduced class set: ``kind``, one name per class, and the release ids each class takes.

    :param kind: ``"official"`` or ``"frequency"``.
    :param class_names: the name of class ``i`` is ``class_names[i]`` (the mask value ``i``).
    :param source_ids: ``source_ids[i]`` are the release ids that become class ``i``.
    :param source_names: the release's name of each of those ids (``""`` if ``objects.txt`` has none).
    """

    kind: str
    class_names: Tuple[str, ...]
    source_ids: Tuple[Tuple[int, ...], ...]
    source_names: Tuple[Tuple[str, ...], ...]

    def lut(self) -> np.ndarray:
        """Return the ``uint8`` table from a release id to a class index (``VOID_INDEX`` elsewhere)."""
        table = np.full(MAX_CLASS_ID + 1, VOID_INDEX, dtype=np.uint8)
        for index, ids in enumerate(self.source_ids):
            table[list(ids)] = index
        return table


# ---------------------------------------------------------------------
# The coding of a _seg.png
# ---------------------------------------------------------------------

def decode_class_ids(seg: np.ndarray) -> np.ndarray:
    """Return the release class id of every pixel of an RGB ``_seg.png`` array; see the module contract."""
    if seg.ndim != 3 or seg.shape[2] < 3:
        raise AdeError(f"a _seg.png decodes to (H, W, 3), got an array of shape {tuple(seg.shape)}")
    # DECISION plan-2026-11-03T221936-98b866fb/D-017
    # The class is `(R // 10) * 256 + G`: the red channel holds ten times the high byte. Do NOT read
    # it as `R * 256 + G` (ids of 100 and more, off by a factor of ten) and do NOT round `R / 10`
    # as MATLAB's `uint16(R) / 10` does in the benchmark's convertFromADE.m: R is a multiple of 10 in
    # every file measured, so the floor and the rounding agree, and the floor is the release's own
    # starter code (utils_ade20k.py) and README.
    red = seg[..., 0].astype(np.int32)
    return (red // 10) * 256 + seg[..., 1].astype(np.int32)


def read_seg(path: str) -> np.ndarray:
    """Return the class ids of one ``_seg.png`` file."""
    try:
        with Image.open(path) as image:
            mode = image.mode
            pixels = np.asarray(image) if mode == "RGB" else None
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise AdeError(f"{path!r} cannot be decoded as an image ({exc})") from exc
    if pixels is None:
        raise AdeError(f"{path!r} is a {mode} image; a _seg.png is RGB")
    return decode_class_ids(pixels)


# ---------------------------------------------------------------------
# The release on disk
# ---------------------------------------------------------------------

def _walk_split(directory: str) -> List[Tuple[str, str]]:
    """Return ``(stem, directory)`` of every ``_seg.png`` under ``directory``, leaving instance directories alone."""
    found: List[Tuple[str, str]] = []
    for here, subdirectories, files in os.walk(directory):
        # An image's own directory (``ADE_train_00000001/``) holds its instance masks, by the thousand.
        subdirectories[:] = sorted(name for name in subdirectories if not _STEM.match(name))
        for name in files:
            if name.endswith(SEG_SUFFIX) and _STEM.match(name[: -len(SEG_SUFFIX)]):
                found.append((name[: -len(SEG_SUFFIX)], here))
    return found


def list_release(root: str) -> Dict[str, List[AdePair]]:
    """Return the ``train`` and ``val`` pairs of the release at ``root``; see the module contract."""
    pairs: Dict[str, List[AdePair]] = {}
    for split, folder in _SPLIT_DIRS.items():
        directory = os.path.join(root, "images", "ADE", folder)
        if not os.path.isdir(directory):
            raise AdeError(f"{directory!r} is not a directory: {root!r} is not the ADE20K 2021_17_01 release "
                           "(it holds images/ADE/training and images/ADE/validation)")
        found = []
        for stem, here in sorted(_walk_split(directory)):
            image = os.path.join(here, stem + ".jpg")
            if not os.path.isfile(image):
                raise AdeError(f"{os.path.join(here, stem + SEG_SUFFIX)!r} has no image {image!r} beside it")
            found.append(AdePair(stem, image, os.path.join(here, stem + SEG_SUFFIX)))
        if not found:
            raise AdeError(f"{directory!r} holds no <stem>{SEG_SUFFIX} file")
        stems = [pair.stem for pair in found]
        if len(set(stems)) != len(stems):
            raise AdeError(f"{directory!r} holds one stem twice; a stem names one mask")
        pairs[split] = found
    return pairs


def read_object_names(objects_txt: str) -> Dict[int, str]:
    """Return ``{id: name}`` from the release's ``objects.txt``; see the module contract."""
    try:
        with open(objects_txt, "rb") as handle:
            rows = handle.read().decode("latin-1").splitlines()
    except OSError as exc:
        raise AdeError(f"{objects_txt!r} cannot be read ({exc})") from exc
    names: Dict[int, str] = {}
    for number, row in enumerate(rows[1:], start=2):
        cells = row.split("\t")
        try:
            names[int(cells[1])] = cells[0].strip()
        except (IndexError, ValueError) as exc:
            raise AdeError(f"{objects_txt!r} line {number} is not 'name<TAB>id<TAB>...': {row[:60]!r}") from exc
    if not names:
        raise AdeError(f"{objects_txt!r} lists no class")
    return names


# ---------------------------------------------------------------------
# Pixel counts
# ---------------------------------------------------------------------

def _count_file(path: str) -> np.ndarray:
    return np.bincount(read_seg(path).ravel(), minlength=MAX_CLASS_ID + 1).astype(np.int64)


def class_pixel_counts(seg_paths: Sequence[str], workers: int = 1) -> np.ndarray:
    """Return the pixels of every release id over ``seg_paths``; see the module contract."""
    total = np.zeros(MAX_CLASS_ID + 1, dtype=np.int64)
    if workers <= 1:
        for path in seg_paths:
            total += _count_file(path)
        return total
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for counts in pool.map(_count_file, seg_paths, chunksize=16):
            total += counts
    return total


# ---------------------------------------------------------------------
# The class sets
# ---------------------------------------------------------------------

# The MIT SceneParse150 classes, in the benchmark's order (class index 1..150 is row 0..149 here, so a
# row's position is the 0-based mask value), each with the release ids its pixels come from.
# Names: objectInfo150.txt of github.com/CSAILVision/sceneparsing (commit e202229). Ids: convertFromADE/mapFromADE.txt of the
# same commit. The ids are the "Name index" of ADE20K's objects.txt: for all 150 classes the 2021
# release's name for at least one id is a synonym of the benchmark's name (checked on every use by
# `official_mapping`), so the ids need no translation from the 2016 release the benchmark was cut from.
SCENE_PARSE_150: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("wall", (2978,)),
    ("building, edifice", (312,)),
    ("sky", (2420,)),
    ("floor, flooring", (976,)),
    ("tree", (2855,)),
    ("ceiling", (447,)),
    ("road, route", (2131,)),
    ("bed ", (165,)),
    ("windowpane, window ", (3055,)),
    ("grass", (1125,)),
    ("cabinet", (350,)),
    ("sidewalk, pavement", (2377,)),
    ("person, individual, someone, somebody, mortal, soul", (1831,)),
    ("earth, ground", (838,)),
    ("door, double door", (774, 783)),
    ("table", (2684,)),
    ("mountain, mount", (1610,)),
    ("plant, flora, plant life", (1910,)),
    ("curtain, drape, drapery, mantle, pall", (687,)),
    ("chair", (471,)),
    ("car, auto, automobile, machine, motorcar", (401,)),
    ("water", (2994,)),
    ("painting, picture", (1735,)),
    ("sofa, couch, lounge", (2473,)),
    ("shelf", (2329,)),
    ("house", (1276,)),
    ("sea", (2264,)),
    ("mirror", (1564,)),
    ("rug, carpet, carpeting", (2178,)),
    ("field", (913,)),
    ("armchair", (57,)),
    ("seat", (2272,)),
    ("fence, fencing", (907,)),
    ("desk", (724,)),
    ("rock, stone", (2138,)),
    ("wardrobe, closet, press", (533, 2985)),
    ("lamp", (1395,)),
    ("bathtub, bathing tub, bath, tub", (155,)),
    ("railing, rail", (2053,)),
    ("cushion", (689,)),
    ("base, pedestal, stand", (137,)),
    ("box", (266,)),
    ("column, pillar", (581,)),
    ("signboard, sign", (2380,)),
    ("chest of drawers, chest, bureau, dresser", (491,)),
    ("counter", (627,)),
    ("sand", (2212,)),
    ("sink", (2388,)),
    ("skyscraper", (2423,)),
    ("fireplace, hearth, open fireplace", (943,)),
    ("refrigerator, icebox", (2096,)),
    ("grandstand, covered stand", (1121,)),
    ("path", (1788,)),
    ("stairs, steps", (2530,)),
    ("runway", (2185,)),
    ("case, display case, showcase, vitrine", (420,)),
    ("pool table, billiard table, snooker table", (1948,)),
    ("pillow", (1869,)),
    ("screen door, screen", (2251,)),
    ("stairway, staircase", (2531,)),
    ("river", (2128,)),
    ("bridge, span", (294,)),
    ("bookcase", (239,)),
    ("blind, screen", (212,)),
    ("coffee table, cocktail table", (571,)),
    ("toilet, can, commode, crapper, pot, potty, stool, throne", (2793,)),
    ("flower", (978,)),
    ("book", (236,)),
    ("hill", (1240,)),
    ("bench", (181,)),
    ("countertop", (629,)),
    ("stove, kitchen stove, range, kitchen range, cooking stove", (2598,)),
    ("palm, palm tree", (1744,)),
    ("kitchen island", (1374,)),
    ("computer, computing machine, computing device, data processor, electronic computer, information processing system", (591,)),
    ("swivel chair", (2679,)),
    ("boat", (223,)),
    ("bar", (123,)),
    ("arcade machine", (47,)),
    ("hovel, hut, hutch, shack, shanty", (1282,)),
    ("bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle", (327,)),
    ("towel", (2821,)),
    ("light, light source", (1451,)),
    ("truck, motortruck", (2880,)),
    ("tower", (2828,)),
    ("chandelier, pendant, pendent", (480,)),
    ("awning, sunshade, sunblind", (77,)),
    ("streetlight, street lamp", (2616,)),
    ("booth, cubicle, stall, kiosk", (246, 247)),
    ("television, television receiver, television set, tv, tv set, idiot box, boob tube, telly, goggle box", (2733,)),
    ("airplane, aeroplane, plane", (14,)),
    ("dirt track", (738,)),
    ("apparel, wearing apparel, dress, clothes", (38,)),
    ("pole", (1936,)),
    ("land, ground, soil", (1401,)),
    ("bannister, banister, balustrade, balusters, handrail", (120,)),
    ("escalator, moving staircase, moving stairway", (868,)),
    ("ottoman, pouf, pouffe, puff, hassock", (1702,)),
    ("bottle", (249,)),
    ("buffet, counter, sideboard", (308,)),
    ("poster, posting, placard, notice, bill, card", (1969,)),
    ("stage", (2526,)),
    ("van", (2928,)),
    ("ship", (2337,)),
    ("fountain", (1023,)),
    ("conveyer belt, conveyor belt, conveyer, conveyor, transporter", (609,)),
    ("canopy", (389,)),
    ("washer, automatic washer, washing machine", (2989,)),
    ("plaything, toy", (1930,)),
    ("swimming pool, swimming bath, natatorium", (2668,)),
    ("stool", (2586,)),
    ("barrel, cask", (131,)),
    ("basket, handbasket", (146,)),
    ("waterfall, falls", (3016,)),
    ("tent, collapsible shelter", (2739,)),
    ("bag", (95,)),
    ("minibike, motorbike", (1563,)),
    ("cradle", (642,)),
    ("oven", (1708,)),
    ("ball", (103,)),
    ("food, solid food", (1002,)),
    ("step, stair", (2569,)),
    ("tank, storage tank", (2704,)),
    ("trade name, brand name, brand, marque", (2833,)),
    ("microwave, microwave oven", (1551,)),
    ("pot, flowerpot", (1981,)),
    ("animal, animate being, beast, brute, creature, fauna", (29,)),
    ("bicycle, bike, wheel, cycle ", (187,)),
    ("lake", (1393,)),
    ("dishwasher, dish washer, dishwashing machine", (747,)),
    ("screen, silver screen, projection screen", (2254,)),
    ("blanket, cover", (206,)),
    ("sculpture", (2262,)),
    ("hood, exhaust hood", (1260,)),
    ("sconce", (2243,)),
    ("vase", (2932,)),
    ("traffic light, traffic signal, stoplight", (2836,)),
    ("tray", (2850,)),
    ("ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin", (64,)),
    ("fan", (894,)),
    ("pier, wharf, wharfage, dock", (1858,)),
    ("crt screen", (3109,)),
    ("plate", (1919,)),
    ("monitor, monitoring device", (1583,)),
    ("bulletin board, notice board", (318,)),
    ("shower", (2356,)),
    ("radiator", (2046,)),
    ("glass, drinking glass", (1098,)),
    ("clock", (530,)),
    ("flag", (954,)),
)


def _synonyms(name: str) -> frozenset:
    return frozenset(part.strip().lower() for part in name.split(",") if part.strip())


def _first_name(name: str) -> str:
    return name.split(",")[0].strip()


def official_mapping(names: Mapping[int, str]) -> ClassMapping:
    """Return the SceneParse150 mapping, checked against the release's ``names``; see the module contract."""
    problems: List[str] = []
    source_names: List[Tuple[str, ...]] = []
    for index, (benchmark, ids) in enumerate(SCENE_PARSE_150):
        wanted = _synonyms(benchmark)
        released = tuple(names.get(identifier, "") for identifier in ids)
        source_names.append(released)
        # DECISION plan-2026-11-03T221936-98b866fb/D-017
        # One id of a class may be a narrower name (`closet` for "wardrobe, closet, press", `booth` for
        # "booth, cubicle, stall, kiosk"): the class is sound when ANY of its ids shares a synonym with the
        # benchmark's name. Do NOT require every id to match (two of the 150 classes would be refused for
        # nothing), and do NOT drop the check: it is what tells this release from another one whose ids moved.
        if not any(_synonyms(name) & wanted for name in released):
            problems.append(f"class {index} {benchmark!r}: ids {list(ids)} are named {list(released)} in this release")
    if problems:
        raise AdeError(
            f"this release's names do not match the SceneParse150 ids for {len(problems)} of 150 classes "
            f"(it is another release than 2021_17_01?): {'; '.join(problems[:3])}")
    return ClassMapping(
        "official", tuple(_first_name(name) for name, _ in SCENE_PARSE_150),
        tuple(ids for _, ids in SCENE_PARSE_150), tuple(source_names))


def frequency_mapping(counts: np.ndarray, names: Mapping[int, str], k: int) -> ClassMapping:
    """Return the ``k`` release classes with the most pixels as a mapping; see the module contract."""
    if counts.shape != (MAX_CLASS_ID + 1,):
        raise AdeError(f"counts must hold {MAX_CLASS_ID + 1} values, got shape {tuple(counts.shape)}")
    candidates = np.nonzero(counts[1:] > 0)[0] + 1
    if k < 2 or k > 254:
        raise AdeError(f"k must be between 2 and 254 (255 is the void value), got {k}")
    if len(candidates) < k:
        raise AdeError(f"only {len(candidates)} classes have a pixel, so k={k} cannot be chosen")
    # Most pixels first; the smaller id wins a tie, so the choice does not depend on the file order.
    order = np.lexsort((candidates, -counts[candidates]))
    chosen = [int(identifier) for identifier in candidates[order][:k]]
    labels = [_first_name(names.get(identifier, "")) or f"class_{identifier}" for identifier in chosen]
    if len(set(labels)) != len(labels):
        labels = [f"{label} ({identifier})" if labels.count(label) > 1 else label
                  for label, identifier in zip(labels, chosen)]
    return ClassMapping("frequency", tuple(labels), tuple((i,) for i in chosen),
                        tuple((names.get(i, ""),) for i in chosen))


def convert_seg(seg: np.ndarray, lut: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the mask of a ``_seg.png`` array and the count of each mask value; see the module contract."""
    mask = lut[decode_class_ids(seg)]
    return mask, np.bincount(mask.ravel(), minlength=256).astype(np.int64)
