"""The TensorFlow-free half of the text data: the tokenizer record, its digest and the check of the pinned directory.

``train.common`` imports TensorFlow (``train/common/__init__.py`` loads ``gpu`` and ``nlp``), so any import of
``train.common.text_data`` loads it. The check of a tokenizer directory needs only paths and ``hashlib``, and a run that is
going to fail on it has to fail BEFORE the GPU is claimed. So the record and that check live here, with the standard library
only, and ``train.common.text_data`` imports them: one implementation, as ``forecast_series`` is for the forecasting data.
Nothing here imports numpy, ``train.common`` or a tokenizer library (``test_cli_help_allocates_nothing`` reads ``sys.modules``).

Interface contract
------------------
``TOKENIZERS``
    ``("ascii95", "gpt2")``: the tokenizers the text tasks offer. ``ascii95`` is the offline character tokenizer (101 ids, no
    file), ``gpt2`` the tiktoken BPE read from the pinned directory.

``TokenizerSpec``
    A frozen record: ``encoding``, ``n_vocab`` (the tokenizer's own ids), ``vocab_size`` (the model's: ``n_vocab`` plus the four
    specials a classifier needs), ``eot_id``, ``cls_id``, ``sep_id``, ``pad_id``, ``mask_id`` and ``files`` (``((cache file name,
    sha256), ...)`` sorted by name, empty for ``ascii95``). ``digest`` is the sha256 of the canonical json of all of them, the
    file hashes included, so a changed id, encoding or file is another tokenizer. The model's ``vocab_size`` is ALWAYS this
    one, never a model default (the library GPT-2 defaults to 100,277, the legacy trainers use 50,261).

``tokenizer_spec(name)``
    :return: the spec of ``name``. :raises ConfigError: for a name that is not in :data:`TOKENIZERS`.

``check_files(files, directory)``
    :param files: ``((name, sha256), ...)``; ``directory``: the directory, ``None`` meaning :data:`DEFAULT_TOKENIZER_DIR`.
    :return: the directory as a ``Path``. :raises ConfigError: naming ``data.tokenizer_dir``, the path and, for a file of another
        hash, the digest it found and the one pinned: a directory that is not one, a file that is missing, not a regular file,
        unreadable, or of another hash. It reads files and never writes, moves or removes one.

``check_tokenizer_dir(name, directory=None)``
    :return: ``None`` for a tokenizer with no file, else :func:`check_files` of its files.
"""

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from .errors import ConfigError
from .init_from import file_sha256

ASCII95, GPT2 = "ascii95", "gpt2"

#: The tokenizers the text tasks offer (``cl100k_base`` is not one of them: a 100,277-wide logits tensor, D-009).
TOKENIZERS: Tuple[str, ...] = (ASCII95, GPT2)

#: The key a refusal names.
TOKENIZER_DIR_KEY = "data.tokenizer_dir"

# DECISION plan-2026-11-24T092348-7806a290/D-004
# Real runs read the BPE files from THIS directory, not from tiktoken's default (``/tmp/data-gym-cache``, which is wiped at
# boot): the user pinned a copy outside /tmp (ruling U3), three files, never on data_fast. Do NOT point the default at /tmp, and
# do NOT drop the hash table below: a file of another content under the same name would pass a name-only check.
DEFAULT_TOKENIZER_DIR = "/media/arxwn/data0_4tb/datasets/tiktoken_cache"

#: The cache file name tiktoken looks for (the sha1 of the URL it fetches) and the sha256 of its bytes, for the two files the
#: ``gpt2`` encoding reads. Generated from the pin record of Step 2 (``scratch_tokenizer/pin.json``), never typed; a test holds
#: the table equal to the hashes tiktoken checks its own downloads against.
GPT2_FILES: Tuple[Tuple[str, str], ...] = (
    # gpt2 encoder.json, 1,042,301 bytes
    ("6c7ea1a7e38e3a7f062df639a5b80947f075ffe6",
     "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"),
    # gpt2 vocab.bpe, 456,318 bytes
    ("6d1cbeee0f20b3d9449abfede4726ed8212e3aee",
     "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"),
)


# DECISION plan-2026-11-24T092348-7806a290/D-016
# This record lives on the ``train.api`` side, not in ``train.common.text_data`` where the plan first put it: the package
# ``__init__`` of ``train.common`` loads TensorFlow, so a pre-claim check imported from there would load it too (measured:
# ``import train.common.seed`` leaves tensorflow, keras and tiktoken in ``sys.modules``). Do NOT move the record or the check
# back under ``train.common``, and do NOT import numpy, keras or tiktoken at module scope here. The ids below are written
# out, not read from ``clm_pretrain`` or ``ascii_char`` (both import Keras at module scope); tests hold them equal.
@dataclass(frozen=True)
class TokenizerSpec:
    """The identity of a tokenizer: the ids a model and a record depend on, and the files that define its vocabulary."""

    encoding: str
    n_vocab: int
    vocab_size: int
    eot_id: int
    cls_id: int
    sep_id: int
    pad_id: int
    mask_id: int
    files: Tuple[Tuple[str, str], ...]

    @property
    def digest(self) -> str:
        """The sha256 (64 hex digits) of every field as json (sorted keys, no spaces; ``files`` an object), the file hashes included."""
        fields = {**dataclasses.asdict(self), "files": dict(self.files)}
        return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


_SPECS = {
    # the character tokenizer's own layout (``dl_techniques.layers.tokenizers.ascii_char``): it has no end-of-text id, so a
    # document ends with its separator
    ASCII95: TokenizerSpec(ASCII95, n_vocab=101, vocab_size=101, eot_id=2, cls_id=1, sep_id=2, pad_id=0, mask_id=3, files=()),
    # the legacy trainers' layout (``train.common.clm_pretrain.ClmPretrainConfig``): tiktoken's 50,257 ids with the end-of-text
    # id inside, then cls, sep, pad and mask after them, 50,261 in all
    GPT2: TokenizerSpec(GPT2, n_vocab=50257, vocab_size=50261, eot_id=50256, cls_id=50257, sep_id=50258, pad_id=50259,
                        mask_id=50260, files=GPT2_FILES),
}


def tokenizer_spec(name: str) -> TokenizerSpec:
    """Return the :class:`TokenizerSpec` of ``name``; see the module contract."""
    if name not in _SPECS:
        raise ConfigError(
            f"Invalid value for 'data.tokenizer': {name!r} is not offered; the text tasks take "
            f"{' or '.join(repr(offered) for offered in TOKENIZERS)}")
    return _SPECS[name]


def _refuse(why: str) -> ConfigError:
    return ConfigError(f"Invalid value for {TOKENIZER_DIR_KEY!r}: {why}")


def check_files(files: Sequence[Tuple[str, str]], directory: Union[str, Path, None]) -> Path:
    """Check that ``directory`` holds every file of ``files`` with its pinned sha256; see the module contract."""
    root = Path(DEFAULT_TOKENIZER_DIR if directory is None else directory)
    try:
        is_directory = root.is_dir()
    except OSError as exc:
        raise _refuse(f"{str(root)!r} cannot be read ({exc.strerror or exc})") from exc
    if not is_directory:
        raise _refuse(
            f"{str(root)!r} is not a directory; the tokenizer reads its BPE files from it (the pinned copy is "
            f"{DEFAULT_TOKENIZER_DIR})")
    for name, expected in files:
        path = root / name
        if not path.exists():
            raise _refuse(f"{str(path)!r} does not exist, but the tokenizer needs it (sha256 {expected})")
        if not path.is_file():
            raise _refuse(f"{str(path)!r} is not a file; the tokenizer needs the BPE file of sha256 {expected}")
        try:
            found = file_sha256(path)
        except OSError as exc:
            raise _refuse(f"{str(path)!r} cannot be read ({exc.strerror or exc})") from exc
        if found != expected:
            raise _refuse(
                f"{str(path)!r} has sha256 {found}, not the pinned {expected}: it is not the file this tokenizer was built "
                "from, so no run, record or checkpoint made with it would match one made with the pinned copy")
    return root


def check_tokenizer_dir(name: str, directory: Union[str, Path, None] = None) -> Optional[Path]:
    """Check the files of the tokenizer ``name`` under ``directory``; see the module contract."""
    spec = tokenizer_spec(name)
    return check_files(spec.files, directory) if spec.files else None
