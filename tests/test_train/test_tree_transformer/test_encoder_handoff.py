"""The tree_transformer pretrain -> finetune encoder hand-off (F-24).

`pretrain.py` used to write the encoder ONLY into the timestamped run directory
returned by `create_nlp_callbacks`, while `finetune.py`'s default
`pretrained_encoder_path` named the static `results/tree_transformer_pretrain/...`
-- a directory that `pretrain.py` created and then left EMPTY. So the hand-off was
broken by default and a plain `pretrain` followed by a plain `finetune` could not
work without passing `--pretrained-encoder-path` explicitly.

These guards compare the two PRODUCERS, not a hard-coded string: a test asserting a
literal path would re-create exactly the hand-maintained agreement whose failure
caused the defect. They are the first tests of any kind for `src/train/tree_transformer/`.
"""

import ast
import inspect
import os

from train.tree_transformer import finetune as ft
from train.tree_transformer import pretrain as pt


class TestEncoderHandoff:
    """The path fine-tuning reads must be the path pre-training writes."""

    def test_finetune_default_is_the_path_pretrain_writes(self) -> None:
        """The default read path equals the static write path, via one producer."""
        written = pt.pretrained_encoder_path(pt.TrainingConfig.save_dir)
        read = ft.FinetuneConfig.pretrained_encoder_path
        assert read == written, (
            "fine-tuning's default encoder path is NOT the path pre-training "
            f"writes (F-24): reads {read!r}, writes {written!r}"
        )

    def test_saving_produces_a_file_at_the_handoff_path(self, tmp_path) -> None:
        """EXECUTED: the save actually lands where fine-tuning will look.

        Drives `save_pretrained_encoder` with a recording stand-in rather than
        grepping the source -- a substring check passes as long as the path is
        COMPUTED, even if nothing is ever written to it, which is the precise shape
        of the original defect.
        """
        written = []

        class _RecordingEncoder:
            def save(self, path: str) -> None:
                written.append(path)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as handle:
                    handle.write("stub")

        results_dir = str(tmp_path / "tree_transformer_pretrain_TT-tiny_20260812_120000")
        save_dir = str(tmp_path / "tree_transformer_pretrain")
        os.makedirs(results_dir, exist_ok=True)

        run_copy, handoff_copy = pt.save_pretrained_encoder(
            _RecordingEncoder(), results_dir, save_dir
        )

        assert os.path.isfile(handoff_copy), (
            "pre-training did not WRITE the encoder to the static hand-off "
            f"location {handoff_copy!r}; fine-tuning's default would resolve to a "
            "file that does not exist (F-24)"
        )
        assert os.path.isfile(run_copy), (
            "the timestamped per-run encoder copy was not written; that copy is a "
            "run's own evidence and must be kept alongside the hand-off copy"
        )
        assert written == [run_copy, handoff_copy], (
            f"expected exactly two saves (run copy then hand-off copy), got {written}"
        )

    def test_the_handoff_file_is_the_one_finetune_will_open(self, tmp_path) -> None:
        """The written file's basename matches fine-tuning's default basename."""
        written = []

        class _RecordingEncoder:
            def save(self, path: str) -> None:
                written.append(path)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                open(path, "w").close()

        save_dir = str(tmp_path / "static")
        _, handoff_copy = pt.save_pretrained_encoder(
            _RecordingEncoder(), str(tmp_path / "run"), save_dir
        )
        assert os.path.basename(handoff_copy) == os.path.basename(
            ft.FinetuneConfig.pretrained_encoder_path
        ), (
            "the file pre-training writes and the file fine-tuning opens have "
            "different names (F-24)"
        )

    def test_the_filename_has_a_single_producer(self) -> None:
        """Neither trainer spells the encoder filename out by hand."""
        literal = pt.PRETRAINED_ENCODER_FILENAME
        for module in (pt, ft):
            source = inspect.getsource(module)
            # The one legitimate occurrence is the constant's own definition.
            occurrences = source.count(literal)
            expected = 1 if module is pt else 0
            assert occurrences == expected, (
                f"{module.__name__} spells the encoder filename literally "
                f"{occurrences} time(s), expected {expected}; a second hand-written "
                "copy is what allowed the two paths to drift apart (F-24)"
            )

    def test_finetune_declares_no_config_field_it_never_reads(self) -> None:
        """No silent no-op knobs on `FinetuneConfig`.

        `save_dir` used to be declared and read exactly once -- by an
        `os.makedirs` that created a directory nothing was ever written into
        (every artefact goes to the timestamped `results_dir`). A user setting it
        would have seen an empty directory appear and nothing else change. This
        guard is deliberately general: it fails for ANY field that is declared and
        then never read, which is the shape of a knob that silently does nothing.
        """
        tree = ast.parse(inspect.getsource(ft))
        config_cls = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "FinetuneConfig"
        )
        declared = [
            node.target.id for node in config_cls.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        ]
        assert declared, "found no annotated fields — the parse is wrong, not the code"

        # Only accesses on the CONFIG itself count. A module-wide `node.attr` set
        # is vacuous here: `_PRETRAIN_SAVE_DIR = _PretrainConfig.save_dir` puts
        # "save_dir" in it, so a dead `FinetuneConfig.save_dir` reads as live.
        # Measured — the first draft of this guard passed against exactly that
        # mutation.
        config_names = {"config", "cfg", "FinetuneConfig", "self"}
        read = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in config_names
        }
        dead = sorted(set(declared) - read)
        assert not dead, (
            f"FinetuneConfig declares {dead} but never reads them anywhere in the "
            "module — a config field nothing consumes is a knob that silently does "
            "nothing when a user sets it"
        )

    def test_the_handoff_directory_is_the_one_pretrain_creates(self) -> None:
        """`makedirs(config.save_dir)` and the hand-off write target agree.

        Before the fix that `makedirs` was effectively dead -- it created an empty
        directory nothing ever wrote into.
        """
        written = pt.pretrained_encoder_path(pt.TrainingConfig.save_dir)
        assert os.path.dirname(written) == pt.TrainingConfig.save_dir
