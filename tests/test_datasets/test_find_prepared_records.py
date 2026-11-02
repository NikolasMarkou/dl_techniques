"""``find_prepared_records``: are the Imagenette records under a TFDS directory, by path tests only.

It replaced a lookup that ``train.api.data_args`` did itself with the constant ``IMAGENETTE_TFDS_NAME``,
which ``test_the_image_classification_trainers_share_one_ingestion_path`` forbids outside the library
(the guard was red at HEAD from step 15 until this function moved the lookup here).
"""

from pathlib import Path

from dl_techniques.datasets.vision.imagenette import IMAGENETTE_TFDS_NAME, find_prepared_records


def test_a_directory_with_no_records_is_not_found_and_nothing_is_created(tmp_path):
    here, found = find_prepared_records(tmp_path)
    assert here == tmp_path / IMAGENETTE_TFDS_NAME and found is False
    assert list(tmp_path.iterdir()) == []


def test_a_version_directory_without_dataset_info_is_not_found(tmp_path):
    (tmp_path / IMAGENETTE_TFDS_NAME / "1.0.0").mkdir(parents=True)
    assert find_prepared_records(tmp_path)[1] is False


def test_a_version_directory_with_dataset_info_is_found_whatever_the_version(tmp_path):
    version = tmp_path / IMAGENETTE_TFDS_NAME / "9.9.9"
    version.mkdir(parents=True)
    (version / "dataset_info.json").write_text("{}")
    here, found = find_prepared_records(str(tmp_path))
    assert found is True and isinstance(here, Path)


def test_a_file_where_the_directory_should_be_is_not_found(tmp_path):
    (tmp_path / "imagenette").mkdir()
    (tmp_path / "imagenette" / "320px-v2").write_text("not a directory")
    assert find_prepared_records(tmp_path)[1] is False
