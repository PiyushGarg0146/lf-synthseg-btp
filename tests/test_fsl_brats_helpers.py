import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fsl_brats as B  # noqa: E402


def make_case(root, cid, seg_name=None):
    d = root / "BraTS2020_TrainingData" / "MICCAI" / cid
    d.mkdir(parents=True)
    for suffix in ("flair", "t1", "t1ce", "t2"):
        (d / f"{cid}_{suffix}.nii").write_bytes(b"x")
    (d / (seg_name or f"{cid}_seg.nii")).write_bytes(b"x")
    return d


def test_find_case_files_standard_and_odd_seg_name(tmp_path):
    make_case(tmp_path, "BraTS20_Training_001")
    make_case(tmp_path, "BraTS20_Training_355", seg_name="W39_1998.09.19_Segm.nii")
    make_case(tmp_path, "BraTS20_Training_002")            # not requested
    found = B.find_case_files(tmp_path, ["BraTS20_Training_001", "BraTS20_Training_355"])
    assert set(found) == {"BraTS20_Training_001", "BraTS20_Training_355"}
    assert found["BraTS20_Training_001"][0].name.endswith("_t2.nii")
    assert found["BraTS20_Training_355"][1].name == "W39_1998.09.19_Segm.nii"


def test_find_case_files_reports_missing(tmp_path):
    make_case(tmp_path, "BraTS20_Training_001")
    with pytest.raises(FileNotFoundError, match="not found"):
        B.find_case_files(tmp_path, ["BraTS20_Training_001", "BraTS20_Training_999"])


def test_find_case_files_needs_exactly_one_seg(tmp_path):
    d = make_case(tmp_path, "BraTS20_Training_001")
    (d / "extra_seg.nii").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="exactly one segmentation"):
        B.find_case_files(tmp_path, ["BraTS20_Training_001"])


def test_read_case_ids_ignores_comments_and_blank_lines(tmp_path):
    f = tmp_path / "ids.txt"
    f.write_text("# frozen\nBraTS20_Training_013\n\nBraTS20_Training_014\n")
    assert B.read_case_ids(f) == ["BraTS20_Training_013", "BraTS20_Training_014"]


def test_real_split_file_has_31_unique_ids():
    ids = B.read_case_ids(Path(__file__).resolve().parents[1] / "configs" / "test_split_brats2020.txt")
    assert len(ids) == 31 and len(set(ids)) == 31


def test_tumor_volume_counts_all_nonzero_labels():
    seg = np.zeros((4, 4, 4)); seg[0, 0, :2] = 1; seg[1, 1, :1] = 2; seg[2, 2, :3] = 4
    assert B.tumor_volume(seg) == 6
