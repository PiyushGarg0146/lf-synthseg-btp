import numpy as np
import pytest

from data import io


def test_nifti_roundtrip(tmp_path):
    arr = np.random.default_rng(0).random((6, 7, 8)).astype(np.float32)
    aff = np.diag([0.7, 0.7, 0.7, 1.0])
    p = io.save_nifti(arr, aff, tmp_path / "sub" / "x.nii.gz")
    data, affine, zooms = io.load_nifti(p)
    assert np.allclose(data, arr)
    assert np.allclose(affine, aff)
    assert zooms == pytest.approx((0.7, 0.7, 0.7))
    assert io.orientation(affine) == ("R", "A", "S")


def test_load_paths_reads_yaml(tmp_path):
    cfg = tmp_path / "p.yaml"
    cfg.write_text("hcp_raw: /a/b\n")
    assert io.load_paths(cfg)["hcp_raw"].name == "b"
    with pytest.raises(FileNotFoundError):
        io.load_paths(tmp_path / "missing.yaml")
