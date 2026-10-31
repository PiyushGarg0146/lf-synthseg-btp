import nibabel as nib
import numpy as np
import pytest

from data import preprocess as P


def make_img(shape, zoom, axcodes):
    """Image whose voxel-grid orientation is `axcodes`, with a marker voxel for tracking flips."""
    arr = np.zeros(shape, np.float32)
    arr[1, 2, 3] = 5.0
    ornt = nib.orientations.axcodes2ornt(axcodes)
    affine = nib.orientations.inv_ornt_aff(ornt, shape) @ np.diag([*zoom, 1.0])
    return nib.Nifti1Image(arr, affine)


@pytest.mark.parametrize("axcodes", [("L", "A", "S"), ("L", "P", "S"), ("R", "A", "S")])
def test_reorient_gives_ras_and_keeps_world_position(axcodes):
    img = make_img((8, 9, 10), (1, 1, 1), axcodes)
    out = P.reorient_ras(img)
    assert nib.aff2axcodes(out.affine) == ("R", "A", "S")
    # the marker voxel must stay at the same physical (world) position
    before = nib.affines.apply_affine(img.affine, [1, 2, 3])
    idx = np.argwhere(np.asarray(out.dataobj) == 5.0)[0]
    after = nib.affines.apply_affine(out.affine, idx)
    assert np.allclose(before, after)


def test_resample_to_1mm_changes_shape_and_voxel_size():
    img = make_img((20, 20, 20), (0.7, 0.7, 0.7), ("R", "A", "S"))
    out = P.resample_iso(img, 1.0, order=1)
    assert tuple(round(z, 3) for z in out.header.get_zooms()[:3]) == (1.0, 1.0, 1.0)
    assert all(13 <= s <= 15 for s in out.shape)   # 20 * 0.7 = 14 mm


def test_resample_nearest_keeps_label_ids():
    arr = np.zeros((10, 10, 10), np.float32); arr[:5] = 2; arr[5:] = 3
    img = nib.Nifti1Image(arr, np.diag([0.7, 0.7, 0.7, 1.0]))
    out = np.asarray(P.resample_iso(img, 1.0, order=0).dataobj)
    assert set(np.unique(out).tolist()) <= {0.0, 2.0, 3.0}


def test_normalize_range_and_background():
    rng = np.random.default_rng(0)
    d = np.zeros((10, 10, 10), np.float32)
    d[2:8, 2:8, 2:8] = rng.random((6, 6, 6)) * 800 + 100
    out = P.normalize_intensity(d)
    assert out.min() == 0.0 and out.max() == pytest.approx(1.0)
    assert (out[d == 0] == 0).all()
    assert out.dtype == np.float32


def test_normalize_empty_mask_raises():
    with pytest.raises(ValueError):
        P.normalize_intensity(np.zeros((4, 4, 4)))


def test_flirt_missing_gives_clear_error(monkeypatch):
    monkeypatch.setattr(P.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="FSL tool"):
        P.register_flirt("a", "b", "c", "d")


def test_standardize_native_keeps_background_empty(tmp_path):
    arr = np.zeros((20, 20, 20), np.float32)
    arr[6:14, 6:14, 6:14] = 500.0
    arr[8:12, 8:12, 8:12] = 900.0          # sharp edges make cubic interpolation ring
    src = tmp_path / "in.nii.gz"
    nib.save(nib.Nifti1Image(arr, np.diag([0.7, 0.7, 0.7, 1.0])), str(src))
    out = np.asarray(nib.load(str(P.standardize_native(src, tmp_path / "out.nii.gz"))).dataobj)
    assert out.min() >= 0
    frac_in = (arr > 0).mean()
    assert abs((out > 0).mean() - frac_in) < 0.05    # background did not fill with ringing
