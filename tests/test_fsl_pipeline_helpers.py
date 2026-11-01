import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fsl_pipeline as F  # noqa: E402
from data import labels as L  # noqa: E402


def _blob():
    d = np.zeros((30, 30, 30), np.float32)
    d[8:22, 8:22, 8:22] = 500
    d[14, 14, 14] = 0     # hole inside the brain
    d[1, 1, 1] = 3        # stray speck outside
    return d


def test_brain_mask_fills_hole_and_drops_speck():
    m = F.brain_mask(_blob())
    assert m[14, 14, 14] and not m[1, 1, 1]
    assert m.sum() == 14 ** 3


def test_fill_unlabeled_leaves_valid_label_map():
    m = F.brain_mask(_blob())
    lab = np.zeros(m.shape, np.uint8); lab[m] = 3; lab[14, 14, 14] = 0
    out = F.fill_unlabeled(lab, m)
    assert out[14, 14, 14] == 3
    assert L.check_label_map(out, m)


def test_row_from_outputs_roundtrip(tmp_path):
    import nibabel as nib

    mask = np.zeros((6, 6, 6), bool); mask[1:5, 1:5, 1:5] = True
    labels = np.zeros(mask.shape, np.uint8); labels[mask] = L.WM; labels[2, 2, 2] = L.CSF
    t2 = np.where(mask, 0.5, 0).astype(np.float32)
    work = tmp_path / "123"; work.mkdir()
    assert F.row_from_outputs(work) is None            # incomplete -> None
    aff = np.eye(4)
    nib.save(nib.Nifti1Image(t2, aff), str(work / "t2.nii.gz"))
    nib.save(nib.Nifti1Image(labels, aff), str(work / "labels.nii.gz"))
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), aff), str(work / "mask.nii.gz"))
    (work / "qc.png").write_bytes(b"x")
    row = F.row_from_outputs(work)
    expected = F.make_row("123", labels, t2, mask)
    assert row.keys() == expected.keys()
    for k in row:   # nan != nan, so compare with allclose-style equality
        assert row[k] == expected[k] or (row[k] != row[k] and expected[k] != expected[k])
    assert row["brain_voxels"] == 64 and row["CSF_frac"] == round(1 / 64, 4)
