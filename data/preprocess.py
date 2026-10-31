"""Bring every scan into one common space: RAS orientation, 1 mm isotropic, MNI152, [0, 1] intensity.

Order of operations (same for HCP training scans and BraTS test scans):
  1. reorient_ras      - fixes the HCP (LAS) vs BraTS (LPS) axis mismatch
  2. resample_iso      - 1 mm isotropic voxels
  3. register_flirt    - affine (12 dof) registration to the MNI152 1 mm brain (T1 template, mutual-information cost) (needs FSL; run on Linux/Kaggle)
  4. normalize_intensity - clip at a high brain percentile, scale to [0, 1]; background stays 0

Steps 1, 2 and 4 are plain Python and run anywhere. Step 3 calls FSL's `flirt`.
"""
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output


def reorient_ras(img):
    """Return the image reoriented to the closest RAS+ orientation (data is flipped/permuted, not interpolated)."""
    return nib.as_closest_canonical(img)


def resample_iso(img, voxel=1.0, order=3):
    """Resample to `voxel` mm isotropic. Use order=3 for images, order=0 for label maps."""
    return resample_to_output(img, voxel_sizes=voxel, order=order, mode="constant", cval=0.0)


def normalize_intensity(data, mask=None, pct=99.5):
    """Scale to [0, 1] using the `pct` percentile of brain voxels; voxels outside the mask become 0."""
    data = np.asarray(data, dtype=np.float32)
    mask = data > 0 if mask is None else np.asarray(mask).astype(bool)
    if not mask.any():
        raise ValueError("empty brain mask")
    hi = float(np.percentile(data[mask], pct))
    out = np.clip(data / hi, 0.0, 1.0)
    out[~mask] = 0.0
    return out.astype(np.float32)


def standardize_native(in_path, out_path, order=3):
    """Steps 1-2 on a file. Returns the output path.

    For images (order > 0) the brain mask (voxels > 0) is resampled with nearest-neighbour and used to
    zero the background, because spline interpolation leaves faint ringing outside the brain.
    """
    img = reorient_ras(nib.load(str(in_path)))
    res = resample_iso(img, 1.0, order)
    if order > 0:
        brain = nib.Nifti1Image((np.asarray(img.dataobj) > 0).astype(np.uint8), img.affine)
        mask = np.asarray(resample_iso(brain, 1.0, order=0).dataobj) > 0
        res = nib.Nifti1Image(np.where(mask, np.clip(np.asarray(res.dataobj), 0, None), 0).astype(np.float32), res.affine)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(res, str(out_path))
    return out_path


def _need_fsl(tool):
    exe = shutil.which(tool)
    if exe is None:
        raise RuntimeError(f"FSL tool '{tool}' not found on PATH. Run this step on Linux with FSL installed (see notebooks/).")
    return exe


def register_flirt(moving, reference, out_img, out_mat, dof=12, cost="normmi"):
    """Affine-register `moving` to `reference` (MNI152 1 mm brain). Saves image and matrix.

    The conda FSL package ships only the T1 MNI template, so a T2 scan is matched to it with a
    multi-modal cost (normalized mutual information) instead of the default correlation ratio.
    """
    subprocess.run([_need_fsl("flirt"), "-in", str(moving), "-ref", str(reference), "-out", str(out_img),
                    "-omat", str(out_mat), "-dof", str(dof), "-cost", cost, "-interp", "spline"], check=True)


def apply_flirt(moving, reference, mat, out_img, nearest=False):
    """Apply a saved flirt matrix (use nearest=True for label maps)."""
    subprocess.run([_need_fsl("flirt"), "-in", str(moving), "-ref", str(reference), "-applyxfm", "-init", str(mat),
                    "-out", str(out_img), "-interp", "nearestneighbour" if nearest else "spline"], check=True)
