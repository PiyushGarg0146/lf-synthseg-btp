"""NIfTI loading/saving and machine-specific path lookup."""
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]


def load_paths(config=None):
    """Read configs/paths.yaml (or $LFSS_PATHS if set) into a dict of Paths."""
    import os

    cfg = Path(config or os.environ.get("LFSS_PATHS") or _ROOT / "configs" / "paths.yaml")
    if not cfg.exists():
        raise FileNotFoundError(f"{cfg} not found. Copy configs/paths.example.yaml to configs/paths.yaml and edit it.")
    with open(cfg) as f:
        return {k: Path(v) for k, v in yaml.safe_load(f).items()}


def load_nifti(path):
    """Return (data float32, affine 4x4, voxel size tuple in mm)."""
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    return data, img.affine, tuple(float(z) for z in img.header.get_zooms()[:3])


def save_nifti(data, affine, path, dtype=None):
    """Save an array with the given affine; creates parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data) if dtype is None else np.asarray(data).astype(dtype)
    nib.save(nib.Nifti1Image(arr, affine), str(path))
    return path


def orientation(affine):
    """Axis codes such as ('R','A','S'), useful for checking HCP vs BraTS vs MNI."""
    return nib.aff2axcodes(affine)
