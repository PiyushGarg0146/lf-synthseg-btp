"""HCP healthy scans -> MNI space, normalized T2, FAST tissue labels, brain mask, QC pictures.

Needs FSL (`flirt`, `fast`) on PATH and $FSLDIR set - run on Linux/Kaggle (see notebooks/01_fsl_fast_hcp.ipynb).

Per subject, outputs in <dst>/<id>/:
  t2.nii.gz     normalized T2 in MNI152 1 mm space ([0, 1], background 0)
  labels.nii.gz uint8 labels 0-4 (see data/labels.py)
  mask.nii.gz   brain mask
  flirt.mat     the affine used
  qc.png        slices of image and labels for visual checking
Also writes <dst>/summary.csv with class volumes and mean intensity per class.

    python scripts/fsl_pipeline.py --src /kaggle/input/hcp --dst /kaggle/working/hcp_processed --limit 1
"""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import labels as L  # noqa: E402
from data import preprocess as P  # noqa: E402


def brain_mask(img, frac=0.01):
    """Voxels above 1% of the 99th percentile of positive voxels, holes filled, largest component kept."""
    pos = img[img > 0]
    m = img > frac * np.percentile(pos, 99)
    m = ndimage.binary_closing(m, iterations=2)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return m


def fill_unlabeled(labels, mask):
    """Give mask voxels that FAST left at 0 the label of the nearest labelled voxel."""
    todo = mask & (labels == 0)
    if todo.any():
        idx = ndimage.distance_transform_edt(labels == 0, return_distances=False, return_indices=True)
        labels = labels.copy()
        labels[todo] = labels[tuple(i[todo] for i in idx)]
    return labels


def run_fast(img_path, prefix):
    subprocess.run([P._need_fsl("fast"), "-n", "4", "-t", "2", "-o", str(prefix), str(img_path)], check=True)
    return Path(f"{prefix}_seg.nii.gz")


def qc_png(t2, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = plt.get_cmap("tab10", 6)
    cx = [s // 2 for s in t2.shape]
    views = [(t2[:, :, cx[2]], labels[:, :, cx[2]]), (t2[:, cx[1], :], labels[:, cx[1], :]), (t2[cx[0], :, :], labels[cx[0], :, :])]
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    for j, (a, b) in enumerate(views):
        ax[0, j].imshow(np.rot90(a), cmap="gray"); ax[0, j].axis("off")
        ax[1, j].imshow(np.rot90(b), cmap=cm, vmin=0, vmax=5, interpolation="nearest"); ax[1, j].axis("off")
    fig.suptitle("T2 (top) / labels (bottom): 1 CSF, 2 GM, 3 WM, 4 dark")
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def process(src_file, out_dir, mni):
    sid = Path(src_file).name.split("_")[0]
    work = Path(out_dir) / sid
    work.mkdir(parents=True, exist_ok=True)
    native = P.standardize_native(src_file, work / "native_ras1mm.nii.gz")
    reg = work / "t2_mni_raw.nii.gz"
    P.register_flirt(native, mni, reg, work / "flirt.mat")

    img = nib.load(str(reg))
    data = np.clip(np.asarray(img.dataobj, dtype=np.float32), 0, None)
    mask = brain_mask(data)
    data[~mask] = 0
    masked = work / "t2_mni_masked.nii.gz"
    nib.save(nib.Nifti1Image(data, img.affine), str(masked))

    seg = np.asarray(nib.load(str(run_fast(masked, work / "fast"))).dataobj).astype(np.int16)
    seg[~mask] = 0
    labels = fill_unlabeled(L.fast_to_project(seg), mask)
    L.check_label_map(labels, mask)

    t2n = P.normalize_intensity(data, mask)
    nib.save(nib.Nifti1Image(t2n, img.affine), str(work / "t2.nii.gz"))
    nib.save(nib.Nifti1Image(labels.astype(np.uint8), img.affine), str(work / "labels.nii.gz"))
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), img.affine), str(work / "mask.nii.gz"))
    qc_png(t2n, labels, work / "qc.png")

    row = {"id": sid, "brain_voxels": int(mask.sum())}
    for k in (L.CSF, L.GM, L.WM, L.DARK):
        sel = labels == k
        row[f"{L.NAMES[k]}_frac"] = round(float(sel.mean() / mask.mean()), 4)
        row[f"{L.NAMES[k]}_meanT2"] = round(float(t2n[sel].mean()), 4) if sel.any() else float("nan")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with *_T2w_*.nii.gz HCP files (searched recursively)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mni", default=None, help="default: $FSLDIR/data/standard/MNI152_T2_1mm_brain.nii.gz")
    args = ap.parse_args()

    mni = Path(args.mni or Path(os.environ["FSLDIR"]) / "data" / "standard" / "MNI152_T2_1mm_brain.nii.gz")
    if not mni.exists():
        sys.exit(f"MNI template not found: {mni}")
    files = sorted(Path(args.src).rglob("*.nii.gz"))[: args.limit]
    if not files:
        sys.exit(f"no .nii.gz files under {args.src}")
    rows = []
    for f in files:
        print("processing", f.name, flush=True)
        rows.append(process(f, args.dst, mni))
        print(rows[-1], flush=True)
    out = Path(args.dst) / "summary.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", out)
    r = rows[0]
    order = [r["CSF_meanT2"], r["GM_meanT2"], r["WM_meanT2"], r["dark_meanT2"]]
    print("mean T2 CSF, GM, WM, dark:", order)
    if not (order[0] > order[1] > order[2] > order[3]):
        print("WARNING: expected CSF > GM > WM > dark on T2; check the label mapping in data/labels.py")


if __name__ == "__main__":
    main()
