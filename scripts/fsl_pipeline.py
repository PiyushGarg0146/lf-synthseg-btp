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
    subprocess.run([P._need_fsl("fast"), "-n", "3", "-t", "2", "-o", str(prefix), str(img_path)], check=True)
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
    from matplotlib.patches import Patch

    names = {0: "background", 1: "CSF", 2: "GM", 3: "WM", 4: "dark", 5: "tumor"}
    fig.legend(handles=[Patch(color=cm(i / 5), label=f"{i} {names[i]}") for i in range(5)], loc="lower center", ncol=5)
    fig.suptitle("T2 (top) / labels (bottom)")
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(path, dpi=90); plt.close(fig)


def make_row(sid, labels, t2n, mask):
    row = {"id": sid, "brain_voxels": int(mask.sum())}
    for k in (L.CSF, L.GM, L.WM, L.DARK):
        sel = labels == k
        row[f"{L.NAMES[k]}_frac"] = round(float(sel.mean() / mask.mean()), 4)
        row[f"{L.NAMES[k]}_meanT2"] = round(float(t2n[sel].mean()), 4) if sel.any() else float("nan")
    return row


def row_from_outputs(work):
    """Rebuild the summary row of an already finished subject from its saved files (None if incomplete)."""
    work = Path(work)
    names = ("t2.nii.gz", "labels.nii.gz", "mask.nii.gz", "qc.png")
    if not all((work / n).exists() for n in names):
        return None
    t2n = np.asarray(nib.load(str(work / "t2.nii.gz")).dataobj, dtype=np.float32)
    labels = np.asarray(nib.load(str(work / "labels.nii.gz")).dataobj).astype(np.uint8)
    mask = np.asarray(nib.load(str(work / "mask.nii.gz")).dataobj).astype(bool)
    return make_row(work.name, labels, t2n, mask)


def process(src_file, out_dir, mni, dark_frac=L.DARK_FRAC):
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
    dark = L.dark_voxels(data, mask, dark_frac)
    ids = sorted(set(np.unique(seg).tolist()) - {0})
    print(f"{sid} FAST classes (id: share of brain, mean raw T2 | excluding dark voxels): "
          + str({i: (round(float((seg == i).sum() / mask.sum()), 3), round(float(data[seg == i].mean()), 1),
                     round(float(data[(seg == i) & ~dark].mean()), 1)) for i in ids}), flush=True)
    labels = fill_unlabeled(L.fast3_by_intensity(seg, data, ignore=dark), mask)
    labels[dark] = L.DARK
    L.check_label_map(labels, mask)

    t2n = P.normalize_intensity(data, mask)
    nib.save(nib.Nifti1Image(t2n, img.affine), str(work / "t2.nii.gz"))
    nib.save(nib.Nifti1Image(labels.astype(np.uint8), img.affine), str(work / "labels.nii.gz"))
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), img.affine), str(work / "mask.nii.gz"))
    qc_png(t2n, labels, work / "qc.png")
    return make_row(sid, labels, t2n, mask)


def write_summary(rows, out):
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with HCP T2 .nii or .nii.gz files (searched recursively; Kaggle may unpack .gz uploads)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="skip subjects whose outputs already exist")
    ap.add_argument("--workers", type=int, default=1, help="subjects processed in parallel (FSL tools are single-threaded)")
    ap.add_argument("--dark-frac", type=float, default=L.DARK_FRAC, help="dark class = brain voxels below this fraction of median WM T2")
    ap.add_argument("--mni", default=None, help="default: $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz (no T2 template ships with conda FSL)")
    args = ap.parse_args()

    mni = Path(args.mni or Path(os.environ["FSLDIR"]) / "data" / "standard" / "MNI152_T1_1mm_brain.nii.gz")
    if not mni.exists():
        sys.exit(f"MNI template not found: {mni}")
    files = sorted(f for f in Path(args.src).rglob("*") if f.name.endswith((".nii", ".nii.gz")))[: args.limit]
    if not files:
        sys.exit(f"no .nii / .nii.gz files under {args.src}")
    out = Path(args.dst) / "summary.csv"
    Path(args.dst).mkdir(parents=True, exist_ok=True)

    def job(f):
        sid = f.name.split("_")[0]
        row = row_from_outputs(Path(args.dst) / sid) if args.resume else None
        return (f, row, True) if row is not None else (f, process(f, args.dst, mni, args.dark_frac), False)

    done = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(job, f) for f in files]
        for n, fut in enumerate(as_completed(futs), 1):
            f, row, skipped = fut.result()
            done[f] = row
            print(f"[{n}/{len(files)}] {'skipped (already done)' if skipped else 'done'} {f.name}", flush=True)
            print(row, flush=True)
            write_summary([done[g] for g in files if g in done], out)   # saved after every subject
    rows = [done[g] for g in files]
    print("wrote", out)
    for r in rows:
        order = [r["CSF_meanT2"], r["GM_meanT2"], r["WM_meanT2"], r["dark_meanT2"]]
        flag = "" if order[0] > order[1] > order[2] > order[3] else "   <-- WARNING: not CSF > GM > WM > dark"
        print(r["id"], "mean T2 CSF, GM, WM, dark:", order, "dark share:", r["dark_frac"], flag)


if __name__ == "__main__":
    main()
