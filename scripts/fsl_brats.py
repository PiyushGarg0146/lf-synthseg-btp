"""BraTS 2020 test cases -> the same MNI152 1 mm space and intensity scaling as the HCP training scans.

Needs FSL (`flirt`) on PATH and $FSLDIR set - run on Linux/Kaggle (see notebooks/02_brats_preprocess.ipynb).

Per case, outputs in <dst>/<id>/:
  t2.nii.gz      normalized T2 in MNI152 1 mm space ([0, 1], background 0)
  tumor.nii.gz   whole-tumor mask (BraTS labels 1, 2, 4 -> 1), uint8
  seg.nii.gz     original BraTS labels (0, 1, 2, 4), uint8, same space
  mask.nii.gz    brain mask (includes the tumor)
  flirt.mat      the affine used
  qc.png         slices with the tumor outline
Also writes <dst>/summary.csv, including the tumor volume before/after registration (should agree within ~2%).

    python scripts/fsl_brats.py --src /kaggle/input --cases configs/test_split_brats2020.txt --dst /kaggle/working/brats_processed
"""
import argparse
import csv
import os
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from data import preprocess as P  # noqa: E402
from fsl_pipeline import brain_mask  # noqa: E402

CASE_T2 = re.compile(r"(BraTS20_Training_\d+)_t2\.nii(\.gz)?$")


def read_case_ids(path):
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]


def find_case_files(root, case_ids):
    """Locate each case's T2 and segmentation under `root` (searched recursively).

    BraTS 2020 case BraTS20_Training_355 has a non-standard segmentation name (..._Segm.nii), so the
    segmentation is the file in the same folder whose name contains 'seg' (any case).
    """
    wanted = set(case_ids)
    found = {}
    for t2 in Path(root).rglob("*_t2.nii*"):
        m = CASE_T2.match(t2.name)
        if not m or m.group(1) not in wanted:
            continue
        cid = m.group(1)
        segs = [f for f in t2.parent.iterdir() if "seg" in f.name.lower() and f.name.endswith((".nii", ".nii.gz"))]
        if len(segs) != 1:
            raise RuntimeError(f"{cid}: expected exactly one segmentation next to {t2.name}, found {[s.name for s in segs]}")
        found[cid] = (t2, segs[0])
    missing = sorted(wanted - set(found))
    if missing:
        raise FileNotFoundError(f"{len(missing)} case(s) not found under {root}: {missing[:5]}")
    return found


def tumor_volume(seg):
    return int((np.asarray(seg) > 0).sum())


def qc_png(t2, tumor, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = [int(round(x)) for x in ndimage.center_of_mass(tumor)] if tumor.any() else [s // 2 for s in t2.shape]
    views = [(t2[:, :, c[2]], tumor[:, :, c[2]]), (t2[:, c[1], :], tumor[:, c[1], :]), (t2[c[0], :, :], tumor[c[0], :, :])]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.5))
    for a, (img, tm) in zip(ax, views):
        a.imshow(np.rot90(img), cmap="gray", vmin=0, vmax=1)
        if tm.any():
            a.contour(np.rot90(tm), levels=[0.5], colors="r", linewidths=1)
        a.axis("off")
    fig.suptitle("T2 with whole-tumor outline (red), MNI space")
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def process(cid, t2_path, seg_path, out_dir, mni):
    work = Path(out_dir) / cid
    work.mkdir(parents=True, exist_ok=True)
    t2_nat = P.standardize_native(t2_path, work / "t2_native_ras1mm.nii.gz", order=3)
    seg_nat = P.standardize_native(seg_path, work / "seg_native_ras1mm.nii.gz", order=0)
    n_native = tumor_volume(nib.load(str(seg_nat)).dataobj)

    t2_reg, mat = work / "t2_mni_raw.nii.gz", work / "flirt.mat"
    P.register_flirt(t2_nat, mni, t2_reg, mat)
    seg_reg = work / "seg_mni.nii.gz"
    P.apply_flirt(seg_nat, mni, mat, seg_reg, nearest=True)

    img = nib.load(str(t2_reg))
    data = np.clip(np.asarray(img.dataobj, dtype=np.float32), 0, None)
    seg = np.rint(np.asarray(nib.load(str(seg_reg)).dataobj)).astype(np.uint8)
    tumor = seg > 0
    mask = ndimage.binary_fill_holes(brain_mask(data) | tumor)
    data[~mask] = 0

    t2n = P.normalize_intensity(data, mask)
    aff = img.affine
    nib.save(nib.Nifti1Image(t2n, aff), str(work / "t2.nii.gz"))
    nib.save(nib.Nifti1Image(tumor.astype(np.uint8), aff), str(work / "tumor.nii.gz"))
    nib.save(nib.Nifti1Image(seg, aff), str(work / "seg.nii.gz"))
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), aff), str(work / "mask.nii.gz"))
    qc_png(t2n, tumor, work / "qc.png")
    n_mni = int(tumor.sum())
    return {"id": cid, "tumor_native": n_native, "tumor_mni": n_mni,
            "volume_ratio": round(n_mni / max(n_native, 1), 4), "brain_voxels": int(mask.sum())}


def main():
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder containing the BraTS 2020 training cases (searched recursively)")
    ap.add_argument("--cases", required=True, help="text file with one case id per line (the frozen test split)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--mni", default=None, help="default: $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz")
    args = ap.parse_args()

    mni = Path(args.mni or Path(os.environ["FSLDIR"]) / "data" / "standard" / "MNI152_T1_1mm_brain.nii.gz")
    if not mni.exists():
        sys.exit(f"MNI template not found: {mni}")
    ids = read_case_ids(args.cases)[: args.limit]
    files = find_case_files(args.src, ids)
    Path(args.dst).mkdir(parents=True, exist_ok=True)
    out = Path(args.dst) / "summary.csv"

    done = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(process, c, *files[c], args.dst, mni) for c in ids]
        for n, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            done[row["id"]] = row
            print(f"[{n}/{len(ids)}] {row}", flush=True)
            rows = [done[c] for c in ids if c in done]
            with open(out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", out)
    for r in done.values():
        if abs(r["volume_ratio"] - 1) > 0.02:
            print(f"WARNING {r['id']}: tumor volume changed by {100 * (r['volume_ratio'] - 1):+.1f}% after registration (cut by the field of view?)")


if __name__ == "__main__":
    main()
