"""Make a picture grid of synthetic tumors pasted into the HCP healthy scans (visual realism check).

    python scripts/synth_demo.py --hcp "C:/.../hcp_processed" --out synth_demo.png --n 6 --seed 0
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.io import load_nifti  # noqa: E402
from synth import paste as P  # noqa: E402
from synth import tumor as T  # noqa: E402


def load_subject(d):
    d = Path(d)
    t2 = load_nifti(d / "t2.nii.gz")[0]
    lab = load_nifti(d / "labels.nii.gz")[0].astype(np.uint8)
    mask = load_nifti(d / "mask.nii.gz")[0] > 0
    return t2, lab, mask


def find_subjects(root):
    return sorted(p.parent for p in Path(root).rglob("labels.nii.gz"))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--hcp", required=True)
    ap.add_argument("--out", default="synth_demo.png")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    subs = find_subjects(args.hcp)
    fig, ax = plt.subplots(args.n, 3, figsize=(9, 3 * args.n))
    for i in range(args.n):
        t2, lab, mask = load_subject(subs[rng.integers(len(subs))])
        t0 = time.time()
        entry = None
        while entry is None:
            entry = T.build_tumor(rng, t2, lab, mask)
        t1 = time.time()
        res = None
        while res is None:
            res = P.paste(rng, t2, lab, mask, P.augment(rng, entry))
        x, labels, parts = res
        print(f"tumor {i}: build {t1 - t0:.1f}s, paste {time.time() - t1:.1f}s, volume {int((labels == 5).sum())}")
        c = [int(round(v)) for v in ndimage.center_of_mass(parts > 0)]
        for j, (img, tm) in enumerate([(x[:, :, c[2]], parts[:, :, c[2]]), (x[:, c[1], :], parts[:, c[1], :]), (x[c[0]], parts[c[0]])]):
            ax[i, j].imshow(np.rot90(img), cmap="gray", vmin=0, vmax=1)
            ax[i, j].contour(np.rot90(tm > 0), levels=[0.5], colors="r", linewidths=0.6)
            ax[i, j].axis("off")
    fig.tight_layout(); fig.savefig(args.out, dpi=80)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
