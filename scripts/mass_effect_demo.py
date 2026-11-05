"""Picture of the mass-effect deformation on real HCP scans: the same tumor pasted with the push off / normal / strong.

    python scripts/mass_effect_demo.py --hcp "C:/.../hcp_processed" --bank "C:/.../tumor_bank" --out mass_effect_demo.png --n 3 --seed 1

Row = one tumor placement; columns = no mass effect, strength 1, strength 2 (axial slice through the tumor centre, red = tumor outline).
Also prints how often the no-folding safeguard had to reduce the push (det J stayed >= 0.3 on the brain).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.io import find_nifti, load_nifti  # noqa: E402
from data.labels import TUMOR  # noqa: E402
from synth import bank  # noqa: E402
from synth import paste as P  # noqa: E402


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--hcp", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", default="mass_effect_demo.png")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    scans = sorted({p.parent for p in Path(args.hcp).rglob("labels.nii*")})
    files = bank.list_entries(args.bank)
    fig, ax = plt.subplots(args.n, 3, figsize=(9, 3 * args.n))
    for i in range(args.n):
        d = scans[rng.integers(len(scans))]
        t2 = load_nifti(find_nifti(d, "t2"))[0].astype(np.float32)
        lab = load_nifti(find_nifti(d, "labels"))[0].astype(np.uint8)
        mask = load_nifti(find_nifti(d, "mask"))[0] > 0
        entry = bank.load_entry(files[rng.integers(len(files))])
        seed = int(rng.integers(1 << 30))
        for j, strength in enumerate((0.0, 1.0, 2.0)):
            res = P.paste(np.random.default_rng(seed), t2, lab, mask, entry, mass=strength)
            if res is None:
                ax[i, j].axis("off"); continue
            x, labels, parts = res
            c = [int(round(v)) for v in ndimage.center_of_mass(labels == TUMOR)]
            sl = np.s_[:, :, c[2]]
            ax[i, j].imshow(np.rot90(x[sl]), cmap="gray", vmin=0, vmax=1)
            ax[i, j].contour(np.rot90((labels == TUMOR)[sl]), levels=[0.5], colors="r", linewidths=0.6)
            ax[i, j].set_title(f"push strength {strength:g}", fontsize=8); ax[i, j].axis("off")
    fig.tight_layout(); fig.savefig(args.out, dpi=80)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
