"""Freeze the BraTS 2020 test split and extract only those cases' T2 + segmentation.

Draws N cases (default 31, as in the paper) from the 369 labelled training cases with a
fixed seed, writes the IDs to configs/test_split_brats2020.txt, and unzips just their
T2 and seg files. Refuses to overwrite an existing split file: the split is frozen.

    python scripts/make_test_split.py
"""
import argparse
import random
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.io import load_paths  # noqa: E402

PREFIX = "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"
SPLIT_FILE = ROOT / "configs" / "test_split_brats2020.txt"


def list_cases(z):
    ids = {m.group(1) for n in z.namelist() if (m := re.match(re.escape(PREFIX) + r"(BraTS20_Training_\d+)/", n))}
    return sorted(ids)


def seg_member(z, case):
    """Name of the segmentation file of a case (case 355 has a non-standard name)."""
    names = [n for n in z.namelist() if n.startswith(f"{PREFIX}{case}/") and n.endswith(".nii")]
    segs = [n for n in names if n.endswith("_seg.nii")] or [n for n in names if "Segm" in n]
    if len(segs) != 1:
        raise RuntimeError(f"{case}: expected one segmentation, found {segs}")
    return segs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=31)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths = load_paths()
    out_dir = paths["brats2020_test"]
    with zipfile.ZipFile(paths["brats2020_zip"]) as z:
        cases = list_cases(z)
        if SPLIT_FILE.exists():
            chosen = [l.strip() for l in SPLIT_FILE.read_text().splitlines() if l.strip() and not l.startswith("#")]
            print(f"Split already frozen ({len(chosen)} cases); only (re)extracting files.")
        else:
            assert len(cases) == 369, f"expected 369 training cases, found {len(cases)}"
            chosen = sorted(random.Random(args.seed).sample(cases, args.n))
            SPLIT_FILE.write_text(f"# BraTS 2020 test split: {args.n} of {len(cases)} training cases, seed {args.seed}. FROZEN - do not edit.\n"
                                  + "\n".join(chosen) + "\n")
            print(f"Wrote {SPLIT_FILE}")
        for c in chosen:
            dst = out_dir / c
            dst.mkdir(parents=True, exist_ok=True)
            (dst / f"{c}_t2.nii").write_bytes(z.read(f"{PREFIX}{c}/{c}_t2.nii"))
            (dst / f"{c}_seg.nii").write_bytes(z.read(seg_member(z, c)))
    print(f"Extracted {len(chosen)} cases to {out_dir}")


if __name__ == "__main__":
    main()
