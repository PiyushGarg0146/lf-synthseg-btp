"""Copy only what training/evaluation needs into three folders (and zips) for upload as Kaggle datasets.

    python scripts/pack_kaggle_data.py --root "C:/.../btp data"

Creates <root>/kaggle_upload/{hcp_lean, tumor_bank, brats_test}[.zip]:
  hcp_lean    <id>/t2.nii.gz, labels.nii.gz, mask.nii.gz   (20 healthy scans; FAST/QC extras left out)
  tumor_bank  tumor_xxxxx.npz                              (the 1000 synthetic tumors)
  brats_test  <case>/t2.nii.gz, tumor.nii.gz, mask.nii.gz  (the 31 frozen test cases)
"""
import argparse
import shutil
from pathlib import Path


def copy_tree(src_dirs, names, dst):
    n = 0
    for d in src_dirs:
        (dst / d.name).mkdir(parents=True, exist_ok=True)
        for f in names:
            shutil.copy2(d / f, dst / d.name / f)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    up = root / "kaggle_upload"
    up.mkdir(exist_ok=True)

    hcp = sorted(p.parent for p in (root / "hcp_processed").rglob("labels.nii.gz"))
    brats = sorted(p.parent for p in (root / "brats_processed").rglob("tumor.nii.gz"))
    bank = sorted((root / "tumor_bank").glob("*.npz"))
    assert len(hcp) == 20 and len(brats) == 31 and len(bank) >= 1000, (len(hcp), len(brats), len(bank))
    for name in ("hcp_lean", "tumor_bank", "brats_test"):
        shutil.rmtree(up / name, ignore_errors=True)
    copy_tree(hcp, ["t2.nii.gz", "labels.nii.gz", "mask.nii.gz"], up / "hcp_lean")
    copy_tree(brats, ["t2.nii.gz", "tumor.nii.gz", "mask.nii.gz"], up / "brats_test")
    (up / "tumor_bank").mkdir()
    for f in bank:
        shutil.copy2(f, up / "tumor_bank" / f.name)
    for name in ("hcp_lean", "tumor_bank", "brats_test"):
        z = shutil.make_archive(str(up / name), "zip", up / name)
        print(f"{name}: {sum(f.stat().st_size for f in (up / name).rglob('*') if f.is_file()) / 1e6:.0f} MB -> {Path(z).name} {Path(z).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
