"""Evaluate a trained model on the preprocessed BraTS test cases (whole tumor): DSC, HD95, AUPRC, SEN.

    python eval/run_eval.py --ckpt <run>/final.pt --brats <brats_processed> --out <run>/eval_brats2020

Run it once per finished model (never to pick checkpoints). Sliding window 96^3 with 25% overlap (paper).
Writes <out>/per_case.csv, <out>/summary.json and <out>/pred_<id>.nii.gz (predicted whole-tumor mask).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from monai.inferers import sliding_window_inference

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import labels as L  # noqa: E402
from data.io import find_nifti  # noqa: E402
from eval import metrics as M  # noqa: E402
from train.model import build_model  # noqa: E402


@torch.no_grad()
def predict_tumor_prob(model, t2, classes, device, roi=96, overlap=0.25, sw_batch=2):
    """t2 (D,H,W) numpy in [0,1] -> tumor probability map (D,H,W) numpy (softmax of the tumor channel)."""
    x = torch.from_numpy(np.ascontiguousarray(t2, dtype=np.float32))[None, None].to(device)
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
        logits = sliding_window_inference(x, (roi,) * 3, sw_batch, model, overlap=overlap)
    prob = torch.softmax(logits.float(), 1)[0]
    if classes == 8:                                     # NCR / ET / ED (classes 5, 6, 7) -> one tumor class
        prob = torch.cat([prob[:5], prob[5:].sum(0, keepdim=True)])
    return prob[L.TUMOR if classes in (6, 8) else 1].cpu().numpy(), prob.argmax(0).cpu().numpy()


def largest_component(pred):
    """Keep only the biggest connected piece of the prediction (removes scattered false-positive specks)."""
    lab, k = ndimage.label(pred)
    return lab == (1 + np.bincount(lab.ravel())[1:].argmax()) if k else pred


def save_pictures(items, path):
    """Rows = cases (worst first): T2 with the true tumor (green) and the predicted tumor (red), at the true tumor's centre slice."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage
    fig, ax = plt.subplots(len(items), 3, figsize=(9, 3 * len(items)))
    for i, (name, t2, gt, pred) in enumerate(items):
        c = [int(round(v)) for v in ndimage.center_of_mass(gt)]
        for j, sl in enumerate((np.s_[:, :, c[2]], np.s_[:, c[1], :], np.s_[c[0]])):
            a = ax[i, j]
            a.imshow(np.rot90(t2[sl]), cmap="gray", vmin=0, vmax=1)
            if gt[sl].any(): a.contour(np.rot90(gt[sl]), levels=[0.5], colors="lime", linewidths=0.8)
            if pred[sl].any(): a.contour(np.rot90(pred[sl]), levels=[0.5], colors="red", linewidths=0.8)
            a.axis("off")
        ax[i, 0].set_title(name, fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=70); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--brats", required=True, help="folder with one sub-folder per case (t2.nii.gz, tumor.nii.gz, mask.nii.gz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.ckpt, map_location=dev)
    classes = state["classes"]
    model = build_model(classes).to(dev)
    model.load_state_dict(state["model"]); model.eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pics, rows_lcc = [], []
    cases, rows = sorted(p for p in Path(args.brats).iterdir() if p.is_dir() and find_nifti(p, "t2"))[: args.limit], []
    for d in cases:
        img = nib.load(str(find_nifti(d, "t2")))
        t2 = np.asarray(img.dataobj, dtype=np.float32)
        gt = np.asarray(nib.load(str(find_nifti(d, "tumor"))).dataobj) > 0
        brain = np.asarray(nib.load(str(find_nifti(d, "mask"))).dataobj) > 0
        prob, arg = predict_tumor_prob(model, t2, classes, dev)
        pred = (arg == L.TUMOR) if classes in (6, 8) else (prob > 0.5)
        r = {"id": d.name, **M.evaluate_case(pred, gt, prob, spacing=img.header.get_zooms()[:3], brain_mask=brain)}
        r["pred_voxels"], r["gt_voxels"] = int(pred.sum()), int(gt.sum())
        rows.append(r)
        keep = largest_component(pred)
        rows_lcc.append(M.evaluate_case(keep, gt, prob * keep, spacing=img.header.get_zooms()[:3], brain_mask=brain))
        pics.append((d.name, t2, gt, pred, r["dsc"]))
        nib.save(nib.Nifti1Image(pred.astype(np.uint8), img.affine), str(out / f"pred_{d.name}.nii.gz"))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    pics.sort(key=lambda t: t[4])
    save_pictures([p[:4] for p in pics[:3] + pics[len(pics) // 2: len(pics) // 2 + 1] + pics[-1:]], out / "pictures.png")   # 3 worst, 1 middle, 1 best
    with open(out / "per_case.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = M.summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    lcc = M.summarize(rows_lcc)
    (out / "summary_largest_piece.json").write_text(json.dumps(lcc, indent=2))     # extra post-processing, reported separately
    print(f"\n{len(rows)} cases")
    for k, v in summary.items():
        print(f"{k.upper():6s} {v['mean']:.4f} +- {v['std']:.4f}" + (f"  ({v['n_excluded']} excluded: inf/nan)" if v["n_excluded"] else ""))
    print("with only the largest connected piece kept (extra, reported separately): " + "  ".join(f"{k.upper()} {v['mean']:.4f}" for k, v in lcc.items()))


if __name__ == "__main__":
    main()
