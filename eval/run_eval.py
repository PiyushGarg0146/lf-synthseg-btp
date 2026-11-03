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
from monai.inferers import sliding_window_inference

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import labels as L  # noqa: E402
from eval import metrics as M  # noqa: E402
from train.model import build_model  # noqa: E402


@torch.no_grad()
def predict_tumor_prob(model, t2, classes, device, roi=96, overlap=0.25, sw_batch=2):
    """t2 (D,H,W) numpy in [0,1] -> tumor probability map (D,H,W) numpy (softmax of the tumor channel)."""
    x = torch.from_numpy(np.ascontiguousarray(t2, dtype=np.float32))[None, None].to(device)
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
        logits = sliding_window_inference(x, (roi,) * 3, sw_batch, model, overlap=overlap)
    prob = torch.softmax(logits.float(), 1)[0]
    return prob[L.TUMOR if classes == 6 else 1].cpu().numpy(), prob.argmax(0).cpu().numpy()


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

    cases, rows = sorted(p for p in Path(args.brats).iterdir() if (p / "t2.nii.gz").exists())[: args.limit], []
    for d in cases:
        img = nib.load(str(d / "t2.nii.gz"))
        t2 = np.asarray(img.dataobj, dtype=np.float32)
        gt = np.asarray(nib.load(str(d / "tumor.nii.gz")).dataobj) > 0
        brain = np.asarray(nib.load(str(d / "mask.nii.gz")).dataobj) > 0
        prob, arg = predict_tumor_prob(model, t2, classes, dev)
        pred = (arg == L.TUMOR) if classes == 6 else (prob > 0.5)
        r = {"id": d.name, **M.evaluate_case(pred, gt, prob, spacing=img.header.get_zooms()[:3], brain_mask=brain)}
        r["pred_voxels"], r["gt_voxels"] = int(pred.sum()), int(gt.sum())
        rows.append(r)
        nib.save(nib.Nifti1Image(pred.astype(np.uint8), img.affine), str(out / f"pred_{d.name}.nii.gz"))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    with open(out / "per_case.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = M.summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{len(rows)} cases")
    for k, v in summary.items():
        print(f"{k.upper():6s} {v['mean']:.4f} +- {v['std']:.4f}" + (f"  ({v['n_excluded']} excluded: inf/nan)" if v["n_excluded"] else ""))


if __name__ == "__main__":
    main()
