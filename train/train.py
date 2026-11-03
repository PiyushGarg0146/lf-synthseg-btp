"""Train the segmentation network on synthetic tumors (C2 = LF-SynthSeg baseline with tissue labels, C1 = tumor-only).

    python train/train.py --hcp <hcp_processed> --bank <tumor_bank> --out <run_dir> --iters 20000 --batch 2 --workers 3

Loss: Dice + cross-entropy (paper Eq. 8). Schedule: linear warm-up then cosine annealing (paper). AMP on GPU.
Checkpoints every --ckpt-every iterations (latest.pt holds model, optimizer, scheduler, iteration) and the run resumes
from it automatically. The final model is saved as final.pt. No BraTS data is used here (no model selection on test data).
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch
from monai.losses import DiceCELoss
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.data import SynthDataset, collate_list, load_scans, render_batch  # noqa: E402
from train.model import build_model  # noqa: E402


def lr_at(it, total, base, warmup):
    if it < warmup:
        return base * (it + 1) / warmup
    return 0.5 * base * (1 + math.cos(math.pi * (it - warmup) / max(1, total - warmup)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hcp", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", type=int, choices=(2, 6), default=6, help="6 = LF-SynthSeg (C2: outside-brain 0 + CSF, GM, WM, dark, tumor), 2 = tumor-only (C1)")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-scans", type=int, default=None, help="use only the first N healthy scans (quick tests)")
    ap.add_argument("--max-minutes", type=float, default=None, help="stop (after saving) once this much time has passed")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.classes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    it0 = 0
    ck = out / "latest.pt"
    if ck.exists():
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); scaler.load_state_dict(s["scaler"]); it0 = s["iter"]
        print(f"resumed from iteration {it0}", flush=True)
    if it0 >= args.iters:
        print("already finished"); return

    scans = load_scans(args.hcp, args.max_scans)
    print(f"{len(scans)} healthy scans, device {dev}, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters", flush=True)
    ds = SynthDataset(scans, args.bank, seed=args.seed + it0, patch=args.patch)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None, collate_fn=collate_list)

    log = out / "log.csv"
    new_log = not log.exists()
    fh = open(log, "a", newline="")
    w = csv.writer(fh)
    if new_log:
        w.writerow(["iter", "loss", "lr", "seconds"]); fh.flush()

    def save(path, it):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "iter": it, "classes": args.classes}, path)

    t0, run, n = time.time(), 0.0, 0
    model.train()
    for it, batch in zip(range(it0, args.iters), dl):
        lr = lr_at(it, args.iters, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        b = render_batch(batch, dev, tumor_only=args.classes == 2)        # tumor rotation/blending runs on the GPU
        x, y = b["image"], b["label"]
        with torch.autocast(device_type=dev, dtype=torch.float16, enabled=dev == "cuda"):
            loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        run += loss.item(); n += 1
        if (it + 1) % 50 == 0:
            w.writerow([it + 1, round(run / n, 4), f"{lr:.2e}", round(time.time() - t0)]); fh.flush()
            print(f"iter {it + 1}/{args.iters} loss {run / n:.4f} lr {lr:.2e} {(time.time() - t0) / (it + 1 - it0):.2f}s/it", flush=True)
            run, n = 0.0, 0
        stop = args.max_minutes is not None and (time.time() - t0) / 60 > args.max_minutes
        if (it + 1) % args.ckpt_every == 0 or stop or it + 1 == args.iters:
            save(ck, it + 1)
        if stop:
            print("time limit reached - checkpoint saved, rerun the same command to continue"); return
    torch.save({"model": model.state_dict(), "classes": args.classes}, out / "final.pt")
    print("training finished ->", out / "final.pt")


if __name__ == "__main__":
    main()
