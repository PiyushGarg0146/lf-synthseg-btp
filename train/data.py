"""On-the-fly synthetic training data: healthy scan + augmented bank tumor -> 96^3 patch with 5-class labels.

Split of work (so a few CPU cores can feed a GPU): data-loader workers only cut a host patch out of a healthy scan,
choose where the tumor goes and draw the augmentation numbers (cheap); the rotation / deformation / blending of the
tumor (`synth.paste.render`) then runs on the GPU in the training process (`render_batch`).

Patch placement (our choice; the paper only says "random 96^3 patches"): with probability P_NEAR the patch is
centred within +-NEAR_MM of the tumor centre (so most patches contain tumor), otherwise on a random brain voxel,
which gives tumor-free patches that teach the model not to fire on normal anatomy.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info

from data import labels as L
from data.io import find_nifti, load_nifti
from synth import bank
from synth import mass_effect as ME
from synth import paste as P
from synth import shapes as S
from synth import tumor as T

PATCH = 96
P_NEAR = 0.7
NEAR_MM = 48


def find_scans(root):
    return sorted({p.parent for p in Path(root).rglob("labels.nii*")})


def load_scans(root, limit=None):
    scans = []
    for d in find_scans(root)[:limit]:
        t2 = load_nifti(find_nifti(d, "t2"))[0].astype(np.float32)
        lab = load_nifti(find_nifti(d, "labels"))[0].astype(np.uint8)
        mask = load_nifti(find_nifti(d, "mask"))[0] > 0
        scans.append({"id": d.name, "t2": t2, "labels": lab, "mask": mask, "inside": np.argwhere(mask),
                      "mu_gm": float(t2[lab == L.GM].mean()), "mu_csf": float(t2[lab == L.CSF].mean())})
    if not scans:
        raise FileNotFoundError(f"no scans (labels.nii.gz) under {root}")
    return scans


def make_item(rng, scan, raw, patch=PATCH, mass=0.0):
    """Everything needed to render one training sample (numpy only); None if the tumor does not fit the brain."""
    g = S.GRID
    lo = raw["lo"]
    off = np.argwhere(raw["parts"] > 0) + lo - g // 2                  # tumor voxels relative to its centre voxel
    c = T.place_center(rng, scan["mask"], inside=scan["inside"], off=off)
    if c is None:
        return None
    if rng.random() < P_NEAR:
        p = c + rng.uniform(-NEAR_MM, NEAR_MM, 3)
    else:
        p = scan["inside"][rng.integers(len(scan["inside"]))]
    p = np.clip(np.round(p), 0, np.array(scan["mask"].shape) - 1).astype(int)
    params = P.sample_params(rng)
    if mass > 0:
        params["mass"] = ME.sample_mass(rng, mass)
    origin = p - patch // 2
    return {"t2": T.crop(scan["t2"], p, patch), "labels": T.crop(scan["labels"], p, patch), "mask": T.crop(scan["mask"], p, patch),
            "raw": raw, "ctr": (g - 1) / 2 - g // 2 + c - origin,       # tumor-grid centre in patch coordinates
            "params": params, "mu_gm": scan["mu_gm"], "mu_csf": scan["mu_csf"]}


class SynthDataset(IterableDataset):
    """Endless stream of raw items (see `make_item`). Each worker has its own random stream (seed, worker id)."""

    def __init__(self, scans, bank_dir, seed=0, patch=PATCH, mass=0.0):
        self.scans, self.files, self.seed, self.patch, self.mass = scans, bank.list_entries(bank_dir), seed, patch, mass
        if not self.files:
            raise FileNotFoundError(f"no tumor bank files in {bank_dir}")

    def __iter__(self):
        info = get_worker_info()
        torch.set_num_threads(1)
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        while True:
            item = make_item(rng, self.scans[rng.integers(len(self.scans))], bank.load_raw(self.files[rng.integers(len(self.files))]), self.patch, self.mass)
            if item is not None:
                yield item


def collate_list(items):
    return items


def render_item(item, device="cpu"):
    """Raw item -> (image, labels, parts) tensors of the patch on `device`."""
    t = lambda a, dt: torch.as_tensor(a, device=device).to(dt)
    return P.render(P.entry_tensors(item["raw"], device), t(item["t2"], torch.float32), t(item["labels"], torch.long),
                    t(item["mask"], torch.bool), item["ctr"], item["params"], item["mu_gm"], item["mu_csf"])


def widen_csf(x, label, rng_k):
    """Make the fluid spaces (ventricles, sulci) wider in some samples, so enlarged CSF counts as normal anatomy.
    The healthy HCP scans are young adults with small ventricles; BraTS patients are older. New CSF voxels take the
    scan's own CSF brightness and the label becomes CSF; tumor and outside-brain voxels are left alone."""
    csf = (label == L.CSF).float()
    grown = F.max_pool3d(csf, rng_k, 1, rng_k // 2) > 0
    new = grown & (label > 0) & (label != L.TUMOR) & (csf == 0)
    level = (x * csf).sum((1, 2, 3, 4), keepdim=True) / csf.sum((1, 2, 3, 4), keepdim=True).clamp_min(1)
    return torch.where(new, level.expand_as(x), x), torch.where(new, torch.full_like(label, L.CSF), label)


def augment_image(x, label):
    """Appearance changes so the network cannot rely on the exact look of the clean HCP scans (real BraTS scans are
    noisier, blurrier, have a slow brightness drift and different tissue contrast, and older brains have bigger fluid spaces).
    x (B,1,D,H,W) in [0,1], label (B,1,D,H,W) long (0 = outside brain, 5 = tumor). Each sample gets its own random numbers.
    Returns the new image and the (possibly changed) label."""
    b = x.shape[0]
    u = lambda lo, hi: torch.empty(b, 1, 1, 1, 1, device=x.device).uniform_(lo, hi)
    if torch.rand(1).item() < 0.5:                                             # wider ventricles / sulci (whole batch)
        x, label = widen_csf(x, label, int(torch.randint(1, 3, (1,))) * 2 + 1)
    brain = label > 0
    f = torch.ones_like(x)                                                     # brightness factor per tissue (CSF, GM, WM, dark, tumor)
    for cls, (lo, hi) in {L.CSF: (0.85, 1.3), L.GM: (0.8, 1.25), L.WM: (0.8, 1.25), 4: (0.8, 1.2), L.TUMOR: (0.85, 1.2)}.items():
        f = torch.where(label == cls, u(lo, hi).expand_as(x), f)
    out = x.clamp(0, 1) * F.avg_pool3d(f, 3, 1, 1)                             # smoothed so tissue borders stay soft
    out = out.clamp(0, 1) ** u(0.7, 1.5)                                       # global contrast (gamma)
    out = out * u(0.8, 1.2)                                                    # global brightness
    low = torch.randn(b, 1, 4, 4, 4, device=x.device)                          # slow brightness drift (bias field), +-25%
    out = out * (1 + 0.25 * torch.tanh(F.interpolate(low, size=x.shape[2:], mode="trilinear", align_corners=True)))
    k = int(torch.randint(0, 3, (1,)))                                         # blur: none / light / medium (same for the batch)
    if k:
        r = k + 1
        w = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / (0.5 * k + 0.3)) ** 2)
        w = w / w.sum()
        for ax in (2, 3, 4):
            shape = [1, 1, 1, 1, 1]; shape[ax] = -1
            pad = [0, 0, 0, 0, 0, 0]; pad[2 * (4 - ax) + 1] = r; pad[2 * (4 - ax)] = r
            out = F.conv3d(F.pad(out, pad, mode="replicate"), w.view(shape))
    out = out + u(0.0, 0.04) * torch.randn_like(out)                           # noise
    return out.clamp(0, 1) * brain, label


def render_batch(items, device="cpu", tumor_only=False, augment=False, subregions=False):
    """List of raw items -> dict(image (B,1,D,H,W) float, label (B,1,D,H,W) long, parts (B,1,D,H,W) long)."""
    xs, ls, ps = zip(*(render_item(it, device) for it in items))
    label = torch.stack(ls)[:, None]
    if augment:
        img, label = augment_image(torch.stack(xs)[:, None], label)
        xs = list(img[:, 0])
    parts = torch.stack(ps)[:, None].long()
    if tumor_only:
        label = (label == L.TUMOR).long()
    elif subregions:                                          # tumor split into NCR / ET / ED = classes 5 / 6 / 7
        label = torch.where(label == L.TUMOR, 4 + parts, label)
    return {"image": torch.stack(xs)[:, None], "label": label, "parts": parts}
