"""Augment a bank entry and paste it into a healthy scan (paper Eq. 7), producing image + 5-class labels.

Five random augmentations (paper III-A5): deformation, scaling, rotation, contrast, movement.
Contrast = gamma on the tumor + a brightness rescale (tumors are bright on T2: mean between GM and CSF of the host scan).
Movement = the tumor is pasted at a random place that is generally NOT where its intensities were taken from.
Ranges below are our choices (the paper gives none).

`render` does everything in one resampling step on any torch device (so training can run it on the GPU):
for every voxel of the host window it looks up the matching point of the (rotated, scaled, elastically bent)
tumor and blends it in. `paste` is the plain numpy wrapper.
"""
import numpy as np
import torch
import torch.nn.functional as F

from data import labels as L
from synth import mass_effect as ME
from synth import shapes as S
from synth import tumor as T

SCALE = (0.45, 1.3)      # volume changes with the cube: ~0.1x to ~2.2x (the bank alone has few tumors below ~30 mL)
ROTATE_DEG = 180
ELASTIC_MM = (0.0, 4.0)
GAMMA = (0.7, 1.5)
BRIGHT = (0.3, 1.0)     # tumor mean T2 = GM mean + u x (CSF mean - GM mean) of the host scan, u ~ U(BRIGHT)
_NOISE_CELLS = S.GRID // 28 + 3


def sample_params(rng):
    """All random numbers of one augmentation, as plain numpy (so they can be made in a data-loader worker)."""
    ang = np.deg2rad(rng.uniform(-ROTATE_DEG, ROTATE_DEG, 3))
    cx, cy, cz = np.cos(ang); sx, sy, sz = np.sin(ang)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    s = rng.uniform(*SCALE, size=3) if rng.random() < 0.5 else np.full(3, rng.uniform(*SCALE))
    return {"A": np.linalg.inv(Rz @ Ry @ Rx @ np.diag(s)).astype(np.float32),      # host voxel -> tumor grid
            "noise": rng.normal(size=(3, _NOISE_CELLS, _NOISE_CELLS, _NOISE_CELLS)).astype(np.float32),
            "amp": float(rng.uniform(*ELASTIC_MM)), "gamma": float(rng.uniform(*GAMMA)), "bright": float(rng.uniform(*BRIGHT))}


def entry_tensors(entry, device="cpu"):
    """Full GRID^3 entry (numpy: tumor, alpha, parts) or stored crop (with `lo`) -> float tensors on `device`."""
    g = S.GRID
    if "lo" in entry:
        lo = [int(v) for v in entry["lo"]]
        sl = tuple(slice(l, l + n) for l, n in zip(lo, entry["parts"].shape))
        full = {k: np.zeros((g,) * 3, np.float32) for k in ("tumor", "alpha", "parts")}
        full["tumor"][sl] = entry["tumor"]; full["alpha"][sl] = entry["alpha"] / 255.0; full["parts"][sl] = entry["parts"]
        entry = full
    return {k: torch.as_tensor(np.asarray(entry[k], dtype=np.float32), device=device) for k in ("tumor", "alpha", "parts")}


def render(entry_t, t2, labels, mask, ctr, params, mu_gm, mu_csf):
    """Blend one augmented tumor into a host window. All inputs are torch tensors on one device.

    entry_t: output of `entry_tensors`; t2/labels/mask: host window (D, H, W); ctr: position of the tumor grid's
    centre in window voxel coordinates; mu_gm/mu_csf: mean T2 of GM / CSF of the host scan.
    Returns (image, labels, parts) for the window: image = (1 - a) t2 + a tumor (a = alpha inside the brain),
    labels = host labels with tumor voxels = 5, parts = 0 none / 1 NCR / 2 ET / 3 ED (0 outside the brain).
    """
    dev, g = t2.device, S.GRID
    A = torch.as_tensor(params["A"], device=dev)
    ax = [torch.arange(n, device=dev, dtype=torch.float32) - float(ctr[i]) for i, n in enumerate(t2.shape)]
    base = torch.stack(torch.meshgrid(*ax, indexing="ij"), 0)
    coords = torch.einsum("ij,jxyz->ixyz", A, base) + (g - 1) / 2
    if params["amp"] > 0:                                          # smooth random displacement (deformation)
        d = F.interpolate(torch.as_tensor(params["noise"], device=dev)[None], size=tuple(t2.shape), mode="trilinear", align_corners=True)[0]
        coords = coords + d * (params["amp"] * 0.5)
    norm = 2 * coords / (g - 1) - 1                                # grid_sample wants (x, y, z) = (axis 2, 1, 0)
    grid = torch.stack([norm[2], norm[1], norm[0]], -1)[None]
    img = torch.stack([entry_t["tumor"], entry_t["alpha"]])[None]
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
    parts = F.grid_sample(entry_t["parts"][None, None], grid, mode="nearest", padding_mode="zeros", align_corners=True)[0, 0].round()
    tum, alpha = out[0], out[1].clamp(0, 1)
    if params.get("mass"):                                         # mass effect: push the host's tissue away first, then paste
        t2, labels, mask = ME.apply(t2, labels, mask, (parts > 0) & mask.bool(), params["mass"])
    brain = mask.bool()
    parts = parts * brain
    inside = parts > 0
    if inside.any():
        mu = tum[inside].mean().clamp_min(1e-4)                    # gamma changes the contrast, then the mean is set
        tum = (mu * (tum.clamp(1e-4, 1) / mu) ** params["gamma"]).clamp(0, 1)
        target = mu_gm + params["bright"] * (mu_csf - mu_gm)
        tum = (tum * (target / tum[inside].mean().clamp_min(1e-4))).clamp(0, 1)
    a = alpha * brain
    x = ((1 - a) * t2 + a * tum) * brain
    lab = torch.where(inside, torch.full_like(labels, L.TUMOR), labels)
    return x, lab, parts.to(torch.uint8)


def paste(rng, t2, labels, mask, entry, mass=0.0):
    """Numpy wrapper: paste `entry` (full GRID^3 arrays) at a random brain location (movement); mass > 0 turns mass effect on.

    Returns (image, labels, parts) of the full volume, or None if the tumor does not fit in the brain.
    """
    union = entry["parts"] > 0
    if not union.any():
        return None
    c = T.place_center(rng, mask, union)
    if c is None:
        return None
    g = S.GRID
    lo = [int(v) - g // 2 for v in c]
    src, dst = [], []
    for a, l in enumerate(lo):
        s0, s1 = max(l, 0), min(l + g, t2.shape[a])
        src.append(slice(s0, s1)); dst.append(slice(s0 - l, s1 - l))
    src = tuple(src)
    win = lambda v: torch.from_numpy(np.ascontiguousarray(v[src]))
    ctr = [(g - 1) / 2 - d.start for d in dst]                     # tumor-grid centre in window coordinates
    mu_gm, mu_csf = float(t2[labels == L.GM].mean()), float(t2[labels == L.CSF].mean())
    params = sample_params(rng)
    if mass > 0:
        params["mass"] = ME.sample_mass(rng, mass)
    x, lab, parts_w = render(entry_tensors(entry), win(t2), win(labels).long(), win(mask), ctr, params, mu_gm, mu_csf)
    out_x, out_lab = t2.copy(), labels.copy()
    out_x[src], out_lab[src] = x.numpy(), lab.numpy().astype(labels.dtype)
    parts = np.zeros(t2.shape, np.uint8)
    parts[src] = parts_w.numpy()
    return out_x, out_lab, parts
