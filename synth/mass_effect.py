"""Mass effect: push the host brain's tissue away from a (synthetic) tumor, smoothly and without folding (Stage D).

Real tumors do not only replace tissue, they also displace it: the surroundings are pushed outward, ventricles near the
lesion are squeezed, and the push fades with distance and stays inside the skull.

Field (all on any torch device, so it can run on the GPU inside the training loop):
  phi = Gaussian blur of the tumor mask          (smooth, 1 inside, 0 far away)
  u   = -grad(phi), scaled so that its largest length is `amp` voxels   (points away from the tumor, peaks at its surface)
  u  *= taper                                    (0 at the brain edge, 1 deeper than ~`taper` voxels inside)
The host scan is warped backwards, x_source = x - u(x): trilinear for the image, nearest neighbour for labels and
mask (the same field for all three). The tumor is pasted AFTER this, so the tumor itself is not distorted.
Folding is prevented by measurement: `safe_field` halves the amplitude until det(I - Du) stays above `min_det`
everywhere in the brain.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

AMP_FRAC = (0.0, 0.3)       # peak push as a fraction of the tumor's equivalent radius (before the strength factor)
AMP_MAX = 15.0              # voxels (= mm); real mass effect is a few mm up to about 1 cm
SIGMA_FRAC = (0.4, 0.9)     # reach (blur sigma) as a fraction of the equivalent radius
SIGMA_MAX = 20.0
TAPER = 6.0                 # blur sigma of the brain mask used for the edge taper


def blur(x, sigma):
    """Gaussian blur of a (D,H,W) float tensor (replicate borders)."""
    r = max(1, int(math.ceil(3 * sigma)))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma) ** 2)
    k = k / k.sum()
    y = x[None, None]
    for ax in (2, 3, 4):
        shape = [1, 1, 1, 1, 1]; shape[ax] = -1
        pad = [0] * 6; pad[2 * (4 - ax)] = r; pad[2 * (4 - ax) + 1] = r
        y = F.conv3d(F.pad(y, pad, mode="replicate"), k.view(shape))
    return y[0, 0]


def field(tumor, brain, amp, sigma, taper=TAPER):
    """Displacement u (3,D,H,W) in voxels, pointing away from the tumor, peak length `amp` before the edge taper."""
    if amp <= 0:
        return torch.zeros((3,) + tuple(tumor.shape), device=tumor.device)
    g = torch.stack(torch.gradient(blur(tumor.float(), sigma)))          # points into the tumor
    u = -g * (amp / g.norm(dim=0).max().clamp_min(1e-8))
    w = ((blur(brain.float(), taper) - 0.5) / 0.5).clamp(0, 1)           # 0 at / outside the brain edge, 1 well inside
    return u * (w * w * (3 - 2 * w))


def jacobian_det(u):
    """det of the Jacobian of the backward map x -> x - u(x), per voxel (D,H,W)."""
    J = torch.stack([torch.stack(torch.gradient(u[i])) for i in range(3)])   # J[i, j] = d u_i / d x_j
    M = torch.eye(3, device=u.device).view(3, 3, 1, 1, 1) - J
    return torch.linalg.det(M.permute(2, 3, 4, 0, 1))


def safe_field(tumor, brain, amp, sigma, taper=TAPER, min_det=0.3, tries=5):
    """`field`, with the amplitude halved until det J >= min_det on every brain voxel (zero field if it never holds)."""
    for _ in range(tries):
        u = field(tumor, brain, amp, sigma, taper)
        if amp <= 0 or float(jacobian_det(u)[brain].min()) >= min_det:
            return u
        amp *= 0.5
    return torch.zeros_like(u)


def warp(t2, labels, mask, u):
    """Warp host window (t2 (D,H,W) float, labels (D,H,W) int, mask (D,H,W) bool) with displacement u (3,D,H,W)."""
    n = torch.tensor(t2.shape, device=t2.device, dtype=torch.float32).view(3, 1, 1, 1)
    ax = [torch.arange(s, device=t2.device, dtype=torch.float32) for s in t2.shape]
    src = torch.stack(torch.meshgrid(*ax, indexing="ij"), 0) - u
    norm = 2 * src / (n - 1) - 1
    grid = torch.stack([norm[2], norm[1], norm[0]], -1)[None]                # grid_sample wants (x, y, z) = (axis 2, 1, 0)
    sample = lambda v, mode: F.grid_sample(v[None, None].float(), grid, mode=mode, padding_mode="border", align_corners=True)[0, 0]
    return sample(t2, "bilinear"), sample(labels, "nearest").round().to(labels.dtype), sample(mask, "nearest") > 0.5


def sample_mass(rng, strength=1.0):
    """Random numbers for one sample (plain numpy, made in a data-loader worker). `strength` scales the push (0 = off)."""
    return {"frac": float(rng.uniform(*AMP_FRAC) * strength), "sigma_frac": float(rng.uniform(*SIGMA_FRAC))}


def apply(t2, labels, mask, tumor, mass):
    """Warp the host window for the tumor mask `tumor` (bool, window grid). Returns (t2, labels, mask) unchanged if the field is zero."""
    vol = int(tumor.sum())
    if vol == 0 or mass["frac"] <= 0:
        return t2, labels, mask
    radius = (3 * vol / (4 * math.pi)) ** (1 / 3)
    amp = min(mass["frac"] * radius, AMP_MAX)
    u = safe_field(tumor, mask.bool(), amp, min(max(mass["sigma_frac"] * radius, 3.0), SIGMA_MAX))
    return warp(t2, labels, mask, u)
