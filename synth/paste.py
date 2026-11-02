"""Augment a bank entry and paste it into a healthy scan (paper Eq. 7), producing image + 5-class labels.

Five random augmentations (paper III-A5): deformation, scaling, rotation, contrast, movement.
Contrast = gamma in `augment` + a brightness rescale in `paste` (tumors are bright on T2: mean between GM and CSF of the host scan).
Movement = the tumor is pasted at a random place that is generally NOT where its intensities were taken from.
Ranges below are our choices (the paper gives none).
"""
import numpy as np
from scipy import ndimage

from data import labels as L
from synth import shapes as S
from synth import tumor as T

SCALE = (0.8, 1.25)
ROTATE_DEG = 180
ELASTIC_MM = (0.0, 4.0)
GAMMA = (0.7, 1.5)
BRIGHT = (0.3, 1.0)     # tumor mean T2 = GM mean + u x (CSF mean - GM mean) of the host scan, u ~ U(BRIGHT)


def augment(rng, entry):
    """Deform / scale / rotate the entry's arrays together (nearest for parts) and change tumor contrast."""
    g = S.GRID
    c = (g - 1) / 2
    ang = np.deg2rad(rng.uniform(-ROTATE_DEG, ROTATE_DEG, 3))
    cx, cy, cz = np.cos(ang); sx, sy, sz = np.sin(ang)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    s = rng.uniform(*SCALE, size=3) if rng.random() < 0.5 else np.full(3, rng.uniform(*SCALE))
    A = np.linalg.inv(Rz @ Ry @ Rx @ np.diag(s))                  # output coords -> input coords
    ax = np.arange(g, dtype=np.float32) - c
    grid = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), 0)
    coords = np.tensordot(A, grid, axes=(1, 0)) + c
    amp = rng.uniform(*ELASTIC_MM)
    if amp > 0:
        for a in range(3):
            coords[a] += S.smooth_noise(rng, g, 28) * (amp * 0.5)
    warp = lambda v, o: ndimage.map_coordinates(v, coords, order=o, mode="constant", cval=0)
    tumor, alpha = warp(entry["tumor"], 1), np.clip(warp(entry["alpha"], 1), 0, 1)
    parts = warp(entry["parts"].astype(np.uint8), 0).astype(np.uint8)
    inside = parts > 0
    if inside.any():                                              # gamma changes the contrast inside the tumor
        mu = max(float(tumor[inside].mean()), 1e-4)
        tumor = np.clip(mu * (np.clip(tumor, 1e-4, 1) / mu) ** rng.uniform(*GAMMA), 0, 1)
    return {"tumor": tumor.astype(np.float32), "alpha": alpha, "parts": parts}


def paste(rng, t2, labels, mask, entry):
    """Paste `entry` at a random brain location (movement). Returns (image, labels, parts) or None if it doesn't fit.

    image = (1 - alpha) * t2 + alpha * tumor inside the brain; labels = healthy labels with tumor voxels = 5;
    parts = full-size uint8 map (0 none, 1 NCR, 2 ET, 3 ED), 0 outside the brain.
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
    src, dst = tuple(src), tuple(dst)
    brain = mask[src]
    a = entry["alpha"][dst] * brain
    tum = entry["tumor"][dst]
    inside = (entry["parts"][dst] > 0) & brain
    mu_gm, mu_csf = float(t2[labels == L.GM].mean()), float(t2[labels == L.CSF].mean())
    target = mu_gm + rng.uniform(*BRIGHT) * (mu_csf - mu_gm)
    if inside.any() and tum[inside].mean() > 1e-4:
        tum = np.clip(tum * (target / tum[inside].mean()), 0, 1)
    x = t2.copy()
    x[src] = (1 - a) * t2[src] + a * tum
    parts = np.zeros(t2.shape, np.uint8)
    parts[src] = np.where(brain, entry["parts"][dst], 0)
    out = labels.copy()
    out[parts > 0] = L.TUMOR
    x[~mask] = 0
    return x, out, parts
