"""Build one synthetic tumor from a healthy scan (paper III-A2..4, Eq. 3-6).

Each sub-shape takes its intensities from one tissue of the healthy scan: NCR <- CSF, ET <- GM, ED <- WM
(Eq. 3-5). Voxels of the shape that are not that tissue ("missing regions") are filled by cubic spline
interpolation from the voxels that are. The three sub-tumors are blended with random weights into one smooth
tumor (Eq. 6). The result is a "bank entry": a GRID^3 crop with the tumor intensities, a blend weight, and a
parts map (1 = NCR, 2 = ET, 3 = ED, 0 = none; kept so ET/TC/WT labels can be derived later).
"""
import numpy as np
from scipy import ndimage
from scipy.interpolate import RBFInterpolator

from data import labels as L
from synth import shapes as S

SOURCE_TISSUE = (L.CSF, L.GM, L.WM)      # tissue that feeds m1 (NCR), m2 (ET), m3 (ED)
MIN_SAMPLES = 20                          # fewer valid voxels than this -> use the tissue's mean intensity
MAX_SAMPLES = 800                         # spline is fitted on at most this many valid voxels
COARSE = 4                                # spline is evaluated on a grid with this stride, then cubic-upsampled
MIN_INSIDE = 0.7                          # a pasted tumor must lie at least 70% inside the brain


def crop(vol, center, grid=S.GRID, fill=0):
    """grid^3 window of `vol` centred at `center`; parts outside the volume are `fill`."""
    out = np.full((grid,) * 3, fill, dtype=vol.dtype)
    lo = [int(c) - grid // 2 for c in center]
    src, dst = [], []
    for a, l in enumerate(lo):
        s0, s1 = max(l, 0), min(l + grid, vol.shape[a])
        if s1 <= s0:
            return out
        src.append(slice(s0, s1)); dst.append(slice(s0 - l, s1 - l))
    out[tuple(dst)] = vol[tuple(src)]
    return out


def fill_missing(values, valid, region, rng):
    """Cubic spline fill: keep `values` where `valid`, interpolate the rest of `region` from the valid voxels."""
    pts = np.argwhere(valid)
    if len(pts) > MAX_SAMPLES:
        pts = pts[rng.choice(len(pts), MAX_SAMPLES, replace=False)]
    vals = values[tuple(pts.T)]
    lo = np.argwhere(region).min(0)
    hi = np.argwhere(region).max(0) + 1
    axes = [np.arange(l, h, COARSE, dtype=np.float32) for l, h in zip(lo, hi)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    rbf = RBFInterpolator(pts.astype(np.float32), vals, neighbors=min(32, len(pts)), kernel="cubic")
    coarse = np.clip(rbf(grid), vals.min(), vals.max()).reshape([len(a) for a in axes]).astype(np.float32)
    idx = np.meshgrid(*[(np.arange(l, h) - l) / COARSE for l, h in zip(lo, hi)], indexing="ij")
    fine = ndimage.map_coordinates(coarse, idx, order=3, mode="nearest")
    out = np.array(values, dtype=np.float32)
    box = tuple(slice(l, h) for l, h in zip(lo, hi))
    sel = region[box] & ~valid[box]
    out[box][sel] = fine[sel]
    return out


def place_center(rng, mask, union, tries=50):
    """Pick a brain voxel so that >= 70% of `union` (a GRID^3 shape) falls inside the brain; None if not found."""
    inside = np.argwhere(mask)
    n = union.sum()
    pts = np.argwhere(union)
    off = pts - S.GRID // 2
    for _ in range(tries):
        c = inside[rng.integers(len(inside))]
        q = off + c
        ok = np.all((q >= 0) & (q < mask.shape), 1)
        if ok.sum() / n >= MIN_INSIDE and mask[tuple(q[ok].T)].sum() / n >= MIN_INSIDE:
            return c
    return None


def build_tumor(rng, t2, labels, mask, shapes=None, weights=None):
    """One bank entry from a healthy scan (t2 in [0,1], labels 0-4, brain mask); None if no good place was found."""
    m1, m2, m3 = shapes if shapes is not None else S.random_tumor_shapes(rng)
    union = m1 | m2 | m3
    c = place_center(rng, mask, union)
    if c is None:
        return None
    x, lab = crop(t2, c), crop(labels, c)
    w = np.asarray(weights if weights is not None else rng.uniform(0.5, 1.5, size=3), dtype=np.float32)
    num = np.zeros_like(x); den = np.zeros_like(x)
    for m, tissue, wi in zip((m1, m2, m3), SOURCE_TISSUE, w):
        valid = m & (lab == tissue)
        inten = None
        if valid.sum() >= MIN_SAMPLES:
            try:
                inten = fill_missing(x, valid, m, rng)
            except np.linalg.LinAlgError:         # valid voxels lie in a plane/line: the spline is ill-posed
                pass
        if inten is None:
            inten = np.full_like(x, t2[labels == tissue].mean())
        s = ndimage.gaussian_filter(m.astype(np.float32), 2.0) * wi
        num += s * inten; den += s
    tumor = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0).astype(np.float32)
    alpha = np.clip(ndimage.gaussian_filter(union.astype(np.float32), 1.0), 0, 1)
    alpha[alpha < 0.01] = 0                   # cut the Gaussian tails so the image is untouched away from the tumor
    parts = np.zeros(union.shape, np.uint8)
    parts[m3] = 3; parts[m2] = 2; parts[m1] = 1
    return {"tumor": tumor, "alpha": alpha, "parts": parts}
