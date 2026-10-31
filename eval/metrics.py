"""Segmentation metrics: DSC, HD95, AUPRC, sensitivity.

Conventions (for the report):
- Masks are binarised with `> 0.5`; shapes must match.
- Both masks empty: DSC = 1, sensitivity = 1, HD95 = 0.
- Ground truth non-empty, prediction empty: DSC = 0, sensitivity = 0, HD95 = inf.
- Ground truth empty, prediction non-empty: DSC = 0, sensitivity = nan, HD95 = inf.
- HD95 is the 95th percentile of the pooled surface distances in both directions
  (surface = mask voxels removed by one 3D erosion), in mm via `spacing`.
- AUPRC is average precision of the probability map over all voxels (or those in `mask`).
"""
import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score


def _bin(x):
    return np.asarray(x) > 0.5


def _check(a, b):
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")


def dice(pred, gt):
    p, g = _bin(pred), _bin(gt)
    _check(p, g)
    total = p.sum() + g.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(p, g).sum() / total)


def sensitivity(pred, gt):
    p, g = _bin(pred), _bin(gt)
    _check(p, g)
    if g.sum() == 0:
        return 1.0 if p.sum() == 0 else float("nan")
    return float(np.logical_and(p, g).sum() / g.sum())


def _surface(mask):
    return mask & ~ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1))


def hd95(pred, gt, spacing=(1.0, 1.0, 1.0)):
    p, g = _bin(pred), _bin(gt)
    _check(p, g)
    if p.sum() == 0 and g.sum() == 0:
        return 0.0
    if p.sum() == 0 or g.sum() == 0:
        return float("inf")
    sp, sg = _surface(p), _surface(g)
    d_to_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    d_to_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    dists = np.concatenate([d_to_g[sp], d_to_p[sg]])
    return float(np.percentile(dists, 95))


def auprc(prob, gt, mask=None):
    g = _bin(gt)
    prob = np.asarray(prob, dtype=np.float64)
    _check(prob, g)
    if mask is not None:
        m = _bin(mask)
        prob, g = prob[m], g[m]
    if g.sum() == 0:
        return float("nan")
    return float(average_precision_score(g.ravel(), prob.ravel()))


def evaluate_case(pred_mask, gt_mask, prob=None, spacing=(1.0, 1.0, 1.0), brain_mask=None):
    """All four metrics for one case. `prob` defaults to the binary prediction."""
    return {
        "dsc": dice(pred_mask, gt_mask),
        "hd95": hd95(pred_mask, gt_mask, spacing),
        "auprc": auprc(pred_mask if prob is None else prob, gt_mask, brain_mask),
        "sen": sensitivity(pred_mask, gt_mask),
    }


def summarize(cases):
    """Mean/std per metric over a list of evaluate_case dicts.

    Non-finite values (inf, nan) are excluded from mean/std and counted in `n_excluded`.
    """
    out = {}
    for key in ("dsc", "hd95", "auprc", "sen"):
        vals = np.array([c[key] for c in cases], dtype=float)
        ok = np.isfinite(vals)
        out[key] = {
            "mean": float(vals[ok].mean()) if ok.any() else float("nan"),
            "std": float(vals[ok].std()) if ok.any() else float("nan"),
            "n_excluded": int((~ok).sum()),
        }
    return out
