"""Tumor bank on disk: one compressed .npz per entry, cropped to the tumor's bounding box (restored to GRID^3 on load)."""
from pathlib import Path

import numpy as np

from synth import shapes as S


def save_entry(entry, path):
    box = np.argwhere(entry["alpha"] > 0)
    lo, hi = box.min(0), box.max(0) + 1
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    np.savez_compressed(path, lo=lo, tumor=entry["tumor"][sl].astype(np.float16),
                        alpha=(entry["alpha"][sl] * 255).round().astype(np.uint8), parts=entry["parts"][sl])


def load_entry(path):
    z = np.load(path)
    lo = z["lo"]
    out = {"tumor": np.zeros((S.GRID,) * 3, np.float32), "alpha": np.zeros((S.GRID,) * 3, np.float32),
           "parts": np.zeros((S.GRID,) * 3, np.uint8)}
    sl = tuple(slice(int(l), int(l) + n) for l, n in zip(lo, z["parts"].shape))
    out["tumor"][sl] = z["tumor"]
    out["alpha"][sl] = z["alpha"] / 255.0
    out["parts"][sl] = z["parts"]
    return out


def list_entries(root):
    return sorted(Path(root).glob("*.npz"))
