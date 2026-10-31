import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fsl_pipeline as F  # noqa: E402
from data import labels as L  # noqa: E402


def _blob():
    d = np.zeros((30, 30, 30), np.float32)
    d[8:22, 8:22, 8:22] = 500
    d[14, 14, 14] = 0     # hole inside the brain
    d[1, 1, 1] = 3        # stray speck outside
    return d


def test_brain_mask_fills_hole_and_drops_speck():
    m = F.brain_mask(_blob())
    assert m[14, 14, 14] and not m[1, 1, 1]
    assert m.sum() == 14 ** 3


def test_fill_unlabeled_leaves_valid_label_map():
    m = F.brain_mask(_blob())
    lab = np.zeros(m.shape, np.uint8); lab[m] = 3; lab[14, 14, 14] = 0
    out = F.fill_unlabeled(lab, m)
    assert out[14, 14, 14] == 3
    assert L.check_label_map(out, m)
