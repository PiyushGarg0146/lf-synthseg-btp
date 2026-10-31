"""Label definitions for the 5-class LF-SynthSeg target.

WORKING DEFINITION (to be confirmed visually on the first FAST outputs):
FSL FAST is run with 4 classes on the skull-stripped T2 image (`fast -n 4 -t 2`).
FAST numbers classes by increasing intensity, so on T2 the order is
dark -> WM -> GM -> CSF. We remap to the project's label ids below.
"""
import numpy as np

BACKGROUND = 0  # outside the brain mask (image == 0)
CSF = 1
GM = 2
WM = 3
DARK = 4  # "non-brain dark matter": the darkest FAST class
TUMOR = 5

NAMES = {BACKGROUND: "background", CSF: "CSF", GM: "GM", WM: "WM", DARK: "dark", TUMOR: "tumor"}

# FAST output id (ascending T2 intensity) -> project id
FAST_T2_TO_PROJECT = {1: DARK, 2: WM, 3: GM, 4: CSF}


def fast_to_project(fast_seg):
    """Map a FAST `-n 4 -t 2` hard segmentation to project labels (0 stays 0)."""
    fast_seg = np.asarray(fast_seg)
    unexpected = set(np.unique(fast_seg).tolist()) - {0, *FAST_T2_TO_PROJECT}
    if unexpected:
        raise ValueError(f"Unexpected FAST labels: {sorted(unexpected)}")
    out = np.zeros(fast_seg.shape, dtype=np.uint8)
    for src, dst in FAST_T2_TO_PROJECT.items():
        out[fast_seg == src] = dst
    return out


def check_label_map(labels, brain_mask):
    """Raise AssertionError unless the label map is valid.

    Rules: only ids 0-4; every brain-mask voxel has a label 1-4 (no gaps);
    nothing outside the brain mask is labelled.
    """
    labels = np.asarray(labels)
    brain_mask = np.asarray(brain_mask).astype(bool)
    assert labels.shape == brain_mask.shape, "shape mismatch"
    ids = set(np.unique(labels).tolist())
    assert ids <= {0, 1, 2, 3, 4}, f"invalid ids: {sorted(ids - {0, 1, 2, 3, 4})}"
    assert (labels[brain_mask] >= 1).all(), "unlabelled voxels inside the brain mask"
    assert (labels[~brain_mask] == 0).all(), "labelled voxels outside the brain mask"
    return True
