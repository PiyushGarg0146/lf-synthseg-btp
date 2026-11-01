"""Label definitions for the 5-class LF-SynthSeg target.

WORKING DEFINITION (the paper does not define class 4 precisely):
FSL FAST is run with 3 classes on the skull-stripped T2 image (`fast -n 3 -t 2`) giving CSF/GM/WM.
FAST with 4 classes was tried first, but its 4th class was unstable across subjects (mean T2 0.08-0.24,
up to 20% bright voxels), so "dark" is instead defined by a fixed rule, see `add_dark_class`.
FAST's numeric class ids are not reliably ordered by intensity, so CSF/GM/WM are assigned per subject
by ranking class mean brightness (see `fast3_by_intensity`).
"""
import numpy as np

BACKGROUND = 0  # outside the brain mask (image == 0)
CSF = 1
GM = 2
WM = 3
DARK = 4  # "non-brain dark matter": the darkest FAST class
TUMOR = 5

NAMES = {BACKGROUND: "background", CSF: "CSF", GM: "GM", WM: "WM", DARK: "dark", TUMOR: "tumor"}

def fast3_by_intensity(seg, t2):
    """Map a FAST `-n 3` hard segmentation (ids 1-3, 0 = outside) to CSF/GM/WM by RANKING class brightness.

    FAST's numeric ids are NOT ordered by intensity in a reliable way (on HCP T2, id 3 was the brightest class
    and id 2 the darkest), so each subject's three classes are sorted by mean T2: brightest -> CSF,
    middle -> GM, darkest -> WM.
    """
    seg = np.asarray(seg)
    t2 = np.asarray(t2)
    ids = sorted(set(np.unique(seg).tolist()) - {0})
    if ids != [1, 2, 3]:
        raise ValueError(f"expected FAST classes 1, 2, 3, got {ids}")
    means = {i: float(t2[seg == i].mean()) for i in ids}
    brightest_first = sorted(ids, key=lambda i: -means[i])
    mapping = dict(zip(brightest_first, (CSF, GM, WM)))
    out = np.zeros(seg.shape, dtype=np.uint8)
    for src, dst in mapping.items():
        out[seg == src] = dst
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


DARK_FRAC = 0.5  # "dark" = brain voxels below this fraction of the subject's median white-matter T2


def add_dark_class(labels, t2, mask, frac=DARK_FRAC):
    """Relabel brain-mask voxels darker than `frac` x (median T2 of WM) as DARK.

    Same rule in every subject, so the class does not depend on how FAST happens to cluster.
    """
    labels = np.asarray(labels).copy()
    t2 = np.asarray(t2)
    mask = np.asarray(mask).astype(bool)
    wm = labels == WM
    if not wm.any():
        raise ValueError("no WM voxels to calibrate the dark threshold")
    thr = frac * float(np.median(t2[wm]))
    labels[mask & (t2 < thr)] = DARK
    return labels
