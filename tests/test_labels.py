import numpy as np
import pytest

from data import labels as L


def test_fast3_ranks_classes_by_brightness_not_by_id():
    # FAST id 3 is brightest, id 1 middle, id 2 darkest (the pattern seen on real HCP T2)
    seg = np.array([[[0, 1, 2, 3]]])
    t2 = np.array([[[0.0, 0.48, 0.32, 0.56]]])
    out = L.fast3_by_intensity(seg, t2)
    assert out.ravel().tolist() == [L.BACKGROUND, L.GM, L.WM, L.CSF]


def test_fast3_identity_when_ids_already_bright_to_dark():
    seg = np.array([[[1, 2, 3]]])
    t2 = np.array([[[0.9, 0.5, 0.2]]])
    assert L.fast3_by_intensity(seg, t2).ravel().tolist() == [L.CSF, L.GM, L.WM]


def test_fast3_rejects_wrong_class_set():
    with pytest.raises(ValueError):
        L.fast3_by_intensity(np.array([0, 1, 2]), np.array([0.0, 0.4, 0.3]))
    with pytest.raises(ValueError):
        L.fast3_by_intensity(np.array([1, 2, 3, 4]), np.array([0.9, 0.5, 0.3, 0.1]))


def test_valid_label_map_passes():
    brain = np.zeros((4, 4, 4), bool); brain[1:3, 1:3, 1:3] = True
    lab = np.zeros((4, 4, 4), np.uint8); lab[brain] = 3
    lab[1, 1, 1] = L.CSF
    assert L.check_label_map(lab, brain)


def test_gap_inside_brain_fails():
    brain = np.zeros((4, 4, 4), bool); brain[1:3, 1:3, 1:3] = True
    lab = np.zeros((4, 4, 4), np.uint8); lab[brain] = 2
    lab[1, 1, 1] = 0
    with pytest.raises(AssertionError, match="unlabelled"):
        L.check_label_map(lab, brain)


def test_label_outside_brain_or_invalid_id_fails():
    brain = np.zeros((4, 4, 4), bool); brain[1:3, 1:3, 1:3] = True
    lab = np.zeros((4, 4, 4), np.uint8); lab[brain] = 2
    lab[0, 0, 0] = 1
    with pytest.raises(AssertionError, match="outside"):
        L.check_label_map(lab, brain)
    lab[0, 0, 0] = 0
    lab[1, 1, 1] = L.TUMOR
    with pytest.raises(AssertionError, match="invalid ids"):
        L.check_label_map(lab, brain)


def test_dark_voxels_use_median_brain_threshold():
    mask = np.ones((4, 4, 4), bool)
    t2 = np.full((4, 4, 4), 0.3, np.float32)    # brain median 0.3 -> threshold 0.4 * 0.3 = 0.12
    t2[3, 3, 3] = 0.10                           # dark
    t2[3, 3, 2] = 0.20                           # above threshold
    d = L.dark_voxels(t2, mask)
    assert d[3, 3, 3] and not d[3, 3, 2] and d.sum() == 1


def test_add_dark_class_ignores_outside_mask():
    mask = np.zeros((4, 4, 4), bool); mask[1:3, 1:3, 1:3] = True
    lab = np.zeros((4, 4, 4), np.uint8); lab[mask] = L.WM
    t2 = np.zeros((4, 4, 4), np.float32); t2[mask] = 0.3
    t2[1, 1, 1] = 0.01
    out = L.add_dark_class(lab, t2, mask)
    assert out[1, 1, 1] == L.DARK and (out[~mask] == 0).all()
    assert L.check_label_map(out, mask)
    with pytest.raises(ValueError):
        L.dark_voxels(t2, np.zeros_like(mask))


def test_ranking_fails_with_dark_contamination_but_works_with_ignore():
    """Real pattern seen on HCP T2: FAST id 3 = bright CSF + dark edge voxels, so its mean is below GM's."""
    n = 1000
    seg = np.zeros(n, int); t2 = np.zeros(n, float)
    seg[:430], t2[:430] = 1, 390.0          # GM-like: 43%, bright-ish
    seg[430:850], t2[430:850] = 2, 260.0    # WM-like: 42%
    seg[850:920], t2[850:920] = 3, 600.0    # true CSF: 7%
    seg[920:970], t2[920:970] = 3, 30.0     # dark edge voxels lumped into class 3
    dark = t2 < 100
    naive = L.fast3_by_intensity(seg[None, None], t2[None, None])[0, 0]
    assert naive[860] == L.GM                # wrong: CSF ranked below GM because of the dark voxels
    fixed = L.fast3_by_intensity(seg[None, None], t2[None, None], ignore=dark[None, None])[0, 0]
    assert fixed[860] == L.CSF and fixed[10] == L.GM and fixed[500] == L.WM
