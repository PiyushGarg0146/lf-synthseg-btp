import numpy as np
import pytest

from data import labels as L


def test_fast_remap_known_values():
    fast = np.array([[[0, 1, 2, 3, 4]]])
    out = L.fast_to_project(fast)
    assert out.ravel().tolist() == [L.BACKGROUND, L.CSF, L.GM, L.WM, L.DARK]


def test_fast_remap_rejects_unknown_label():
    with pytest.raises(ValueError):
        L.fast_to_project(np.array([0, 5]))


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


def test_add_dark_class_uses_wm_relative_threshold():
    mask = np.ones((4, 4, 4), bool)
    lab = np.full((4, 4, 4), L.WM, np.uint8)
    lab[0] = L.CSF
    t2 = np.full((4, 4, 4), 0.3, np.float32)   # WM median 0.3 -> threshold 0.15
    t2[0] = 0.9
    t2[3, 3, 3] = 0.10                           # dark voxel
    t2[3, 3, 2] = 0.20                           # above threshold, stays WM
    out = L.add_dark_class(lab, t2, mask)
    assert out[3, 3, 3] == L.DARK and out[3, 3, 2] == L.WM and (out[0] == L.CSF).all()
    assert (out == L.DARK).sum() == 1
    assert L.check_label_map(out, mask)


def test_add_dark_class_ignores_outside_mask_and_needs_wm():
    mask = np.zeros((4, 4, 4), bool); mask[1:3, 1:3, 1:3] = True
    lab = np.zeros((4, 4, 4), np.uint8); lab[mask] = L.WM
    t2 = np.zeros((4, 4, 4), np.float32); t2[mask] = 0.3
    assert (L.add_dark_class(lab, t2, mask) == lab).all()      # nothing dark; outside stays 0
    with pytest.raises(ValueError):
        L.add_dark_class(np.zeros((4, 4, 4), np.uint8), t2, mask)
