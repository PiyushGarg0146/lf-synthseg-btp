import numpy as np
import pytest

from data import labels as L


def test_fast_remap_known_values():
    fast = np.array([[[0, 1, 2, 3, 4]]])
    out = L.fast_to_project(fast)
    assert out.ravel().tolist() == [L.BACKGROUND, L.DARK, L.WM, L.GM, L.CSF]


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
