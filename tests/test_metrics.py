import math

import numpy as np
import pytest

from eval import metrics as M


def cube(shape=(20, 20, 20), lo=5, hi=10):
    m = np.zeros(shape, dtype=bool)
    m[lo:hi, lo:hi, lo:hi] = True
    return m


def test_dice_identical_and_disjoint():
    a = cube()
    assert M.dice(a, a) == 1.0
    assert M.dice(a, cube(lo=12, hi=17)) == 0.0


def test_dice_half_overlap_known_value():
    a = np.zeros((10, 10, 10), bool); a[:, :, :4] = True    # 400 voxels
    b = np.zeros((10, 10, 10), bool); b[:, :, 2:6] = True   # 400 voxels, 200 shared
    assert M.dice(a, b) == pytest.approx(0.5)


def test_dice_empty_conventions():
    z = np.zeros((5, 5, 5), bool)
    assert M.dice(z, z) == 1.0
    assert M.dice(z, cube((5, 5, 5), 1, 3)) == 0.0


def test_sensitivity_known_value_and_empty():
    gt = cube()
    pred = np.zeros_like(gt); pred[5:10, 5:10, 5:8] = True   # 3 of 5 slices
    assert M.sensitivity(pred, gt) == pytest.approx(0.6)
    z = np.zeros_like(gt)
    assert M.sensitivity(z, z) == 1.0
    assert math.isnan(M.sensitivity(gt, z))
    assert M.sensitivity(z, gt) == 0.0


def test_hd95_identical_is_zero():
    a = cube()
    assert M.hd95(a, a) == 0.0


def test_hd95_shifted_cube_known_value():
    a = cube(lo=5, hi=10)
    b = np.zeros_like(a); b[8:13, 5:10, 5:10] = True   # shifted by 3 along axis 0
    assert M.hd95(a, b) == pytest.approx(3.0, abs=1.0)  # surface-to-surface, <=3 by construction
    assert M.hd95(a, b) > 0


def test_hd95_respects_spacing():
    a = cube(lo=5, hi=10)
    b = np.zeros_like(a); b[7:12, 5:10, 5:10] = True
    d1 = M.hd95(a, b, (1, 1, 1))
    d2 = M.hd95(a, b, (2, 2, 2))
    assert d2 == pytest.approx(2 * d1)


def test_hd95_single_voxels_exact():
    a = np.zeros((10, 10, 10), bool); a[2, 2, 2] = True
    b = np.zeros((10, 10, 10), bool); b[2, 2, 7] = True
    assert M.hd95(a, b) == pytest.approx(5.0)


def test_hd95_empty_conventions():
    z = np.zeros((6, 6, 6), bool)
    assert M.hd95(z, z) == 0.0
    assert math.isinf(M.hd95(z, cube((6, 6, 6), 1, 3)))


def test_auprc_perfect_and_random():
    gt = cube()
    assert M.auprc(gt.astype(float), gt) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    rand = rng.random(gt.shape)
    assert M.auprc(rand, gt) < 0.2        # chance level ~ prevalence (0.016)


def test_auprc_beats_chance_for_noisy_signal_and_mask():
    gt = cube()
    rng = np.random.default_rng(1)
    prob = gt * 0.6 + rng.random(gt.shape) * 0.4
    assert M.auprc(prob, gt) > 0.99
    brain = np.ones_like(gt)
    assert M.auprc(prob, gt, brain) == pytest.approx(M.auprc(prob, gt))
    assert math.isnan(M.auprc(prob, np.zeros_like(gt)))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        M.dice(np.zeros((3, 3, 3)), np.zeros((4, 4, 4)))


def test_evaluate_case_and_summarize_excludes_inf():
    gt = cube()
    perfect = M.evaluate_case(gt, gt)
    assert perfect == {"dsc": 1.0, "hd95": 0.0, "auprc": 1.0, "sen": 1.0}
    missed = M.evaluate_case(np.zeros_like(gt), gt)
    s = M.summarize([perfect, missed])
    assert s["dsc"]["mean"] == pytest.approx(0.5)
    assert s["hd95"]["n_excluded"] == 1
    assert s["hd95"]["mean"] == 0.0
