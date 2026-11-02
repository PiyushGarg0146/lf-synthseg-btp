import numpy as np
import pytest
from scipy import ndimage

from data import labels as L
from synth import paste as P
from synth import shapes as S
from synth import tumor as T


def fake_brain(n=170, seed=0):
    """Sphere-shaped 'brain' with WM core, GM shell, CSF blobs and T2-like intensities."""
    rng = np.random.default_rng(seed)
    ax = np.arange(n) - n / 2
    r = np.sqrt(ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2)
    mask = r < 65
    lab = np.zeros((n,) * 3, np.uint8)
    lab[mask] = L.GM
    lab[r < 50] = L.WM
    csf = ndimage.gaussian_filter(rng.normal(size=(n,) * 3), 4) > 0.03
    lab[mask & csf] = L.CSF
    means = {L.CSF: 0.8, L.GM: 0.45, L.WM: 0.25}
    t2 = np.zeros((n,) * 3, np.float32)
    for k, m in means.items():
        t2[lab == k] = m
    t2 += rng.normal(scale=0.02, size=t2.shape).astype(np.float32) * mask
    return np.clip(t2, 0, 1), lab, mask


def test_shapes_fit_grid_and_match_requested_size():
    rng = np.random.default_rng(1)
    m1, m2, m3 = S.random_tumor_shapes(rng, diameter=50.0)
    want = np.pi / 6 * 50.0 ** 3
    assert abs(m3.sum() - want) / want < 0.3
    assert m1.any() and m2.any() and m3.sum() > m2.sum() > 0
    assert m3.shape == (S.GRID,) * 3
    for m in (m1, m2, m3):
        assert not (m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any() or m[:, :, 0].any() or m[:, :, -1].any())


def test_shapes_are_reproducible_per_seed():
    a = S.random_tumor_shapes(np.random.default_rng(5), diameter=40.0)
    b = S.random_tumor_shapes(np.random.default_rng(5), diameter=40.0)
    assert all((x == y).all() for x, y in zip(a, b))


def test_crop_pads_outside_the_volume():
    v = np.ones((10, 10, 10), np.float32)
    c = T.crop(v, (0, 5, 5), grid=8)
    assert c.shape == (8, 8, 8)
    assert c[:4].sum() == 0                        # x < 0 is outside the volume
    assert c[4:, 1:7, 1:7].min() == 1              # volume voxels are copied


def test_fill_missing_recovers_smooth_field_and_keeps_valid_voxels():
    n = 40
    ax = np.arange(n, dtype=np.float32)
    field = (0.2 + 0.01 * ax[:, None, None] + 0.005 * ax[None, :, None] + 0.003 * ax[None, None, :]).astype(np.float32)
    region = np.ones((n,) * 3, bool)
    valid = np.random.default_rng(0).random((n,) * 3) < 0.1
    vals = np.where(valid, field, 0).astype(np.float32)
    out = T.fill_missing(vals, valid, region, np.random.default_rng(0))
    assert np.allclose(out[valid], field[valid])
    assert np.abs(out - field).mean() < 0.02


def test_build_and_paste_make_consistent_image_and_labels():
    t2, lab, mask = fake_brain()
    rng = np.random.default_rng(3)
    entry = None
    while entry is None:
        entry = T.build_tumor(rng, t2, lab, mask)
    assert set(np.unique(entry["parts"])) <= {0, 1, 2, 3}
    assert entry["tumor"].min() >= 0 and entry["tumor"].max() <= 1
    res = None
    while res is None:
        res = P.paste(rng, t2, lab, mask, P.augment(rng, entry))
    x, labels, parts = res
    tumor = labels == L.TUMOR
    assert tumor.sum() > 1000
    assert (tumor == (parts > 0)).all()
    assert mask[tumor].all()                                   # tumor only inside the brain
    assert (x[~mask] == 0).all()
    far = ~ndimage.binary_dilation(tumor, iterations=6)
    assert np.array_equal(x[far], t2[far])                     # image unchanged away from the tumor
    assert np.array_equal(labels[~tumor], lab[~tumor])         # healthy labels untouched
    assert tumor.sum() / mask.sum() < 0.3


def test_place_center_rejects_when_brain_too_small():
    mask = np.zeros((170,) * 3, bool)
    mask[80:84, 80:84, 80:84] = True
    union = np.zeros((S.GRID,) * 3, bool)
    union[40:90, 40:90, 40:90] = True
    assert T.place_center(np.random.default_rng(0), mask, union) is None


def test_bank_roundtrip(tmp_path):
    from synth import bank
    t2, lab, mask = fake_brain()
    rng = np.random.default_rng(4)
    entry = None
    while entry is None:
        entry = T.build_tumor(rng, t2, lab, mask)
    bank.save_entry(entry, tmp_path / "a.npz")
    back = bank.load_entry(tmp_path / "a.npz")
    assert np.array_equal(back["parts"], entry["parts"])
    assert np.abs(back["alpha"] - entry["alpha"]).max() < 0.01
    vis = entry["alpha"] > 0                                      # outside the tumor box the values are dropped
    assert np.abs(back["tumor"] - entry["tumor"])[vis].max() < 2e-3
    assert bank.list_entries(tmp_path) == [tmp_path / "a.npz"]
