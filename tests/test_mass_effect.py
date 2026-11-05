import numpy as np
import pytest
import torch

from synth import mass_effect as ME

N = 56


def scene(tumor_center=(28, 28, 22), radius=7):
    g = torch.stack(torch.meshgrid(*[torch.arange(N, dtype=torch.float32)] * 3, indexing="ij"))
    brain = ((g - 27.5) / torch.tensor([24.0, 22.0, 20.0]).view(3, 1, 1, 1)).pow(2).sum(0) < 1
    tumor = ((g - torch.tensor(tumor_center, dtype=torch.float32).view(3, 1, 1, 1)) ** 2).sum(0) < radius ** 2
    t2 = torch.where(brain, 0.4 + 0.1 * torch.sin(g[0] / 3) * torch.cos(g[1] / 4), torch.zeros(()))
    labels = torch.where(brain, torch.full((N,) * 3, 3), torch.zeros((N,) * 3, dtype=torch.long)).long()
    return t2, labels, brain, tumor


def test_zero_amplitude_is_identity():
    t2, labels, brain, tumor = scene()
    t2w, lw, mw = ME.apply(t2, labels, brain, tumor, {"frac": 0.0, "sigma_frac": 0.6})
    assert torch.equal(t2w, t2) and torch.equal(lw, labels) and torch.equal(mw, brain)
    u = ME.field(tumor, brain, 0.0, 5.0)
    t2w, lw, mw = ME.warp(t2, labels, brain, u)
    assert torch.allclose(t2w, t2, atol=1e-4) and torch.equal(lw, labels) and torch.equal(mw, brain)


@pytest.mark.parametrize("amp,sigma", [(2, 4), (6, 5), (12, 8), (15, 4), (15, 15)])
def test_safe_field_never_folds_inside_the_brain(amp, sigma):
    _, _, brain, tumor = scene()
    u = ME.safe_field(tumor, brain, float(amp), float(sigma))
    det = ME.jacobian_det(u)[brain]
    assert float(det.min()) >= 0.3 - 1e-4          # the guarantee: no folding, measured
    assert float(u.norm(dim=0).max()) <= amp + 1e-4


def test_field_is_zero_at_the_brain_edge_and_pushes_away_from_the_tumor():
    _, _, brain, tumor = scene()
    u = ME.field(tumor, brain, 8.0, 6.0)
    outside = ~brain
    assert float(u.norm(dim=0)[outside].max()) < 0.2             # nothing moves outside the brain
    c = torch.tensor([28.0, 28.0, 22.0])
    pt = (28, 28, 34)                                           # a voxel 12 above the tumor centre
    assert float(u[(slice(None),) + pt][2]) > 0                 # pushed away (+z), not towards the tumor


def test_warped_labels_stay_aligned_with_the_image_and_csf_is_squeezed_away_from_the_tumor():
    t2, labels, brain, tumor = scene()
    labels = labels.clone(); labels[(slice(24, 32), slice(24, 32), slice(32, 38))] = 1      # a CSF block just above the tumor
    t2 = torch.where(labels == 1, torch.full_like(t2, 0.9), t2)
    t2w, lw, mw = ME.apply(t2, labels, brain, tumor, {"frac": 0.3, "sigma_frac": 0.8})
    assert float(t2w[lw == 1].mean()) > 0.8 and float(t2w[lw == 3].mean()) < 0.6        # image and labels moved together
    z = lambda l: torch.nonzero(l == 1)[:, 2].float().mean()
    assert z(lw) > z(labels) + 0.3                                                      # the block moved up, away from the tumor
    assert int((mw ^ brain).sum()) < 0.002 * int(brain.sum())                           # brain mask is (almost) unchanged


def test_off_by_default_leaves_render_untouched():
    from synth import paste as P
    rng = np.random.default_rng(0)
    p = P.sample_params(rng)
    assert "mass" not in p
    assert ME.sample_mass(rng, 0.0)["frac"] == 0.0
