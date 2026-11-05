import pytest
import torch

from train.hier_loss import HierLoss


def onehot_logits(y, classes, strength=40.0):
    return torch.nn.functional.one_hot(y.squeeze(1), classes).permute(0, 4, 1, 2, 3).float() * strength


def target(classes):
    y = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
    y[..., 2:6, 2:6, 2:6] = 3                                    # healthy tissue
    y[..., 3:5, 3:5, 3:5] = 5 if classes == 6 else 6              # tumor (6 classes) / ET (8 classes)
    return y


@pytest.mark.parametrize("classes", [6, 8])
def test_loss_is_nonzero_at_init_and_gradient_flows(classes):
    torch.manual_seed(0)
    logits = torch.randn(1, classes, 8, 8, 8, requires_grad=True)
    loss = HierLoss(classes, 1.0)(logits, target(classes))
    assert loss.item() > 0
    loss.backward()
    assert logits.grad.abs().sum() > 0


@pytest.mark.parametrize("classes", [6, 8])
def test_perfect_prediction_is_nearly_zero_and_wrong_group_costs_more_than_wrong_sibling(classes):
    y = target(classes)
    f = HierLoss(classes, 1.0)
    assert f(onehot_logits(y, classes), y).item() < 0.05
    if classes == 8:                                              # ET predicted as NCR: right group, wrong leaf
        sib = y.clone(); sib[y == 6] = 5
        grp = y.clone(); grp[y == 6] = 3                          # ET predicted as healthy WM: wrong group
        l_sib, l_grp = f(onehot_logits(sib, 8), y).item(), f(onehot_logits(grp, 8), y).item()
        assert 0 < l_sib < l_grp
        # the coarse levels see no error at all for the sibling swap
        only_coarse = HierLoss(8, 1.0)
        f0 = HierLoss(8, 0.0)
        assert (l_sib - f0(onehot_logits(sib, 8), y).item()) < 0.1 * (l_grp - f0(onehot_logits(grp, 8), y).item())


def test_lambda_zero_equals_plain_loss():
    from monai.losses import DiceCELoss
    torch.manual_seed(1)
    logits, y = torch.randn(1, 6, 8, 8, 8), target(6)
    assert torch.allclose(HierLoss(6, 0.0)(logits, y), DiceCELoss(to_onehot_y=True, softmax=True)(logits, y))
