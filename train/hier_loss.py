"""Hierarchy-aware multi-level loss (Stage E, design H-A).

Classes form a tree: background | healthy brain {CSF, GM, WM, dark} | tumor (6 classes: one tumor class; 8 classes: tumor
split into NCR, ET, ED). Besides the usual Dice + cross-entropy on the leaves, the same loss is computed on COARSER views of
the same prediction, made by adding up the probabilities of the classes that belong together:

    8 classes:  leaves -> {bg, CSF, GM, WM, dark, tumor} -> {bg, healthy brain, tumor} -> {not tumor, tumor}
    6 classes:                                              {bg, healthy brain, tumor} -> {not tumor, tumor}

loss = leaf loss + lam * mean(coarser levels). A prediction that confuses two classes of the same group (say NCR vs ET)
is punished only at the leaves, while one that confuses groups (tumor vs healthy tissue) is punished at every level, so
the tumor / healthy-brain boundary, which is what the whole-tumor score depends on, gets extra weight.
(The mid-term containment loss ET <= TC <= WT is always 0 for a softmax head, so we do not use it.)
"""
import torch
import torch.nn as nn
from monai.losses import DiceCELoss

LEVELS = {
    6: [[[0], [1, 2, 3, 4], [5]], [[0, 1, 2, 3, 4], [5]]],
    8: [[[0], [1], [2], [3], [4], [5, 6, 7]], [[0], [1, 2, 3, 4], [5, 6, 7]], [[0, 1, 2, 3, 4], [5, 6, 7]]],
}


class HierLoss(nn.Module):
    def __init__(self, classes, lam=1.0):
        super().__init__()
        if classes not in LEVELS:
            raise ValueError("hierarchy loss needs 6 or 8 classes")
        self.levels, self.lam = LEVELS[classes], lam
        self.base = DiceCELoss(to_onehot_y=True, softmax=True)

    def forward(self, logits, y):
        loss = self.base(logits, y)
        if self.lam == 0:
            return loss
        p = torch.softmax(logits.float(), 1)
        coarse = []
        for groups in self.levels:
            lut = torch.zeros(p.shape[1], dtype=torch.long, device=y.device)
            for g, members in enumerate(groups):
                lut[members] = g
            pg = torch.stack([p[:, m].sum(1) for m in groups], 1)           # (B, groups, ...) probabilities that add up to 1
            coarse.append(self.base(torch.log(pg.clamp_min(1e-12)), lut[y.squeeze(1)].unsqueeze(1)))    # softmax(log p) = p
        return loss + self.lam * torch.stack(coarse).mean()
