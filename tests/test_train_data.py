import numpy as np
import pytest
import torch

from data import labels as L
from synth import shapes as S
from synth import tumor as T
from synth.bank import load_raw, save_entry
from tests.test_synth import fake_brain
from train import data as D
from train.train import lr_at


def make_scan():
    t2, lab, mask = fake_brain()
    return {"id": "fake", "t2": t2, "labels": lab, "mask": mask, "inside": np.argwhere(mask),
            "mu_gm": float(t2[lab == L.GM].mean()), "mu_csf": float(t2[lab == L.CSF].mean())}


@pytest.fixture(scope="module")
def raw_entry(tmp_path_factory):
    scan = make_scan()
    rng = np.random.default_rng(1)
    entry = None
    while entry is None:
        entry = T.build_tumor(rng, scan["t2"], scan["labels"], scan["mask"])
    f = tmp_path_factory.mktemp("bank") / "t.npz"
    save_entry(entry, f)
    return load_raw(f)


def test_items_render_to_consistent_patches(raw_entry):
    scan = make_scan()
    rng = np.random.default_rng(0)
    items = []
    while len(items) < 3:
        it = D.make_item(rng, scan, raw_entry)
        if it is not None:
            items.append(it)
    b = D.render_batch(items)
    assert b["image"].shape == (3, 1, 96, 96, 96) and b["label"].shape == b["parts"].shape == b["image"].shape
    assert b["image"].min() >= 0 and b["image"].max() <= 1
    assert int(b["label"].max()) <= L.TUMOR
    assert ((b["label"] == L.TUMOR) == (b["parts"] > 0)).all()
    for it, x in zip(items, b["image"]):
        assert (x[0][~torch.as_tensor(it["mask"])] == 0).all()           # nothing outside the brain


def test_tumor_only_labels_are_binary(raw_entry):
    scan = make_scan()
    rng = np.random.default_rng(2)
    it = None
    while it is None:
        it = D.make_item(rng, scan, raw_entry)
    b = D.render_batch([it], tumor_only=True)
    assert set(b["label"].unique().tolist()) <= {0, 1}
    assert int(b["label"].sum()) == int((b["parts"] > 0).sum())


def test_render_is_deterministic_for_one_item(raw_entry):
    scan = make_scan()
    rng = np.random.default_rng(3)
    it = None
    while it is None:
        it = D.make_item(rng, scan, raw_entry)
    a, b = D.render_batch([it]), D.render_batch([it])
    assert torch.equal(a["image"], b["image"]) and torch.equal(a["label"], b["label"])


def test_lr_schedule_warms_up_then_decays():
    assert lr_at(0, 1000, 1e-3, 100) == pytest.approx(1e-5)
    assert lr_at(99, 1000, 1e-3, 100) == pytest.approx(1e-3)
    assert lr_at(1000, 1000, 1e-3, 100) == pytest.approx(0, abs=1e-9)
    assert lr_at(550, 1000, 1e-3, 100) == pytest.approx(5e-4)


def test_predict_tumor_prob_shapes_and_range():
    from eval.run_eval import predict_tumor_prob
    from train.model import build_model
    model = build_model(6, filters=(4, 8, 8, 8, 8, 8)).eval()
    t2 = np.random.default_rng(0).random((70, 80, 100)).astype(np.float32)
    prob, arg = predict_tumor_prob(model, t2, 6, "cpu", roi=64)
    assert prob.shape == arg.shape == t2.shape
    assert prob.min() >= 0 and prob.max() <= 1 and arg.max() <= 5


def test_scans_found_with_either_nifti_extension(tmp_path):
    from data.io import find_nifti
    for name, ext in (("a", ".nii.gz"), ("b", ".nii")):
        d = tmp_path / "hcp" / name
        d.mkdir(parents=True)
        (d / f"labels{ext}").write_bytes(b"x")
    assert [p.name for p in D.find_scans(tmp_path / "hcp")] == ["a", "b"]
    assert find_nifti(tmp_path / "hcp" / "b", "labels").name == "labels.nii"
    assert find_nifti(tmp_path / "hcp" / "a", "labels").name == "labels.nii.gz"
    assert find_nifti(tmp_path / "hcp" / "a", "t2") is None
