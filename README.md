# LF-SynthSeg Reproduction + Mass-Effect Deformation + Hierarchy-Aware Loss

B.Tech project (NSUT). We re-implement LF-SynthSeg (Xu et al., IEEE JBHI 2025), then test two
improvements: 3D mass-effect deformation of the host brain and a hierarchy-aware loss.
See [PLAN.md](PLAN.md) for the full plan, gates and timeline.

## Layout
| Folder | Purpose |
|---|---|
| `synth/` | Tumor synthesis: shapes, intensities, refinement, paste, deformation |
| `data/` | Dataset loaders and preprocessing scripts (raw/processed data are git-ignored) |
| `train/` | Training loops, losses, model setup |
| `eval/` | Metrics (DSC, HD95, AUPRC, SEN) and evaluation scripts |
| `tests/` | Unit tests (run with `pytest`) |
| `configs/` | One YAML per experiment |
| `notebooks/` | Kaggle/Colab notebooks and visualization |

## Setup (local PC)
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pytest
```
GPU training runs on Kaggle; see `notebooks/`.

## Rules
- Never commit medical images, checkpoints or credentials. Data lives outside git.
- Work on a branch, open a PR, one reviewer from the team before merging to `main`.
- Test case IDs are frozen once created (`configs/test_split_*.txt`); do not edit them.

## Label classes (working definition)
Label maps use ids 0-5 (`data/labels.py`):

| id | class | source |
|---|---|---|
| 0 | background (outside brain mask) | |
| 1 | CSF | brightest of FAST's 3 classes (per subject, dark voxels excluded from the ranking) |
| 2 | GM | middle FAST class |
| 3 | WM | darkest FAST class |
| 4 | "non-brain dark matter" | our rule: brain voxels with T2 < 0.4 x median brain T2 (`dark_voxels`) |
| 5 | tumor | added by the synthesizer |

FAST is run as `fast -n 3 -t 2` on the skull-stripped, MNI-registered T2. FAST's numeric class ids are not
ordered by intensity, and its CSF class also absorbs very dark edge voxels (class id 3 on subject 100206:
12% of the brain, mean raw T2 330 vs 392 for GM). So the dark voxels are found first by a fixed rule, the
three classes are ranked by brightness *excluding* them (`fast3_by_intensity(..., ignore=dark)`), and the
dark voxels are then labelled 4. The paper does not define class 4; a 4-class FAST run was unstable across
subjects. The fraction (0.4) is tunable (`--dark-frac`).

## Test split
`configs/test_split_brats2020.txt` holds the 31 frozen BraTS 2020 test cases (seed 42, drawn from the
369 labelled training cases). Created by `scripts/make_test_split.py`; do not edit.

## Preprocessing
`data/preprocess.py`: reorient to RAS (HCP is LAS, BraTS is LPS) -> 1 mm isotropic -> affine
registration to MNI152 1 mm (FSL FLIRT, T1 template + mutual information, since conda FSL has no T2 template) -> intensity scaled to [0, 1]. The FSL part runs on Kaggle via
`notebooks/01_fsl_fast_hcp.ipynb` and `scripts/fsl_pipeline.py`.
