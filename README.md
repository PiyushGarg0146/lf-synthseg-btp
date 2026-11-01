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
| 1 | CSF | brightest of FAST's 3 classes (per subject) |
| 2 | GM | middle FAST class |
| 3 | WM | darkest FAST class |
| 4 | "non-brain dark matter" | our rule: brain voxels with T2 < 0.5 x median WM T2 (`add_dark_class`) |
| 5 | tumor | added by the synthesizer |

FAST is run as `fast -n 3 -t 2` on the skull-stripped, MNI-registered T2. **FAST's numeric class ids are not
ordered by intensity** (on HCP T2 id 3 was the brightest class, id 2 the darkest), so CSF/GM/WM are assigned
per subject by ranking the classes' mean T2 (`fast3_by_intensity`). The paper does not define class 4. A
4-class FAST run was unstable across subjects (4th class mean T2 0.08-0.24; up to 20% bright voxels), so
class 4 is defined by the fixed rule above; its fraction (0.5) is tunable (`--dark-frac`).

## Test split
`configs/test_split_brats2020.txt` holds the 31 frozen BraTS 2020 test cases (seed 42, drawn from the
369 labelled training cases). Created by `scripts/make_test_split.py`; do not edit.

## Preprocessing
`data/preprocess.py`: reorient to RAS (HCP is LAS, BraTS is LPS) -> 1 mm isotropic -> affine
registration to MNI152 1 mm (FSL FLIRT, T1 template + mutual information, since conda FSL has no T2 template) -> intensity scaled to [0, 1]. The FSL part runs on Kaggle via
`notebooks/01_fsl_fast_hcp.ipynb` and `scripts/fsl_pipeline.py`.
