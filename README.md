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

## Synthesizer design choices (Stage B; the paper gives no numbers for these)
- Shapes (`synth/shapes.py`): 2-4 randomly rotated ellipsoids per set, boundary roughened with smooth noise; whole-tumor diameter U(35, 85) mm; ET volume = U(0.35, 0.7) x ED, NCR volume = U(0.08, 0.3) x ED; ET and NCR shifted by up to 0.2 radius.
- Intensity (`synth/tumor.py`): NCR <- CSF, ET <- GM, ED <- WM voxels of a healthy scan; shape voxels that are not that tissue are filled by cubic RBF spline (fitted on <= 800 voxels, evaluated on a stride-4 grid, cubic-upsampled); tissue with < 20 voxels in the shape falls back to its mean T2. Sub-tumors are blended with weights U(0.5, 1.5).
- "70% rule": we read it as "at least 70% of the tumor must lie inside the brain" (a tissue-coverage rule is impossible for CSF); the missing-region interpolation handles the rest.
- Augmentation (`synth/paste.py`): rotation +-180 deg, scale 0.8-1.25, elastic up to ~2 mm, gamma 0.7-1.5 on tumor contrast, then tumor mean T2 set to GM mean + U(0.3, 1.0) x (CSF mean - GM mean) of the host scan (tumors are bright on T2; chosen by eye on HCP pictures, NOT tuned on the BraTS test cases), random movement (pasted anywhere in the brain, not where it was built).
- Each bank entry keeps a `parts` map (1 NCR, 2 ET, 3 ED) so ET/TC/WT labels are available later for the hierarchy loss.
- Picture check: `python scripts/synth_demo.py --hcp <hcp_processed> --out demo.png --n 6`.

## Training and evaluation (Stage C)
- `scripts/pack_kaggle_data.py` copies only the needed files into three zips for upload as Kaggle datasets (healthy scans, tumor bank, BraTS test cases).
- `train/train.py`: MONAI DynUNet (`train/model.py`), Dice + CE loss, linear warm-up + cosine LR, AMP, checkpoint/resume. Output classes: 6 = outside-brain 0, CSF, GM, WM, dark, tumor (C2) or 2 = tumor-only (C1).
- `train/data.py`: workers cut 96^3 host patches and draw augmentation numbers; the tumor rotation/deformation/blending (`synth/paste.py: render`) runs on the GPU. 70% of patches are centred near the tumor, 30% on a random brain voxel.
- `eval/run_eval.py`: sliding window (96^3, 25% overlap), whole-tumor DSC / HD95 / AUPRC / SEN on the 31 BraTS cases. Run once per finished model; no checkpoint is chosen on BraTS data.
- Kaggle notebook: `notebooks/03_train_baseline.ipynb` (pilot of 300 iterations first, then the full run).
