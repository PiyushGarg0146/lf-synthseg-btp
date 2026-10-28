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
