# PLAN.md — LF-SynthSeg Reproduction + Mass-Effect Deformation + Hierarchy-Aware Loss

Team: Piyush Garg, Sahaj Sharma, Goutam Jain · Supervisor: Dr. Vijay · NSUT BTP
Written: 2026-10-29 · Budget: **under 6 weeks** · Compute: **Kaggle (primary)**, university GPU as an optional backup

---

## 0. Context and ground rules

**Goal.** (1) Re-implement LF-SynthSeg (Xu et al., IEEE JBHI 2025) from scratch and show our numbers land near the paper's. (2) Add two changes — 3D mass-effect deformation of the host brain, and a hierarchy-aware loss — and measure honestly whether they help.

**Paper targets for the baseline (T2-only, 3D, BraTS 2020, 31 test cases).** DSC 0.8232 · HD95 9.861 · AUPRC 0.9164 · SEN 0.8668. Fully supervised nnU-Net is about 8% higher (the paper's "92.3% of supervised").

**Ground rules**
- Nothing in the mid-term repo is trusted. We reuse an idea only after it passes the check listed for it here.
- Each stage has a gate. We do not start the next stage until the gate passes.
- A negative result is a valid outcome. The report claims only what the numbers support.
- One config file per experiment, a fixed seed, and a results CSV per run. Nothing is reported from memory.

---

## 1. Problems found in the mid-term work

I read the mid-term report and the paper side by side. These need fixing or an explicit decision **before** we build on them.

| # | Issue | Why it matters | Action |
|---|---|---|---|
| 1 | **The hierarchy loss as written is vacuous.** With a softmax over mutually exclusive classes, P_TC = P_NCR+P_ET and P_WT = P_TC+P_ED, so P_ET ≤ P_TC ≤ P_WT always holds and L_containment is identically 0. | Training with it would change nothing, and we would wrongly conclude it "doesn't help". | Redesign in §7 (Stage E). First measure how often the baseline actually violates containment. |
| 2 | **The paper's synthetic ground truth has no ET/TC/WT labels.** The target is 5 classes: CSF, GM, WM, non-brain dark matter, and one tumor class. NCR/ET/ED are used only to build tumor *intensities*. | The hierarchy loss needs sub-region labels we don't currently produce. Evaluation on BraTS is also whole-tumor only, from T2. | Decision in §7. Subregion masks exist inside the synthesizer and can be exported as labels. |
| 3 | **Radial Gaussian warp can fold.** The mid-term map is x_source = x − u(x), with |u| up to 15 vox and a *non-zero* displacement right at the centroid. Near the centre |u| > r, so the map crosses itself. The reported min det(J)=0.882 does not look credible for α≈15 and σ=22. | Folded warps give bad images and misaligned labels, which would poison training. | Re-derive and test det(J) numerically (§6). Design the warp around the tumor boundary, not the centroid. |
| 4 | **The warp is not constrained to the brain.** It runs over the whole volume, including the skull-stripped boundary. | Brain edge and background get pushed. Real mass effect stays inside the skull. | Mask the field to the brain and taper it to zero at the brain edge. |
| 5 | **Unverified claims.** The report lists an RTX 4090, 64 GB RAM and Ubuntu. This PC has Intel UHD graphics, 15.7 GB RAM and Windows. Results were shown on one subject (HCP 100206), and no network was trained. | There is no evidence yet that anything trains or improves. | Treat the mid-term as a design sketch only. |
| 6 | FSL FAST gives CSF/GM/WM (3 tissues). The paper's 4th class, "non-brain dark matter", has to be defined by us. | Label mapping errors silently break the tissue-assisted task. | Pin the class definition in §3 (Stage A) and unit-test it. |
| 7 | Likely wrong claim: "diffeomorphic". A smoothed displacement field is not automatically a diffeomorphism. | Report wording. | Say "no folding (det J > 0 on the brain)" only if measured. |

---

## 2. Where to run this (answering "local or Kaggle?")

**Recommendation: split the work by what each place is good at.**

| Task | Where | Why |
|---|---|---|
| Code editing, unit tests, quick 2D/small-array checks, plots | **Your PC** (VS Code + Python venv) | No GPU needed. Fast iteration. |
| One-time preprocessing: FSL FAST + MNI registration + skull-strip for the 20 HCP subjects | **WSL2 (Ubuntu) on your PC** — or Colab/Kaggle if WSL is a hassle | FSL and ANTs don't run natively on Windows. It is CPU-only, one-off, ~20 subjects. |
| All GPU training and evaluation | **Kaggle notebooks (T4×2 / P100)** | Free, Linux, PyTorch+CUDA preinstalled, datasets persist. Limits: ~30 GPU-h/week, ~9–12 h per session, so runs must checkpoint and resume. |
| University GPU | **Optional, in parallel** | Ask Dr. Vijay in week 1 what the access process is (SSH? queue? time limits?). Use it only if it arrives early. Do not make the plan depend on it. |

Practical workflow: preprocess once → upload the results as a **private Kaggle Dataset** → notebooks pull the git repo, read the dataset, write checkpoints and CSVs to `/kaggle/working`, and you download them. Keep the code in a GitHub repo so the PC and Kaggle share it.

Storage: BraTS 2020 T2 volumes plus HCP preprocessed volumes should fit Kaggle's dataset limits if we keep only T2 and labels (crop to the brain, store float16). Check this in week 1.

---

## 3. Stage A — Setup, data and the reference (Week 1)

1. **Repo skeleton**: `configs/`, `synth/` (shape, intensity, refine, paste, deform), `data/`, `train/`, `eval/`, `tests/`, `notebooks/`. Git, pinned `requirements.txt`.
2. **Environment**: Python 3.10/3.11, PyTorch, MONAI, nibabel, SimpleITK, scipy, antspyx (optional). Check that Kaggle's versions match.
3. **Data access** (start now — approvals can be slow):
   - HCP Young Adult T2w: need a ConnectomeDB account and data-use terms. We need ~20 healthy subjects.
   - BraTS 2020 (primary), plus BraTS 2019/2021, BraTS-PED and BraTS-SSA (Synapse/Kaggle) for the secondary tables. **T2 only.**
   - If HCP is delayed: fall back to any healthy T2 set (e.g. IXI) and say so in the report.
4. **Official code check**: the paper says code will be at `github.com/xu1px/LF-Synthseg`. If it exists, read it for decisions the paper leaves open. We still write our own implementation. Record every place where we follow the repo versus the paper text.
5. **Preprocessing pipeline** (script, not manual): affine registration to MNI152 1 mm → skull-strip (ANTs for HCP; BraTS is already stripped) → histogram/intensity normalization → crop. Apply the **same** normalization to HCP and BraTS. A mismatch here is a classic cause of "synthetic-to-real" failure.
6. **FAST outputs for the 20 HCP volumes**: CSF/GM/WM masks + the 4th "dark/non-brain" class. Write the exact definition of the 4 classes in the README and add a test that labels cover the volume with no gaps.
7. **Fixed test split**: draw 31 random cases per dataset once with a fixed seed, save the case IDs to a file, and never touch them again. Training uses HCP only (no BraTS data in training at any point).

**Gate A.** Preprocessed HCP volumes + FAST labels visually checked on at least 5 subjects; BraTS test list frozen; `evaluate.py` reproduces known Dice/HD95/AUPRC/SEN on a toy example (hand-made masks with known answers).

---

## 4. Stage B — Reproduce the LF-SynthSeg synthesizer (Weeks 1–2)

Implement in this order, with a unit test and a visual check for each piece.

1. **Shape generation** (paper §III-A1): three ellipsoid sets D1, D2, D3 (replicated n1,n2,n3 times), random deformation/scale/rotation, union → ẽ1, ẽ2, ẽ3, then size scaling and displacement Z(·) → m1, m2, m3.
2. **Mask/overlap** (Eq. 3–4): M_NCR = M_CSF·m1, M_ET = M_GM·m2, M_ED = M_WM·m3. Accept a sub-mask only if it covers ≥70% of its shape; otherwise regenerate the shape.
3. **Intensity + "missing regions"** (Eq. 5): I_x = I0·M*_x, then **third-order spline interpolation** along x, y, z to fill the voxels of m_i that are missing from M*.
4. **Refinement** (Eq. 6): align the centers, scale, and take a **weighted sum** of the three sub-tumors → a smooth single tumor. Build the **tumor bank** from the 20 healthy volumes.
5. **Paste** (Eq. 7): X* = (1−M*)·X + T*(T), with the five random augmentations (deformation, scaling, rotation, contrast, movement). Movement matters: the paper shows CSF false positives without it.
6. **Labels**: L* = FAST labels (1–4) with tumor voxels set to 5.

**Details the paper does not give — decide during development and write each choice down** (ellipsoid count/size ranges, weight values, augmentation ranges, how the 70% rule interacts with overlap, how union masks resolve overlaps between NCR/ET/ED). Tune them against the realism checks below, never against the test set.

**Realism checks (run on the training side only):**
- 3 × 4 grid of synthetic vs. real BraTS T2 slices, looked at by all three team members.
- Replicate the paper's diversity analysis: shape (volume, surface area, tightness, concavity) and intensity (mean, std, skewness, kurtosis) for ~120 tumors; compare with the paper's entropy values (synthetic ≈2.43 shape / 2.49 intensity).
- Intensity histogram of synthetic tumor vs. real BraTS tumor on T2.

**Gate B.** The synthesizer produces ≥1000 distinct tumors without crashes; labels are aligned with images by overlay; realism checks look like the paper's Figs. 1, 3, 7.

---

## 5. Stage C — Baseline training and reproduction (Weeks 2–3)

**Framework decision.** Use **MONAI (DynUNet / 3D U-Net)** as the main framework: it supports on-the-fly synthesis and custom losses, which stages E–F need. The paper itself ran U-Net, Swin UNETR (MONAI) and nnU-Net. nnU-Net v2 is a *stretch goal*: it expects a pre-generated dataset and a custom trainer for any custom loss.

**Paper settings to follow.** 20 healthy training volumes; random 96³ patches; lr 1e-3 with linear warm-up and cosine annealing; CE + Dice over 5 classes (Eq. 8); sliding-window inference with 0.25 overlap; the output tumor class is the prediction.

**Scale down for the budget.** The paper trained 500 epochs on a TITAN RTX. On a Kaggle T4, plan roughly: mixed precision (AMP), batch size 1–2, ~100–150 "epochs" of ~250 synthetic iterations each, with checkpoints every N steps and resume support. Run a 1-hour pilot first to measure throughput, then size the full run. State the reduced schedule in the report.

**Runs**
- **C1 — Tumor-only baseline** (no tissue labels, 2 classes). Needed for the paper's ablation (Fig. 9).
- **C2 — LF-SynthSeg baseline** (5-class, tissue-assisted). This is our reference for everything after.

**Evaluation** (3D, whole tumor vs. BraTS GT>0): DSC, HD95, AUPRC, SEN, mean ± std over the 31 cases, on BraTS 2020 first. Then BraTS 2019/2021, PED, SSA if time allows.

**Debug order if the Dice is far below ~0.75:**
1. Orientation/space mismatch between MNI-registered synthetic data and BraTS test data.
2. Intensity normalization mismatch.
3. Tumor realism or location (CSF false positives → check the movement augmentation).
4. Label mapping bugs.
5. Under-training.
Do **not** fix a low score by touching the test set.

**Gate C (the key gate).** C2 reaches roughly DSC ≥ 0.75 on BraTS 2020 (paper 0.82), and tissue-assisted C2 is ≥ tumor-only C1, matching the paper's trend. If we are within ~0.05–0.07 of the paper, we call the reproduction successful and say honestly how large the gap is. Below that, spend up to 3–4 more days debugging before moving on, because every later claim depends on this baseline.

---

## 6. Stage D — Mass-effect deformation module (Weeks 3–4, can start in Week 2)

Unit-test it **in isolation** before it goes anywhere near training.

**Design intent.** Real mass effect: tissue around the lesion is pushed outward, ventricles near the lesion are compressed, sulci are effaced, and the displacement fades with distance and stays inside the skull. We approximate this with a smooth, brain-confined, outward radial field relative to the **tumor boundary**, not the centroid.

**Requirements**
- The field is zero (or tapers to zero) at the brain boundary.
- No folding: **det(J) > 0 over the brain mask**, measured, not argued. Report min/max/percentiles of det(J). Reduce the amplitude until the property holds.
- Image warped with cubic interpolation, labels with nearest-neighbor, both with the same field. Warp the *host brain and FAST labels first*, then paste the tumor, so the tumor is not itself distorted.
- Amplitude and reach tied to tumor size, with random ranges rather than the fixed α/σ in the mid-term report. Whether to keep the Gaussian shape, make the push anisotropic, or add a ventricle-specific term is **left for development**; the tests below decide.
- Off by default and controlled by a config switch so that the baseline stays untouched.

**Tests / sanity checks**
1. Identity: amplitude 0 gives exactly the original image and labels.
2. Label–image alignment: overlay warped labels on the warped image; compare ventricle volume before/after (should shrink near the lesion).
3. det(J) statistics, plus a check that no brain voxel is mapped outside the brain.
4. Tissue volume change vs. distance plot (compare the shape to the mid-term Fig. 4, but only after recomputing it with the new field).
5. Visual comparison against real BraTS cases with obvious mass effect (midline shift, ventricle compression): does it *look* plausible, and is the magnitude in a realistic range (a few mm to ~1 cm)?
6. Speed: warp cost per sample. If it is too slow on CPU, do it on the GPU inside the training loop.

**Gate D.** det(J) > 0 everywhere in the brain for the whole sampled parameter range; labels and image aligned; team and supervisor agree that the examples look plausible.

---

## 7. Stage E — Hierarchy-aware loss design (Weeks 3–4, in parallel with D)

**Step 1 — diagnose first (cheap, uses C2 outputs).** On the BraTS test predictions of the baseline, count the topological problems we claim to fix: tumor voxels disconnected from the main component, and predicted tumor outside the brain mask. If the baseline barely has them, the loss has little to improve and the report should say so.

**Step 2 — decide what "hierarchy" means here.** The mid-term loss (ET ⊂ TC ⊂ WT) is vacuous with a softmax head (issue #1) and needs sub-region labels we don't have (issue #2). Candidate designs, in order of my recommendation:

| Option | Idea | Needs |
|---|---|---|
| **H-A. Multi-level Dice** (recommended first) | Add Dice/CE terms on *merged* classes at each level of a tissue/tumor ontology: brain ⊃ {CSF, GM, WM, tumor}, plus tumor ⊃ {NCR, ET, ED} if we export those labels. Related work from the same group: Lyu et al., MICCAI 2023 (cited in the mid-term report). | Export NCR/ET/ED as synthetic labels (they exist inside the synthesizer). |
| **H-B. Nested sigmoid head with containment penalty** | Predict overlapping ET, TC, WT channels (BraTS style) with sigmoid, so `max(0, P_ET − P_TC)²` is no longer zero. | Same sub-region labels; changes the output head. |
| **H-C. Tumor-in-brain constraint** | Penalize tumor probability outside the brain mask / outside tissue classes. | Brain mask only. No extra labels. |

All of them must be implemented so that the loss is **not identically zero** at initialization (test this). Weights (λ) are tuned on a small validation split of *synthetic* data, never on BraTS test cases.

**Important caveat to write down now.** We train and test on **T2 only**, and evaluation is **whole-tumor only**. ET is barely visible in T2, and synthetic NCR/ET/ED are defined by CSF/GM/WM-like intensities, so they are not guaranteed to match BraTS sub-regions. The hierarchy loss is therefore a *regularizer* whose benefit may be small. That is a legitimate finding.

**Gate E.** Chosen design justified in one paragraph; unit tests show the loss is non-zero on a deliberately inconsistent prediction and zero on a consistent one; loss gradient flows.

---

## 8. Stage F — Experiments (Weeks 4–5)

All use the same architecture, patch size, schedule, optimizer, and test list as C2. Only the listed factor changes.

| ID | Synthesis | Loss | Purpose |
|---|---|---|---|
| **C2** | static (paper) | CE+Dice | Baseline (from Stage C) |
| **E1** | **+ mass-effect deformation** | CE+Dice | Effect of deformation alone |
| **E2** | static | **+ hierarchy loss** | Effect of the loss alone |
| **E3** | **+ deformation** | **+ hierarchy loss** | The proposed method |

**Controls and extras (priority order, drop from the bottom if time runs out)**
1. **Seeds**: baseline and proposed method with ≥3 seeds each. If only a few runs fit, use 3 seeds for C2 and E3, and 1 seed for E1/E2.
2. **Parameter sweep** for the deformation amplitude (e.g. none / low / medium / high). A good result should vary smoothly with amplitude, not be a lucky point.
3. **Stratified analysis** on BraTS: split test cases by tumor volume (or by midline shift, if a simple estimate is possible). The central hypothesis is that deformation helps most for large tumors with mass effect.
4. **Framework check**: repeat C2 vs. E3 once with a second architecture (e.g. U-Net or Swin UNETR) as the paper did.
5. **Other datasets**: BraTS 2019/2021, PED, SSA for C2 vs. best variant.
6. **Qualitative**: failure cases where each variant is worse, not only wins.

**Statistics.** Paired comparison on the same 31 test cases: Wilcoxon signed-rank (or paired t-test if the differences look normal), bootstrap 95% CI of the mean DSC difference, effect size. With 31 cases and noisy 3D training, small gains (<0.01 DSC) cannot be claimed as real, so say that explicitly.

**Hierarchy-specific metrics** (besides DSC/HD95): containment-violation rate, number of connected components, tumor volume outside the brain mask.

**Compute budget** (to be refined by the Stage C pilot): E1, E2, E3 each cost about one C2 run. With ~30 GPU-hours/week on Kaggle, plan for about 4–6 full runs per week, so the grid above plus seeds fits in ~2 weeks only if each run is ≤ 4–5 hours. If a run is slower, shrink the schedule *for all variants equally*.

---

## 9. Stage G — Report and viva (Week 6, with writing started earlier)

- Methods: our own pipeline with every undefined paper detail listed; the redesigned deformation and loss with the reasons (issues #1–#7).
- Results: reproduction table next to the paper's numbers; ablation table (C1, C2, E1–E3) with mean ± std, CIs and p-values; stratified plots; qualitative figures.
- Honest limitations: T2-only, 20 training subjects, reduced training schedule, whole-tumor evaluation only, deformation is a phenomenological model rather than biomechanics.
- Reproducibility package: configs, seeds, test case IDs, a `run_all.md` with exact commands.
- Fix the mid-term report wording where it contradicted what was actually verified (hardware claims, "diffeomorphic").

---

## 10. Timeline (≈ 6 weeks from 2026-10-29)

| Week | Dates (approx.) | Work | Gate |
|---|---|---|---|
| 1 | Oct 29 – Nov 4 | Repo, env, data requests, official-code check, preprocessing, FAST, frozen test list, evaluation code. Ask Dr. Vijay about university GPU. | A |
| 2 | Nov 5 – Nov 11 | Synthesizer (shape → intensity → refine → paste → labels), realism checks. Start deformation prototype and the hierarchy diagnostic in parallel. | B |
| 3 | Nov 12 – Nov 18 | Throughput pilot, train C1 and C2, evaluate on BraTS 2020. Debug until the reproduction is acceptable. Deformation tests. | **C** |
| 4 | Nov 19 – Nov 25 | Finish deformation (gate D) and hierarchy loss (gate E). Launch E1/E2/E3. | D, E |
| 5 | Nov 26 – Dec 2 | Seeds, amplitude sweep, other datasets, stratified analysis, statistics. | — |
| 6 | Dec 3 – Dec 9 | Final figures/tables, report, viva slides, buffer. Freeze all experiments by the start of the week. | — |

**If we fall behind (cut in this order):** other datasets (PED, SSA) → second-architecture check → amplitude sweep → extra seeds for E1/E2. Never cut: Gate C, C2 vs. E3 on BraTS 2020 with ≥3 seeds each, and the statistics.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| HCP access delayed | Use IXI or another healthy T2 set; document the substitution. Start the request on day 1. |
| FSL/ANTs won't install on Windows | WSL2 for the one-time preprocessing; fallback is running it on Colab/Kaggle with conda. |
| Baseline does not reproduce | Follow the debug order in §5; budget extra days; report the gap honestly. |
| Kaggle quota/session limits | Checkpoint and resume, AMP, smaller patch batch, throughput pilot first, shrink all variants equally. |
| Deformation or hierarchy loss gives no gain | Valid result. Report with CIs, stratified analysis and the diagnostic from §7 explaining why. |
| Sub-region labels from the synthesizer are ill-defined | Start with H-C/H-A on merged classes before sub-region-level losses. |
| Data leakage | Test IDs frozen in week 1; training uses only HCP; hyper-parameters tuned on synthetic validation data only. |

---

## 12. Left for development (deliberately not decided here)

- Exact ellipsoid/augmentation/weight ranges in the synthesizer.
- The functional form and parameters of the deformation field (Gaussian vs. other, anisotropy, ventricle-specific term).
- Which hierarchy design (H-A / H-B / H-C) and its λ.
- Training schedule length, batch size, and whether nnU-Net v2 is attempted.
- Whether the university GPU becomes useful.

---

## 13. Immediate next actions (first 3 days)

1. Create the repo and the Python environment on the PC.
2. Open HCP / BraTS data-access requests; download BraTS 2020 T2 + labels.
3. Email or meet Dr. Vijay about university GPU access and the revised scope (reproduce first, then improve; T2 only).
4. Install WSL2 + FSL + ANTs (or decide on the Colab fallback) and run FAST on one HCP subject end-to-end.
5. Write `evaluate.py` with unit tests and freeze the 31-case test lists.
