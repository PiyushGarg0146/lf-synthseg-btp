"""Build the synthetic tumor bank from the healthy HCP scans (one .npz per tumor in --dst).

    python scripts/build_bank.py --hcp "C:/.../hcp_processed" --dst "C:/.../tumor_bank" --n 1000 --workers 3

Each tumor is built from one scan (cycling through the scans) with its own seed, so a run can be resumed:
existing files are skipped. Entries are augmented and pasted later, on the fly (synth/paste.py).
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from synth import bank  # noqa: E402
from synth import tumor as T  # noqa: E402
from synth_demo import find_subjects, load_subject  # noqa: E402

_cache = {}


def make_one(args):
    i, sub, dst, seed = args
    out = Path(dst) / f"tumor_{i:05d}.npz"
    if out.exists():
        return i, "skipped"
    if sub not in _cache:
        _cache.clear()
        _cache[sub] = load_subject(sub)
    t2, lab, mask = _cache[sub]
    rng = np.random.default_rng([seed, i])
    for _ in range(10):
        entry = T.build_tumor(rng, t2, lab, mask)
        if entry is not None:
            bank.save_entry(entry, out)
            return i, f"{Path(sub).name} volume {int((entry['parts'] > 0).sum())}"
    return i, "FAILED (no valid location)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hcp", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    subs = find_subjects(args.hcp)
    Path(args.dst).mkdir(parents=True, exist_ok=True)
    # consecutive tumors use the same scan so each worker reloads a scan rarely
    jobs = [(i, str(subs[(i * len(subs)) // args.n]), args.dst, args.seed) for i in range(args.n)]
    with ProcessPoolExecutor(args.workers) as ex:
        for k, (i, msg) in enumerate(ex.map(make_one, jobs, chunksize=4), 1):
            print(f"[{k}/{args.n}] tumor {i}: {msg}", flush=True)


if __name__ == "__main__":
    main()
