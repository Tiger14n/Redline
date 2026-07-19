"""Probe a red-head (notation) checkpoint directly: peak-pick the red curve and score
redline precision/recall vs ground truth on the cohorts, before wiring it into the solver.

  python -m redline.probe_redhead --ckpt checkpoints/redline_v1/last.pt
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from redline.osu_format import gt_from_osz
from redline.eval_cohorts import COHORT_DIR
from redline.gridmetrics import redline_pr

FRAME_MS = 20.0


def peaks(curve, th, min_gap_ms=350.0):
    c = np.asarray(curve)
    gap = int(min_gap_ms / FRAME_MS)
    out = []
    i = 1
    while i < len(c) - 1:
        if c[i] >= th and c[i] >= c[i - 1] and c[i] >= c[i + 1]:
            out.append(i * FRAME_MS)
            i += gap
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cohorts", nargs="+", default=["general", "metal", "live"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--th", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.6])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from redline.audio_io import load_osz_audio
    from redline.infer import Model2Beats
    from redline.curves import model_curves

    m2b = Model2Beats(args.ckpt, device=args.device)
    assert getattr(m2b.model, "use_red", False), "checkpoint has no red head"

    for c in args.cohorts:
        paths = json.load(open(COHORT_DIR / f"{c}.json"))
        if args.limit:
            paths = paths[:args.limit]
        agg = {t: {"pr": [], "rc": [], "f1": []} for t in args.th}
        for p in paths:
            try:
                gt, title, artist, _ = gt_from_osz(p)
                sig, sr = load_osz_audio(p, None)
                if sig is None or len(sig) < sr:
                    continue
                if sig.ndim > 1:
                    sig = sig.mean(0)
                cur = model_curves(m2b, sig, sr, want_hidden=False, want_tempo=False)
                if "red" not in cur:
                    raise RuntimeError("no red curve emitted")
                for t in args.th:
                    P = [(x, 500.0, 4) for x in peaks(cur["red"], t)]
                    pr, rc, f1 = redline_pr(P, gt)
                    for k, v in zip(("pr", "rc", "f1"), (pr, rc, f1)):
                        agg[t][k].append(v)
            except Exception as e:
                print(f"  FAIL {p.split('/')[-1][:40]}: {e}", flush=True)
        print(f"\n[{c}] red-head peak redline match (n={len(agg[args.th[0]]['f1'])}):")
        for t in args.th:
            a = agg[t]
            print(f"  th {t:.1f}: recall {np.mean(a['rc']):.3f}  prec {np.mean(a['pr']):.3f}  f1 {np.mean(a['f1']):.3f}")


if __name__ == "__main__":
    main()
