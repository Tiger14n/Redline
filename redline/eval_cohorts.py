"""End-to-end evaluation on fixed cohorts: audio -> solver -> redlines, scored against
the ranked map's timing (beat-grid p10/p20/median + redline F1).

  python -m redline.eval_cohorts --ckpt models/redline-v1.pt
  python -m redline.eval_cohorts --ckpt A.pt B.pt --labels a b     # compare checkpoints

Cohort map lists live in redline/cohorts/*.json (paths to .osz mapsets on your disk).
Results are appended to redline/cohorts/results.jsonl. Checkpoints trained at a longer
window (chunk_frames) are evaluated at that window automatically.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from redline.osu_format import gt_from_osz
from redline.gridmetrics import redline_f1, grid_fidelity     # noqa: E402

COHORT_DIR = Path(__file__).parent / "cohorts"
COHORTS = ("general", "metal", "live")


def score_map(R, gt, dur_ms):
    gf = grid_fidelity(R, gt, dur_ms)
    return dict(p10=gf["p10"], p20=gf["p20"], med=gf["med"],
                rf=redline_f1(R, gt), nred=len(R), ngt=len(gt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--cohorts", nargs="+", default=list(COHORTS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="cap maps per cohort (quick probe)")
    args = ap.parse_args()
    labels = args.labels or [Path(c).parent.name + "/" + Path(c).stem for c in args.ckpt]

    from redline.infer import Model2Beats
    import redline.curves as adg
    from redline.curves import model_curves
    from redline.solver import TimingSolver
    from redline.audio_io import load_osz_audio
    import torch

    # load cohort audio+gt once, share across checkpoints
    songs = {}
    for c in args.cohorts:
        paths = json.load(open(COHORT_DIR / f"{c}.json"))
        if args.limit:
            paths = paths[:args.limit]
        songs[c] = []
        for p in paths:
            try:
                gt, title, artist, _ = gt_from_osz(p)
                sig, sr = load_osz_audio(p, None)
                if sig is None or len(sig) < sr:
                    raise ValueError("short audio")
                if sig.ndim > 1:
                    sig = sig.mean(0)
                songs[c].append(dict(gt=gt, title=f"{artist} - {title}", sig=sig, sr=sr,
                                     dur_ms=len(sig) / sr * 1000.0))
            except Exception as e:
                print(f"  SKIP [{c}] {os.path.basename(p)[:40]}: {e}", flush=True)
        print(f"cohort {c}: {len(songs[c])} maps loaded", flush=True)

    for ck, label in zip(args.ckpt, labels):
        a = torch.load(ck, map_location="cpu", weights_only=False).get("args", {})
        cf = int(a.get("chunk_frames") or 1500)
        adg.CHUNK_FRAMES = cf                       # widen inference window for 60s models
        m2b = Model2Beats(ck, device=args.device)
        pipe = TimingSolver(ckpt=None)               # deploy defaults (tempo+union+fitgate)
        print(f"\n===== {label}  (step {torch.load(ck, map_location='cpu', weights_only=False).get('step')}, "
              f"window {cf*20/1000:.0f}s) =====")
        rec = dict(ckpt=ck, label=label, time=time.strftime("%Y-%m-%d %H:%M"), cohorts={})
        for c in args.cohorts:
            rows = []
            for s in songs[c]:
                try:
                    cur = model_curves(m2b, s["sig"], s["sr"], want_hidden=False, want_tempo=True)
                    R = pipe.predict_from_curves(cur)
                    rows.append(score_map(R, s["gt"], s["dur_ms"]) | {"title": s["title"]})
                except Exception as e:
                    print(f"  FAIL {s['title'][:40]}: {e}", flush=True)
            p10 = np.array([r["p10"] for r in rows])
            rf = float(np.mean([r["rf"] for r in rows]))
            print(f"  [{c:8s}] p10 mean {p10.mean()*100:5.1f}%  med {np.median(p10)*100:5.1f}%  "
                  f">=90%: {(p10 >= 0.9).sum():2d}/{len(p10)}  rf {rf:.3f}  "
                  f"nred {sum(r['nred'] for r in rows)} (gt {sum(r['ngt'] for r in rows)})", flush=True)
            rec["cohorts"][c] = dict(p10_mean=float(p10.mean()), p10_med=float(np.median(p10)),
                                     rf=rf, rows=rows)
        with open(COHORT_DIR / "results.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        del m2b
        torch.cuda.empty_cache()

    # worst-map table for the last ckpt (where to look next)
    print(f"\nper-map (last ckpt), worst first:")
    for c in args.cohorts:
        rows = sorted(rec["cohorts"][c]["rows"], key=lambda r: r["p10"])[:5]
        for r in rows:
            print(f"  [{c:8s}] {r['title'][:48]:48s} p10 {r['p10']*100:5.1f}%  "
                  f"rf {r['rf']:.2f}  nred {r['nred']}/{r['ngt']}")


if __name__ == "__main__":
    main()
