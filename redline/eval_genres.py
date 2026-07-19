"""Per-genre end-to-end evaluation: audio -> solver -> redlines, scored against the
ranked map's human timing, grouped by the mapset's genre.

  python -m redline.eval_genres --pool /path/to/heldout_osz_dir --train_dir /path/to/train_osz_dir

The pool is song-deduped against --train_dir (same key as training). Genres come from
datasets/osu_genre_ranked.json (setid -> genre); genres with fewer than --min_genre songs
are merged into "other". Per-map rows go to redline/cohorts/genre_results.jsonl.
"""
from __future__ import annotations
import argparse, collections, glob, json, os, random, re, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from redline.osu_format import gt_from_osz
from redline.gridmetrics import redline_pr, grid_fidelity

OUT = Path(__file__).parent / "cohorts" / "genre_results.jsonl"
MAX_LEN_S = 480


def songkey(p):
    nm = re.sub(r"\.osz$", "", os.path.basename(p), flags=re.I)
    m = re.match(r"^\s*\d+\s+(.*)$", nm)
    return (m.group(1) if m else nm).strip().lower()


def setid(p):
    m = re.match(r"^\s*(\d+)\s", os.path.basename(p))
    return int(m.group(1)) if m else None


def build_sample(pool_dir, train_dir, genre_json, per_genre, min_genre, seed=7):
    gmap = {int(k): v for k, v in json.load(open(genre_json)).items()}
    pool = sorted(glob.glob(os.path.join(pool_dir, "**", "*.osz"), recursive=True))
    if train_dir:
        train_keys = set(songkey(p) for p in glob.glob(os.path.join(train_dir, "**", "*.osz"), recursive=True))
        pool = [p for p in pool if songkey(p) not in train_keys]
    seen, dedup = set(), []
    for p in pool:
        k = songkey(p)
        if k not in seen:
            seen.add(k); dedup.append(p)
    by = collections.defaultdict(list)
    for p in dedup:
        g = gmap.get(setid(p) or -1)
        if g:
            by[g].append(p)
    # merge small genres into "other"
    merged = collections.defaultdict(list)
    for g, ps in by.items():
        merged[g if len(ps) >= min_genre else "OTHER*"].extend(ps)
    rng = random.Random(seed)
    sel = []
    for g, ps in sorted(merged.items()):
        rng.shuffle(ps)
        take = ps if len(ps) <= per_genre else ps[:per_genre]
        sel += [(g, p) for p in take]
        print(f"{g:12s} {len(take):3d} of {len(ps)}", flush=True)
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="held-out directory of .osz mapsets")
    ap.add_argument("--train_dir", default="", help="training corpus dir (for song dedup)")
    ap.add_argument("--genre_json", default=str(ROOT / "datasets" / "osu_genre_ranked.json"))
    ap.add_argument("--per_genre", type=int, default=55)
    ap.add_argument("--min_genre", type=int, default=50)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from redline.audio_io import load_osz_audio
    from redline.infer import Model2Beats
    from redline.curves import model_curves
    from redline.solver import TimingSolver, DEFAULT_CKPT

    sel = build_sample(args.pool, args.train_dir, args.genre_json, args.per_genre, args.min_genre)
    print(f"{len(sel)} maps selected", flush=True)
    m2b = Model2Beats(args.ckpt or DEFAULT_CKPT, device=args.device)
    solver = TimingSolver(ckpt=None)

    f = open(OUT, "a")
    t0 = time.time()
    for i, (g, p) in enumerate(sel):
        try:
            gt, title, artist, _ = gt_from_osz(p)
            sig, sr = load_osz_audio(p, None)
            if sig is None or len(sig) < sr:
                raise ValueError("short audio")
            if sig.ndim > 1:
                sig = sig.mean(0)
            if len(sig) / sr > MAX_LEN_S:
                raise ValueError("too long")
            dur = len(sig) / sr * 1000.0
            cur = model_curves(m2b, sig, sr, want_hidden=False, want_tempo=True)
            R = solver.predict_from_curves(cur)
            pr, rc, f1 = redline_pr(R, gt)
            gf = grid_fidelity(R, gt, dur)
            row = dict(genre=g, path=p, title=f"{artist} - {title}", ngt=len(gt), nred=len(R),
                       p10=gf["p10"], med=gf["med"], P=pr, R=rc, F1=f1)
            f.write(json.dumps(row) + "\n"); f.flush()
            el = (time.time() - t0) / 60
            print(f"[{i+1}/{len(sel)} {el:.0f}m] {g:12s} {row['title'][:34]:34s} "
                  f"p10 {gf['p10']*100:5.1f} P {pr:.2f} R {rc:.2f} n {len(R):3d}/{len(gt)}", flush=True)
        except Exception as e:
            print(f"SKIP [{g}] {os.path.basename(p)[:36]}: {e}", flush=True)
    f.close()

    # aggregate
    rows = [json.loads(l) for l in open(OUT)]
    print(f"\n{'genre':12s} {'n':>4s} {'p10':>7s} {'p10med':>7s} {'P':>6s} {'R':>6s} {'nred':>5s} {'gt':>5s}")
    for g in sorted(set(r["genre"] for r in rows)):
        rs = [r for r in rows if r["genre"] == g]
        print(f"{g:12s} {len(rs):4d} {np.mean([r['p10'] for r in rs])*100:6.1f}% "
              f"{np.median([r['p10'] for r in rs])*100:6.1f}% "
              f"{np.mean([r['P'] for r in rs]):6.3f} {np.mean([r['R'] for r in rs]):6.3f} "
              f"{np.mean([r['nred'] for r in rs]):5.1f} {np.mean([r['ngt'] for r in rs]):5.1f}")
    const = [r for r in rows if r["ngt"] == 1]
    if const:
        n = np.array([r["nred"] for r in const])
        print(f"\nconstant-GT maps (n={len(const)}): nred median {np.median(n):.0f}  "
              f"mean {n.mean():.1f}  >3 lines: {(n>3).sum()} maps")
    print("BENCH DONE")


if __name__ == "__main__":
    main()
