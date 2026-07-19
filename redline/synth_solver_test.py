"""Synthetic end-to-end tests of pipeline_onset.TimingSolver (no model, no GPU).

Builds a fully-controlled `curves` dict from known redlines (beat/down/off/change/tempo),
runs predict_from_curves with the default config, and scores the output redlines against
the known truth: redline count, grid coverage <10ms, downbeat/bar-phase correctness,
boundary placement error, meter.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redline.solver import TimingSolver
from redline.targets import beat_grid, BPM_MIN, N_TEMPO_BINS
from redline.dataset_osz import FRAME_MS

rng = np.random.default_rng(0)


def synth_curves(redlines, dur_ms, jitter_ms=2.0, act_noise=0.02, intro_silence_ms=0.0,
                 miss_prob=0.0):
    """redlines=[(off,mpb,meter)] -> curves dict as model_curves would emit."""
    beats, downs = beat_grid(redlines, dur_ms)
    if intro_silence_ms > 0:
        beats = beats[beats >= intro_silence_ms]
        downs = downs[downs >= intro_silence_ms]
    n_frames = int(np.ceil(dur_ms / FRAME_MS)) + 1
    ab = np.zeros(n_frames); ad = np.zeros(n_frames); ao = np.zeros(n_frames)
    jb = beats + rng.normal(0, jitter_ms, len(beats))           # model timing error
    keep = rng.random(len(jb)) >= miss_prob
    for t in jb[keep]:
        f = int(round(t / FRAME_MS))
        if 0 <= f < n_frames:
            ab[f] = max(ab[f], 0.95)
            ao[f] = (t - f * FRAME_MS) / FRAME_MS
            for w, v in ((-1, 0.35), (1, 0.35)):
                g = f + w
                if 0 <= g < n_frames:
                    ab[g] = max(ab[g], v)
    dset = set(np.round(downs / FRAME_MS).astype(int))
    for t in downs:
        f = int(round(t / FRAME_MS))
        if 0 <= f < n_frames:
            ad[f] = max(ad[f], 0.9)
    ab = np.clip(ab + rng.random(n_frames) * act_noise, 0, 1)
    ad = np.clip(ad + rng.random(n_frames) * act_noise, 0, 1)
    ach = np.zeros(n_frames)
    for off, mpb, m in redlines:
        f = int(round(off / FRAME_MS))
        if 0 <= f < n_frames:
            ach[max(0, f - 1):f + 2] = 0.9
    tempo = np.full((n_frames, N_TEMPO_BINS), 1e-4)
    R = sorted(redlines)
    for i, (off, mpb, m) in enumerate(R):
        nxt = R[i + 1][0] if i + 1 < len(R) else dur_ms
        sf, ef = max(0, int(off / FRAME_MS)), min(n_frames, int(nxt / FRAME_MS) + 1)
        b = int(round(60000.0 / mpb - BPM_MIN))
        if 0 <= b < N_TEMPO_BINS:
            tempo[sf:ef, max(0, b - 1):b + 2] = 0.3
            tempo[sf:ef, b] = 0.9
    tempo /= tempo.sum(-1, keepdims=True)
    return dict(beat=ab, down=ad, off=ao, change=ach, conf=ab.copy(), tempo=tempo), beats, downs


def score(pred, gt_redlines, gt_beats, gt_downs, dur_ms):
    """grid coverage + bar-phase vs truth."""
    if not pred:
        return dict(nred=0, cov10=0.0, bar_ok=0.0, first_err=1e9)
    P = sorted(pred)
    pb, pd = beat_grid([(o, m, mt) for o, m, mt in P], dur_ms)
    if len(pb) == 0:
        return dict(nred=len(P), cov10=0.0, bar_ok=0.0, first_err=1e9)
    gb = gt_beats[(gt_beats >= P[0][0] - 1)]                      # score from first pred anchor
    d = np.abs(gb[:, None] - pb[None, :]).min(1) if len(gb) else np.array([1e9])
    cov10 = float((d <= 10.0).mean())
    dd = np.abs(gt_downs[:, None] - pd[None, :]).min(1) if len(pd) and len(gt_downs) else np.array([1e9])
    bar_ok = float((dd <= 10.0).mean())
    fe = min(abs(P[0][0] - g) for g in gt_downs) if len(gt_downs) else 1e9
    return dict(nred=len(P), cov10=cov10, bar_ok=bar_ok, first_err=fe)


def run(name, redlines, dur_ms, gt_nred=None, **synth_kw):
    curves, beats, downs = synth_curves(redlines, dur_ms, **synth_kw)
    for solver in ("tempo", "dp"):
        t = TimingSolver(ckpt=None, solver=solver, max_error_ms=9.0)
        out = t.predict_from_curves(curves)
        s = score(out, redlines, beats, downs, dur_ms)
        bpms = sorted(set(round(60000.0 / m, 2) for _, m, _ in out)) if out else []
        meters = [mt for _, _, mt in out]
        print(f"{name:38s} [{solver:5s}] red {s['nred']:>2} (gt {gt_nred or len(redlines)})  "
              f"cov10 {s['cov10']*100:5.1f}%  barOK {s['bar_ok']*100:5.1f}%  "
              f"firstRL->down {s['first_err']:6.1f}ms  meters {sorted(set(meters))}  bpms {bpms[:4]}")


print(f"{'scenario':38s} {'':7s} {'redlines':>8}  {'beat cov<10ms':>12}  {'downbeat cov':>12}")
# 1. plain constant tempo
run("constant 180bpm 3min", [(1234.0, 60000/180, 4)], 180_000)
# 2. non-round BPM (live-ish but constant)
run("constant 183.7bpm", [(811.0, 60000/183.7, 4)], 180_000)
# 3. hard tempo step at 60s
run("step 180->120 @60s", [(1000.0, 60000/180, 4), (60_100.0, 60000/120, 4)], 150_000)
# 4. waltz meter 3
run("constant 150bpm meter3", [(700.0, 60000/150, 3)], 150_000)
# 5. long map: does grid drift/dedup hurt?
run("constant 200bpm 6min", [(500.0, 300.0, 4)], 360_000)
# 6. quiet intro: music (and beats) start at 10s, redline earlier
run("10s silent intro", [(500.0, 60000/170, 4)], 150_000, intro_silence_ms=10_000)
# 7. heavier jitter (hard audio)
run("constant 180bpm jitter4ms", [(900.0, 60000/180, 4)], 180_000, jitter_ms=4.0)
# 8. dropped beats (weak onsets): 15% misses
run("constant 180bpm 15% missed", [(900.0, 60000/180, 4)], 180_000, miss_prob=0.15)
# 9. three tempo sections
run("180->200->140", [(1000.0, 60000/180, 4), (50_000.0, 60000/200, 4), (110_000.0, 60000/140, 4)], 160_000)
# 10. meter change 4 -> 3 same bpm (meter-only redline)
run("meter change 4->3 @60s", [(1000.0, 60000/160, 4), (60_025.0, 60000/160, 3)], 120_000, gt_nred=2)
