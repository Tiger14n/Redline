"""Grid/redline metrics: beat-line fidelity + redline offset P/R (the release metrics)."""
import numpy as np


def gt_redlines_full(d):
    return sorted([(o, 60000.0/b, m) for o, b, m in zip(d["gt_offsets_ms"], d["gt_bpms"], d["gt_meters"]) if b > 0],
                  key=lambda x: x[0])



def human_lines(redlines, start_t, end_t, measure=False):
    """All beat (or downbeat) times the human grid produces over the span."""
    if not redlines:
        return []
    out = []
    for i, (off, mpb, meter) in enumerate(redlines):
        if mpb <= 0:
            continue
        step = mpb * (meter if measure else 1)
        if step <= 0:
            continue
        nxt = redlines[i+1][0] if i+1 < len(redlines) else end_t + step
        lo = start_t if i == 0 else off
        t = off + int(np.floor((lo - off) / step)) * step
        while t < nxt - 1e-6:
            if start_t - step <= t <= end_t + step:
                out.append(t)
            t += step
    return sorted(out)



def nearest_line_err(t, redlines, measure=False):
    """Distance from time t to the nearest generated beat/downbeat line."""
    active = redlines[0]
    for r in redlines:
        if r[0] <= t:
            active = r
        else:
            break
    off, mpb, meter = active
    step = mpb * (meter if measure else 1)
    if step <= 0:
        return float("inf")
    return abs((t - off) - round((t - off) / step) * step)



def redline_pr(ours, gt, tol_ms=150.0):
    """P/R/F1 of redline offsets (greedy nearest, greedy nearest)."""
    o = np.array(sorted(r[0] for r in ours)); g = np.array(sorted(r[0] for r in gt))
    if len(o) == 0 or len(g) == 0:
        return 0.0, 0.0, 0.0
    used = np.zeros(len(o), bool); tp = 0
    for gi in g:
        d = np.abs(o - gi); d[used] = 1e12
        k = int(d.argmin())
        if d[k] <= tol_ms:
            used[k] = True; tp += 1
    pr = tp / len(o); rc = tp / len(g)
    f1 = 0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc)
    return pr, rc, f1



def redline_f1(ours, gt, tol_ms=150.0):
    """Redline-offset F1 at +-tol_ms (greedy nearest match). ours/gt = [(off,mpb,meter)]."""
    o = np.array(sorted(r[0] for r in ours)); g = np.array(sorted(r[0] for r in gt))
    if len(o) == 0 or len(g) == 0:
        return 0.0
    used = np.zeros(len(o), bool); tp = 0
    for gi in g:
        d = np.abs(o - gi); d[used] = 1e12
        k = int(d.argmin())
        if d[k] <= tol_ms:
            used[k] = True; tp += 1
    fp = len(o) - tp; fn = len(g) - tp
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    return 0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc)




def grid_fidelity(ours, gt, dur_ms):
    """For each GT beat line, distance to nearest generated beat line (bench_grid style)."""
    gt_beats = human_lines(gt, 0, dur_ms, measure=False)
    if not gt_beats:
        return {"med": float("nan"), "p10": 0.0, "p20": 0.0, "n": 0}
    errs = np.array([nearest_line_err(t, ours, measure=False) for t in gt_beats])
    return {"med": float(np.median(errs)), "p10": float((errs < 10).mean()),
            "p20": float((errs < 20).mean()), "n": len(errs)}


