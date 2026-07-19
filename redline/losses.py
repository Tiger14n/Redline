"""Losses for the timing model: beat/down/change/red BCE, offset smooth-L1, soft-CE tempo."""
from __future__ import annotations
import torch
import torch.nn.functional as F

from .targets import LOG_MPB_CENTER, BPM_MIN, N_TEMPO_BINS, TEMPO_SOFT_STD

BORDER = 2          # frames masked at each chunk edge



def timing_loss(out, batch, w_beat=1.0, w_down=1.0, w_off=2.0, w_tempo=1.0, w_change=1.0,
                w_sub=0.0, pos_weight_beat=5.0, pos_weight_down=20.0, pos_weight_change=40.0,
                w_red=1.0, pos_weight_red=15.0):
    beat_t, down_t = batch["beat"], batch["down"]
    off_t, off_m = batch["off"], batch["mask"]
    dev = out["beat"].device
    F_ = beat_t.shape[1]
    m = torch.ones(F_, device=dev)
    if BORDER > 0:
        m[:BORDER] = 0
        m[-BORDER:] = 0
    m = m[None]                                          # [1,F]

    pwb = torch.tensor(pos_weight_beat, device=dev)
    pwd = torch.tensor(pos_weight_down, device=dev)
    lb = F.binary_cross_entropy_with_logits(out["beat"], beat_t, pos_weight=pwb, reduction="none")
    ld = F.binary_cross_entropy_with_logits(out["down"], down_t, pos_weight=pwd, reduction="none")
    lb = (lb * m).sum() / m.sum() / beat_t.shape[0]
    ld = (ld * m).sum() / m.sum() / down_t.shape[0]

    # offset: smooth-L1 only where there is a beat (masked), within the border
    omask = off_m * m
    lo = F.smooth_l1_loss(out["off"], off_t, reduction="none", beta=0.1)
    lo = (lo * omask).sum() / omask.sum().clamp_min(1.0)

    parts = {"beat": lb.detach(), "down": ld.detach(), "off": lo.detach()}
    total = w_beat * lb + w_down * ld + w_off * lo

    # subdivision suppression: extra penalty on the beat head firing at off-beat midpoints
    # (eighth-note positions) -> teaches tempo-consistency in-model, killing over-detection
    # WITHOUT inference-time pruning (so recall is preserved, unlike decode-time gating).
    if w_sub > 0 and "sub" in batch:
        sub = batch["sub"] * m
        lsub = F.binary_cross_entropy_with_logits(
            out["beat"], torch.zeros_like(out["beat"]), reduction="none")
        lsub = (lsub * sub).sum() / sub.sum().clamp_min(1.0)
        total = total + w_sub * lsub
        parts["sub"] = lsub.detach()

    # v2 structure heads (skipped gracefully if the batch/model lack them)
    if "tempo" in out and "tempo" in batch:
        tmask = batch["tmask"] * m
        # tempo as classification: soft Gaussian target over BPM bins + cross-entropy
        std = TEMPO_SOFT_STD
        tbpm = 60000.0 / torch.exp(batch["tempo"] + LOG_MPB_CENTER)        # [B,F] true BPM
        bin_pos = (tbpm - BPM_MIN).clamp(0, N_TEMPO_BINS - 1)              # continuous bin index
        bins = torch.arange(N_TEMPO_BINS, device=dev).float().view(1, 1, -1)
        soft = torch.exp(-0.5 * ((bins - bin_pos.unsqueeze(-1)) / std) ** 2)
        soft = soft / soft.sum(-1, keepdim=True).clamp_min(1e-8)           # [B,F,N_BINS]
        logp = F.log_softmax(out["tempo"], dim=-1)
        lt = -(soft * logp).sum(-1)                                        # [B,F] soft-CE
        lt = (lt * tmask).sum() / tmask.sum().clamp_min(1.0)
        pwc = torch.tensor(pos_weight_change, device=dev)
        lc = F.binary_cross_entropy_with_logits(out["change"], batch["change"], pos_weight=pwc, reduction="none")
        lc = (lc * m).sum() / m.sum() / batch["change"].shape[0]
        total = total + w_tempo * lt + w_change * lc
        parts["tempo"] = lt.detach()
        parts["change"] = lc.detach()

    # REDLINE-PLACEMENT (notation) head: BCE against ALL GT redlines (major_only=False),
    # i.e. including the convention per-barline re-anchors the change head deliberately
    # excludes. pos_weight lower than change (positives are ~10x denser on live maps).
    if "red" in out and "red" in batch:
        pwr = torch.tensor(pos_weight_red, device=dev)
        lr_ = F.binary_cross_entropy_with_logits(out["red"], batch["red"], pos_weight=pwr, reduction="none")
        lr_ = (lr_ * m).sum() / m.sum() / batch["red"].shape[0]
        total = total + w_red * lr_
        parts["red"] = lr_.detach()

    return total, parts


@torch.no_grad()
def beat_f1(logits, target, mask, tol_frames=2, thresh=0.0):
    """Rough peak-agnostic F1 at frame tolerance (for monitoring, not the real metric)."""
    pred = (logits > thresh).float()
    if mask is not None:
        pred = pred * mask
    # dilate target by tol for tolerant matching
    t = target.clone()
    for s in range(1, tol_frames + 1):
        t[:, s:] = torch.maximum(t[:, s:], target[:, :-s])
        t[:, :-s] = torch.maximum(t[:, :-s], target[:, s:])
    tp = (pred * t).sum()
    fp = (pred * (1 - t)).sum()
    fn = ((1 - pred) * target).sum()
    prec = tp / (tp + fp).clamp_min(1)
    rec = tp / (tp + fn).clamp_min(1)
    return (2 * prec * rec / (prec + rec).clamp_min(1e-6)).item()
