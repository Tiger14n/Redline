"""Model -> per-frame activation curves (beat/down/off/change/red [+tempo]), chunked with
overlap-add over the model's trained inference window.
Same npz format (x[14], y, t, e[384]), same song selection, GT beats -> clean A/B on
whether better perception features lift the non-AR redline GRU above 0.40.
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from redline.dataset_osz import list_osz, _read_set, SR, CHUNK_FRAMES, FRAME_MS, add_onset  # noqa
from redline.targets import beat_grid, major_change_mask, OFFSET_RANGE                       # noqa
from redline.infer import Model2Beats, BORDER                                                # noqa
from redline.lattice import Lattice, viterbi, upsample2, LFRAME_MS                        # noqa


@torch.no_grad()
def model_curves(m2b, sig, sr, want_hidden=True, want_tempo=False, progress_cb=None):
    """progress_cb(frames_done, total_frames, partial_beat_curve) — optional, called after each
    chunk (used by the GUI to stream 'the model is listening' progress). No-op when None."""
    import torchaudio
    from redline.targets import N_TEMPO_BINS
    wav = torch.as_tensor(sig, dtype=torch.float32)
    if sr != m2b.sr:
        wav = torchaudio.functional.resample(wav, sr, m2b.sr)
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak
    mel = torch.log1p(m2b.mel(wav)).T.to(m2b.device)
    mel = add_onset(mel)                                  # onset rep -> [F, 2*nmels]
    F = mel.shape[0]
    chans = {k: np.zeros(F) for k in ("beat", "down", "off", "change", "conf")}
    if getattr(m2b.model, "use_red", False):
        chans["red"] = np.zeros(F)                       # notation head (ALL-redline placement)
    wsum = np.zeros(F)
    hid = np.zeros((F, m2b.model.norm.normalized_shape[0]), np.float32) if want_hidden else None
    tempo = np.zeros((F, N_TEMPO_BINS), np.float32) if want_tempo else None
    # window = the model's TRAINED crop length (60s-context ckpts report chunk_frames=3000);
    # falls back to the module default for older/30s models.
    cf = int(getattr(m2b, "chunk_frames", CHUNK_FRAMES))
    hop = cf - 2 * BORDER
    _ov = float(os.environ.get("INFER_OVERLAP", "0"))     # >0 -> overlapping windows, averaged
    if _ov > 0:
        hop = max(1, int(round(cf * (1.0 - _ov))))
    s = 0
    while s < F:
        seg = mel[s:s + cf]; n = seg.shape[0]
        if n < cf:
            seg = torch.cat([seg, seg.new_zeros(cf - n, seg.shape[1])])
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if m2b.use_bf16 else torch.autocast("cuda", enabled=False)
        with ctx:
            x = m2b.model.norm(_blocks(m2b.model, m2b.model.frontend(seg[None])))
            beat = m2b.model.beat_head(x).squeeze(-1)
            down = m2b.model.down_head(x).squeeze(-1)
            if getattr(m2b.model, "sum_head", False):
                beat = beat + down                        # sum head: downbeat logit adds into beat
            off = torch.tanh(m2b.model.off_head(x).squeeze(-1)) * OFFSET_RANGE
            change = m2b.model.change_head(x).squeeze(-1)
            red = m2b.model.red_head(x).squeeze(-1) if "red" in chans else None
            if want_tempo:
                if getattr(m2b.model, "autocorr_tempo", False):
                    # AutocorrTempoHead consumes the DETACHED sigmoid beat curve [B,F]
                    # (periodicity of the pulse), mirroring model.forward — NOT trunk features.
                    tlog = m2b.model.tempo_head(torch.sigmoid(beat).detach())
                elif getattr(m2b.model, "local_tempo", False):
                    tlog = m2b.model.tempo_head(m2b.model.frontend(seg[None]))
                else:
                    tlog = m2b.model.tempo_head(x)
                tp = torch.softmax(tlog.float(), -1)[0, :n].cpu().numpy()
        a, b = (BORDER if s > 0 else 0), (n - BORDER if s + cf < F else n)
        pb = torch.sigmoid(beat.float())[0, :n].cpu().numpy()
        chans["beat"][s + a:s + b] += pb[a:b]
        chans["conf"][s + a:s + b] += pb[a:b]                    # conf mirrors beat activation
        chans["down"][s + a:s + b] += torch.sigmoid(down.float())[0, :n].cpu().numpy()[a:b]
        chans["change"][s + a:s + b] += torch.sigmoid(change.float())[0, :n].cpu().numpy()[a:b]
        if red is not None:
            chans["red"][s + a:s + b] += torch.sigmoid(red.float())[0, :n].cpu().numpy()[a:b]
        chans["off"][s + a:s + b] += off.float()[0, :n].cpu().numpy()[a:b]
        if want_hidden:
            hid[s + a:s + b] = x.float()[0, :n].cpu().numpy()[a:b]
        if want_tempo:
            tempo[s + a:s + b] += tp[a:b]
        wsum[s + a:s + b] += 1
        if progress_cb is not None:
            done = min(F, s + b)
            progress_cb(done, F, chans["beat"][:done] / np.maximum(wsum[:done], 1))
        s += hop
        if n < cf:
            break
    w = np.maximum(wsum, 1)
    out = {k: v / w for k, v in chans.items()}
    if want_hidden:
        out["hidden"] = hid
    if want_tempo:
        out["tempo"] = tempo / w[:, None]
    return out


def _blocks(model, x):
    for blk in model.blocks:
        x = blk(x)
    return x


