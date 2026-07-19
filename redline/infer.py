"""Model loading + a minimal beat-tracker interface.

Model2Beats loads a Redline checkpoint and rebuilds the exact data representation it was
trained with (sample rate, mel config, onset channel, inference window). Calling it returns
(beats, downbeats) in seconds; the full timing pipeline consumes the richer per-frame curves
from redline.curves.model_curves instead.
"""
from __future__ import annotations
from contextlib import nullcontext

import numpy as np
import torch

from redline.dataset_osz import _mel_transform, add_onset
from redline.model import TimingModel
from redline.targets import BPM_MIN, N_TEMPO_BINS

BORDER = 6


class Model2Beats:
    def __init__(self, ckpt, device="cuda"):
        self.device = device
        sd = torch.load(ckpt, map_location=device)
        a = sd.get("args", {}) if isinstance(sd, dict) else {}
        # rebuild the exact DATA REPRESENTATION the ckpt was trained with.
        # HOP = sr//50 keeps 50 fps so FRAME_MS stays 20 ms.
        import redline.dataset_osz as _ds
        self.sr = int(a.get("sr") or 22050)
        _ds.SR = self.sr; _ds.N_FFT = int(a.get("n_fft") or 1024)
        _ds.N_MELS = int(a.get("n_mels") or 128); _ds.FMAX = (a.get("fmax") or None)
        self.onset = bool(a.get("onset")); _ds.ONSET = self.onset
        _ds.HOP = self.sr // 50; _ds.FRAME_MS = _ds.HOP / _ds.SR * 1000.0
        kw = {k: a[k] for k in ("d_model", "depth", "n_heads", "ffn",
                                "sum_head", "red_head") if a.get(k)}
        kw["n_mels"] = _ds.N_MELS * (2 if self.onset else 1)   # onset doubles the freq axis
        self.model = TimingModel(**kw).to(device).eval()
        self.model.load_state_dict(sd["model"] if "model" in sd else sd)
        self.mel = _mel_transform()
        self.use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
        # inference window = the crop length the ckpt was TRAINED at (60s-context models use
        # chunk_frames=3000). Rotary attention extrapolates, but matching the training window
        # is what the model was validated at. model_curves reads this per-model.
        self.chunk_frames = int(a.get("chunk_frames") or 1500)

    @torch.no_grad()
    def _frame_probs(self, signal, sr):
        """signal -> per-frame (beat, down, offset, bpm, change) via chunked overlap-add."""
        import torchaudio
        from redline.dataset_osz import FRAME_MS  # noqa: F401  (module-level, ckpt-adjusted)
        wav = torch.as_tensor(signal, dtype=torch.float32)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak
        mel = torch.log1p(self.mel(wav)).T.to(self.device)        # [F, M]
        mel = add_onset(mel)                                      # [F,2M] if onset rep
        ftot = mel.shape[0]
        cf = self.chunk_frames
        ab = torch.zeros(ftot); ad = torch.zeros(ftot)
        ao = torch.zeros(ftot); cnt = torch.zeros(ftot)
        at = torch.zeros(ftot); ac = torch.zeros(ftot)            # tempo (BPM) + change prob
        bins = torch.arange(N_TEMPO_BINS).float()
        hop = cf - 2 * BORDER
        for start in range(0, max(1, ftot), hop):
            seg = mel[start:start + cf]
            n = seg.shape[0]
            if n < cf:
                seg = torch.cat([seg, seg.new_zeros(cf - n, seg.shape[1])])
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if self.use_bf16 else nullcontext()
            with ctx:
                out = self.model(seg[None])
            pb = torch.sigmoid(out["beat"][0].float()).cpu()
            pd = torch.sigmoid(out["down"][0].float()).cpu()
            of = out["off"][0].float().cpu()
            ptempo = torch.softmax(out["tempo"][0].float().cpu(), dim=-1)     # [n, N_BINS]
            bpm = BPM_MIN + (ptempo * bins).sum(-1)                           # soft-argmax -> BPM
            pc = torch.sigmoid(out["change"][0].float()).cpu()
            lo = 0 if start == 0 else BORDER
            hi = n if start + cf >= ftot else cf - BORDER
            idx = slice(start + lo, start + min(hi, n))
            seg_idx = slice(lo, min(hi, n))
            ab[idx] += pb[seg_idx]; ad[idx] += pd[seg_idx]
            ao[idx] += of[seg_idx]; cnt[idx] += 1
            at[idx] += bpm[seg_idx]; ac[idx] += pc[seg_idx]
            if start + cf >= ftot:
                break
        cnt = cnt.clamp_min(1)
        return ((ab / cnt).numpy(), (ad / cnt).numpy(), (ao / cnt).numpy(),
                (at / cnt).numpy(), (ac / cnt).numpy())

    def __call__(self, signal, sr, thresh=0.4, min_dist=5):
        """Beat-tracker interface: (signal, sr) -> (beat times, downbeat times) in seconds."""
        from redline.npsignal import find_peaks
        from redline.dataset_osz import FRAME_MS
        pb, pd, off, _tempo, _change = self._frame_probs(signal, sr)
        peaks, _ = find_peaks(pb, height=thresh, distance=min_dist)
        if len(peaks) == 0:
            return np.array([]), np.array([])
        beats = (peaks + off[peaks]) * FRAME_MS / 1000.0
        dmask = pd[peaks] > 0.5
        return beats, beats[dmask]
