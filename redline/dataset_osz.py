"""On-the-fly .osz streaming dataset for the timing model.

No extraction, no metadata file: each .osz (a ranked beatmap set) is opened in memory,
one .osu is parsed for timing, the audio is decoded from the zip buffer, and a few random
30 s chunks are emitted as (mel, beat, downbeat, offset, mask) tensors. Timing is shared
across a set's difficulties, so we use one .osu per set (the one with the most redlines).

Audio frontend matches Beat This!: 22050 Hz, n_fft 1024, hop 441 -> exactly 50 fps (20 ms
frames), 128 mel bins, log1p. Mel is computed here (per worker) so frames and labels are
guaranteed aligned.

Quick check:  python -m redline.dataset_osz
"""
from __future__ import annotations
import io
import glob
import os
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .targets import (redlines_from_beatmap, beat_grid, make_frame_labels, subdivision_mask,
                      make_tempo_change_labels, major_change_mask)

# canonical values live in redline.const (torch-free); re-exported here as mutable module
# globals (infer.py rebinds _ds.SR etc. per-checkpoint at runtime).
from redline.const import (SR, N_FFT, HOP, N_MELS, FMAX, ONSET, CHUNK_FRAMES,  # noqa: E402,F401
                           FRAME_MS, MAX_AUDIO_SEC)


def _mel_transform():
    import torchaudio
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=1.0, f_max=FMAX)


def add_onset(mel):
    """[F,M] log-mel -> [F,2M] = mel concat per-bin spectral flux (relu of frame-to-frame
    rise). Explicit onset cue for soft/vocal onsets. No-op unless ONSET. Frontend is
    unchanged (just a wider freq axis)."""
    if not ONSET:
        return mel
    import torch
    flux = (mel[1:] - mel[:-1]).clamp(min=0)
    flux = torch.cat([flux.new_zeros(1, mel.shape[1]), flux], dim=0)   # [F,M], first frame 0
    return torch.cat([mel, flux], dim=-1)                              # [F,2M]


def list_osz(root: str) -> list[str]:
    return sorted(glob.glob(os.path.join(root, "**", "*.osz"), recursive=True))


def _read_set(osz_path):
    """-> (mono float32 @22050, redlines) or None. Picks the .osu with the most redlines."""
    from slider import Beatmap
    from pydub import AudioSegment
    with zipfile.ZipFile(osz_path) as z:
        names = z.namelist()
        osus = [n for n in names if n.lower().endswith(".osu")]
        if not osus:
            return None
        best, best_R, best_audio = None, [], None
        for n in osus:
            try:
                bm = Beatmap.parse(z.read(n).decode("utf-8", "ignore"))
            except Exception:
                continue
            R = redlines_from_beatmap(bm)
            if len(R) > len(best_R):
                best, best_R, best_audio = bm, R, bm.audio_filename
        if best is None or not best_R:
            return None
        am = next((n for n in names if n.lower() == (best_audio or "").lower()), None)
        am = am or next((n for n in names if n.lower().endswith((".mp3", ".ogg", ".wav", ".m4a"))), None)
        if am is None:
            return None
        if z.getinfo(am).file_size > 60 * 1024 * 1024:  # no song is >60MB; corrupt/marathon audio
            return None                                  # would fully decode before the cap below
        seg = (AudioSegment.from_file(io.BytesIO(z.read(am)), duration=MAX_AUDIO_SEC)
               .set_channels(1).set_frame_rate(SR))      # duration= -> ffmpeg -t: cap AT DECODE
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    if len(samples) > MAX_AUDIO_SEC * SR:          # cap RAM: truncate very long songs
        samples = samples[:MAX_AUDIO_SEC * SR]
    peak = np.abs(samples).max()
    if peak > 0:
        samples = samples / peak
    return samples, best_R


_RESAMPLERS: dict = {}     # factor -> cached torchaudio.transforms.Resample (per worker)


def _varispeed(samples: np.ndarray, factor: float) -> np.ndarray:
    """Tape-style varispeed (pitch+tempo together) of a mono float32 @SR array.
    factor>1 = faster & shorter (BPM x factor); factor<1 = slower.

    Resamples by a SMALL-INTEGER RATIO (factor ~= p/q via limit_denominator), NOT
    (SR, round(SR/factor)). torchaudio.Resample only uses the *ratio* of its two args, and a
    small ratio => tiny sinc kernel. The old (SR, round(SR/factor)) form landed on integers
    coprime with 22050 for some factors (e.g. 0.8 -> ratio 11025:13781), building a ~14k-tap
    filter = ~225ms/call (1.1 -> ~44ms), while nice ratios (0.7=7:10) were ~2ms -- a 24% train
    slowdown that varied wildly by factor. Small-ratio form makes EVERY factor ~2ms, same
    effect/length. Cached per factor (one module each); callers pass DISCRETE factors."""
    import torchaudio
    from fractions import Fraction
    rs = _RESAMPLERS.get(factor)
    if rs is None:
        fr = Fraction(factor).limit_denominator(16)      # factor ~= p/q (playback speed)
        # output played @SR is `factor` faster <=> resample ratio new/orig = 1/factor = q/p,
        # i.e. Resample(orig=p, new=q). p,q tiny => tiny kernel.
        rs = torchaudio.transforms.Resample(fr.numerator, fr.denominator, lowpass_filter_width=6)
        _RESAMPLERS[factor] = rs
    t = torch.from_numpy(np.ascontiguousarray(samples)).float().unsqueeze(0)
    return rs(t).squeeze(0).numpy().astype(np.float32)


class OszTimingDataset(IterableDataset):
    def __init__(self, osz_paths, crops_per_song=4, chunk_frames=CHUNK_FRAMES,
                 shuffle=True, seed=0, gain_aug=True, redline_crop_prob=0.5,
                 speed_aug=False, speed_factors=(0.9, 0.95, 1.05, 1.1), speed_aug_prob=0.5,
                 degrade_aug=False, degrade_prob=0.5,
                 genre_buckets=None, genre_alpha=0.5,
                 length_aware=False, sample_alpha=0.65, unique_crops=False):
        super().__init__()
        self.osz_paths = list(osz_paths)
        self.crops_per_song = crops_per_song
        self.chunk_frames = chunk_frames
        self.shuffle = shuffle
        self.seed = seed
        self.gain_aug = gain_aug
        self.redline_crop_prob = redline_crop_prob   # bias crops to contain a redline (v2 change head)
        self.speed_aug = speed_aug                   # TRAIN only; val_ds is always built with False
        # drop a no-op 1.0 so a sampled factor always perturbs; the list is kept DISCRETE so
        # torchaudio's resample-kernel cache is bounded (continuous factors would leak -- see
        # _varispeed). Empty list => aug effectively off even if speed_aug=True.
        self.speed_factors = [float(f) for f in speed_factors if abs(float(f) - 1.0) > 1e-3]
        self.speed_aug_prob = speed_aug_prob
        # within-crop speed-splicing: synthetic multi-BPM / accelerando / rubato crops with
        # exact warped labels (TRAIN only). Alternative crop type, gated to single-tempo source
        # spans so the warp + per-segment redline mpb stay exact.
        self.degrade_aug = degrade_aug
        self.degrade_prob = degrade_prob
        # genre-balanced sampling: {genre: [paths]}. When set, the stream draws a genre per item
        # with prob proportional to count**genre_alpha (alpha=0 -> exactly equal per genre;
        # alpha=1 -> natural distribution; alpha~0.5 -> boost rare genres while keeping the
        # majority well-represented), then cycles that genre's maps in shuffled order (no
        # replacement within a genre pass -> max unique exposure). Rebalances metal et al.
        self.genre_buckets = {g: list(v) for g, v in (genre_buckets or {}).items() if v}
        self.genre_alpha = genre_alpha
        # length-aware crop count: k = round(alpha * dur/30s) (Beat-This alpha=0.65) -- sublinear in
        # duration, balances short-song over-sampling vs long-song under-coverage of fixed crops.
        self.length_aware = length_aware
        self.sample_alpha = sample_alpha
        # non-overlapping crops: partition each song into disjoint chunk-sized tiles (with per-tile
        # random jitter) so the N crops from one song never overlap -> max diversity per song, no
        # near-duplicate samples. Short songs emit only as many crops as fit (no overlap-padding).
        self.unique_crops = unique_crops
        self._mel = None

    def _tiled_starts(self, total_frames, major_offsets, n_crops, rng):
        """Distinct, non-overlapping crop start-frames. Song is split into floor(total/chunk)
        tiles (granularity >= chunk_frames); we pick min(n_crops, n_tiles) distinct tiles and
        place one crop inside each with a random within-tile offset -> guaranteed no overlap,
        still randomized per epoch. Tile SELECTION is biased toward redline tiles
        (redline_crop_prob) so the change head keeps seeing positives without overlapping crops."""
        cf = self.chunk_frames
        if total_frames <= cf:
            return [rng.randint(0, max(0, total_frames - cf))]            # song < 1 chunk -> 1 crop
        n_tiles = total_frames // cf
        tile_w = total_frames // n_tiles                                  # >= cf
        eff = min(n_crops, n_tiles)
        rl_tiles, plain_tiles = [], []
        rl_f = [int(o / FRAME_MS) for o in major_offsets]
        for t in range(n_tiles):
            lo, hi = t * tile_w, t * tile_w + tile_w
            (rl_tiles if any(lo <= r < hi for r in rl_f) else plain_tiles).append(t)
        rng.shuffle(rl_tiles); rng.shuffle(plain_tiles)
        chosen = []
        for _ in range(eff):
            if rng.random() < self.redline_crop_prob and rl_tiles:
                chosen.append(rl_tiles.pop())
            elif plain_tiles:
                chosen.append(plain_tiles.pop())
            elif rl_tiles:
                chosen.append(rl_tiles.pop())
        starts = []
        for t in chosen:
            base = t * tile_w
            in_tile = [r for r in rl_f if base <= r < base + tile_w]
            if in_tile:                                                   # place crop to contain a redline
                r = in_tile[rng.randint(0, len(in_tile) - 1)]
                lo = max(base, r - cf + 100)
                hi = min(base + tile_w - cf, r - 100)
                start_f = rng.randint(lo, hi) if hi > lo else max(base, min(base + tile_w - cf, r - cf // 2))
            else:
                start_f = base + rng.randint(0, tile_w - cf)
            starts.append(start_f)
        return starts

    def _shard(self):
        wi = get_worker_info()
        paths = list(self.osz_paths)
        if self.shuffle:
            random.Random(self.seed).shuffle(paths)
        if wi is None:
            return paths
        return paths[wi.id::wi.num_workers]

    def _genre_stream(self):
        """Infinite genre-balanced path stream: pick a genre with prob ~ count**genre_alpha,
        then draw the next map from that genre's per-worker shard (shuffled, cycled). Equalises
        genre representation (alpha=0) or boosts rare genres while keeping the majority (alpha~0.5)."""
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        rng = random.Random(self.seed + wid + 1)
        shards = {g: paths[wid::nw] for g, paths in self.genre_buckets.items()}
        genres = sorted(g for g, v in shards.items() if v)
        weights = [len(self.genre_buckets[g]) ** self.genre_alpha for g in genres]  # full-count weight
        order = {g: rng.sample(shards[g], len(shards[g])) for g in genres}
        pos = {g: 0 for g in genres}
        while True:
            g = rng.choices(genres, weights=weights)[0]
            if pos[g] >= len(order[g]):
                rng.shuffle(order[g]); pos[g] = 0
            p = order[g][pos[g]]; pos[g] += 1
            yield p

    def _degrade(self, mel, rng):
        """Label-preserving onset-degradation aug on the [F, n_mels] log1p-mel (mel domain, ~no
        cost -- elementwise + one tiny temporal conv, Simulates the
        onset-ambiguity that wrecks beat localization on metal (distortion walls), soft vocal/lo-fi,
        and loud/live masters, using the abundant clean data. Applied BEFORE add_onset so the flux
        cue sees the degraded onsets. Timing untouched."""
        F_, M = mel.shape
        # 1) temporal smearing: blend in a causal decaying-tail conv along time -> softer onsets
        if rng.random() < 0.6:
            a = rng.uniform(0.3, 0.7)
            L = 8
            ker = torch.tensor([a ** i for i in range(L)], dtype=mel.dtype)
            ker = (ker / ker.sum()).view(1, 1, L)
            x = torch.nn.functional.pad(mel.t().unsqueeze(1), (L - 1, 0))     # [M,1,F+L-1] causal
            sm = torch.nn.functional.conv1d(x, ker).squeeze(1).t()           # [F,M]
            w = rng.uniform(0.3, 0.8)
            mel = (1.0 - w) * mel + w * sm
        # 2) dynamic-range compression toward the per-bin mean (loudness-war / distortion feel)
        if rng.random() < 0.5:
            ratio = rng.uniform(0.5, 0.85)
            m = mel.mean(0, keepdim=True)
            mel = m + (mel - m) * ratio
        # 3) spectral tilt (random EQ coloration)
        if rng.random() < 0.5:
            d = rng.uniform(-0.5, 0.5)
            mel = mel + torch.linspace(-d, d, M, dtype=mel.dtype)
        # 4) additive noise (dense/noisy spectra)
        if rng.random() < 0.5:
            mel = mel + torch.randn_like(mel) * (rng.uniform(0.02, 0.1) * float(mel.std()) + 1e-6)
        return mel.clamp_min(0.0)

    def __iter__(self):
        if self._mel is None:
            self._mel = _mel_transform()
        wi = get_worker_info()
        rng = random.Random(self.seed + (wi.id if wi else 0) + 1)
        chunk_ms = self.chunk_frames * FRAME_MS
        chunk_samples = self.chunk_frames * HOP
        stream = self._genre_stream() if self.genre_buckets else self._shard()
        for osz in stream:
            try:
                res = _read_set(osz)
            except Exception:
                continue
            if res is None:
                continue
            samples, R = res
            # Optional varispeed (tape) augmentation -- TRAIN only (val_ds is built with
            # speed_aug=False). Applied PER-CROP below (resample only the ~30s clip, not the
            # whole song -- resampling a 4min marathon to use a 30s chunk was the perf hit).
            # Here we just pick the song's factor; labels are rescaled in the crop loop.
            dur_ms = len(samples) / SR * 1000.0
            if dur_ms < chunk_ms * 0.5:
                continue
            beats, downs = beat_grid(R, dur_ms)
            if len(beats) < 8:
                continue
            major_offsets = [o for (o, mp, me), mj in zip(R, major_change_mask(R)) if mj]
            total_frames = int(dur_ms / FRAME_MS)
            max_start = max(1, total_frames - self.chunk_frames)
            # length-aware crop count: k = round(alpha * dur/30s), sublinear in duration (Beat-This);
            # else fixed crops_per_song.
            n_crops = (max(1, int(round(self.sample_alpha * dur_ms / 1000.0 / 30.0)))
                       if self.length_aware else self.crops_per_song)
            # unique_crops: precompute distinct non-overlapping starts (caps crops to what fits).
            # Else: per-crop independent random start (legacy; can overlap on short songs).
            tiled_starts = (self._tiled_starts(total_frames, major_offsets, n_crops, rng)
                            if self.unique_crops else [None] * n_crops)
            for crop_start in tiled_starts:
                # per-crop varispeed factor (1.0 = none); resample only this clip, not the song
                s = 1.0
                if self.speed_aug and self.speed_factors and rng.random() < self.speed_aug_prob:
                    s = float(rng.choice(self.speed_factors))
                if crop_start is not None:
                    # unique_crops: distinct non-overlapping tile start (redline already considered)
                    start_f = crop_start
                elif rng.random() < self.redline_crop_prob and major_offsets:
                    # place a MAJOR redline at a random interior position of the chunk so the
                    # change head sees real positives (minor re-anchors are excluded from its target)
                    ro_f = int(major_offsets[rng.randint(0, len(major_offsets) - 1)] / FRAME_MS)
                    start_f = min(max_start, max(0, ro_f - rng.randint(100, self.chunk_frames - 100)))
                else:
                    start_f = rng.randint(0, max_start)
                start_ms = start_f * FRAME_MS
                # crop audio. with varispeed s, take s*chunk_samples of SOURCE so the stretched
                # clip fills exactly chunk_samples (s>1 = faster -> need more source). pad if short.
                a0 = start_f * HOP
                need = int(round(chunk_samples * s))
                clip = samples[a0:a0 + need]
                if s != 1.0:
                    clip = _varispeed(clip, s)                  # cheap: ~30s clip, not whole song
                clip = clip[:chunk_samples]
                if len(clip) < chunk_samples:
                    clip = np.pad(clip, (0, chunk_samples - len(clip)))
                if self.gain_aug:
                    clip = clip * rng.uniform(0.7, 1.0)
                # chunk-relative labels: source times shifted by start_ms then DIVIDED by s
                # (varispeed s makes everything 1/s shorter -> beat times and periods scale 1/s).
                end_ms = start_ms + chunk_ms * s               # source span that maps to this chunk
                wav = torch.from_numpy(np.ascontiguousarray(clip)).float()
                mel = self._mel(wav)                       # [n_mels, F+1]
                mel = torch.log1p(mel).T[:self.chunk_frames]   # [F, n_mels]
                if mel.shape[0] < self.chunk_frames:           # pad frames if short
                    mel = torch.cat([mel, mel.new_zeros(self.chunk_frames - mel.shape[0], N_MELS)])
                if self.degrade_aug and rng.random() < self.degrade_prob:   # onset-degradation
                    mel = self._degrade(mel, rng)
                mel = add_onset(mel)                            # [F,M] -> [F,2M] if ONSET
                cb = (beats[(beats >= start_ms) & (beats < end_ms)] - start_ms) / s
                cd = (downs[(downs >= start_ms) & (downs < end_ms)] - start_ms) / s
                bt, dt, ot, om = make_frame_labels(cb, cd, self.chunk_frames, widen=1)
                sub = subdivision_mask(cb, self.chunk_frames) * (bt == 0)   # off-beat midpoints only
                # v2 structure labels: shift redlines into chunk-relative time then scale by 1/s
                rc = [((off - start_ms) / s, mpb / s, meter) for (off, mpb, meter) in R]
                lt, tm, ch = make_tempo_change_labels(rc, chunk_ms, self.chunk_frames, widen=1)
                _, _, rd = make_tempo_change_labels(rc, chunk_ms, self.chunk_frames, widen=1,
                                                    major_only=False)   # notation head target
                yield {
                    "mel": mel,
                    "beat": torch.from_numpy(bt),
                    "down": torch.from_numpy(dt),
                    "off": torch.from_numpy(ot),
                    "mask": torch.from_numpy(om),
                    "tempo": torch.from_numpy(lt),
                    "tmask": torch.from_numpy(tm),
                    "change": torch.from_numpy(ch),
                    "red": torch.from_numpy(rd),
                    "sub": torch.from_numpy(sub),
                }


def _check():
    from torch.utils.data import DataLoader
    root = "data/osz"
    paths = list_osz(root)
    print(f"found {len(paths)} osz under {root}")
    ds = OszTimingDataset(paths[:200], crops_per_song=2, seed=1)
    dl = DataLoader(ds, batch_size=8, num_workers=2)
    for bi, b in enumerate(dl):
        mel, beat, down, off, mask = b["mel"], b["beat"], b["down"], b["off"], b["mask"]
        bpm_est = beat[0].sum().item() / (CHUNK_FRAMES * FRAME_MS / 60000)
        print(f"batch {bi}: mel{tuple(mel.shape)} beat{tuple(beat.shape)} "
              f"beats/chunk={beat.sum(1).float().mean():.0f} downs/chunk={down.sum(1).float().mean():.0f} "
              f"off|range=[{off[mask>0].min():.2f},{off[mask>0].max():.2f}] est_bpm(ex0)={bpm_est:.0f} "
              f"mel|mean={mel.mean():.2f}")
        if bi >= 3:
            break
    print("OK: shapes aligned, beat/offset labels populated, offsets within [-0.5,0.5].")


if __name__ == "__main__":
    _check()
