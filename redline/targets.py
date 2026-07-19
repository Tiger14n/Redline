"""osu redlines -> per-frame training targets (and back).

A frame-level tracker predicts, per spectrogram frame (50 fps = 20 ms):
  - beat       : is there a beat in this frame?
  - downbeat   : is there a downbeat (measure start) in this frame?
  - offset     : the beat's sub-frame position in [-0.5, 0.5] (frames), so we recover
                 ms precision on the coarse 20 ms grid. This is also where the osu-
                 convention offset is learned, which removes the ~33 ms latency that
                 off-the-shelf trackers have.

Targets here are HARD (one frame per beat); shift-tolerance is handled in the loss
(max-pool), not by widening the targets, so the offset stays well-defined.

Run `python -m redline.targets` for a round-trip sanity check on a few dataset maps:
labels -> reconstruct -> compare to the original beat grid (error should be ~0 ms).
"""
from __future__ import annotations
import numpy as np

FPS = 50
FRAME_MS = 1000.0 / FPS          # 20.0 ms per frame


def redlines_from_beatmap(bm) -> list[tuple[float, float, int]]:
    """(offset_ms, ms_per_beat, meter) for every uninherited timing point, sorted."""
    reds = []
    for t in bm.timing_points:
        if t.parent is None:                       # uninherited == redline
            mpb = t.ms_per_beat
            meter = int(getattr(t, "meter", 4) or 4)
            reds.append((t.offset.total_seconds() * 1000.0, float(mpb), meter))
    reds.sort(key=lambda x: x[0])
    return reds


def beat_grid(redlines, end_ms) -> tuple[np.ndarray, np.ndarray]:
    """Expand redlines into (beat_times_ms, downbeat_times_ms) up to end_ms."""
    R = sorted(redlines, key=lambda x: x[0])
    beats, downs = [], []
    for i, (off, mpb, meter) in enumerate(R):
        # mpb floor: <10ms/beat = >6000 BPM = gimmick/troll redline, not music. Without it
        # one mpb~1e-4 redline expands to billions of floats (~25GB -> froze the machine).
        if mpb < 10.0 or off >= end_ms:
            continue
        # cap each section at the audio end: osu maps often have trailing timing points
        # placed long after the song, which would otherwise flood the gap with beats.
        nxt = min(R[i + 1][0], end_ms) if i + 1 < len(R) else end_ms
        k = 0
        t = off
        while t < nxt - 1e-6:
            beats.append(t)
            if meter > 0 and k % meter == 0:
                downs.append(t)
            k += 1
            t = off + k * mpb
    return np.asarray(beats, np.float64), np.asarray(downs, np.float64)


def make_frame_labels(beats, downs, n_frames, fps=FPS, widen=0):
    """beat/downbeat times (ms) -> per-frame (beat, downbeat, offset, offset_mask).

    widen>0 smears each beat/downbeat target over +-widen frames (shift-tolerant
    training). The offset target at each smeared frame points to the SAME true beat, so
    its range is [-(0.5+widen), 0.5+widen] frames — whichever frame the model peaks on at
    inference, its offset recovers the exact beat time.
    """
    fr_ms = 1000.0 / fps
    beat_t = np.zeros(n_frames, np.float32)
    down_t = np.zeros(n_frames, np.float32)
    off_t = np.zeros(n_frames, np.float32)
    off_m = np.zeros(n_frames, np.float32)
    for t in beats:
        fc = int(round(t / fr_ms))
        delta = (t - fc * fr_ms) / fr_ms            # in [-0.5, 0.5]
        for w in range(-widen, widen + 1):
            f = fc + w
            if 0 <= f < n_frames:
                beat_t[f] = 1.0
                off_t[f] = delta - w                # points to the true beat from frame f
                off_m[f] = 1.0
    for t in downs:
        fc = int(round(t / fr_ms))
        for w in range(-widen, widen + 1):
            f = fc + w
            if 0 <= f < n_frames:
                down_t[f] = 1.0
    return beat_t, down_t, off_t, off_m


def subdivision_mask(beats, n_frames, fps=FPS):
    """Per-frame mask = 1 at the half-beat MIDPOINTS between consecutive beats (the
    eighth-note positions where the model over-detects beats). Drives an extra beat-
    suppression loss so the beat head learns NOT to fire on subdivisions = tempo-
    consistency in-model (the fundamental fix, vs pruning at decode time)."""
    fr_ms = 1000.0 / fps
    sub = np.zeros(n_frames, np.float32)
    bs = np.sort(np.asarray(beats, np.float64))
    if len(bs) < 2:
        return sub
    mids = (bs[:-1] + bs[1:]) / 2.0
    for t in mids:
        fc = int(round(t / fr_ms))
        if 0 <= fc < n_frames:
            sub[fc] = 1.0
    return sub


OFFSET_RANGE = 1.5    # |offset| bound for widen=1 (= 0.5 + widen); model head clamps to this
LOG_MPB_CENTER = float(np.log(60000.0 / 180.0))   # center the log-tempo target at 180 BPM

# A redline counts as a MAJOR (audio-driven) change if it shifts tempo/meter/phase by at
# least these vs the previous redline. Minor re-anchors below this (eyeballed phase nudges,
# <3% drift corrections) are NOT learnable from audio and are positive-label NOISE for the
# change head -- so we exclude them from the change target (the tempo curve still keeps them).
MAJOR_TEMPO_FRAC = 0.03      # > 3% ms-per-beat change
MAJOR_PHASE_MS = 20.0        # > 20 ms grid phase shift

# Tempo head as per-frame BPM classification (sidesteps regression-loss scaling; soft-argmax
# over bins gives sub-BPM precision). Bins are 1 BPM wide over the osu range.
BPM_MIN = 30.0
N_TEMPO_BINS = 300           # 30..330 BPM, 1 BPM/bin
TEMPO_SOFT_STD = 2.0         # Gaussian soft-target width (bins)


def major_change_mask(redlines):
    """Per-redline bool: is it a major change vs the previous one? redlines = (off_ms, mpb, meter).
    First redline -> True (the initial anchor / music start). Shared by the change-head target
    and the major-recall eval so training and measurement use the SAME definition."""
    R = sorted(redlines, key=lambda x: x[0])
    out = []
    for i, (off, mpb, meter) in enumerate(R):
        if i == 0:
            out.append(True)
            continue
        poff, pmpb, pmeter = R[i - 1]
        if pmpb <= 0:
            out.append(True)
            continue
        tempo_chg = abs(mpb - pmpb) / pmpb
        d = off - poff
        phase = abs(d - round(d / pmpb) * pmpb)
        out.append(bool(tempo_chg > MAJOR_TEMPO_FRAC or meter != pmeter or phase > MAJOR_PHASE_MS))
    return out


def make_tempo_change_labels(redlines, end_ms, n_frames, fps=FPS, widen=1, major_only=True):
    """Per-frame timing-structure targets for the v2 model:
      tempo  : centered log(ms_per_beat) of the redline section each frame falls in
      tmask  : 1 where a tempo target exists
      change : 1 at each redline frame (widened) -> where an uninherited point belongs

    Reconstruction reads the change peaks (redline positions) + the tempo curve (BPM per
    section) directly, instead of differentiating jittery beats -> no spray, octave learned.
    """
    fr_ms = 1000.0 / fps
    logt = np.zeros(n_frames, np.float32)
    tmask = np.zeros(n_frames, np.float32)
    change = np.zeros(n_frames, np.float32)
    R = sorted(redlines, key=lambda x: x[0])
    major = major_change_mask(R) if major_only else [True] * len(R)
    for i, (off, mpb, meter) in enumerate(R):
        if mpb <= 0 or off >= end_ms:
            continue
        nxt = min(R[i + 1][0], end_ms) if i + 1 < len(R) else end_ms
        sf = max(0, int(round(off / fr_ms)))
        ef = min(n_frames, max(sf + 1, int(round(nxt / fr_ms))))
        logt[sf:ef] = np.log(mpb) - LOG_MPB_CENTER   # tempo curve keeps EVERY section (full fidelity)
        tmask[sf:ef] = 1.0
        if not major[i]:                              # change head learns only major redlines
            continue
        fc = int(round(off / fr_ms))
        for w in range(-widen, widen + 1):
            f = fc + w
            if 0 <= f < n_frames:
                change[f] = 1.0
    return logt, tmask, change


def reconstruct_times(act, off_t, fps=FPS, thresh=0.5):
    """Inverse of make_frame_labels for one channel: frames above thresh -> ms times,
    refined by the per-frame offset. (At inference, peak-pick replaces the threshold.)"""
    fr_ms = 1000.0 / fps
    idx = np.where(act >= thresh)[0]
    return idx * fr_ms + off_t[idx] * fr_ms


# --------------------------------------------------------------------------- sanity
def _sanity(n_maps=6):
    import glob
    import io
    import zipfile
    from slider import Beatmap
    from pydub import AudioSegment

    oszs = sorted(glob.glob("data/osz/**/*.osz", recursive=True))
    step = max(1, len(oszs) // 400)
    oszs = oszs[::step][:n_maps]
    print(f"{'map':<46}{'redln':>6}{'beats':>7}{'dbeats':>7}{'recon_err_ms':>14}{'collisions':>11}")
    print("-" * 92)
    for osz in oszs:
        try:
            with zipfile.ZipFile(osz) as z:
                osu = next(n for n in z.namelist() if n.lower().endswith(".osu"))
                bm = Beatmap.parse(z.read(osu).decode("utf-8", "ignore"))
                anm = next((n for n in z.namelist() if n.lower() == (bm.audio_filename or "").lower()), None)
                anm = anm or next((n for n in z.namelist() if n.lower().endswith((".mp3", ".ogg", ".wav"))), None)
                dur_ms = len(AudioSegment.from_file(io.BytesIO(z.read(anm)))) if anm else 0
            R = redlines_from_beatmap(bm)
            if not R or dur_ms <= 0:
                print(f"  skip {osz.split('/')[-1][:40]}"); continue
            beats, downs = beat_grid(R, dur_ms)
            n_frames = int(np.ceil(dur_ms / FRAME_MS)) + 1
            bt, dt, ot, om = make_frame_labels(beats, downs, n_frames)
            rb = reconstruct_times(bt, ot)
            # each reconstructed beat -> nearest original beat error
            err = np.array([abs(x - beats[np.argmin(np.abs(beats - x))]) for x in rb]) if len(rb) else np.array([0.])
            collisions = len(beats) - int(bt.sum())   # beats that fell on an already-set frame (sub-20ms apart)
            name = osz.split("/")[-1][:44]
            print(f"{name:<46}{len(R):>6}{len(beats):>7}{len(downs):>7}"
                  f"{err.mean():>10.2f}/{err.max():>4.1f}{collisions:>11}")
        except Exception as ex:
            print(f"  FAIL {osz.split('/')[-1][:40]}: {type(ex).__name__}: {ex}")
    print("\nrecon_err = mean/max ms between reconstructed and original beats (should be ~0).")
    print("collisions = beats lost to a shared 20ms frame (sub-beat sections; expected tiny).")


if __name__ == "__main__":
    _sanity()
