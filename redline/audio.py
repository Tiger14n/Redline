"""Torch-free, ffmpeg-free audio decode via miniaudio (bundles dr_mp3 / stb_vorbis /
dr_wav / dr_flac). Replaces the pydub+ffmpeg decode and torchaudio resample used for
inference. One miniaudio.decode() call handles format detection, mono downmix and
resample to the model rate.

Formats: mp3, ogg/vorbis, wav, flac. NOT m4a/aac (miniaudio has no AAC decoder) — a rare
osu audio format; those maps must be transcoded first or fall back to the ffmpeg path.
"""
from __future__ import annotations
import zipfile

import numpy as np
import miniaudio
import soxr

SR = 22050
MAX_AUDIO_SEC = 300
_AUDIO_EXTS = (".mp3", ".ogg", ".wav", ".flac")


def load_audio(data: bytes, sr: int = SR, max_sec: int = MAX_AUDIO_SEC) -> np.ndarray:
    """Decode audio bytes -> mono float32 @ sr, capped at max_sec. miniaudio (dr_libs)
    decodes at the file's native rate; soxr (SoX resampler, SIMD C) resamples at high
    quality. Both are small, fast C libraries."""
    dec = miniaudio.decode(data, output_format=miniaudio.SampleFormat.FLOAT32,
                           nchannels=1)                     # native rate, mono
    x = np.asarray(dec.samples, dtype=np.float32)
    if dec.sample_rate != sr:
        x = soxr.resample(x, dec.sample_rate, sr, quality="HQ").astype(np.float32)
    if max_sec:
        x = x[: int(max_sec * sr)]
    return x


def audio_from_osz(osz_path: str, sr: int = SR, max_sec: int = MAX_AUDIO_SEC):
    """Extract and decode the audio track from an .osz -> mono float32 @ sr, or None."""
    with zipfile.ZipFile(osz_path) as z:
        names = z.namelist()
        am = next((n for n in names if n.lower().endswith(_AUDIO_EXTS)), None)
        if am is None:
            return None
        return load_audio(z.read(am), sr, max_sec)
