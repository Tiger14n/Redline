"""Canonical audio/frame constants, torch-free so the curves->solver path can import them
without pulling in torch (dataset_osz defines the dataset class on torch's IterableDataset,
so importing it drags torch; this module does not)."""
from __future__ import annotations

SR = 22050
N_FFT = 1024
HOP = 441                     # 22050 / 441 = 50 fps -> 20 ms frames
N_MELS = 128
FMAX = None                   # mel f_max (None = sr/2)
ONSET = False                 # if True, concat per-bin spectral flux (onset cue) -> mel width 2x
CHUNK_FRAMES = 1500           # 30 s
FRAME_MS = HOP / SR * 1000.0  # 20.0
MAX_AUDIO_SEC = 300           # cap per-song decode
