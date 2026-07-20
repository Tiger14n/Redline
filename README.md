# Redline

Automatic timing detection for osu!. Give it a song, get back redlines (uninherited timing points
with BPM, offset and meter).


This tool was built to make timing accessible for inexperienced people, and save time for experienced ones. 
I don't claim the results are perfect, so users should check and validate the result by ear.

**[Web app demo →](https://redline-kohl.vercel.app/)** 

<img width="936" height="481" alt="image" src="https://github.com/user-attachments/assets/34373307-9b32-467d-8227-9958c265c969" />


Under the hood:

- **redline-v1**, a 45M-parameter frame-level encoder-only model emitting beat, downbeat, sub-frame
  offset, tempo, change and notation activations at 50 fps. The notation head is trained on
  where mappers actually put redlines, which is not where the audio changes.
- a deterministic solver turning those curves into redlines: tempo-curve segmentation,
  octave anchoring, fit-gate plus live re-anchoring, notation placement. Same input always
  gives the same output; the solve takes ~0.2 s after the model pass.


## Quick start

### Online demo
web app demo: **[https://redline-kohl.vercel.app](https://redline-kohl.vercel.app/)** 

this web app demo will remain online and free as long as i can afford the costs. and the web app codebase will be open sourced by then.

### Local install
```bash
pip install -r requirements.txt              # plus ffmpeg on PATH for mp3 decode
hf download Tiger14n/redline-v1 redline-v1.pt --local-dir models/   # weights (180 MB)

python -m redline.generate_timing song.mp3           # -> song.osu (timing only)
python -m redline.generate_timing song.mp3 --osz     # -> drag-in .osz (audio + .osu)
python -m redline.generate_timing song.mp3 --solver rubato   # steep-ramp specialist
```

Weights: [`Tiger14n/redline-v1`](https://huggingface.co/Tiger14n/redline-v1) (path also
settable via `REDLINE_CKPT`).

## Performance

Held-out data throughout: ranked mapsets excluded from training by song-level dedup,
scored against the ranked map's human timing.

Model heads, 1000 held-out mapsets:

| genre | maps | beat F1 | downbeat F1 | offset MAE |
|---|---|---|---|---|
| Pop | 285 | 0.966 | 0.885 | 4.7 ms |
| Anime | 89 | 0.969 | 0.895 | 5.2 ms |
| Rock | 192 | 0.963 | 0.863 | 4.8 ms |
| Electronic | 224 | 0.962 | 0.886 | 5.1 ms |
| Metal | 50 | 0.947 | 0.855 | 5.2 ms |
| Video game | 100 | 0.944 | 0.863 | 5.1 ms |
| other | 60 | 0.916 | 0.801 | 5.6 ms |
| **all** | **1000** | **0.959** | **0.873** | **5.0 ms** |

Frame-level, ±40 ms tolerance. End-to-end (audio in, redlines out), 366 held-out mapsets,
~50 per genre:

| genre | maps | beats within 10 ms | typical map | redline match P / R |
|---|---|---|---|---|
| Pop | 55 | 89.3 % | 100.0 % | 0.51 / 0.54 |
| Metal | 48 | 88.2 % | 96.9 % | 0.55 / 0.43 |
| Rock | 51 | 87.3 % | 95.2 % | 0.42 / 0.42 |
| Electronic | 49 | 85.0 % | 100.0 % | 0.57 / 0.57 |
| Video game | 54 | 82.7 % | 98.7 % | 0.47 / 0.50 |
| Anime | 55 | 80.9 % | 98.8 % | 0.65 / 0.50 |
| other (folk/jazz/hip-hop/classical/novelty) | 54 | 80.2 % | 92.7 % | 0.40 / 0.52 |
| **all** | **366** | **84.7 %** | **98.7 %** | 0.51 / 0.50 |

Genre gaps here are mostly sample size: on the 1000-map run the bootstrap intervals
overlap (metal − anime = +3.2 pt, 95 % CI [−4.1, +10.5]). Read the "all" row.

### What the numbers mean

**Beats within 10 ms** — every beat of the human timing, checked against Redline's grid.
10 ms is about the tightest offset a player can feel, so a beat inside it plays as snapped.
The mean is dragged down by a hard minority: 240 of 366 maps score above 90 %, and 39 below
50 % (see below). On a 42-redline accelerando (84→200 BPM over 20 s) it places 43 redlines
at 93 %.

**Redline match** — where lines were placed, not how the grid sounds. Precision is the
fraction of Redline's lines the human also wrote, recall the fraction of the human's lines
Redline found (±150 ms). These read low because placement is partly personal style: mappers
often add a line at a new section when the BPM barely changes, and skipping it doesn't move
a single beat. Nothing scores near 1.0 here.

Of the 185 songs a human timed with one redline, Redline writes exactly one on 147. The
rest pick up a few extra lines around quiet intros and outros.

### Where it fails, and why

~7 % of maps score below 50 %. Classifying those failures by the *shape* of the timing
error (flat offset, drifting ramp, staircase, or noise) separates four distinct causes:

**1. The reference is offset, not Redline (~27 %).** The error is a perfectly constant
displacement — Redline's grid exactly parallel to the mapper's, ~12 ms away — so a 10 ms
tolerance scores it ~0 % while it sits ~2 ms from right. Adjudicated against the audio
(spectral-flux onsets at 2.9 ms, detector latency calibrated out): **on 19 of 20 such maps
Redline is 2.0 ms from the audio-true beat and the mapper 16.0 ms away**, outside tolerance
from their own song. Not an MP3 delay artifact — containers are mixed and the pipeline has
no global bias (+0.05 ms). Unfixable from audio: per-band onset optima are identical on
failing and passing maps, so nothing acoustic predicts a mapper's offset habit.

**2. Metrical-level disagreement (~17 %).** Redline picks ×2, ÷2 or ×2⁄3 of the mapper's
BPM — the same pulse at a different metrical level, a notation convention call rather than
an acoustic fact. The tempo head usually hears both levels.

**3. Genuinely hard audio (~40 %).** Live recordings, solo-piano rubato, jazz swing. Beat
activation drops to 0.650 versus 0.942 on healthy maps: the pulse isn't clearly there, and
neither grid — Redline's or the mapper's — sits on strong onsets.

**4. Odd tempo ramps (~4 %).** Model hears the song, grid still drifts. Too few maps to
conclude anything.

So the headline is pessimistic by ~1.7 points: excluding only the proven-offset maps moves
the 1000-map benchmark from 88.5 % to 90.2 % and the failure rate from 7.0 % to 5.1 %. The
reference set is not hand-cleaned, so numbers are reported as measured.

### Solvers

`tempo` (default) beat `rubato` on 166 stratified held-out maps: wins beat accuracy 86 to
34 (46 ties), wins redline F1 in every stratum, solves 44× faster. `rubato` remains for
steep monotone ramps; `dp` is a plain constant-tempo fitter.

Reproduce: `python -m redline.eval_genres --pool <heldout_osz_dir> --train_dir <train_osz_dir>`
(or `redline.eval_cohorts` for fixed map lists).

## Training

The released model is the third stage of a warm-start lineage; configs in `configs/`:

```bash
python -m redline.train configs/pretrain.yaml     # base (beat/down/offset/tempo/change)
python -m redline.train configs/ft_dense.yaml     # dense/live oversample fine-tune
python -m redline.train configs/redline_v1.yaml   # + notation head -> released model
```

Override any key inline: `--set steps=600 batch=8`. Training reads `.osz` archives directly
(`root:` in the config). Bring your own corpus; ranked-map audio is not distributable.
Trained end-to-end on a single RTX 3080 (10 GB).

The model was trained using ~20k ranked osu! mapsets.

## Repository layout

```
redline/            the package
  model.py          frame-level transformer (rotary attention, conv frontend)
  dataset_osz.py    streaming .osz dataset (crops, varispeed/degrade aug, genre balance)
  losses.py         beat/down/change/notation BCE + offset smooth-L1 + soft-CE tempo
  train.py          YAML-config trainer (Muon + AdamW)
  infer.py          checkpoint loader + minimal beat-tracker interface
  curves.py         model -> per-frame activation curves (chunked overlap-add)
  lattice.py        bar-pointer viterbi beat decode
  solver.py         curves -> redlines (segmentation, octave anchor, notation placement)
  generate_timing.py / eval_cohorts.py / probe_redhead.py / synth_solver_test.py
  osu_format.py     .osu/.osz read & write, GT parsing
configs/            the 3-stage training lineage
```

## Acknowledgements

- Benchmarked against [Beat This!](https://arxiv.org/abs/2407.21658) (Foscarin et al.,
  ISMIR 2024) during development; a shift-tolerant loss following their Sec. 3.3 was
  evaluated but is not used in the released recipe.
