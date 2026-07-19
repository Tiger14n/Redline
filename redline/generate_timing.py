"""Generate osu! timing from an audio file.

    python -m redline.generate_timing song.mp3
    python -m redline.generate_timing song.mp3 --osz          # pack .osu + audio into an .osz
    python -m redline.generate_timing song.mp3 --solver rubato -o out/

Writes a timing-only beatmap (redlines: offset / BPM / meter) named after the audio file:
  default : song.osu   — drop next to an existing project or import into the editor
  --osz   : song.osz   — self-contained set (the .osu + the audio), drag into osu!
"""
from __future__ import annotations
import argparse
import os
import zipfile
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", help="audio file (mp3 / ogg / wav / m4a / flac)")
    ap.add_argument("-o", "--out", default="", help="output directory (default: next to the audio)")
    ap.add_argument("--osz", action="store_true",
                    help="write a self-contained .osz (audio + .osu) instead of a bare .osu")
    ap.add_argument("--solver", default="tempo", choices=["tempo", "rubato", "dp"],
                    help="tempo (default, recommended) | rubato (steep-ramp specialist) | dp")
    ap.add_argument("--device", default="cuda", help="cuda | cpu")
    ap.add_argument("--ckpt", default="", help="checkpoint override (default: models/redline-v1.pt "
                                               "or $REDLINE_CKPT)")
    args = ap.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        ap.error(f"audio file not found: {audio_path}")
    out_dir = Path(args.out) if args.out else audio_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    from redline.audio_io import load_audio
    from redline.solver import TimingSolver, DEFAULT_CKPT
    from redline.osu_format import osu_text

    print(f"loading audio: {audio_path.name}")
    sig, sr = load_audio(str(audio_path))
    dur_s = len(sig) / sr
    print(f"  {dur_s:.0f}s @ {sr}Hz")

    print(f"listening… (solver={args.solver}, device={args.device})")
    solver = TimingSolver(ckpt=args.ckpt or DEFAULT_CKPT, device=args.device, solver=args.solver)
    redlines = solver.predict(sig, sr)
    if not redlines:
        raise SystemExit("no redlines produced — audio too short or silent?")
    bpms = [60000.0 / m for _, m, _ in redlines]
    print(f"  {len(redlines)} redline{'s' if len(redlines) != 1 else ''} "
          f"(median {np.median(bpms):.1f} BPM)")
    for off, mpb, meter in redlines[:12]:
        print(f"    {off:9.0f}ms  {60000.0 / mpb:7.2f} BPM  {int(meter)}/4")
    if len(redlines) > 12:
        print(f"    … {len(redlines) - 12} more")

    title = audio_path.stem
    first_obj = redlines[0][0]
    osu = osu_text(audio_path.name, title, "Unknown", "Redline timing", redlines, first_obj)

    if args.osz:
        out = out_dir / f"{title} (Redline).osz"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(audio_path.name, audio_path.read_bytes())
            z.writestr(f"Unknown - {title} (Redline) [Redline timing].osu", osu)
    else:
        out = out_dir / f"{title}.osu"
        out.write_text(osu, encoding="utf-8")
    print(f"wrote {out}")
    if args.osz:
        print("  -> drag into osu! to import, then open the diff in the editor (F4 for timing).")
    else:
        print("  -> timing-only .osu; import alongside its audio, or use --osz for a drag-in set.")


if __name__ == "__main__":
    main()
