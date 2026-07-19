"""Reading and writing osu! beatmap timing: .osu text, .osz packing, GT redline parsing."""
from __future__ import annotations
import zipfile
from pathlib import Path


def osu_text(audio_name, title, artist, version, redlines, first_obj_ms):
    """A minimal valid .osu: headers + uninherited timing points + one placeholder circle."""
    tp_lines = []
    for off, mpb, meter in redlines:
        # time,beatLength,meter,sampleSet,sampleIndex,volume,uninherited,effects
        tp_lines.append(f"{int(round(off))},{mpb:.6f},{int(meter) if meter else 4},2,0,60,1,0")
    return "\n".join([
        "osu file format v14", "",
        "[General]",
        f"AudioFilename: {audio_name}",
        "AudioLeadIn: 0", "PreviewTime: -1", "Countdown: 0",
        "SampleSet: Soft", "Mode: 0", "",
        "[Metadata]",
        f"Title:{title}", f"TitleUnicode:{title}",
        f"Artist:{artist}", f"ArtistUnicode:{artist}",
        "Creator:Redline", f"Version:{version}",
        "Source:", "Tags:redline ai-timing", "BeatmapID:0", "BeatmapSetID:-1", "",
        "[Difficulty]",
        "HPDrainRate:5", "CircleSize:4", "OverallDifficulty:5",
        "ApproachRate:5", "SliderMultiplier:1.4", "SliderTickRate:1", "",
        "[TimingPoints]", *tp_lines, "",
        "[HitObjects]",
        f"256,192,{int(round(first_obj_ms))},1,0,0:0:0:0:", "",
    ])




def write_osz_set(out_dir, audio_name, audio_bytes, title, artist, diffs):
    """diffs = list of (version_label, redlines, first_obj_ms). Returns the .osz path."""
    safe = "".join(c for c in f"{artist} - {title}" if c.isalnum() or c in " -_()").strip()
    out_osz = Path(out_dir) / f"{safe} (AI-timing).osz"
    with zipfile.ZipFile(out_osz, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(audio_name, audio_bytes)
        for label, redlines, first_obj in diffs:
            z.writestr(f"{safe} [{label}].osu",
                       osu_text(audio_name, title, artist, label, redlines, first_obj))
    return out_osz




def read_audio_member(osz_path, audio_filename):
    """Raw audio bytes + its basename from the source .osz (no re-encode)."""
    with zipfile.ZipFile(osz_path) as z:
        names = z.namelist()
        m = next((n for n in names if n.lower() == (audio_filename or "").lower()), None)
        if m is None:
            m = next((n for n in names if n.lower().endswith(AUDIO_EXTS)), None)
        if m is None:
            raise FileNotFoundError("no audio in osz")
        return z.read(m), Path(m).name




def _gt_from_text(txt):
    """Direct [TimingPoints]/[Metadata] parse (fallback when slider rejects the
    format, e.g. BOM or a non-standard 'osu file format vNN'). Only uninherited
    (red) timing points -> (offset_ms, ms_per_beat, meter)."""
    def sec(name):
        if name not in txt:
            return ""
        return txt.split(name, 1)[1].split("[", 1)[0]
    md = sec("[Metadata]")
    title = next((l.split(":", 1)[1].strip() for l in md.splitlines()
                  if l.startswith("Title:")), "Unknown")
    artist = next((l.split(":", 1)[1].strip() for l in md.splitlines()
                   if l.startswith("Artist:")), "Unknown")
    gen = sec("[General]")
    audio = next((l.split(":", 1)[1].strip() for l in gen.splitlines()
                  if l.startswith("AudioFilename:")), None)
    tp = sec("[TimingPoints]")
    R = []
    for line in tp.splitlines():
        p = line.strip().split(",")
        if len(p) < 7:
            continue
        try:
            off, bl, meter, unhin = float(p[0]), float(p[1]), float(p[2]), int(float(p[6]))
        except ValueError:
            continue
        if unhin == 1 and bl > 0:
            R.append((off, bl, int(meter) if meter else 4))
    return sorted(R), title, artist, audio




def gt_from_osz(osz_path):
    """GT redlines + metadata straight from an .osz (no benchmark cache needed):
    picks the .osu with the most redlines, like dataset_osz._read_set.
    Falls back to a direct TimingPoints parse if slider rejects the format."""
    from slider import Beatmap
    from redline.targets import redlines_from_beatmap
    best, best_R = None, []
    fb_R, fb_meta = [], None
    with zipfile.ZipFile(osz_path) as z:
        for n in (m for m in z.namelist() if m.lower().endswith(".osu")):
            raw = z.read(n).decode("utf-8-sig", "ignore")          # utf-8-sig strips BOM
            try:
                bm = Beatmap.parse(raw)
                R = redlines_from_beatmap(bm)
                if len(R) > len(best_R):
                    best, best_R = bm, R
            except Exception:
                R, title, artist, audio = _gt_from_text(raw)
                if len(R) > len(fb_R):
                    fb_R, fb_meta = (R, title, artist, audio)
    if best is not None:
        return sorted(best_R), best.title, best.artist, best.audio_filename
    if fb_R:
        return fb_R, fb_meta[1], fb_meta[2], fb_meta[3]
    raise ValueError("no parseable .osu in the set")


