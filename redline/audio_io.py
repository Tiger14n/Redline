"""Audio extraction from .osz archives (song member via .osu AudioFilename, else largest)."""
import io, zipfile
import numpy as np
from pydub import AudioSegment


def load_osz_audio(osz_path, audio_filename):
    """Decode the set's audio straight from the .osz in memory (no extraction)."""
    with zipfile.ZipFile(osz_path) as z:
        names = z.namelist()
        m = next((n for n in names if n.lower() == (audio_filename or "").lower()), None)
        if m is None:
            m = next((n for n in names if n.lower().endswith((".mp3", ".ogg", ".wav", ".m4a"))), None)
        if m is None:
            raise FileNotFoundError("no audio member in osz")
        data = z.read(m)
    seg = AudioSegment.from_file(io.BytesIO(data)).set_channels(1)
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    samples /= float(1 << (8 * seg.sample_width - 1))   # int -> [-1, 1]
    return samples, seg.frame_rate



def load_audio(path):
    """Load a plain audio file (mp3/ogg/wav/m4a/flac...) -> (mono float32, sample_rate)."""
    seg = AudioSegment.from_file(path).set_channels(1)
    sig = np.array(seg.get_array_of_samples()).astype(np.float32) / float(1 << (8 * seg.sample_width - 1))
    return sig, seg.frame_rate
