from pathlib import Path
from pydub import AudioSegment, effects
import numpy as np

def pitch_shift(
    input_path: Path,
    semitones: float,
    output_path: Path | None = None
) -> Path:
    """
    Shift the pitch of an audio file by `semitones`.
    Returns the Path to the new WAV.
    """
    sound = AudioSegment.from_file(input_path)
    # Calculate new frame rate
    new_rate = int(sound.frame_rate * (2 ** (semitones / 12)))
    # Create a “spawned” sound at the new rate
    pitched = sound._spawn(sound.raw_data, overrides={"frame_rate": new_rate})
    # Resample back to original frame rate so it plays back at normal speed
    pitched = pitched.set_frame_rate(sound.frame_rate)

    if output_path is None:
        # TODO: get rid of intermediate paths
        stem = input_path.stem  # + f"_pitch{semitones:+}"
        output_path = input_path.with_name(stem).with_suffix(".wav")

    pitched.export(output_path, format="wav")
    return output_path


def high_shelf_boost(input_path: str | Path, gain_db: float = 6.0) -> str:
    """
    Applies a gentle high‐shelf boost above ~4kHz for sparkle,
    overlays it subtly, and writes out a new file.
    """
    input_path = Path(input_path)
    sound = AudioSegment.from_file(input_path)

    # 1) isolate high frequencies
    filtered = effects.high_pass_filter(sound, cutoff=4000).apply_gain(gain_db)

    # 2) blend back with original (filtered is louder by gain_db, so drop by a bit)
    overlay_gain = gain_db - 2
    combined = sound.overlay(filtered - overlay_gain)

    # 3) construct output filename
    out_name = f"{input_path.stem}_eq{int(gain_db)}dB.wav"
    output_path = input_path.with_name(out_name)

    # 4) export
    combined.export(output_path, format="wav")
    return str(output_path)

def add_vibrato(sound: AudioSegment, rate=5.0, depth=0.002):
    samples = np.array(sound.get_array_of_samples(), dtype=float)
    sr = sound.frame_rate
    t = np.arange(len(samples)) / sr
    mod = 1 + depth * np.sin(2 * np.pi * rate * t)
    warped = (samples * mod).astype(np.int16)
    vi = sound._spawn(warped.tobytes())
    return vi

def _post_process(wav_path: str, speech_tone=0) -> str:
    """
    Apply teen-girl FX: gentle pitch up, EQ sparkle, vibrato,
    write out a final file and return its path.
    """
    # 1) cambio de pitch
    # TODO: get rid of intermediate paths
    shifted = pitch_shift(wav_path, semitones=speech_tone)
    # 2) vibrato (si quieres)
    segment = add_vibrato(AudioSegment.from_file(shifted))
    final_path = wav_path.with_name(f"{wav_path.stem}_final.wav")
    segment.export(final_path, format="wav")
    return str(final_path)