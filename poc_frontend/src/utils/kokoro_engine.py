import os
import io
import numpy as np
import torch
import soundfile as sf
from kokoro import KPipeline
from pydub import AudioSegment, effects
from pathlib import Path

class KokoroEngine:
    """
    Wrapper for Kokoro TTS with:
      - single or multiple voices with weights
      - optional tone effects via pitch shift, EQ boost, vibrato
      - streaming vs one-shot output
      - saving to file or returning raw bytes
    """

    def __init__(self,
                 lang_code: str = 'e',
                 voices_path: str = './kokoro-voices',
                 default_speed: float = 1.0,
                 max_semitone_shift: float = 4.0,
                 vibrato_rate: float = 5.0,
                 vibrato_depth: float = 0.002,
                 eq_gain_db: float = 6.0):
        """
        :param lang_code: Kokoro language code ('e' for Spanish, etc.)
        :param voices_path: folder where *.pt voice tensors live
        :param default_speed: default speech rate
        :param max_semitone_shift: semitones for tone=±1
        :param vibrato_rate: Hz for vibrato LFO
        :param vibrato_depth: fraction for vibrato amplitude
        :param eq_gain_db: gain for high‐shelf boost when tone>0
        """
        self.lang_code = lang_code
        self.voices_path = Path(voices_path)
        self.pipeline = KPipeline(lang_code=lang_code, repo_id='hexgrad/Kokoro-82M')
        self.default_speed = default_speed
        self.max_semitone_shift = max_semitone_shift
        self.vibrato_rate = vibrato_rate
        self.vibrato_depth = vibrato_depth
        self.eq_gain_db = eq_gain_db

    def load_voice(self, voice):
        if isinstance(voice, torch.Tensor):
            return voice
        path = self.voices_path / f"{voice}.pt"
        return torch.load(str(path), weights_only=True)

    def blend_voices(self, voices, weights):
        tensors = [self.load_voice(v) for v in voices]
        w = [float(x) for x in weights]
        total = sum(w)
        return sum(t * weight for t, weight in zip(tensors, w)) / total

    def _to_numpy(self, audio):
        """Asegura que `audio` sea un np.ndarray, no Tensor."""
        if isinstance(audio, torch.Tensor):
            return audio.detach().cpu().numpy()
        return audio

    def _numpy_to_audiosegment(self, audio: np.ndarray, sr: int) -> AudioSegment:
        if audio.dtype != np.int16:
            int16 = (audio * 32767).astype(np.int16)
        else:
            int16 = audio
        return AudioSegment(
            data=int16.tobytes(),
            sample_width=2,
            frame_rate=sr,
            channels=1
        )

    def _audiosegment_to_numpy(self, seg: AudioSegment) -> np.ndarray:
        arr = np.frombuffer(seg.raw_data, dtype=np.int16)
        return arr.astype(np.float32) / 32767

    def _pitch_shift_segment(self, seg: AudioSegment, semitones: float) -> AudioSegment:
        new_rate = int(seg.frame_rate * (2 ** (semitones / 12)))
        pitched = seg._spawn(seg.raw_data, overrides={"frame_rate": new_rate})
        return pitched.set_frame_rate(seg.frame_rate)

    def _high_shelf_boost_segment(self, seg: AudioSegment, gain_db: float) -> AudioSegment:
        filtered = effects.high_pass_filter(seg, cutoff=4000).apply_gain(gain_db)
        overlay_gain = gain_db - 2
        return seg.overlay(filtered - overlay_gain)

    def _add_vibrato_segment(self, seg: AudioSegment) -> AudioSegment:
        samples = np.array(seg.get_array_of_samples(), dtype=float)
        sr = seg.frame_rate
        t = np.arange(len(samples)) / sr
        mod = 1 + self.vibrato_depth * np.sin(2 * np.pi * self.vibrato_rate * t)
        warped = (samples * mod).astype(np.int16)
        return seg._spawn(warped.tobytes())

    def _run_stream(self, gen, tone, apply_pitch, apply_eq, apply_vibrato):
        for gs, ps, audio in gen:
            audio = self._to_numpy(audio)
            seg = self._numpy_to_audiosegment(audio, 24000)

            # saturar tone a [-1,1]
            tone = max(-10.0, min(10.0, tone))

            if tone != 0.0 and apply_pitch:
                semis = tone * self.max_semitone_shift
                seg = self._pitch_shift_segment(seg, semis)
            if tone > 0 and apply_eq:
                seg = self._high_shelf_boost_segment(seg, self.eq_gain_db * tone)
            if tone != 0.0 and apply_vibrato:
                seg = self._add_vibrato_segment(seg)

            audio_np = self._audiosegment_to_numpy(seg)
            yield gs, ps, audio_np
            
    def synthesize(self,
                   text: str,
                   voices,
                   weights: list[float] = None,
                   tone: float = 0.0,
                   speed: float = None,
                   split_pattern: str = r'\n+',
                   stream: bool = False,
                   output_path: str = None,
                   apply_eq: bool = False,
                   apply_vibrato: bool = False):
        """
        Convert text to speech.

        :param text: the text to synthesize
        :param voices: single voice name/tensor or list thereof
        :param weights: blend weights if multiple voices
        :param tone: float in [-1,+1]; controls semitone shift direction & magnitude
        :param speed: speech rate override
        :param split_pattern: regex for text splitting
        :param stream: if True, yields (gs, ps, audio) per segment
        :param output_path: if set & stream=False, writes WAV here
        :param apply_pitch: whether to apply pitch shift when tone!=0
        :param apply_eq: whether to apply high‐shelf boost when tone>0
        :param apply_vibrato: whether to apply vibrato when tone!=0
        :return: 
          - generator yielding (gs, ps, audio) if stream=True
          - output_path (str) if stream=False and output_path provided
          - raw WAV bytes if stream=False and no output_path
        """
        # 1) build voice tensor
        if isinstance(voices, (list, tuple)):
            w = weights or [1.0/len(voices)]*len(voices)
            voice_tensor = self.blend_voices(voices, w)
        else:
            if weights is not None:
                raise ValueError("Weights only allowed when voices is list.")
            voice_tensor = self.load_voice(voices)

        # 2) generate raw segments
        rate = speed or self.default_speed
        gen = self.pipeline(text, voice=voice_tensor, speed=rate, split_pattern=split_pattern)

        # 3) streaming mode
        if stream:
            self._run_stream(gen, tone, apply_eq, apply_vibrato)
            return

        # 4) one-shot: collect & process
        processed = []
        for _, _, audio in gen:
            audio = self._to_numpy(audio)
            seg = self._numpy_to_audiosegment(audio, 24000)

            tone = max(-1.0, min(1.0, tone))
            if tone != 0.0:
                semis = tone * self.max_semitone_shift
                seg = self._pitch_shift_segment(seg, semis)
            if tone > 0 and apply_eq:
                seg = self._high_shelf_boost_segment(seg, self.eq_gain_db * tone)
            if tone != 0.0 and apply_vibrato:
                seg = self._add_vibrato_segment(seg)

            processed.append(self._audiosegment_to_numpy(seg))

        full = np.concatenate(processed, axis=0)
        sr = 24000

        # 5) output
        if output_path:
            sf.write(output_path, full, sr)
            return output_path

        buf = io.BytesIO()
        sf.write(buf, full, sr, format='WAV', subtype='PCM_16')
        return buf.getvalue()